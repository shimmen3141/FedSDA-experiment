"""モデルを呼び出さず、分類予測に必要な数値を計算する。"""

import math
from collections.abc import Mapping

import torch


def _validate_class_count(*, class_count: int) -> None:
    if type(class_count) is not int:
        raise TypeError("class_countはbool以外のbuiltin intにしてください。")
    if class_count < 2:
        raise ValueError("class_countは2以上にしてください。")


def _validate_prediction_tensor(
    *, prediction_tensor: torch.Tensor, class_count: int, require_probabilities: bool,
) -> None:
    if not isinstance(prediction_tensor, torch.Tensor):
        raise TypeError("prediction_tensorはTensorにしてください。")
    if prediction_tensor.device.type != 'cpu' or prediction_tensor.dtype != torch.float32:
        raise ValueError("prediction_tensorはCPU float32にしてください。")
    if prediction_tensor.layout != torch.strided:
        raise ValueError("prediction_tensorはdenseなstrided表にしてください。")
    if prediction_tensor.ndim != 2 or prediction_tensor.shape[0] == 0 or prediction_tensor.shape[1] != (1 if class_count == 2 else class_count):
        raise ValueError("prediction_tensorの形は二値[N,1]、多クラス[N,K]、N>0にしてください。")
    if not torch.isfinite(prediction_tensor).all().item():
        raise ValueError("prediction_tensorは有限値にしてください。")
    if require_probabilities:
        if not ((prediction_tensor >= 0) & (prediction_tensor <= 1)).all().item():
            raise ValueError("モデル別確率は0～1にしてください。")
        if class_count > 2 and not (torch.abs(prediction_tensor.sum(dim=1) - 1) <= 1e-6).all().item():
            raise ValueError("モデル別確率の行総和は1との差1e-6以内にしてください。")


def _validated_model_prediction_tensor_ids(
    *, model_tensors_by_model_id: Mapping[int, torch.Tensor], class_count: int,
    require_probabilities: bool,
) -> tuple[int, ...]:
    _validate_class_count(class_count=class_count)
    if not isinstance(model_tensors_by_model_id, Mapping):
        raise TypeError("モデル別TensorはMappingにしてください。")
    if not model_tensors_by_model_id:
        raise ValueError("モデル別Tensorは非空にしてください。")
    if any(type(model_id) is not int for model_id in model_tensors_by_model_id):
        raise TypeError("モデルIDはbool以外のbuiltin intにしてください。")
    model_ids = tuple(sorted(model_tensors_by_model_id))
    for model_id in model_ids:
        _validate_prediction_tensor(
            prediction_tensor=model_tensors_by_model_id[model_id],
            class_count=class_count, require_probabilities=require_probabilities,
        )
    batch_sample_count = model_tensors_by_model_id[model_ids[0]].shape[0]
    if any(model_tensors_by_model_id[model_id].shape[0] != batch_sample_count for model_id in model_ids):
        raise ValueError("全モデルの標本数Nを一致させてください。")
    return model_ids


def _validated_model_prediction_weights(
    *, prediction_weights_by_model_id: Mapping[int, float],
) -> dict[int, float]:
    if not isinstance(prediction_weights_by_model_id, Mapping):
        raise TypeError("prediction_weights_by_model_idはMappingにしてください。")
    if not prediction_weights_by_model_id:
        raise ValueError("予測重みは非空にしてください。")
    if any(type(model_id) is not int for model_id in prediction_weights_by_model_id):
        raise TypeError("モデルIDはbool以外のbuiltin intにしてください。")
    for prediction_weight in prediction_weights_by_model_id.values():
        if type(prediction_weight) not in (int, float):
            raise TypeError("予測重みはbool以外のbuiltin int/floatにしてください。")
        if not 0 <= prediction_weight <= 1 or not math.isfinite(prediction_weight):
            raise ValueError("予測重みは有限な0～1にしてください。")
    if abs(math.fsum(prediction_weights_by_model_id.values()) - 1) > 1e-12:
        raise ValueError("予測重みの総和は1との差1e-12以内にしてください。")
    return {model_id: float(prediction_weights_by_model_id[model_id]) for model_id in sorted(prediction_weights_by_model_id)}


@torch.no_grad()
def convert_model_outputs_to_prediction_probabilities(
    *, model_outputs_by_model_id: Mapping[int, torch.Tensor], class_count: int,
) -> dict[int, torch.Tensor]:
    """二値確率をコピーし、多クラスlogitをモデルごとに確率化する。"""
    model_ids = _validated_model_prediction_tensor_ids(
        model_tensors_by_model_id=model_outputs_by_model_id, class_count=class_count,
        require_probabilities=class_count == 2,
    )
    return {
        model_id: model_outputs_by_model_id[model_id].detach().clone() if class_count == 2
        else torch.softmax(model_outputs_by_model_id[model_id], dim=1)
        for model_id in model_ids
    }


def normalize_model_prediction_weights(
    *, prediction_weights_by_model_id: Mapping[int, float],
) -> dict[int, float]:
    """旧経路と同じ昇順の通常加算・個別除算で重みを正規化する。"""
    prediction_weights_by_model_id = _validated_model_prediction_weights(
        prediction_weights_by_model_id=prediction_weights_by_model_id,
    )
    total_prediction_weight = sum(prediction_weights_by_model_id.values())
    return {model_id: prediction_weight / total_prediction_weight for model_id, prediction_weight in prediction_weights_by_model_id.items()}
