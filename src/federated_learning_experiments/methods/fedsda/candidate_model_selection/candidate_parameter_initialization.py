"""完成した候補初期parameter snapshotを選択・コピーする。"""

import math

import torch

from .candidate_parameter_initialization_settings import CandidateParameterInitializationSettings


def _validate_model_id(*, model_id: int, input_name: str) -> None:
    if type(model_id) is not int:
        raise TypeError(f"{input_name}はbool以外のbuiltin intが必要です。")


def _validate_parameter_snapshot(
    *,
    parameter_snapshot: dict[str, torch.Tensor],
    input_name: str,
) -> None:
    if type(parameter_snapshot) is not dict:
        raise TypeError(f"{input_name}はexact dictが必要です。")
    if not parameter_snapshot:
        raise ValueError(f"{input_name}は空でないparameter集合が必要です。")
    allowed_parameter_dtypes = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    )
    for parameter_name, parameter_value in parameter_snapshot.items():
        if type(parameter_name) is not str:
            raise TypeError(f"{input_name}のparameter名はexact strが必要です。")
        if not parameter_name:
            raise ValueError(f"{input_name}のparameter名は空文字を許可しません。")
        if type(parameter_value) is not torch.Tensor:
            raise TypeError(f"{input_name}[{parameter_name!r}]はexact torch.Tensorが必要です。")
        if parameter_value.device.type != "cpu":
            raise ValueError(f"{input_name}[{parameter_name!r}]はCPU Tensorが必要です。")
        if parameter_value.layout != torch.strided or parameter_value.is_nested:
            raise ValueError(f"{input_name}[{parameter_name!r}]はdense strided Tensorが必要です。")
        if parameter_value.dtype not in allowed_parameter_dtypes:
            raise ValueError(f"{input_name}[{parameter_name!r}]のdtypeは許可されていません。")
        if parameter_value.is_floating_point() or parameter_value.is_complex():
            if not torch.isfinite(parameter_value).all().item():
                raise ValueError(f"{input_name}[{parameter_name!r}]は有限値が必要です。")


def _validate_initialization_inputs(
    *,
    settings: CandidateParameterInitializationSettings,
    available_parameter_snapshots_by_model_id: dict[int, dict[str, torch.Tensor]],
    current_training_model_id: int,
    evaluated_mean_losses_by_model_id: tuple[tuple[int, float], ...],
) -> tuple[CandidateParameterInitializationSettings, dict[int, dict[str, torch.Tensor]]]:
    if type(settings) is not CandidateParameterInitializationSettings:
        raise TypeError("settingsはexact CandidateParameterInitializationSettingsが必要です。")
    validated_settings = CandidateParameterInitializationSettings(
        candidate_parameter_initialization_source=settings.candidate_parameter_initialization_source
    )
    _validate_model_id(model_id=current_training_model_id, input_name="current_training_model_id")
    if type(available_parameter_snapshots_by_model_id) is not dict:
        raise TypeError("available_parameter_snapshots_by_model_idはexact dictが必要です。")
    validated_parameter_snapshots_by_model_id = {}
    first_parameter_snapshot = None
    for model_id, parameter_snapshot in available_parameter_snapshots_by_model_id.items():
        _validate_model_id(
            model_id=model_id, input_name="available_parameter_snapshots_by_model_id"
        )
        input_name = f"available_parameter_snapshots_by_model_id[{model_id}]"
        _validate_parameter_snapshot(parameter_snapshot=parameter_snapshot, input_name=input_name)
        if first_parameter_snapshot is None:
            first_parameter_snapshot = parameter_snapshot
            parameter_names = tuple(first_parameter_snapshot)
        else:
            if set(parameter_snapshot) != set(parameter_names):
                raise ValueError(f"{input_name}のparameter名集合が一致しません。")
            for parameter_name in parameter_names:
                parameter_value = parameter_snapshot[parameter_name]
                first_parameter_value = first_parameter_snapshot[parameter_name]
                if (
                    parameter_value.shape != first_parameter_value.shape
                    or parameter_value.dtype != first_parameter_value.dtype
                    or parameter_value.device != first_parameter_value.device
                ):
                    raise ValueError(
                        f"{input_name}[{parameter_name!r}]のshape/dtype/deviceが一致しません。"
                    )
        validated_parameter_snapshots_by_model_id[model_id] = parameter_snapshot
    if type(evaluated_mean_losses_by_model_id) is not tuple:
        raise TypeError("evaluated_mean_losses_by_model_idはexact tupleが必要です。")
    seen_evaluated_model_ids = set()
    for evaluated_model_loss in evaluated_mean_losses_by_model_id:
        if type(evaluated_model_loss) is not tuple:
            raise TypeError("evaluated_mean_losses_by_model_idの要素はexact tupleが必要です。")
        if len(evaluated_model_loss) != 2:
            raise ValueError("evaluated_mean_losses_by_model_idの要素はIDとlossのpairが必要です。")
        evaluated_model_id, mean_loss = evaluated_model_loss
        _validate_model_id(
            model_id=evaluated_model_id, input_name="evaluated_mean_losses_by_model_id"
        )
        if evaluated_model_id in seen_evaluated_model_ids:
            raise ValueError("evaluated_mean_losses_by_model_idのIDが重複しています。")
        seen_evaluated_model_ids.add(evaluated_model_id)
        if evaluated_model_id not in validated_parameter_snapshots_by_model_id:
            raise ValueError("evaluated_mean_losses_by_model_idのIDが保有集合にありません。")
        if type(mean_loss) not in (int, float):
            raise TypeError(
                "evaluated_mean_losses_by_model_idのlossはbool以外のbuiltin int/floatが必要です。"
            )
        if not 0 <= mean_loss <= 1 or not math.isfinite(mean_loss):
            raise ValueError("evaluated_mean_losses_by_model_idのlossは有限の[0,1]値が必要です。")
    initialization_source = validated_settings.candidate_parameter_initialization_source
    if initialization_source == "assigned_training_model" or (
        initialization_source == "lowest_evaluated_mean_loss_model"
        and not evaluated_mean_losses_by_model_id
    ):
        if current_training_model_id not in validated_parameter_snapshots_by_model_id:
            raise ValueError("current_training_model_idが保有集合にありません。")
    return validated_settings, validated_parameter_snapshots_by_model_id


def _copy_parameter_snapshot(
    *,
    parameter_snapshot: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        parameter_name: parameter_value.detach().clone()
        for parameter_name, parameter_value in parameter_snapshot.items()
    }


def _average_parameter_snapshots(
    *,
    parameter_snapshots: tuple[dict[str, torch.Tensor], ...],
) -> dict[str, torch.Tensor] | None:
    if not parameter_snapshots:
        return None
    first_parameter_snapshot = parameter_snapshots[0]
    averaged_parameter_snapshot = {}
    for parameter_name in first_parameter_snapshot:
        parameter_values = [
            parameter_snapshot[parameter_name].detach()
            for parameter_snapshot in parameter_snapshots
        ]
        first_parameter_value = parameter_values[0]
        if first_parameter_value.is_floating_point() or first_parameter_value.is_complex():
            averaged_parameter_value = torch.stack(parameter_values).mean(dim=0)
            if not torch.isfinite(averaged_parameter_value).all().item():
                raise ValueError(
                    f"available_parameter_snapshots_by_model_idの平均[{parameter_name!r}]が非有限です。"
                )
        else:
            averaged_parameter_value = first_parameter_value
        averaged_parameter_snapshot[parameter_name] = averaged_parameter_value.detach().clone()
    return averaged_parameter_snapshot


@torch.no_grad()
def select_candidate_initial_parameter_snapshot(
    *,
    settings: CandidateParameterInitializationSettings,
    available_parameter_snapshots_by_model_id: dict[int, dict[str, torch.Tensor]],
    current_training_model_id: int,
    evaluated_mean_losses_by_model_id: tuple[tuple[int, float], ...],
) -> dict[str, torch.Tensor] | None:
    """全入力検査後、明示方式から独立した初期値だけを返す。"""
    validated_settings, validated_parameter_snapshots_by_model_id = _validate_initialization_inputs(
        settings=settings,
        available_parameter_snapshots_by_model_id=available_parameter_snapshots_by_model_id,
        current_training_model_id=current_training_model_id,
        evaluated_mean_losses_by_model_id=evaluated_mean_losses_by_model_id,
    )
    initialization_source = validated_settings.candidate_parameter_initialization_source
    if initialization_source == "equal_mean_of_available_models":
        return _average_parameter_snapshots(
            parameter_snapshots=tuple(validated_parameter_snapshots_by_model_id.values())
        )
    selected_model_id = current_training_model_id
    if (
        initialization_source == "lowest_evaluated_mean_loss_model"
        and evaluated_mean_losses_by_model_id
    ):
        selected_model_id = min(
            evaluated_mean_losses_by_model_id,
            key=lambda evaluated_model_loss: evaluated_model_loss[1],
        )[0]
    selected_parameter_snapshot = validated_parameter_snapshots_by_model_id[selected_model_id]
    return _copy_parameter_snapshot(parameter_snapshot=selected_parameter_snapshot)
