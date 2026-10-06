"""準備済み分類器を一回呼び、観測ラベルへの標本別有界損失を返す。"""

from torch import Tensor, abs, float32, isfinite, no_grad, softmax, strided, trunc

from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)


def _validate_classifier_bounded_loss_inputs(
    *,
    classifier: ResidualAdapterClassifier,
    input_features: Tensor,
    observed_class_labels: Tensor,
) -> None:
    """forward前にモデルの実行環境と観測batchを検査する。"""
    if type(classifier) is not ResidualAdapterClassifier:
        raise TypeError("classifierはexact ResidualAdapterClassifierが必要です。")
    if type(classifier.class_count) is not int or classifier.class_count < 2:
        raise ValueError("classifier.class_countは2以上のbuiltin intが必要です。")
    for classifier_parameter in classifier.parameters():
        if (
            classifier_parameter.device.type != "cpu"
            or classifier_parameter.dtype != float32
            or classifier_parameter.layout != strided
            or classifier_parameter.is_nested
        ):
            raise ValueError("classifierのparameterはCPU float32 stridedが必要です。")
    if not isinstance(input_features, Tensor):
        raise TypeError("input_featuresはTensorが必要です。")
    if (
        input_features.device.type != "cpu"
        or input_features.dtype != float32
        or input_features.layout != strided
        or input_features.is_nested
    ):
        raise ValueError("input_featuresはCPU float32 strided Tensorが必要です。")
    if (
        input_features.ndim != 2
        or input_features.shape[0] == 0
        or input_features.shape[1] != classifier.feature_extractor.input_feature_count
    ):
        raise ValueError("input_featuresは非空で特徴数が一致するshape[N,F]が必要です。")
    if not isfinite(input_features).all().item():
        raise ValueError("input_featuresは有限値が必要です。")
    if not isinstance(observed_class_labels, Tensor):
        raise TypeError("observed_class_labelsはTensorが必要です。")
    if (
        observed_class_labels.device.type != "cpu"
        or observed_class_labels.dtype != float32
        or observed_class_labels.layout != strided
        or observed_class_labels.is_nested
    ):
        raise ValueError("observed_class_labelsはCPU float32 strided Tensorが必要です。")
    if observed_class_labels.shape != (input_features.shape[0], 1):
        raise ValueError("observed_class_labelsは同じ標本数のshape[N,1]が必要です。")
    if not isfinite(observed_class_labels).all().item():
        raise ValueError("observed_class_labelsは有限値が必要です。")
    if (observed_class_labels != trunc(observed_class_labels)).any().item():
        raise ValueError("observed_class_labelsは整数のクラス値が必要です。")
    if (observed_class_labels < 0).any().item() or (
        observed_class_labels >= classifier.class_count
    ).any().item():
        raise ValueError("observed_class_labelsは0以上class_count未満が必要です。")


def _validate_classifier_outputs_for_bounded_loss(
    *, classifier_outputs: Tensor, batch_sample_count: int, class_count: int
) -> None:
    """分類出力を補正せず、損失計算に必要な契約を検査する。"""
    if not isinstance(classifier_outputs, Tensor):
        raise TypeError("classifier_outputsはTensorが必要です。")
    if (
        classifier_outputs.device.type != "cpu"
        or classifier_outputs.dtype != float32
        or classifier_outputs.layout != strided
        or classifier_outputs.is_nested
    ):
        raise ValueError("classifier_outputsはCPU float32 strided Tensorが必要です。")
    expected_output_count = 1 if class_count == 2 else class_count
    if classifier_outputs.shape != (batch_sample_count, expected_output_count):
        raise ValueError("classifier_outputsの標本数と出力列数が一致しません。")
    if not isfinite(classifier_outputs).all().item():
        raise ValueError("classifier_outputsは有限値が必要です。")
    if class_count == 2 and (
        (classifier_outputs < 0).any().item() or (classifier_outputs > 1).any().item()
    ):
        raise ValueError("classifier_outputsの二値確率は0以上1以下が必要です。")


@no_grad()
def evaluate_classifier_per_sample_bounded_losses(
    *,
    classifier: ResidualAdapterClassifier,
    input_features: Tensor,
    observed_class_labels: Tensor,
) -> Tensor:
    """学習状態を保持した一forwardから、標本順の独立した損失を返す。"""
    _validate_classifier_bounded_loss_inputs(
        classifier=classifier,
        input_features=input_features,
        observed_class_labels=observed_class_labels,
    )
    classifier_outputs = classifier(input_features)
    _validate_classifier_outputs_for_bounded_loss(
        classifier_outputs=classifier_outputs,
        batch_sample_count=input_features.shape[0],
        class_count=classifier.class_count,
    )
    if classifier.class_count == 2:
        return abs(classifier_outputs - observed_class_labels).reshape(-1)
    class_probabilities = softmax(classifier_outputs, dim=1)
    correct_class_probabilities = class_probabilities.gather(
        1, observed_class_labels.reshape(-1).long().unsqueeze(1)
    ).squeeze(1)
    return 1.0 - correct_class_probabilities
