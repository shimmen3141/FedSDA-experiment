"""準備済み分類器の全parameter現在値を独立コピーする。"""

from torch import Tensor, float32, isfinite, no_grad, strided

from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)


def _validate_classifier_parameter_snapshot_inputs(
    *, classifier: ResidualAdapterClassifier
) -> None:
    if type(classifier) is not ResidualAdapterClassifier:
        raise TypeError("classifierはexact ResidualAdapterClassifierが必要です。")
    for parameter_name, parameter_values in classifier.named_parameters():
        if (
            parameter_values.device.type != "cpu"
            or parameter_values.dtype != float32
            or parameter_values.layout != strided
            or parameter_values.is_nested
        ):
            raise ValueError(f"classifier.{parameter_name}はCPU float32 stridedが必要です。")
        if not isfinite(parameter_values).all().item():
            raise ValueError(f"classifier.{parameter_name}は有限値が必要です。")


@no_grad()
def snapshot_classifier_parameters(*, classifier: ResidualAdapterClassifier) -> dict[str, Tensor]:
    """native順を保持し、モデル・勾配から独立した値を返す。"""
    _validate_classifier_parameter_snapshot_inputs(classifier=classifier)
    classifier_parameter_values = classifier.state_dict()
    parameter_snapshot = {
        parameter_name: parameter_values.detach().clone()
        for parameter_name, parameter_values in classifier_parameter_values.items()
    }
    return parameter_snapshot
