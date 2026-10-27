"""FedDriftの検出バッチを評価し、最小損失とドリフトを判断する。"""

from dataclasses import dataclass
from math import isfinite

from torch import Tensor

from federated_learning_experiments.learning.models.model_architecture_classifier import (
    INDEPENDENT_MODELS_ARCHITECTURE_NAME,
    ModelArchitectureClassifier,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.prediction.class_probability_calculations import (
    predict_class_labels_from_prediction_scores,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
    evaluate_classifier_per_sample_bounded_losses_and_outputs,
    validate_classifier_bounded_loss_inputs,
)


def _validate_bounded_loss(value: float) -> None:
    if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("平均有界損失は0以上1以下の有限なbuiltin floatが必要です。")


@dataclass(frozen=True, kw_only=True)
class FeddriftDetectionBatchAssessment:
    """保有順の平均基準値と、最小モデルの選択・ドリフトの判断。

    lossの既存fieldには、選んだ基準の値（有界損失または誤り率）が入る。
    """

    model_mean_bounded_losses: tuple[tuple[int, float], ...]
    minimum_loss_model_id: int
    minimum_mean_bounded_loss: float
    drift_detected: bool
    batch_sample_count: int

    def __post_init__(self) -> None:
        if type(self.model_mean_bounded_losses) is not tuple or not self.model_mean_bounded_losses:
            raise ValueError("モデル別平均損失は非空のtupleが必要です。")
        seen_model_ids: set[int] = set()
        for item in self.model_mean_bounded_losses:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("モデル別平均損失の各要素はIDと損失のtupleが必要です。")
            model_id, mean_loss = item
            if type(model_id) is not int or model_id in seen_model_ids:
                raise ValueError("モデルIDは重複しないbuiltin intが必要です。")
            seen_model_ids.add(model_id)
            _validate_bounded_loss(mean_loss)
        if type(self.minimum_loss_model_id) is not int:
            raise ValueError("最小損失のモデルIDはbuiltin intが必要です。")
        _validate_bounded_loss(self.minimum_mean_bounded_loss)
        first_minimum = min(self.model_mean_bounded_losses, key=lambda item: item[1])
        if first_minimum != (self.minimum_loss_model_id, self.minimum_mean_bounded_loss):
            raise ValueError("選択モデルと損失は、保有順で最初の最小平均損失と一致させてください。")
        if type(self.drift_detected) is not bool:
            raise ValueError("ドリフトの判断はbuiltin boolが必要です。")
        if type(self.batch_sample_count) is not int or self.batch_sample_count < 1:
            raise ValueError("バッチの標本数は1以上のbuiltin intが必要です。")


def assess_feddrift_detection_batch(
    *,
    held_models: tuple[tuple[int, ModelArchitectureClassifier], ...],
    input_features: Tensor,
    observed_class_labels: Tensor,
    previous_minimum_mean_bounded_loss: float | None,
    loss_increase_threshold: float,
    model_comparison_criterion: str = "bounded_loss",
) -> FeddriftDetectionBatchAssessment:
    """全モデルの評価前に入力を検査し、旧と同じ保有順・平均・比較で判断する。"""
    if type(model_comparison_criterion) is not str or model_comparison_criterion not in (
        "bounded_loss",
        "classification_error_rate",
    ):
        raise ValueError("モデルの比較の基準が未対応です。")
    if type(held_models) is not tuple or not held_models:
        raise ValueError("保有モデルは非空のtupleが必要です。")
    if previous_minimum_mean_bounded_loss is not None:
        _validate_bounded_loss(previous_minimum_mean_bounded_loss)
    if (
        type(loss_increase_threshold) is not float
        or not isfinite(loss_increase_threshold)
        or loss_increase_threshold < 0.0
    ):
        raise ValueError("損失増加の閾値は0以上の有限なbuiltin floatが必要です。")
    seen_model_ids: set[int] = set()
    model_dimensions: tuple[int, int] | None = None
    for held_model in held_models:
        if type(held_model) is not tuple or len(held_model) != 2:
            raise ValueError("保有モデルの各要素はIDと分類器のtupleが必要です。")
        model_id, classifier = held_model
        if type(model_id) is not int or model_id in seen_model_ids:
            raise ValueError("モデルIDは重複しないbuiltin intが必要です。")
        seen_model_ids.add(model_id)
        if type(classifier) is not ModelArchitectureClassifier:
            raise TypeError("分類器はexact ModelArchitectureClassifierが必要です。")
        if (
            type(classifier.model_architecture_settings) is not ModelArchitectureSettings
            or classifier.model_architecture_settings.model_architecture_name
            != INDEPENDENT_MODELS_ARCHITECTURE_NAME
        ):
            raise ValueError("FedDriftの判定には共有部を持たないモデル構造が必要です。")
        dimensions = (classifier.input_feature_count, classifier.class_count)
        if model_dimensions is not None and dimensions != model_dimensions:
            raise ValueError("保有モデルの入力幅とクラス数は一致させてください。")
        model_dimensions = dimensions
        validate_classifier_bounded_loss_inputs(
            classifier=classifier,
            input_features=input_features,
            observed_class_labels=observed_class_labels,
        )
    # 全モデルの入力検査が終わってから、保有順の評価を始める。
    model_mean_bounded_losses: list[tuple[int, float]] = []
    for model_id, classifier in held_models:
        if model_comparison_criterion == "classification_error_rate":
            _, outputs = evaluate_classifier_per_sample_bounded_losses_and_outputs(
                classifier=classifier,
                input_features=input_features,
                observed_class_labels=observed_class_labels,
            )
            predictions = predict_class_labels_from_prediction_scores(
                prediction_scores=outputs,
                class_count=classifier.class_count,
            )
            losses = (predictions != observed_class_labels).float()
        else:
            losses = evaluate_classifier_per_sample_bounded_losses(
                classifier=classifier,
                input_features=input_features,
                observed_class_labels=observed_class_labels,
            )
        model_mean_bounded_losses.append((model_id, float(losses.mean().item())))
    minimum_loss_model_id, minimum_mean_bounded_loss = min(
        model_mean_bounded_losses, key=lambda item: item[1]
    )
    return FeddriftDetectionBatchAssessment(
        model_mean_bounded_losses=tuple(model_mean_bounded_losses),
        minimum_loss_model_id=minimum_loss_model_id,
        minimum_mean_bounded_loss=minimum_mean_bounded_loss,
        drift_detected=(
            previous_minimum_mean_bounded_loss is not None
            and minimum_mean_bounded_loss
            > previous_minimum_mean_bounded_loss + loss_increase_threshold
        ),
        batch_sample_count=input_features.shape[0],
    )
