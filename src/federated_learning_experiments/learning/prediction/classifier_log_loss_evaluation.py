"""分類器の標本ごとの対数損失（−log 真のクラスの確率）を、有界の損失から計算する。"""

from torch import Tensor, clamp, log

from federated_learning_experiments.learning.models.model_architecture_classifier import (
    ModelArchitectureClassifier,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
)

# 真のクラスの確率の下限（対数損失を有限に保つ）。
MINIMUM_TRUE_CLASS_PROBABILITY = 1e-12


def evaluate_classifier_per_sample_log_losses(
    *,
    classifier: ModelArchitectureClassifier,
    input_features: Tensor,
    observed_class_labels: Tensor,
) -> Tensor:
    """標本ごとの対数損失（nats）。有界の損失（1−真のクラスの確率）から、−log(max(1−有界の損失, 下限))。"""
    bounded_losses = evaluate_classifier_per_sample_bounded_losses(
        classifier=classifier,
        input_features=input_features,
        observed_class_labels=observed_class_labels,
    )
    return -log(clamp(1.0 - bounded_losses, min=MINIMUM_TRUE_CLASS_PROBABILITY))
