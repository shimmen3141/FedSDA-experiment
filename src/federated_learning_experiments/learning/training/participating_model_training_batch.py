"""一回の共同更新へ参加するモデル・optimizer・batchの借用記録。"""

from dataclasses import dataclass

from torch import Tensor
from torch.optim import Optimizer

from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)


@dataclass(frozen=True, kw_only=True)
class ParticipatingModelTrainingBatch:
    """入力参照を束ね、更新関数で毎回整合性を検査する。"""

    classifier: ResidualAdapterClassifier
    concept_specific_parameter_optimizer: Optimizer
    input_features: Tensor
    observed_class_labels: Tensor
