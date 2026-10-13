"""保有モデルのIDと学習実体への参照を束ねる。"""

from dataclasses import dataclass

from torch.optim import Optimizer

from federated_learning_experiments.learning.models.model_architecture_classifier import (
    ModelArchitectureClassifier,
)


@dataclass(frozen=True, kw_only=True)
class HeldModelTrainingBinding:
    """分類器と個別optimizerの所有を外側に残した借用記録。"""

    model_id: int
    classifier: ModelArchitectureClassifier
    concept_specific_parameter_optimizer: Optimizer
