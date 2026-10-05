"""保有モデルのIDと学習実体への参照を束ねる。"""

from dataclasses import dataclass

from torch.optim import Optimizer

from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)


@dataclass(frozen=True, kw_only=True)
class HeldModelTrainingBinding:
    """分類器と個別optimizerの所有を外側に残した借用記録。"""

    model_id: int
    classifier: ResidualAdapterClassifier
    concept_specific_parameter_optimizer: Optimizer
