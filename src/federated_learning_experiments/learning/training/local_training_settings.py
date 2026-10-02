"""ローカル学習のパラメータ更新と共有部の勾配統合に関する固定条件。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)


@dataclass(frozen=True, kw_only=True)
class LocalTrainingSettings:
    """共同学習と概念別勾配の統合方式を保持し、構築時に正式値を検証する。"""

    local_model_parameter_update_strategy: str = field(
        metadata={
            "allowed_parameter_values": ("joint_backbone_adapter_and_head_training",),
        },
    )
    shared_backbone_gradient_combination_strategy: str = field(
        metadata={
            "allowed_parameter_values": ("sample_weighted_mean_per_concept_gradients",),
        },
    )

    def __post_init__(self) -> None:
        validate_settings_field_values(self)
