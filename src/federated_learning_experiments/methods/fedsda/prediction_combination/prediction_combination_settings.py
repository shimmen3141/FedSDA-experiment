"""予測混合方式・予測状態管理方針・share時間尺度の固定条件。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)


@dataclass(frozen=True, kw_only=True)
class PredictionCombinationSettings:
    """予測処理の状態を持たず、構築時に正式名と時間尺度を検証する。"""

    prediction_combination_strategy: str = field(
        metadata={"allowed_parameter_values": ("fixed_share_weighted_prediction",)},
    )
    prediction_mixture_activation_policy: str = field(
        metadata={"allowed_parameter_values": ("always",)},
    )
    prediction_weight_recalibration_after_aggregation_policy: str = field(
        metadata={"allowed_parameter_values": ("recompute_buffer_losses_and_replay_weight_updates",)},
    )
    prediction_state_reset_on_training_assignment_change_policy: str = field(
        metadata={"allowed_parameter_values": ("restart_adahedge_preserve_fixed_share_prediction_state",)},
    )
    fixed_share_weight_redistribution_time_scale_samples: int = field(
        metadata={
            "parameter_unit": "sample/client",
            "minimum_allowed_value": 2,
            "minimum_value_is_inclusive": True,
        },
    )

    def __post_init__(self) -> None:
        validate_settings_field_values(self)
