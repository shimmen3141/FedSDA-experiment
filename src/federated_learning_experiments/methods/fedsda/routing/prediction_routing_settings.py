"""予測混合方式・routing状態管理方針・share時間尺度の固定条件。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)


@dataclass(frozen=True, kw_only=True)
class PredictionRoutingSettings:
    """予測処理の状態を持たず、構築時に正式名と時間尺度を検証する。"""

    prediction_routing_strategy: str = field(
        metadata={"allowed_parameter_values": ("switching_fixed_share_mixture",)},
    )
    prediction_mixture_activation_policy: str = field(
        metadata={"allowed_parameter_values": ("always",)},
    )
    post_aggregation_routing_recalibration_policy: str = field(
        metadata={"allowed_parameter_values": ("fifo_loss_replay",)},
    )
    routing_reset_on_assignment_change_policy: str = field(
        metadata={"allowed_parameter_values": ("restart_adahedge_preserve_switching",)},
    )
    switching_share_horizon_samples: int = field(
        metadata={
            "parameter_unit": "sample/client",
            "minimum_allowed_value": 2,
            "minimum_value_is_inclusive": True,
        },
    )

    def __post_init__(self) -> None:
        validate_settings_field_values(self)
