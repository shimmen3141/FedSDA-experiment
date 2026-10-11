"""検出方式・監視対象・誤警報制御値の固定条件。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)


@dataclass(frozen=True, kw_only=True)
class LossChangeDetectionSettings:
    """検出器の実行状態を持たず、構築時に正式名とalphaの値域を検証する。"""

    drift_detector_name: str = field(
        metadata={"allowed_parameter_values": ("e_sr",)},
    )
    loss_monitoring_scope: str = field(
        metadata={
            "allowed_parameter_values": (
                # 全体の損失の系列と、正解クラス別の損失の系列を、固定の重みで混合する（提案構成）。
                "overall_and_true_class_losses",
                # 全体の損失の系列だけを監視する。
                "overall_loss",
            )
        },
    )
    e_sr_false_alarm_control_alpha: float = field(
        metadata={
            "parameter_unit": "dimensionless",
            "minimum_allowed_value": 0,
            "minimum_value_is_inclusive": False,
            "maximum_allowed_value": 1,
            "maximum_value_is_inclusive": False,
        },
    )

    def __post_init__(self) -> None:
        validate_settings_field_values(self)
