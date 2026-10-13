"""検出方式・監視対象・誤警報制御値と、合図を警報とする時機の固定条件。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)

# 検出器の合図があり、推定変化区間が最小件数より短いときに、警報とするかどうか。
# 合図があれば、すぐに警報とする（短い警報として処理し、検出器を再開する。既定。固定旧と同じ）。
RAISE_ALARM_IMMEDIATELY_POLICY_NAME = "raise_alarm_immediately"
# 推定変化区間が最小件数に達するまで、警報としない（検出器の証拠は残す。検出器がe-SRのときだけ選べる）。
WAIT_UNTIL_CHANGE_INTERVAL_REACHES_MINIMUM_POLICY_NAME = (
    "wait_until_change_interval_reaches_minimum"
)
SHORT_CHANGE_INTERVAL_ALARM_POLICIES = (
    RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
    WAIT_UNTIL_CHANGE_INTERVAL_REACHES_MINIMUM_POLICY_NAME,
)


@dataclass(frozen=True, kw_only=True)
class LossChangeDetectionSettings:
    """検出器の実行状態を持たず、構築時に正式名とalphaの値域を検証する。"""

    drift_detector_name: str = field(
        metadata={
            "allowed_parameter_values": (
                # 有界な損失の平均の増加を、e値（Shiryaev-Roberts型の和）で検定する（提案構成）。
                "e_sr",
                # 損失の窓を2つに分けたときの、平均の差で検定する（ADWIN）。
                "adwin",
            )
        },
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

    # ADWINの信頼度のパラメータ。e-SRでは使わない。
    adwin_confidence_delta: float = field(
        metadata={
            "parameter_unit": "dimensionless",
            "minimum_allowed_value": 0,
            "minimum_value_is_inclusive": False,
            "maximum_allowed_value": 1,
            "maximum_value_is_inclusive": False,
        },
    )
    # ADWINが、分割の検定を始める、窓の標本数。e-SRでは使わない。
    adwin_minimum_test_window_sample_count: int = field(
        metadata={
            "parameter_unit": "sample/client",
            "minimum_allowed_value": 1,
            "minimum_value_is_inclusive": True,
        },
    )

    short_change_interval_alarm_policy: str = field(
        metadata={"allowed_parameter_values": SHORT_CHANGE_INTERVAL_ALARM_POLICIES},
    )

    def __post_init__(self) -> None:
        validate_settings_field_values(self)
        # ADWINは、検知したときに窓の古い側を削るので、再開せずに待っても、合図が続かない。
        if (
            self.drift_detector_name == "adwin"
            and self.short_change_interval_alarm_policy
            == WAIT_UNTIL_CHANGE_INTERVAL_REACHES_MINIMUM_POLICY_NAME
        ):
            raise RunSettingsValidationError(
                configuration_parameter_name="short_change_interval_alarm_policy",
                specified_parameter_value=self.short_change_interval_alarm_policy,
                validation_failure_reason=(
                    "drift_detector_name=adwinでは、"
                    f"{RAISE_ALARM_IMMEDIATELY_POLICY_NAME}を指定してください。"
                ),
            )
