"""帰属確定を保留するFIFOの、容量と、短い警報のあとの扱いに関する固定条件。"""

from dataclasses import dataclass, field

from federated_learning_experiments.core.settings_field_validation import (
    validate_settings_field_values,
)

# 短い警報（推定変化区間が最小件数より少ない警報）のあと、保留に何を残すか。
# 現行モデルへ取り込んだ前区間を保留から外し、変化区間を残す（FedSDAの既定）。
RELEASE_EARLIER_SAMPLES_ASSIGNED_TO_CURRENT_MODEL_POLICY_NAME = (
    "release_earlier_samples_assigned_to_current_model"
)
# 保留を全部残す（固定旧FedSDAとの照合用。取り込んだ前区間も残り、後で、もう一度取り込まれる）。
KEEP_ALL_PENDING_SAMPLES_POLICY_NAME = "keep_all_pending_samples"
PENDING_SAMPLE_RETENTION_AFTER_TOO_SHORT_CHANGE_INTERVAL_POLICIES = (
    RELEASE_EARLIER_SAMPLES_ASSIGNED_TO_CURRENT_MODEL_POLICY_NAME,
    KEEP_ALL_PENDING_SAMPLES_POLICY_NAME,
)


@dataclass(frozen=True, kw_only=True)
class TrainingDataAssignmentSettings:
    """バッファ内容や割当先を持たず、構築時に設定容量を検証する。"""

    pending_assignment_buffer_capacity_samples: int = field(
        metadata={
            "parameter_unit": "sample/client",
            "minimum_allowed_value": 1,
            "minimum_value_is_inclusive": True,
        },
    )

    pending_sample_retention_after_too_short_change_interval_policy: str = field(
        metadata={
            "allowed_parameter_values": (
                PENDING_SAMPLE_RETENTION_AFTER_TOO_SHORT_CHANGE_INTERVAL_POLICIES
            ),
        },
    )

    def __post_init__(self) -> None:
        validate_settings_field_values(self)
