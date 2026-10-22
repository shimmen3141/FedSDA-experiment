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

# 警報の後の変化区間の長さと帰属先の選び方。
# 検出器の推定した変化の長さと、履歴の平均損失からの増加の許容量で選ぶ（基準版9までの扱い）。
DETECTOR_CHANGE_SPAN_AND_MEAN_LOSS_INCREASE_TOLERANCE_POLICY_NAME = (
    "detector_change_span_and_mean_loss_increase_tolerance"
)
# 最小記述長の原理で選ぶ（基準版10の既定）。
MINIMUM_DESCRIPTION_LENGTH_POLICY_NAME = "minimum_description_length"
ALARM_INTERVAL_ASSIGNMENT_POLICY_NAMES: tuple[str, ...] = (
    DETECTOR_CHANGE_SPAN_AND_MEAN_LOSS_INCREASE_TOLERANCE_POLICY_NAME,
    MINIMUM_DESCRIPTION_LENGTH_POLICY_NAME,
)


# 警報の後の最小記述長の選択で、「新しいモデル」の仮説の符号長を測る学習者の組。
# 概念ごとの部分を初期化し直した学習者だけ（基準版10までの扱い）。
REINITIALIZED_LEARNER_POLICY_NAME = "reinitialized_learner"
# 初期化し直した学習者と、現在のモデルの写しを初期化せずに学ばせるwarm startの学習者の、短い方＋ln 2
# （基準版11の既定）。
REINITIALIZED_AND_WARM_STARTED_LEARNERS_POLICY_NAME = "reinitialized_and_warm_started_learners"
NEW_MODEL_HYPOTHESIS_LEARNER_POLICY_NAMES: tuple[str, ...] = (
    REINITIALIZED_LEARNER_POLICY_NAME,
    REINITIALIZED_AND_WARM_STARTED_LEARNERS_POLICY_NAME,
)


def validate_new_model_hypothesis_learners(*, new_model_hypothesis_learners: str) -> None:
    """学習者の組が、builtin strの2つの値のどちらかであることを確かめる。"""
    if type(new_model_hypothesis_learners) is not str:
        raise TypeError("new_model_hypothesis_learners must be builtin str")
    if new_model_hypothesis_learners not in NEW_MODEL_HYPOTHESIS_LEARNER_POLICY_NAMES:
        raise ValueError("new_model_hypothesis_learners must be a declared value")


def validate_alarm_interval_assignment_policy(*, alarm_interval_assignment_policy: str) -> None:
    """選び方が、builtin strの2つの値のどちらかであることを確かめる。"""
    if type(alarm_interval_assignment_policy) is not str:
        raise TypeError("alarm_interval_assignment_policy must be builtin str")
    if alarm_interval_assignment_policy not in ALARM_INTERVAL_ASSIGNMENT_POLICY_NAMES:
        raise ValueError("alarm_interval_assignment_policy must be a declared value")


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
    # 警報の後の変化区間の長さと帰属先の選び方（`minimum_description_length_assignment_selection`の値）。
    alarm_interval_assignment_policy: str = field(
        metadata={"allowed_parameter_values": ALARM_INTERVAL_ASSIGNMENT_POLICY_NAMES},
    )
    # 警報の後の最小記述長の選択の、「新しいモデル」の仮説の学習者の組。最小記述長の原理で選ぶ扱いだけが使う。
    new_model_hypothesis_learners: str = field(
        metadata={"allowed_parameter_values": NEW_MODEL_HYPOTHESIS_LEARNER_POLICY_NAMES},
    )

    def __post_init__(self) -> None:
        validate_settings_field_values(self)
