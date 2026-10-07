"""警報の変化区間へ区間評価の結果を適用し、再利用・維持・候補検証開始のいずれかを行う。"""

from dataclasses import dataclass

from federated_learning_experiments.learning.training.current_training_model_assignment import (
    TrainingModelAssignmentChange,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment import (
    AlarmIntervalModelReuseAssessment,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import (
    PostAlarmCandidateValidationSession,
)

ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES: tuple[str, ...] = (
    "alarm_interval_held_model_reused",
    "alarm_interval_current_model_maintained",
    "alarm_interval_candidate_validation_started",
)


@dataclass(frozen=True, kw_only=True)
class AlarmChangeIntervalResolution:
    """適用した結果種別、区間評価の情報、標本の吸収先、学習帰属の変更記録、開始したsession。"""

    resolution_outcome: str
    alarm_interval_reuse_assessment: AlarmIntervalModelReuseAssessment
    assigned_model_id: int | None
    training_model_assignment_change: TrainingModelAssignmentChange | None
    started_validation_session: PostAlarmCandidateValidationSession | None

    def __post_init__(self) -> None:
        if self.resolution_outcome not in ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES:
            raise ValueError("resolution_outcomeは正式な結果種別のいずれかが必要です。")
