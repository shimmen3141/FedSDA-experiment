"""警報時の保留標本への応答を、既存の準備・吸収・解決から組み立てる。"""

from dataclasses import dataclass

from federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals import (
    PreparedAlarmTrainingIntervals,
)
from federated_learning_experiments.runtime.alarm_change_interval_resolution import (
    ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES,
    AlarmChangeIntervalResolution,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import (
    PostAlarmCandidateValidationSession,
)

ALARM_BUFFER_RESPONSE_OUTCOMES: tuple[str, ...] = (
    "alarm_during_candidate_validation",
    "alarm_change_interval_too_short",
    *ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES,
)


@dataclass(frozen=True, kw_only=True)
class AlarmBufferResponse:
    """応答と借用session、呼出側のFIFO後始末判断を返す。"""

    response_outcome: str
    prepared_alarm_training_intervals: PreparedAlarmTrainingIntervals | None
    change_interval_resolution: AlarmChangeIntervalResolution | None
    active_validation_session: PostAlarmCandidateValidationSession | None

    def __post_init__(self) -> None:
        if (
            type(self.response_outcome) is not str
            or self.response_outcome not in ALARM_BUFFER_RESPONSE_OUTCOMES
        ):
            raise ValueError("response_outcome must be a declared value")
        if self.response_outcome == "alarm_during_candidate_validation":
            if (
                self.prepared_alarm_training_intervals is not None
                or self.change_interval_resolution is not None
                or type(self.active_validation_session) is not PostAlarmCandidateValidationSession
            ):
                raise ValueError("active response needs the unchanged session only")
        elif self.response_outcome == "alarm_change_interval_too_short":
            if (
                type(self.prepared_alarm_training_intervals) is not PreparedAlarmTrainingIntervals
                or self.change_interval_resolution is not None
                or self.active_validation_session is not None
            ):
                raise ValueError("insufficient response needs prepared intervals only")
        elif (
            type(self.prepared_alarm_training_intervals) is not PreparedAlarmTrainingIntervals
            or type(self.change_interval_resolution) is not AlarmChangeIntervalResolution
        ):
            raise ValueError("resolved response needs prepared intervals and resolution")
        elif (
            self.change_interval_resolution.resolution_outcome != self.response_outcome
            or self.active_validation_session
            is not self.change_interval_resolution.started_validation_session
        ):
            raise ValueError("response must match its actual resolution and session")

    @property
    def pending_assignment_buffer_should_be_cleared(self) -> bool:
        return self.response_outcome != "alarm_change_interval_too_short"
