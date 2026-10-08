"""進行中の候補検証sessionを高々1つ保持する。開始・進行・確定の判断は行わない。"""

from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import (
    PostAlarmCandidateValidationSession,
)


class CandidateValidationSessionHolder:
    """保持と解除だけを行う。sessionの中身は借用で、深い不変性は保証しない。"""

    def __init__(self) -> None:
        self._held_validation_session: PostAlarmCandidateValidationSession | None = None

    @property
    def held_validation_session(self) -> PostAlarmCandidateValidationSession | None:
        return self._held_validation_session

    def hold_validation_session(
        self, *, validation_session: PostAlarmCandidateValidationSession
    ) -> None:
        """空のときだけ保持する。進行中のsessionを黙って置き換えない。"""
        if type(validation_session) is not PostAlarmCandidateValidationSession:
            raise TypeError("validation_session must be exact PostAlarmCandidateValidationSession")
        if self._held_validation_session is not None:
            raise ValueError("a candidate validation session is already held")
        self._held_validation_session = validation_session

    def release_validation_session(self) -> PostAlarmCandidateValidationSession:
        """保持中のsessionを外して返す。保持していなければ拒否する。"""
        if self._held_validation_session is None:
            raise LookupError("no candidate validation session is held")
        released_validation_session = self._held_validation_session
        self._held_validation_session = None
        return released_validation_session
