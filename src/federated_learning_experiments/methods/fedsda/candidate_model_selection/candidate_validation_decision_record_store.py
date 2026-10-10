"""候補検証の判定記録（確定と、未完了の終端回収）を、起きた順に保持する。"""

from federated_learning_experiments.methods.fedsda.candidate_model_selection.incomplete_post_alarm_candidate_validation_decision_record import (
    IncompletePostAlarmCandidateValidationDecisionRecord,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record import (
    PostAlarmCandidateValidationDecisionRecord,
)


class CandidateValidationDecisionRecordStore:
    """判定記録を、足した順に所有する。記録は不変の値なので、写しを作らずに持つ。"""

    def __init__(self) -> None:
        self._decision_records: list[
            PostAlarmCandidateValidationDecisionRecord
            | IncompletePostAlarmCandidateValidationDecisionRecord
        ] = []

    def append_candidate_validation_decision_record(
        self,
        *,
        decision_record: PostAlarmCandidateValidationDecisionRecord
        | IncompletePostAlarmCandidateValidationDecisionRecord,
    ) -> None:
        if type(decision_record) not in (
            PostAlarmCandidateValidationDecisionRecord,
            IncompletePostAlarmCandidateValidationDecisionRecord,
        ):
            raise TypeError(
                "decision_record must be exact PostAlarmCandidateValidationDecisionRecord "
                "or IncompletePostAlarmCandidateValidationDecisionRecord"
            )
        self._decision_records.append(decision_record)

    def snapshot_candidate_validation_decision_records(
        self,
    ) -> tuple[
        PostAlarmCandidateValidationDecisionRecord
        | IncompletePostAlarmCandidateValidationDecisionRecord,
        ...,
    ]:
        return tuple(self._decision_records)
