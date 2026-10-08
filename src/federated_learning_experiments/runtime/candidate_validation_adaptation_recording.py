"""候補検証の到達時の確定と未完了の終端回収を適応記録に写し、記録ownerへ一件追加する。"""

from federated_learning_experiments.evaluation.adaptation_record_store import (
    AdaptationOutcome,
    AdaptationRecord,
    AdaptationRecordStore,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    TrainingModelAssignmentChange,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.incomplete_post_alarm_candidate_validation_decision_record import (
    IncompletePostAlarmCandidateValidationDecisionRecord,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record import (
    PostAlarmCandidateValidationDecisionRecord,
)
from federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization import (
    IncompletePostAlarmCandidateValidationFinalization,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import (
    PostAlarmCandidateValidationCompletion,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import (
    PostAlarmCandidateValidationResolution,
)

ADAPTATION_OUTCOME_BY_VALIDATION_RESOLUTION_OUTCOME: dict[str, AdaptationOutcome] = {
    "candidate_adopted_as_new_model": "post_alarm_validation_candidate_adopted",
    "held_reference_model_reused": "post_alarm_validation_held_model_reused",
    "current_model_maintained": "post_alarm_validation_current_model_maintained",
    "candidate_rejected": "post_alarm_validation_candidate_rejected",
}


def record_completed_candidate_validation(
    *,
    validation_completion: PostAlarmCandidateValidationCompletion,
    adaptation_record_store: AdaptationRecordStore,
) -> AdaptationRecord:
    if type(validation_completion) is not PostAlarmCandidateValidationCompletion:
        raise TypeError(
            "validation_completion must be exact PostAlarmCandidateValidationCompletion"
        )
    if type(adaptation_record_store) is not AdaptationRecordStore:
        raise TypeError("adaptation_record_store must be exact AdaptationRecordStore")
    # 上流の完了情報・判定記録・確定結果・変更記録はconstructorでfieldを検査しない。
    # 記録へ写す値と、写す値を決める比較に使う値を、履歴の更新より前にここで確かめる。
    decision_record = validation_completion.decision_record
    if type(decision_record) is not PostAlarmCandidateValidationDecisionRecord:
        raise TypeError("decision_record must be exact PostAlarmCandidateValidationDecisionRecord")
    validation_resolution = validation_completion.validation_resolution
    if type(validation_resolution) is not PostAlarmCandidateValidationResolution:
        raise TypeError(
            "validation_resolution must be exact PostAlarmCandidateValidationResolution"
        )
    resolution_outcome = validation_resolution.resolution_outcome
    if type(resolution_outcome) is not str:
        raise TypeError("resolution_outcome must be builtin str")
    if resolution_outcome not in ADAPTATION_OUTCOME_BY_VALIDATION_RESOLUTION_OUTCOME:
        raise ValueError("resolution_outcome must be a declared validation resolution outcome")
    previous_training_model_id = validation_completion.previous_training_model_id
    if type(previous_training_model_id) is not int:
        raise TypeError("previous_training_model_id must be builtin int")
    if type(validation_resolution.assigned_model_id) is not int:
        raise TypeError("assigned_model_id must be builtin int")
    current_training_model_id = previous_training_model_id
    training_model_assignment_change = validation_resolution.training_model_assignment_change
    if training_model_assignment_change is not None:
        if type(training_model_assignment_change) is not TrainingModelAssignmentChange:
            raise TypeError(
                "training_model_assignment_change must be exact TrainingModelAssignmentChange"
            )
        if (
            type(training_model_assignment_change.previous_model_id) is not int
            or type(training_model_assignment_change.current_model_id) is not int
        ):
            raise TypeError("assignment change model IDs must be builtin int")
        if training_model_assignment_change.previous_model_id != previous_training_model_id:
            raise ValueError("assignment change must start from the previous training model")
        # 変更記録は実際の変更だけを表す。前後が同じ記録を許すと、維持・棄却の結果と
        # 組み合わせた入力が、記録のID整合の検査を通過してしまう。
        if (
            training_model_assignment_change.current_model_id
            == training_model_assignment_change.previous_model_id
        ):
            raise ValueError("assignment change must move to a different training model")
        current_training_model_id = training_model_assignment_change.current_model_id
    # 保留標本の帰属先は、どの結果でも確定後の学習帰属と同じモデルになる。
    if validation_resolution.assigned_model_id != current_training_model_id:
        raise ValueError("assigned_model_id must be the training model after the resolution")
    adaptation_record = AdaptationRecord(
        adaptation_sample_index=decision_record.resolution_sample_index,
        detector_name=decision_record.detector_name,
        adaptation_outcome=ADAPTATION_OUTCOME_BY_VALIDATION_RESOLUTION_OUTCOME[resolution_outcome],
        previous_training_model_id=previous_training_model_id,
        current_training_model_id=current_training_model_id,
        estimated_change_point_sample_index=validation_completion.estimated_change_point_sample_index,
        detection_episode_id=validation_completion.detection_episode_id,
    )
    adaptation_record_store.append_adaptation_record(adaptation_record=adaptation_record)
    return adaptation_record


def record_incomplete_candidate_validation_finalization(
    *,
    incomplete_validation_finalization: IncompletePostAlarmCandidateValidationFinalization,
    adaptation_record_store: AdaptationRecordStore,
) -> AdaptationRecord:
    if (
        type(incomplete_validation_finalization)
        is not IncompletePostAlarmCandidateValidationFinalization
    ):
        raise TypeError(
            "incomplete_validation_finalization must be exact IncompletePostAlarmCandidateValidationFinalization"
        )
    if type(adaptation_record_store) is not AdaptationRecordStore:
        raise TypeError("adaptation_record_store must be exact AdaptationRecordStore")
    decision_record = incomplete_validation_finalization.decision_record
    if type(decision_record) is not IncompletePostAlarmCandidateValidationDecisionRecord:
        raise TypeError(
            "decision_record must be exact IncompletePostAlarmCandidateValidationDecisionRecord"
        )
    # 終端回収は学習帰属を変えない。変更前後とも回収時点の学習帰属ID。
    adaptation_record = AdaptationRecord(
        adaptation_sample_index=decision_record.finalization_sample_index,
        detector_name=decision_record.detector_name,
        adaptation_outcome="post_alarm_validation_incomplete_candidate_rejected",
        previous_training_model_id=incomplete_validation_finalization.current_training_model_id,
        current_training_model_id=incomplete_validation_finalization.current_training_model_id,
        estimated_change_point_sample_index=incomplete_validation_finalization.estimated_change_point_sample_index,
        detection_episode_id=incomplete_validation_finalization.detection_episode_id,
    )
    adaptation_record_store.append_adaptation_record(adaptation_record=adaptation_record)
    return adaptation_record
