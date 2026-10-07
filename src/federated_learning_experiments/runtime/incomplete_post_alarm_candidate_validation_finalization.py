"""件数不足の候補検証を終端で棄却し、保留標本を現在の学習先へ回収する。"""

from dataclasses import dataclass

from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.model_training_and_assignment_counts import (
    ModelTrainingAndAssignmentCountsStore,
)
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.incomplete_post_alarm_candidate_validation_decision_record import (
    IncompletePostAlarmCandidateValidationDecisionRecord,
)
from federated_learning_experiments.runtime.assigned_training_sample_absorption import (
    absorb_assigned_training_samples_into_held_model,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import (
    PostAlarmCandidateValidationSession,
)


@dataclass(frozen=True, kw_only=True)
class IncompletePostAlarmCandidateValidationFinalization:
    """回収成功後の記録と、呼出側の終端記録に必要な値だけを返す。"""

    decision_record: IncompletePostAlarmCandidateValidationDecisionRecord
    current_training_model_id: int
    estimated_change_point_sample_index: int | None
    detection_episode_id: int | None


def _validate_incomplete_validation_finalization_inputs(
    *,
    validation_session: PostAlarmCandidateValidationSession,
    processed_sample_count: int,
    current_training_model_assignment: CurrentTrainingModelAssignment,
) -> int:
    if type(validation_session) is not PostAlarmCandidateValidationSession:
        raise TypeError("validation_sessionはexact session型が必要です。")
    if type(current_training_model_assignment) is not CurrentTrainingModelAssignment:
        raise TypeError("current_training_model_assignmentはexact帰属owner型が必要です。")
    if type(processed_sample_count) is not int:
        raise TypeError("processed_sample_countはbool以外のbuiltin intが必要です。")
    if processed_sample_count < 0:
        raise ValueError("processed_sample_countは0以上が必要です。")
    collection_state_snapshot = (
        validation_session.post_alarm_candidate_loss_collection.get_state_snapshot()
    )
    if (
        collection_state_snapshot.validation_sample_count
        >= collection_state_snapshot.required_validation_sample_count
    ):
        raise ValueError("要求件数に到達した候補検証は不足用の終端確定対象ではありません。")
    return collection_state_snapshot.validation_sample_count


def finalize_incomplete_post_alarm_candidate_validation(
    *,
    validation_session: PostAlarmCandidateValidationSession | None,
    processed_sample_count: int,
    pending_assignment_sample_concept_ids: tuple[int | None, ...],
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    training_sample_store: ModelTrainingSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    current_training_model_assignment: CurrentTrainingModelAssignment,
) -> IncompletePostAlarmCandidateValidationFinalization | None:
    """sessionの解除・記録追加・通知は行わず、回収成功後にだけ確定情報を返す。"""
    if validation_session is None:
        return None
    validation_sample_count = _validate_incomplete_validation_finalization_inputs(
        validation_session=validation_session,
        processed_sample_count=processed_sample_count,
        current_training_model_assignment=current_training_model_assignment,
    )
    current_training_model_id = current_training_model_assignment.current_training_model_id
    decision_record = IncompletePostAlarmCandidateValidationDecisionRecord(
        proposal_sample_index=validation_session.proposal_sample_index,
        finalization_sample_index=max(
            validation_session.proposal_sample_index, processed_sample_count - 1
        ),
        detector_name=validation_session.detector_name,
        candidate_training_interval_sample_count=len(validation_session.training_input_features),
        validation_sample_count=validation_sample_count,
    )
    absorb_assigned_training_samples_into_held_model(
        model_id=current_training_model_id,
        assigned_training_samples=validation_session.pending_assignment_training_samples,
        assigned_sample_concept_ids=pending_assignment_sample_concept_ids,
        held_model_training_state_registry=held_model_training_state_registry,
        loss_statistics_store=loss_statistics_store,
        training_sample_store=training_sample_store,
        model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
    )
    return IncompletePostAlarmCandidateValidationFinalization(
        decision_record=decision_record,
        current_training_model_id=current_training_model_id,
        estimated_change_point_sample_index=validation_session.estimated_change_point_sample_index,
        detection_episode_id=validation_session.detection_episode_id,
    )
