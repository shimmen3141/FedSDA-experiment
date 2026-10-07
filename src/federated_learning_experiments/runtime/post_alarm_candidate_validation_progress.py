"""開始済み候補の標本観測と到達時の採否適用を組み立てる。"""

from dataclasses import dataclass
from math import isfinite

from torch import Tensor

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
from federated_learning_experiments.learning.training.temporary_model_id_allocation import (
    TemporaryModelIdAllocator,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import (
    CandidateModelTrainingAndAcceptanceSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import (
    evaluate_candidate_using_post_alarm_losses,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record import (
    PostAlarmCandidateValidationDecisionRecord,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import (
    PendingModelUploadState,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_resolution import (
    PostAlarmCandidateValidationResolution,
    apply_post_alarm_candidate_validation_resolution,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_sample_observation import (
    observe_post_alarm_candidate_validation_sample,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import (
    PostAlarmCandidateValidationSession,
)


@dataclass(frozen=True, kw_only=True)
class PostAlarmCandidateValidationCompletion:
    """適用済み判定と、呼出側の記録・通知に必要な情報。"""

    decision_record: PostAlarmCandidateValidationDecisionRecord
    validation_resolution: PostAlarmCandidateValidationResolution
    previous_training_model_id: int
    estimated_change_point_sample_index: int | None
    detection_episode_id: int | None

    @property
    def training_model_switch_sample_index(self) -> int | None:
        if self.validation_resolution.training_model_assignment_change is None:
            return None
        return self.decision_record.resolution_sample_index

    @property
    def detection_episode_operation_required(self) -> bool:
        return self.validation_resolution.training_model_assignment_change is not None


@dataclass(frozen=True, kw_only=True)
class PostAlarmCandidateValidationProgress:
    """未到達なら同session、到達なら完了情報を返す。"""

    session_to_continue: PostAlarmCandidateValidationSession | None
    completed_validation: PostAlarmCandidateValidationCompletion | None


def _validate_validation_progress_threshold(
    *, specified_value: float, parameter_name: str
) -> float:
    if type(specified_value) not in (int, float):
        raise TypeError(f"{parameter_name}はbool以外のbuiltin int/floatが必要です。")
    try:
        specified_value = float(specified_value)
    except OverflowError as exception:
        raise ValueError(f"{parameter_name}は有限非負値が必要です。") from exception
    if not isfinite(specified_value) or specified_value < 0:
        raise ValueError(f"{parameter_name}は有限非負値が必要です。")
    return specified_value


def _validate_validation_progress_inputs(
    *,
    validation_session: PostAlarmCandidateValidationSession,
    candidate_model_training_and_acceptance_settings: CandidateModelTrainingAndAcceptanceSettings,
    maximum_reference_mean_loss_increase: float,
    minimum_candidate_mean_loss_improvement: float,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    current_training_model_assignment: CurrentTrainingModelAssignment,
) -> tuple[float, float]:
    if type(validation_session) is not PostAlarmCandidateValidationSession:
        raise TypeError("validation_sessionはexact session型が必要です。")
    if (
        type(candidate_model_training_and_acceptance_settings)
        is not CandidateModelTrainingAndAcceptanceSettings
    ):
        raise TypeError("候補の採否設定はexact設定型が必要です。")
    CandidateModelTrainingAndAcceptanceSettings(
        candidate_model_acceptance_policy=candidate_model_training_and_acceptance_settings.candidate_model_acceptance_policy,
        candidate_post_alarm_validation_sample_count=candidate_model_training_and_acceptance_settings.candidate_post_alarm_validation_sample_count,
    )
    if (
        type(
            candidate_model_training_and_acceptance_settings.candidate_post_alarm_validation_sample_count
        )
        is not int
    ):
        raise TypeError("検証標本件数はbool以外のbuiltin intが必要です。")
    collection_state_snapshot = (
        validation_session.post_alarm_candidate_loss_collection.get_state_snapshot()
    )
    if (
        candidate_model_training_and_acceptance_settings.candidate_post_alarm_validation_sample_count
        != collection_state_snapshot.required_validation_sample_count
    ):
        raise ValueError("進行設定と開始sessionの要求検証件数が一致しません。")
    maximum_reference_mean_loss_increase = _validate_validation_progress_threshold(
        specified_value=maximum_reference_mean_loss_increase,
        parameter_name="maximum_reference_mean_loss_increase",
    )
    minimum_candidate_mean_loss_improvement = _validate_validation_progress_threshold(
        specified_value=minimum_candidate_mean_loss_improvement,
        parameter_name="minimum_candidate_mean_loss_improvement",
    )
    if type(held_model_training_state_registry) is not HeldModelTrainingStateRegistry:
        raise TypeError("保有モデルはexact登録owner型が必要です。")
    if type(current_training_model_assignment) is not CurrentTrainingModelAssignment:
        raise TypeError("現在帰属はexact帰属owner型が必要です。")
    return maximum_reference_mean_loss_increase, minimum_candidate_mean_loss_improvement


def advance_post_alarm_candidate_validation(
    *,
    validation_session: PostAlarmCandidateValidationSession | None,
    sample_index: int,
    input_features: Tensor,
    observed_class_labels: Tensor,
    candidate_model_training_and_acceptance_settings: CandidateModelTrainingAndAcceptanceSettings,
    maximum_reference_mean_loss_increase: float,
    minimum_candidate_mean_loss_improvement: float,
    pending_assignment_sample_concept_ids: tuple[int | None, ...],
    temporary_model_id_allocator: TemporaryModelIdAllocator,
    upload_delay_round_count: int,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    training_sample_store: ModelTrainingSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    current_training_model_assignment: CurrentTrainingModelAssignment,
    pending_model_upload_state: PendingModelUploadState,
) -> PostAlarmCandidateValidationProgress:
    """観測後の評価・適用の失敗について、観測や外部ownerのrollbackを行わない。"""
    if validation_session is None:
        return PostAlarmCandidateValidationProgress(
            session_to_continue=None, completed_validation=None
        )
    maximum_reference_mean_loss_increase, minimum_candidate_mean_loss_improvement = (
        _validate_validation_progress_inputs(
            validation_session=validation_session,
            candidate_model_training_and_acceptance_settings=candidate_model_training_and_acceptance_settings,
            maximum_reference_mean_loss_increase=maximum_reference_mean_loss_increase,
            minimum_candidate_mean_loss_improvement=minimum_candidate_mean_loss_improvement,
            held_model_training_state_registry=held_model_training_state_registry,
            current_training_model_assignment=current_training_model_assignment,
        )
    )
    if not observe_post_alarm_candidate_validation_sample(
        sample_index=sample_index,
        input_features=input_features,
        observed_class_labels=observed_class_labels,
        candidate_classifier=validation_session.candidate_training_state.candidate_classifier,
        reference_classifiers_by_model_id=validation_session.fixed_reference_models.reference_classifiers_by_model_id,
        post_alarm_candidate_loss_collection=validation_session.post_alarm_candidate_loss_collection,
    ):
        return PostAlarmCandidateValidationProgress(
            session_to_continue=validation_session, completed_validation=None
        )
    collection_state_snapshot = (
        validation_session.post_alarm_candidate_loss_collection.get_state_snapshot()
    )
    available_reference_model_ids = tuple(
        held_model_training_state.model_id
        for held_model_training_state in held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    )
    previous_training_model_id = current_training_model_assignment.current_training_model_id
    post_alarm_candidate_loss_evaluation = evaluate_candidate_using_post_alarm_losses(
        candidate_model_training_and_acceptance_settings=candidate_model_training_and_acceptance_settings,
        candidate_losses=collection_state_snapshot.candidate_losses,
        reference_losses_by_model_id=dict(collection_state_snapshot.reference_losses_by_model_id),
        reference_historical_mean_losses_by_model_id=validation_session.fixed_reference_models.reference_historical_mean_losses_by_model_id,
        available_reference_model_ids=available_reference_model_ids,
        current_training_model_id=previous_training_model_id,
        maximum_reference_mean_loss_increase=maximum_reference_mean_loss_increase,
        minimum_candidate_mean_loss_improvement=minimum_candidate_mean_loss_improvement,
    )
    decision_record = PostAlarmCandidateValidationDecisionRecord(
        proposal_sample_index=validation_session.proposal_sample_index,
        resolution_sample_index=sample_index,
        detector_name=validation_session.detector_name,
        candidate_training_interval_sample_count=len(validation_session.training_input_features),
        post_alarm_candidate_loss_evaluation=post_alarm_candidate_loss_evaluation,
    )
    validation_resolution = apply_post_alarm_candidate_validation_resolution(
        post_alarm_candidate_loss_evaluation=post_alarm_candidate_loss_evaluation,
        pending_assignment_training_samples=validation_session.pending_assignment_training_samples,
        pending_assignment_sample_concept_ids=pending_assignment_sample_concept_ids,
        temporary_model_id_allocator=temporary_model_id_allocator,
        adopted_candidate_classifier=validation_session.candidate_training_state.candidate_classifier,
        candidate_concept_specific_parameter_optimizer_state=validation_session.candidate_training_state.candidate_concept_specific_parameter_optimizer_state,
        initial_statistics_input_features=validation_session.training_input_features,
        initial_statistics_observed_class_labels=validation_session.training_observed_class_labels,
        upload_delay_round_count=upload_delay_round_count,
        candidate_trained_sample_count=validation_session.candidate_epoch_training_result.candidate_trained_sample_count,
        candidate_parameter_update_step_count=validation_session.candidate_epoch_training_result.candidate_parameter_update_step_count,
        held_model_training_state_registry=held_model_training_state_registry,
        loss_statistics_store=loss_statistics_store,
        training_sample_store=training_sample_store,
        model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
        current_training_model_assignment=current_training_model_assignment,
        pending_model_upload_state=pending_model_upload_state,
    )
    validation_completion = PostAlarmCandidateValidationCompletion(
        decision_record=decision_record,
        validation_resolution=validation_resolution,
        previous_training_model_id=previous_training_model_id,
        estimated_change_point_sample_index=validation_session.estimated_change_point_sample_index,
        detection_episode_id=validation_session.detection_episode_id,
    )
    return PostAlarmCandidateValidationProgress(
        session_to_continue=None, completed_validation=validation_completion
    )
