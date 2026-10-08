"""保持中の候補検証sessionについて、警報応答の反映・標本ごとの進行・終端回収を、記録と保持の更新へ接続する。"""

from dataclasses import dataclass

from torch import Tensor

from federated_learning_experiments.evaluation.adaptation_record_store import (
    AdaptationRecord,
    AdaptationRecordStore,
)
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
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import (
    PendingModelUploadState,
)
from federated_learning_experiments.runtime.alarm_buffer_response import AlarmBufferResponse
from federated_learning_experiments.runtime.candidate_validation_adaptation_recording import (
    record_completed_candidate_validation,
    record_incomplete_candidate_validation_finalization,
)
from federated_learning_experiments.runtime.candidate_validation_session_holder import (
    CandidateValidationSessionHolder,
)
from federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization import (
    IncompletePostAlarmCandidateValidationFinalization,
    finalize_incomplete_post_alarm_candidate_validation,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_progress import (
    PostAlarmCandidateValidationProgress,
    advance_post_alarm_candidate_validation,
)


@dataclass(frozen=True, kw_only=True)
class HeldCandidateValidationAdvance:
    """保持中のsessionの1標本ぶんの進行結果と、確定したときに追加した適応記録。"""

    validation_progress: PostAlarmCandidateValidationProgress
    adaptation_record: AdaptationRecord | None


@dataclass(frozen=True, kw_only=True)
class HeldIncompleteCandidateValidationFinalization:
    """保持中の未完了sessionの終端回収の結果と、追加した適応記録。"""

    incomplete_validation_finalization: IncompletePostAlarmCandidateValidationFinalization
    adaptation_record: AdaptationRecord


def _validate_holder_and_record_store(
    *,
    validation_session_holder: CandidateValidationSessionHolder,
    adaptation_record_store: AdaptationRecordStore,
) -> None:
    if type(validation_session_holder) is not CandidateValidationSessionHolder:
        raise TypeError("validation_session_holder must be exact CandidateValidationSessionHolder")
    if type(adaptation_record_store) is not AdaptationRecordStore:
        raise TypeError("adaptation_record_store must be exact AdaptationRecordStore")


def apply_alarm_response_to_validation_session_holder(
    *,
    alarm_buffer_response: AlarmBufferResponse,
    validation_session_holder: CandidateValidationSessionHolder,
) -> None:
    """警報応答の結果を保持へ反映する。候補検証を開始した応答だけが新しいsessionを保持させる。"""
    if type(alarm_buffer_response) is not AlarmBufferResponse:
        raise TypeError("alarm_buffer_response must be exact AlarmBufferResponse")
    if type(validation_session_holder) is not CandidateValidationSessionHolder:
        raise TypeError("validation_session_holder must be exact CandidateValidationSessionHolder")
    # 手で壊した応答も、結果種別とsessionの組を保持の更新より前に再検査する。
    alarm_buffer_response.__post_init__()
    held_validation_session = validation_session_holder.held_validation_session
    if alarm_buffer_response.response_outcome == "alarm_during_candidate_validation":
        # 候補検証中の警報は保持中のsessionへの応答で、保持を変えない。
        if alarm_buffer_response.active_validation_session is not held_validation_session:
            raise ValueError("response must refer to the held candidate validation session")
        return
    if held_validation_session is not None:
        raise ValueError("response without an active validation requires an empty holder")
    if alarm_buffer_response.active_validation_session is not None:
        validation_session_holder.hold_validation_session(
            validation_session=alarm_buffer_response.active_validation_session
        )


def advance_held_candidate_validation(
    *,
    validation_session_holder: CandidateValidationSessionHolder,
    adaptation_record_store: AdaptationRecordStore,
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
) -> HeldCandidateValidationAdvance:
    """保持中のsessionへ標本1件を観測させ、確定したら適応記録を追加して保持を解除する。"""
    # 進行は観測・確定で多くのownerを更新する。保持と記録のownerの型は、その前に確かめる。
    _validate_holder_and_record_store(
        validation_session_holder=validation_session_holder,
        adaptation_record_store=adaptation_record_store,
    )
    validation_progress = advance_post_alarm_candidate_validation(
        validation_session=validation_session_holder.held_validation_session,
        sample_index=sample_index,
        input_features=input_features,
        observed_class_labels=observed_class_labels,
        candidate_model_training_and_acceptance_settings=candidate_model_training_and_acceptance_settings,
        maximum_reference_mean_loss_increase=maximum_reference_mean_loss_increase,
        minimum_candidate_mean_loss_improvement=minimum_candidate_mean_loss_improvement,
        pending_assignment_sample_concept_ids=pending_assignment_sample_concept_ids,
        temporary_model_id_allocator=temporary_model_id_allocator,
        upload_delay_round_count=upload_delay_round_count,
        held_model_training_state_registry=held_model_training_state_registry,
        loss_statistics_store=loss_statistics_store,
        training_sample_store=training_sample_store,
        model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
        current_training_model_assignment=current_training_model_assignment,
        pending_model_upload_state=pending_model_upload_state,
    )
    if validation_progress.completed_validation is None:
        return HeldCandidateValidationAdvance(
            validation_progress=validation_progress, adaptation_record=None
        )
    # 旧と同じく、適応記録を追加してからsessionを外す。
    adaptation_record = record_completed_candidate_validation(
        validation_completion=validation_progress.completed_validation,
        adaptation_record_store=adaptation_record_store,
    )
    validation_session_holder.release_validation_session()
    return HeldCandidateValidationAdvance(
        validation_progress=validation_progress, adaptation_record=adaptation_record
    )


def finalize_held_incomplete_candidate_validation(
    *,
    validation_session_holder: CandidateValidationSessionHolder,
    adaptation_record_store: AdaptationRecordStore,
    processed_sample_count: int,
    pending_assignment_sample_concept_ids: tuple[int | None, ...],
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    training_sample_store: ModelTrainingSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    current_training_model_assignment: CurrentTrainingModelAssignment,
) -> HeldIncompleteCandidateValidationFinalization | None:
    """実験の終端で、保持中の未完了sessionを回収し、適応記録を追加して保持を解除する。"""
    _validate_holder_and_record_store(
        validation_session_holder=validation_session_holder,
        adaptation_record_store=adaptation_record_store,
    )
    incomplete_validation_finalization = finalize_incomplete_post_alarm_candidate_validation(
        validation_session=validation_session_holder.held_validation_session,
        processed_sample_count=processed_sample_count,
        pending_assignment_sample_concept_ids=pending_assignment_sample_concept_ids,
        held_model_training_state_registry=held_model_training_state_registry,
        loss_statistics_store=loss_statistics_store,
        training_sample_store=training_sample_store,
        model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
        current_training_model_assignment=current_training_model_assignment,
    )
    if incomplete_validation_finalization is None:
        return None
    adaptation_record = record_incomplete_candidate_validation_finalization(
        incomplete_validation_finalization=incomplete_validation_finalization,
        adaptation_record_store=adaptation_record_store,
    )
    validation_session_holder.release_validation_session()
    return HeldIncompleteCandidateValidationFinalization(
        incomplete_validation_finalization=incomplete_validation_finalization,
        adaptation_record=adaptation_record,
    )
