"""警報1回ぶんの処理を、既存の応答・完了処理・適応記録・session保持への反映・診断通知の順につなぐ。"""

from dataclasses import dataclass
from random import Random

from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection import (
    AdaHedgeDiagnosticEvidenceCollection,
)
from federated_learning_experiments.evaluation.adaptation_record_store import (
    AdaptationRecord,
    AdaptationRecordStore,
)
from federated_learning_experiments.evaluation.model_evaluation_sample_store import (
    ModelEvaluationSampleStore,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training.candidate_epoch_training_settings import (
    CandidateEpochTrainingSettings,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
    IndexedObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_and_assignment_counts import (
    ModelTrainingAndAssignmentCountsStore,
)
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import (
    CandidateModelTrainingAndAcceptanceSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import (
    CandidateParameterInitializationSettings,
)
from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
    OverallAndTrueClassLossMonitor,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import (
    PendingTrainingAssignmentBuffer,
)
from federated_learning_experiments.runtime.alarm_adaptation_recording import (
    record_completed_alarm_response,
)
from federated_learning_experiments.runtime.alarm_buffer_response import (
    respond_to_alarm_with_buffered_samples,
)
from federated_learning_experiments.runtime.alarm_response_completion import (
    AlarmResponseCompletion,
    complete_alarm_buffer_response,
)
from federated_learning_experiments.runtime.candidate_validation_session_holder import (
    CandidateValidationSessionHolder,
)
from federated_learning_experiments.runtime.held_candidate_validation_progress import (
    apply_alarm_response_to_validation_session_holder,
)
from federated_learning_experiments.runtime.training_assignment_diagnostic_notification import (
    notify_diagnostics_of_training_assignment_change,
)


@dataclass(frozen=True, kw_only=True)
class AlarmOccurrenceHandling:
    """警報1回ぶんの処理の結果。完了した応答の情報と、追加した適応記録。"""

    alarm_response_completion: AlarmResponseCompletion
    adaptation_record: AdaptationRecord


def handle_alarm_occurrence(
    *,
    validation_session_holder: CandidateValidationSessionHolder,
    adaptation_record_store: AdaptationRecordStore,
    diagnostic_evidence_collection: AdaHedgeDiagnosticEvidenceCollection,
    loss_change_monitor: OverallAndTrueClassLossMonitor,
    alarm_sample_index: int,
    pending_training_assignment_buffer: PendingTrainingAssignmentBuffer,
    pending_sample_observations: tuple[IndexedObservedTrainingSample, ...],
    estimated_change_span_sample_count: int,
    minimum_change_interval_sample_count: int,
    model_evaluation_sample_store: ModelEvaluationSampleStore,
    python_random_generator: Random,
    maximum_alarm_interval_mean_loss_increase: float,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    training_sample_store: ModelTrainingSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    current_training_model_assignment: CurrentTrainingModelAssignment,
    candidate_parameter_initialization_settings: CandidateParameterInitializationSettings,
    architecture_reference_classifier: ResidualAdapterClassifier,
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
    candidate_epoch_training_settings: CandidateEpochTrainingSettings,
    candidate_model_training_and_acceptance_settings: CandidateModelTrainingAndAcceptanceSettings,
    estimated_change_point_sample_index: int | None,
    detection_episode_id: int | None,
    detector_name: str,
) -> AlarmOccurrenceHandling:
    """保持中のsessionを応答へ渡し、応答を一度だけ完了させて、記録・保持・診断へ反映する。"""
    # 応答は多くのownerを更新する。応答より後の段だけが検査するownerと値は、応答の前に確かめる。
    for owner_argument_name, owner, required_owner_type in (
        ("validation_session_holder", validation_session_holder, CandidateValidationSessionHolder),
        ("adaptation_record_store", adaptation_record_store, AdaptationRecordStore),
        (
            "diagnostic_evidence_collection",
            diagnostic_evidence_collection,
            AdaHedgeDiagnosticEvidenceCollection,
        ),
        ("loss_change_monitor", loss_change_monitor, OverallAndTrueClassLossMonitor),
        (
            "pending_training_assignment_buffer",
            pending_training_assignment_buffer,
            PendingTrainingAssignmentBuffer,
        ),
    ):
        if type(owner) is not required_owner_type:
            raise TypeError(f"{owner_argument_name} must be exact {required_owner_type.__name__}")
    # 警報位置・検出器名・推定変化点・episode IDは、適応記録と同じ規則で確かめる（記録は保存しない）。
    AdaptationRecord(
        adaptation_sample_index=alarm_sample_index,
        detector_name=detector_name,
        adaptation_outcome="alarm_change_interval_too_short",
        previous_training_model_id=0,
        current_training_model_id=0,
        estimated_change_point_sample_index=estimated_change_point_sample_index,
        detection_episode_id=detection_episode_id,
    )
    if (
        pending_training_assignment_buffer.get_state_snapshot().last_observed_sample_index
        != alarm_sample_index
    ):
        raise ValueError("alarm_sample_index must be the last observed sample index")
    alarm_buffer_response = respond_to_alarm_with_buffered_samples(
        active_validation_session=validation_session_holder.held_validation_session,
        pending_training_assignment_buffer=pending_training_assignment_buffer,
        pending_sample_observations=pending_sample_observations,
        estimated_change_span_sample_count=estimated_change_span_sample_count,
        minimum_change_interval_sample_count=minimum_change_interval_sample_count,
        model_evaluation_sample_store=model_evaluation_sample_store,
        python_random_generator=python_random_generator,
        maximum_alarm_interval_mean_loss_increase=maximum_alarm_interval_mean_loss_increase,
        held_model_training_state_registry=held_model_training_state_registry,
        loss_statistics_store=loss_statistics_store,
        training_sample_store=training_sample_store,
        model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
        current_training_model_assignment=current_training_model_assignment,
        candidate_parameter_initialization_settings=candidate_parameter_initialization_settings,
        architecture_reference_classifier=architecture_reference_classifier,
        parameter_optimizer_settings=parameter_optimizer_settings,
        candidate_epoch_training_settings=candidate_epoch_training_settings,
        candidate_model_training_and_acceptance_settings=candidate_model_training_and_acceptance_settings,
        proposal_sample_index=alarm_sample_index,
        estimated_change_point_sample_index=estimated_change_point_sample_index,
        detection_episode_id=detection_episode_id,
        detector_name=detector_name,
    )
    alarm_response_completion = complete_alarm_buffer_response(
        alarm_buffer_response=alarm_buffer_response,
        alarm_sample_index=alarm_sample_index,
        estimated_change_point_sample_index=estimated_change_point_sample_index,
        detection_episode_id=detection_episode_id,
        current_training_model_assignment=current_training_model_assignment,
        loss_statistics_store=loss_statistics_store,
        loss_change_monitor=loss_change_monitor,
        pending_training_assignment_buffer=pending_training_assignment_buffer,
    )
    adaptation_record = record_completed_alarm_response(
        alarm_response_completion=alarm_response_completion,
        detector_name=detector_name,
        adaptation_record_store=adaptation_record_store,
    )
    apply_alarm_response_to_validation_session_holder(
        alarm_buffer_response=alarm_buffer_response,
        validation_session_holder=validation_session_holder,
    )
    change_interval_resolution = alarm_buffer_response.change_interval_resolution
    notify_diagnostics_of_training_assignment_change(
        assignment_change=(
            None
            if change_interval_resolution is None
            else change_interval_resolution.training_model_assignment_change
        ),
        diagnostic_evidence_collection=diagnostic_evidence_collection,
    )
    return AlarmOccurrenceHandling(
        alarm_response_completion=alarm_response_completion, adaptation_record=adaptation_record
    )
