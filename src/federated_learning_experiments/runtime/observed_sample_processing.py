"""観測した標本1件を処理する: 予測→候補検証の進行→損失の監視→保留→（警報の処理｜帰属の確定と学習）。"""

from dataclasses import dataclass
from random import Random

from torch import Tensor
from torch.optim import Optimizer

from federated_learning_experiments.evaluation.adahedge_diagnostic_evidence_collection import (
    AdaHedgeDiagnosticEvidenceCollection,
)
from federated_learning_experiments.evaluation.adaptation_record_store import (
    AdaptationRecordStore,
)
from federated_learning_experiments.evaluation.loss_change_alarm_record_store import (
    LossChangeAlarmRecordStore,
)
from federated_learning_experiments.evaluation.model_evaluation_sample_store import (
    ModelEvaluationSampleStore,
)
from federated_learning_experiments.evaluation.sample_prediction_record_store import (
    SamplePredictionRecordStore,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.models.shared_feature_extractor import (
    SharedFeatureExtractor,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
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
from federated_learning_experiments.learning.training.local_training_request_schedule import (
    LocalTrainingRequestSchedule,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.model_training_and_assignment_counts import (
    ModelTrainingAndAssignmentCountsStore,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.temporary_model_id_allocation import (
    TemporaryModelIdAllocator,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import (
    CandidateModelTrainingAndAcceptanceSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import (
    CandidateParameterInitializationSettings,
)
from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
    LossMonitoringObservation,
    OverallAndTrueClassLossMonitor,
)
from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import (
    select_loss_monitoring_baseline_mean_loss,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import (
    PendingModelUploadState,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_prediction_weights import (
    FixedSharePredictionWeightController,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_sample_observation_store import (
    PendingSampleObservationStore,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import (
    PendingTrainingAssignmentBuffer,
)
from federated_learning_experiments.runtime.alarm_occurrence_handling import (
    AlarmOccurrenceHandling,
    handle_alarm_occurrence,
)
from federated_learning_experiments.runtime.candidate_validation_session_holder import (
    CandidateValidationSessionHolder,
)
from federated_learning_experiments.runtime.held_candidate_validation_progress import (
    HeldCandidateValidationAdvance,
    advance_held_candidate_validation,
)
from federated_learning_experiments.runtime.held_model_training_request_handling import (
    record_training_request_and_train_held_models_when_due,
    train_held_models_for_pending_training_requests,
)
from federated_learning_experiments.runtime.observed_sample_prediction import (
    ObservedSamplePrediction,
    predict_observed_sample_and_update_prediction_weights,
)
from federated_learning_experiments.runtime.released_pending_sample_assignment import (
    assign_released_pending_samples_to_current_training_model,
)


@dataclass(frozen=True, kw_only=True)
class ObservedSampleProcessing:
    """標本1件の処理の結果。警報の有無で、警報の処理か、帰属の確定と学習のどちらかが入る。"""

    observed_sample_prediction: ObservedSamplePrediction
    held_validation_advance: HeldCandidateValidationAdvance
    loss_monitoring_observation: LossMonitoringObservation
    alarm_occurrence_handling: AlarmOccurrenceHandling | None
    released_sample_observations: tuple[IndexedObservedTrainingSample, ...]
    completed_joint_update_losses: tuple[float, ...]


# 引数名から、本処理が受け取るownerのexact型への対応（最初の状態更新より前に確かめる）。
_REQUIRED_OWNER_TYPES_BY_ARGUMENT_NAME = {
    "fixed_share_prediction_weight_controller": FixedSharePredictionWeightController,
    "sample_prediction_record_store": SamplePredictionRecordStore,
    "validation_session_holder": CandidateValidationSessionHolder,
    "adaptation_record_store": AdaptationRecordStore,
    "diagnostic_evidence_collection": AdaHedgeDiagnosticEvidenceCollection,
    "loss_change_monitor": OverallAndTrueClassLossMonitor,
    "loss_change_alarm_record_store": LossChangeAlarmRecordStore,
    "pending_training_assignment_buffer": PendingTrainingAssignmentBuffer,
    "pending_sample_observation_store": PendingSampleObservationStore,
    "held_model_training_state_registry": HeldModelTrainingStateRegistry,
    "loss_statistics_store": ModelAndClassLossStatisticsStore,
    "training_sample_store": ModelTrainingSampleStore,
    "model_training_and_assignment_counts_store": ModelTrainingAndAssignmentCountsStore,
    "current_training_model_assignment": CurrentTrainingModelAssignment,
    "model_evaluation_sample_store": ModelEvaluationSampleStore,
    "temporary_model_id_allocator": TemporaryModelIdAllocator,
    "pending_model_upload_state": PendingModelUploadState,
    "local_training_request_schedule": LocalTrainingRequestSchedule,
}


def _validate_observed_sample_processing_inputs(
    *,
    indexed_observation: IndexedObservedTrainingSample,
    owners_by_argument_name: dict[str, object],
    python_random_generator: Random,
    validation_session_holder: CandidateValidationSessionHolder,
    loss_change_monitor: OverallAndTrueClassLossMonitor,
    pending_training_assignment_buffer: PendingTrainingAssignmentBuffer,
    pending_sample_observation_store: PendingSampleObservationStore,
) -> None:
    for owner_argument_name, required_owner_type in _REQUIRED_OWNER_TYPES_BY_ARGUMENT_NAME.items():
        if type(owners_by_argument_name[owner_argument_name]) is not required_owner_type:
            raise TypeError(f"{owner_argument_name} must be exact {required_owner_type.__name__}")
    if type(python_random_generator) is not Random:
        raise TypeError("python_random_generator must be exact random.Random")
    # 標本: 型と、1標本であること。保留へ入った標本は、後の警報や確定まで検査されないので、ここで確かめる。
    if type(indexed_observation) is not IndexedObservedTrainingSample:
        raise TypeError("indexed_observation must be exact IndexedObservedTrainingSample")
    sample_index = indexed_observation.sample_index
    if type(sample_index) is not int:
        raise TypeError("sample_index must be builtin int")
    if sample_index < 0:
        raise ValueError("sample_index must be nonnegative")
    if indexed_observation.observed_concept_id is not None and (
        type(indexed_observation.observed_concept_id) is not int
    ):
        raise TypeError("observed_concept_id must be builtin int or None")
    training_sample = indexed_observation.training_sample
    if type(training_sample) is not ObservedTrainingSample:
        raise TypeError("training_sample must be exact ObservedTrainingSample")
    if (
        type(training_sample.input_features) is not Tensor
        or type(training_sample.observed_class_labels) is not Tensor
    ):
        raise TypeError("input_features and observed_class_labels must be exact torch.Tensor")
    if training_sample.input_features.ndim != 2 or training_sample.input_features.shape[0] != 1:
        raise ValueError("input_features must hold exactly one sample as a 2D tensor")
    if tuple(training_sample.observed_class_labels.shape) != (1, 1):
        raise ValueError("observed_class_labels must have shape (1, 1)")
    # 位置の連続性: 保留位置のownerと損失の監視が、どちらもこの位置を次の位置として受け入れること。
    pending_assignment_state = pending_training_assignment_buffer.get_state_snapshot()
    if (
        pending_assignment_state.last_observed_sample_index is not None
        and sample_index != pending_assignment_state.last_observed_sample_index + 1
    ):
        raise ValueError("sample_index must follow the last observed sample index by one")
    last_monitoring_observation = loss_change_monitor.last_observation
    if (
        last_monitoring_observation is not None
        and sample_index != last_monitoring_observation.sample_index + 1
    ):
        raise ValueError("sample_index must follow the last monitored sample index by one")
    # 保留標本のownerは、保留位置のownerと同じ並びを持つこと。
    if (
        tuple(
            pending_observation.sample_index
            for pending_observation in pending_sample_observation_store.snapshot_pending_sample_observations()
        )
        != pending_assignment_state.pending_sample_indices
    ):
        raise ValueError("pending sample observations must match pending sample indices")
    # 候補検証を保持している間だけ、候補検証へ渡した標本の概念IDを保持していること。
    if (validation_session_holder.held_validation_session is None) != (
        pending_sample_observation_store.validation_assignment_sample_concept_ids is None
    ):
        raise ValueError(
            "validation assignment sample concept IDs must be held exactly while a session is held"
        )


def _select_validation_assignment_sample_concept_ids(
    *,
    pending_sample_observations: tuple[IndexedObservedTrainingSample, ...],
    validation_assignment_training_samples: tuple[ObservedTrainingSample, ...],
) -> tuple[int | None, ...]:
    """候補検証へ渡された標本（警報時の保留標本の末尾）の概念IDを返す。末尾と同一でなければ拒否する。"""
    latest_pending_observations = pending_sample_observations[
        len(pending_sample_observations) - len(validation_assignment_training_samples) :
    ]
    if len(latest_pending_observations) != len(validation_assignment_training_samples) or any(
        pending_observation.training_sample is not training_sample
        for pending_observation, training_sample in zip(
            latest_pending_observations, validation_assignment_training_samples
        )
    ):
        raise ValueError(
            "validation assignment samples must be the latest pending sample observations"
        )
    return tuple(
        pending_observation.observed_concept_id
        for pending_observation in latest_pending_observations
    )


def process_observed_sample(
    *,
    indexed_observation: IndexedObservedTrainingSample,
    fixed_share_prediction_weight_controller: FixedSharePredictionWeightController,
    sample_prediction_record_store: SamplePredictionRecordStore,
    validation_session_holder: CandidateValidationSessionHolder,
    adaptation_record_store: AdaptationRecordStore,
    diagnostic_evidence_collection: AdaHedgeDiagnosticEvidenceCollection,
    loss_change_monitor: OverallAndTrueClassLossMonitor,
    loss_change_alarm_record_store: LossChangeAlarmRecordStore,
    pending_training_assignment_buffer: PendingTrainingAssignmentBuffer,
    pending_sample_observation_store: PendingSampleObservationStore,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    training_sample_store: ModelTrainingSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    current_training_model_assignment: CurrentTrainingModelAssignment,
    model_evaluation_sample_store: ModelEvaluationSampleStore,
    temporary_model_id_allocator: TemporaryModelIdAllocator,
    pending_model_upload_state: PendingModelUploadState,
    local_training_request_schedule: LocalTrainingRequestSchedule,
    candidate_model_training_and_acceptance_settings: CandidateModelTrainingAndAcceptanceSettings,
    maximum_reference_mean_loss_increase: float,
    minimum_candidate_mean_loss_improvement: float,
    upload_delay_round_count: int,
    minimum_change_interval_sample_count: int,
    maximum_alarm_interval_mean_loss_increase: float,
    candidate_parameter_initialization_settings: CandidateParameterInitializationSettings,
    architecture_reference_classifier: ResidualAdapterClassifier,
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
    candidate_epoch_training_settings: CandidateEpochTrainingSettings,
    detector_name: str,
    batch_sample_count: int,
    python_random_generator: Random,
    local_training_settings: LocalTrainingSettings,
    shared_feature_extractor: SharedFeatureExtractor,
    shared_parameter_optimizer: Optimizer | None,
) -> ObservedSampleProcessing:
    """標本1件を、旧の標本処理と同じ順で処理する（最初に予測する。計算量と所要時間の記録は含めない）。"""
    _validate_observed_sample_processing_inputs(
        indexed_observation=indexed_observation,
        owners_by_argument_name=dict(
            fixed_share_prediction_weight_controller=fixed_share_prediction_weight_controller,
            sample_prediction_record_store=sample_prediction_record_store,
            validation_session_holder=validation_session_holder,
            adaptation_record_store=adaptation_record_store,
            diagnostic_evidence_collection=diagnostic_evidence_collection,
            loss_change_monitor=loss_change_monitor,
            loss_change_alarm_record_store=loss_change_alarm_record_store,
            pending_training_assignment_buffer=pending_training_assignment_buffer,
            pending_sample_observation_store=pending_sample_observation_store,
            held_model_training_state_registry=held_model_training_state_registry,
            loss_statistics_store=loss_statistics_store,
            training_sample_store=training_sample_store,
            model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
            current_training_model_assignment=current_training_model_assignment,
            model_evaluation_sample_store=model_evaluation_sample_store,
            temporary_model_id_allocator=temporary_model_id_allocator,
            pending_model_upload_state=pending_model_upload_state,
            local_training_request_schedule=local_training_request_schedule,
        ),
        python_random_generator=python_random_generator,
        validation_session_holder=validation_session_holder,
        loss_change_monitor=loss_change_monitor,
        pending_training_assignment_buffer=pending_training_assignment_buffer,
        pending_sample_observation_store=pending_sample_observation_store,
    )
    sample_index = indexed_observation.sample_index
    input_features = indexed_observation.training_sample.input_features
    observed_class_labels = indexed_observation.training_sample.observed_class_labels
    # (0) この標本を予測し、記録して、予測重みと診断証拠を更新する（学習側のどの段よりも前）。
    observed_sample_prediction = predict_observed_sample_and_update_prediction_weights(
        indexed_observation=indexed_observation,
        fixed_share_prediction_weight_controller=fixed_share_prediction_weight_controller,
        diagnostic_evidence_collection=diagnostic_evidence_collection,
        sample_prediction_record_store=sample_prediction_record_store,
        held_model_training_state_registry=held_model_training_state_registry,
        current_training_model_assignment=current_training_model_assignment,
    )
    # (1) 保持中の候補検証へ、この標本を観測させる。確定したら学習帰属が変わりうる。
    held_validation_advance = advance_held_candidate_validation(
        validation_session_holder=validation_session_holder,
        adaptation_record_store=adaptation_record_store,
        diagnostic_evidence_collection=diagnostic_evidence_collection,
        sample_index=sample_index,
        input_features=input_features,
        observed_class_labels=observed_class_labels,
        candidate_model_training_and_acceptance_settings=candidate_model_training_and_acceptance_settings,
        maximum_reference_mean_loss_increase=maximum_reference_mean_loss_increase,
        minimum_candidate_mean_loss_improvement=minimum_candidate_mean_loss_improvement,
        pending_assignment_sample_concept_ids=(
            pending_sample_observation_store.validation_assignment_sample_concept_ids or ()
        ),
        temporary_model_id_allocator=temporary_model_id_allocator,
        upload_delay_round_count=upload_delay_round_count,
        pending_model_upload_state=pending_model_upload_state,
        held_model_training_state_registry=held_model_training_state_registry,
        loss_statistics_store=loss_statistics_store,
        training_sample_store=training_sample_store,
        model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
        current_training_model_assignment=current_training_model_assignment,
    )
    if held_validation_advance.validation_progress.completed_validation is not None:
        pending_sample_observation_store.release_validation_assignment_sample_concept_ids()
    # (2) (1)の後の現在のモデルで、この標本の損失を評価して監視へ渡す。
    current_training_model_id = current_training_model_assignment.current_training_model_id
    observed_loss = evaluate_classifier_per_sample_bounded_losses(
        classifier=held_model_training_state_registry.get_held_model_training_state(
            model_id=current_training_model_id
        ).classifier,
        input_features=input_features,
        observed_class_labels=observed_class_labels,
    )[0].item()
    current_model_loss_statistics = loss_statistics_store.get_model_loss_statistics(
        model_id=current_training_model_id
    )
    loss_monitoring_observation = loss_change_monitor.observe_loss_after_label_observation(
        observed_loss=observed_loss,
        observed_class_id=int(observed_class_labels.reshape(-1)[0].item()),
        sample_index=sample_index,
        current_model_baseline_loss_mean=select_loss_monitoring_baseline_mean_loss(
            loss_moments=(
                None
                if current_model_loss_statistics is None
                else current_model_loss_statistics.overall_loss_moments
            )
        ),
    )
    loss_change_alarm_record_store.append_monitored_log_e_value(
        log_e_value=loss_monitoring_observation.log_e_value
    )
    # (3) 標本を保留へ足す（警報の判定の後、警報の処理の前）。
    pending_training_assignment_buffer.append_observed_sample_index(sample_index=sample_index)
    pending_sample_observation_store.append_pending_sample_observation(
        indexed_observation=indexed_observation
    )
    pending_sample_observations = (
        pending_sample_observation_store.snapshot_pending_sample_observations()
    )
    alarm_occurrence_handling = None
    released_sample_observations: tuple[IndexedObservedTrainingSample, ...] = ()
    if loss_monitoring_observation.drift_detected:
        # (4a) 警報: 保留中の学習要求を先に学習し、警報の位置を記録してから、警報1回ぶんの処理を行う。
        completed_joint_update_losses = train_held_models_for_pending_training_requests(
            local_training_request_schedule=local_training_request_schedule,
            held_model_training_state_registry=held_model_training_state_registry,
            training_sample_store=training_sample_store,
            model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
            batch_sample_count=batch_sample_count,
            python_random_generator=python_random_generator,
            local_training_settings=local_training_settings,
            shared_feature_extractor=shared_feature_extractor,
            shared_parameter_optimizer=shared_parameter_optimizer,
        )
        estimated_change_span_sample_count = (
            loss_monitoring_observation.estimated_change_span_sample_count
        )
        estimated_change_point_sample_index = max(
            0,
            sample_index
            - min(len(pending_sample_observations), estimated_change_span_sample_count)
            + 1,
        )
        loss_change_alarm_record_store.append_alarm_record(
            alarm_sample_index=sample_index,
            estimated_change_point_sample_index=estimated_change_point_sample_index,
            detector_candidate_start_sample_index=(
                loss_monitoring_observation.detector_candidate_start_sample_index
            ),
        )
        alarm_occurrence_handling = handle_alarm_occurrence(
            validation_session_holder=validation_session_holder,
            adaptation_record_store=adaptation_record_store,
            diagnostic_evidence_collection=diagnostic_evidence_collection,
            loss_change_monitor=loss_change_monitor,
            alarm_sample_index=sample_index,
            pending_training_assignment_buffer=pending_training_assignment_buffer,
            pending_sample_observations=pending_sample_observations,
            estimated_change_span_sample_count=estimated_change_span_sample_count,
            minimum_change_interval_sample_count=minimum_change_interval_sample_count,
            model_evaluation_sample_store=model_evaluation_sample_store,
            python_random_generator=python_random_generator,
            maximum_alarm_interval_mean_loss_increase=maximum_alarm_interval_mean_loss_increase,
            candidate_parameter_initialization_settings=candidate_parameter_initialization_settings,
            architecture_reference_classifier=architecture_reference_classifier,
            parameter_optimizer_settings=parameter_optimizer_settings,
            candidate_epoch_training_settings=candidate_epoch_training_settings,
            candidate_model_training_and_acceptance_settings=candidate_model_training_and_acceptance_settings,
            estimated_change_point_sample_index=estimated_change_point_sample_index,
            detection_episode_id=None,
            detector_name=detector_name,
            held_model_training_state_registry=held_model_training_state_registry,
            loss_statistics_store=loss_statistics_store,
            training_sample_store=training_sample_store,
            model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
            current_training_model_assignment=current_training_model_assignment,
        )
        alarm_buffer_response = (
            alarm_occurrence_handling.alarm_response_completion.alarm_buffer_response
        )
        active_validation_session = alarm_buffer_response.active_validation_session
        if (
            alarm_buffer_response.response_outcome == "alarm_interval_candidate_validation_started"
            and active_validation_session is not None
        ):
            # 開始した候補検証へ渡した標本（警報時の保留標本の末尾）の概念IDを、確定まで保持する。
            pending_sample_observation_store.hold_validation_assignment_sample_concept_ids(
                sample_concept_ids=_select_validation_assignment_sample_concept_ids(
                    pending_sample_observations=pending_sample_observations,
                    validation_assignment_training_samples=active_validation_session.pending_assignment_training_samples,
                )
            )
        # 保留標本を、警報の処理の後に保留位置のownerに残った位置へ合わせる。
        pending_sample_observation_store.retain_latest_pending_sample_observations(
            retained_sample_indices=pending_training_assignment_buffer.get_state_snapshot().pending_sample_indices
        )
    else:
        # (4b) 警報なし: 容量を超えた最古の保留標本を現在のモデルへ確定し、学習要求を1件記録する。
        released_sample_observations = assign_released_pending_samples_to_current_training_model(
            pending_training_assignment_buffer=pending_training_assignment_buffer,
            pending_sample_observations=pending_sample_observations,
            held_model_training_state_registry=held_model_training_state_registry,
            loss_statistics_store=loss_statistics_store,
            training_sample_store=training_sample_store,
            model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
            current_training_model_assignment=current_training_model_assignment,
        )
        # 保留標本を、保留位置のownerに残った位置へ合わせてから、学習へ進む。
        pending_sample_observation_store.retain_latest_pending_sample_observations(
            retained_sample_indices=pending_training_assignment_buffer.get_state_snapshot().pending_sample_indices
        )
        completed_joint_update_losses = record_training_request_and_train_held_models_when_due(
            local_training_request_schedule=local_training_request_schedule,
            held_model_training_state_registry=held_model_training_state_registry,
            training_sample_store=training_sample_store,
            model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
            batch_sample_count=batch_sample_count,
            python_random_generator=python_random_generator,
            local_training_settings=local_training_settings,
            shared_feature_extractor=shared_feature_extractor,
            shared_parameter_optimizer=shared_parameter_optimizer,
        )
    return ObservedSampleProcessing(
        observed_sample_prediction=observed_sample_prediction,
        held_validation_advance=held_validation_advance,
        loss_monitoring_observation=loss_monitoring_observation,
        alarm_occurrence_handling=alarm_occurrence_handling,
        released_sample_observations=released_sample_observations,
        completed_joint_update_losses=completed_joint_update_losses,
    )
