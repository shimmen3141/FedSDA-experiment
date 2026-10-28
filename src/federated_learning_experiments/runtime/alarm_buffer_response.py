"""警報時の保留標本への応答を、既存の準備・吸収・解決から組み立てる。"""

from dataclasses import dataclass
from random import Random

from federated_learning_experiments.evaluation.model_evaluation_sample_store import (
    ModelEvaluationSampleStore,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.model_architecture_classifier import (
    ModelArchitectureClassifier,
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
from federated_learning_experiments.learning.training.temporary_model_id_allocation import (
    TemporaryModelIdAllocator,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import (
    CandidateParameterInitializationSettings,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import (
    PendingModelUploadState,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection import (
    MinimumDescriptionLengthAssignmentSelection,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import (
    PendingTrainingAssignmentBuffer,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals import (
    PreparedAlarmTrainingIntervals,
)
from federated_learning_experiments.runtime.alarm_change_interval_resolution import (
    ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES,
    AlarmChangeIntervalResolution,
    resolve_alarm_change_interval,
)
from federated_learning_experiments.runtime.alarm_training_interval_preparation import (
    prepare_alarm_training_intervals,
)

ALARM_BUFFER_RESPONSE_OUTCOMES: tuple[str, ...] = (
    "alarm_change_interval_too_short",
    *ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES,
)


@dataclass(frozen=True, kw_only=True)
class AlarmBufferResponse:
    """応答と、呼出側のFIFO後始末判断を返す。"""

    response_outcome: str
    prepared_alarm_training_intervals: PreparedAlarmTrainingIntervals | None
    change_interval_resolution: AlarmChangeIntervalResolution | None

    def __post_init__(self) -> None:
        if (
            type(self.response_outcome) is not str
            or self.response_outcome not in ALARM_BUFFER_RESPONSE_OUTCOMES
        ):
            raise ValueError("response_outcome must be a declared value")
        if self.response_outcome == "alarm_change_interval_too_short":
            if (
                type(self.prepared_alarm_training_intervals) is not PreparedAlarmTrainingIntervals
                or self.change_interval_resolution is not None
            ):
                raise ValueError("insufficient response needs prepared intervals only")
        elif (
            type(self.prepared_alarm_training_intervals) is not PreparedAlarmTrainingIntervals
            or type(self.change_interval_resolution) is not AlarmChangeIntervalResolution
        ):
            raise ValueError("resolved response needs prepared intervals and resolution")
        elif self.change_interval_resolution.resolution_outcome != self.response_outcome:
            raise ValueError("response must match its actual resolution")

    @property
    def pending_assignment_buffer_should_be_cleared(self) -> bool:
        return self.response_outcome != "alarm_change_interval_too_short"


def _validate_buffered_alarm_observations(
    *,
    pending_training_assignment_buffer: PendingTrainingAssignmentBuffer,
    pending_sample_observations: tuple[IndexedObservedTrainingSample, ...],
) -> None:
    if type(pending_training_assignment_buffer) is not PendingTrainingAssignmentBuffer:
        raise TypeError(
            "pending_training_assignment_buffer must be exact PendingTrainingAssignmentBuffer"
        )
    if type(pending_sample_observations) is not tuple:
        raise TypeError("pending_sample_observations must be exact tuple")
    for indexed_observation in pending_sample_observations:
        if type(indexed_observation) is not IndexedObservedTrainingSample:
            raise TypeError("observations must be exact IndexedObservedTrainingSample")
        if type(indexed_observation.sample_index) is not int:
            raise TypeError("sample_index must be builtin int")
        if indexed_observation.sample_index < 0:
            raise ValueError("sample_index must be nonnegative")
    if (
        tuple(
            indexed_observation.sample_index for indexed_observation in pending_sample_observations
        )
        != pending_training_assignment_buffer.get_state_snapshot().pending_sample_indices
    ):
        raise ValueError("supplied indices must exactly match pending FIFO indices")


def respond_to_alarm_with_buffered_samples(
    *,
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
    temporary_model_id_allocator: TemporaryModelIdAllocator,
    pending_model_upload_state: PendingModelUploadState,
    upload_delay_round_count: int,
    candidate_parameter_initialization_settings: CandidateParameterInitializationSettings,
    architecture_reference_classifier: ModelArchitectureClassifier,
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
    candidate_epoch_training_settings: CandidateEpochTrainingSettings,
    proposal_sample_index: int,
    estimated_change_point_sample_index: int | None,
    detection_episode_id: int | None,
    detector_name: str,
    alarm_interval_assignment_selection: MinimumDescriptionLengthAssignmentSelection | None = None,
) -> AlarmBufferResponse:

    _validate_buffered_alarm_observations(
        pending_training_assignment_buffer=pending_training_assignment_buffer,
        pending_sample_observations=pending_sample_observations,
    )
    if type(current_training_model_assignment) is not CurrentTrainingModelAssignment:
        raise TypeError(
            "current_training_model_assignment must be exact CurrentTrainingModelAssignment"
        )
    if type(minimum_change_interval_sample_count) is not int:
        raise TypeError("minimum_change_interval_sample_count must be builtin int")
    if minimum_change_interval_sample_count < 1:
        raise ValueError("minimum_change_interval_sample_count must be positive")
    prepared_intervals = prepare_alarm_training_intervals(
        pending_training_assignment_buffer=pending_training_assignment_buffer,
        pending_sample_observations=pending_sample_observations,
        estimated_change_span_sample_count=estimated_change_span_sample_count,
        current_training_model_assignment=current_training_model_assignment,
        model_evaluation_sample_store=model_evaluation_sample_store,
        python_random_generator=python_random_generator,
        held_model_training_state_registry=held_model_training_state_registry,
        training_sample_store=training_sample_store,
        model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
        loss_statistics_store=loss_statistics_store,
    )
    if len(prepared_intervals.change_interval_observations) < minimum_change_interval_sample_count:
        return AlarmBufferResponse(
            response_outcome="alarm_change_interval_too_short",
            prepared_alarm_training_intervals=prepared_intervals,
            change_interval_resolution=None,
        )
    interval_resolution = resolve_alarm_change_interval(
        change_interval_training_samples=tuple(
            indexed_observation.training_sample
            for indexed_observation in prepared_intervals.change_interval_observations
        ),
        change_interval_sample_concept_ids=tuple(
            indexed_observation.observed_concept_id
            for indexed_observation in prepared_intervals.change_interval_observations
        ),
        maximum_alarm_interval_mean_loss_increase=maximum_alarm_interval_mean_loss_increase,
        held_model_training_state_registry=held_model_training_state_registry,
        loss_statistics_store=loss_statistics_store,
        training_sample_store=training_sample_store,
        model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
        current_training_model_assignment=current_training_model_assignment,
        temporary_model_id_allocator=temporary_model_id_allocator,
        pending_model_upload_state=pending_model_upload_state,
        upload_delay_round_count=upload_delay_round_count,
        candidate_parameter_initialization_settings=candidate_parameter_initialization_settings,
        architecture_reference_classifier=architecture_reference_classifier,
        parameter_optimizer_settings=parameter_optimizer_settings,
        candidate_epoch_training_settings=candidate_epoch_training_settings,
        proposal_sample_index=proposal_sample_index,
        estimated_change_point_sample_index=estimated_change_point_sample_index,
        detection_episode_id=detection_episode_id,
        detector_name=detector_name,
        alarm_interval_assignment_selection=alarm_interval_assignment_selection,
    )
    return AlarmBufferResponse(
        response_outcome=interval_resolution.resolution_outcome,
        prepared_alarm_training_intervals=prepared_intervals,
        change_interval_resolution=interval_resolution,
    )
