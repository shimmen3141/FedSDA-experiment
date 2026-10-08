"""警報のない標本で、保留の容量を超えた最古の標本を現在の学習帰属のモデルへ確定する。"""

from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
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
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import (
    PendingTrainingAssignmentBuffer,
)
from federated_learning_experiments.runtime.assigned_training_sample_absorption import (
    absorb_assigned_training_samples_into_held_model,
)


def assign_released_pending_samples_to_current_training_model(
    *,
    pending_training_assignment_buffer: PendingTrainingAssignmentBuffer,
    pending_sample_observations: tuple[IndexedObservedTrainingSample, ...],
    current_training_model_assignment: CurrentTrainingModelAssignment,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    training_sample_store: ModelTrainingSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
) -> tuple[IndexedObservedTrainingSample, ...]:
    """容量を超えた保留標本を古い順に現在のモデルへ吸収し、保留から解放して、解放した標本を返す。"""
    if type(pending_training_assignment_buffer) is not PendingTrainingAssignmentBuffer:
        raise TypeError(
            "pending_training_assignment_buffer must be exact PendingTrainingAssignmentBuffer"
        )
    if type(current_training_model_assignment) is not CurrentTrainingModelAssignment:
        raise TypeError(
            "current_training_model_assignment must be exact CurrentTrainingModelAssignment"
        )
    if type(pending_sample_observations) is not tuple:
        raise TypeError("pending_sample_observations must be exact tuple")
    for indexed_observation in pending_sample_observations:
        if type(indexed_observation) is not IndexedObservedTrainingSample:
            raise TypeError("observations must be exact IndexedObservedTrainingSample")
        if type(indexed_observation.sample_index) is not int:
            raise TypeError("sample_index must be builtin int")
    if (
        tuple(
            indexed_observation.sample_index for indexed_observation in pending_sample_observations
        )
        != pending_training_assignment_buffer.get_state_snapshot().pending_sample_indices
    ):
        raise ValueError("supplied indices must exactly match pending sample indices")
    # 解放する位置を、解放せずに求める。吸収が全検査を終えて更新を済ませてから、保留を解放する。
    released_sample_observations = pending_sample_observations[
        : len(pending_training_assignment_buffer.get_sample_indices_exceeding_capacity())
    ]
    if not released_sample_observations:
        return ()
    absorb_assigned_training_samples_into_held_model(
        model_id=current_training_model_assignment.current_training_model_id,
        assigned_training_samples=tuple(
            indexed_observation.training_sample
            for indexed_observation in released_sample_observations
        ),
        assigned_sample_concept_ids=tuple(
            indexed_observation.observed_concept_id
            for indexed_observation in released_sample_observations
        ),
        held_model_training_state_registry=held_model_training_state_registry,
        training_sample_store=training_sample_store,
        model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
        loss_statistics_store=loss_statistics_store,
    )
    pending_training_assignment_buffer.release_sample_indices_exceeding_capacity()
    return released_sample_observations
