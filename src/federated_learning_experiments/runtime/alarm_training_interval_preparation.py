"""警報の保留位置を照合し、前区間の評価保存と吸収を組み立てる。"""

from random import Random

from torch import Tensor

from federated_learning_experiments.evaluation.model_evaluation_sample_records import (
    ObservedEvaluationSample,
)
from federated_learning_experiments.evaluation.model_evaluation_sample_store import (
    ModelEvaluationSampleStore,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    validate_classifier_bounded_loss_inputs,
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
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import (
    PendingTrainingAssignmentBuffer,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals import (
    PreparedAlarmTrainingIntervals,
)
from federated_learning_experiments.runtime.assigned_training_sample_absorption import (
    absorb_assigned_training_samples_into_held_model,
)


def prepare_alarm_training_intervals(
    *,
    pending_training_assignment_buffer: PendingTrainingAssignmentBuffer,
    pending_sample_observations: tuple[IndexedObservedTrainingSample, ...],
    estimated_change_span_sample_count: int,
    current_training_model_assignment: CurrentTrainingModelAssignment,
    model_evaluation_sample_store: ModelEvaluationSampleStore,
    python_random_generator: Random,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    training_sample_store: ModelTrainingSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
) -> PreparedAlarmTrainingIntervals:
    """全件の入力と前区間損失を検査した後、保存してから吸収する。"""
    for state_owner, expected_owner_type, owner_name in (
        (
            pending_training_assignment_buffer,
            PendingTrainingAssignmentBuffer,
            "pending_training_assignment_buffer",
        ),
        (
            current_training_model_assignment,
            CurrentTrainingModelAssignment,
            "current_training_model_assignment",
        ),
        (
            model_evaluation_sample_store,
            ModelEvaluationSampleStore,
            "model_evaluation_sample_store",
        ),
        (python_random_generator, Random, "python_random_generator"),
        (
            held_model_training_state_registry,
            HeldModelTrainingStateRegistry,
            "held_model_training_state_registry",
        ),
        (training_sample_store, ModelTrainingSampleStore, "training_sample_store"),
        (
            model_training_and_assignment_counts_store,
            ModelTrainingAndAssignmentCountsStore,
            "model_training_and_assignment_counts_store",
        ),
        (loss_statistics_store, ModelAndClassLossStatisticsStore, "loss_statistics_store"),
    ):
        if type(state_owner) is not expected_owner_type:
            raise TypeError(f"{owner_name}はexact {expected_owner_type.__name__}が必要です。")
    if type(estimated_change_span_sample_count) is not int:
        raise TypeError("estimated_change_span_sample_countはbuiltin intが必要です。")
    if estimated_change_span_sample_count < 1:
        raise ValueError("estimated_change_span_sample_countは正の値が必要です。")
    if type(pending_sample_observations) is not tuple:
        raise TypeError("pending_sample_observationsはexact tupleが必要です。")
    for indexed_observation in pending_sample_observations:
        if type(indexed_observation) is not IndexedObservedTrainingSample:
            raise TypeError("各観測はexact IndexedObservedTrainingSampleが必要です。")
        if type(indexed_observation.sample_index) is not int:
            raise TypeError("sample_indexはbuiltin intが必要です。")
        if indexed_observation.sample_index < 0:
            raise ValueError("sample_indexは非負が必要です。")
        if (
            indexed_observation.observed_concept_id is not None
            and type(indexed_observation.observed_concept_id) is not int
        ):
            raise TypeError("observed_concept_idはNoneまたはbuiltin intが必要です。")
        if type(indexed_observation.training_sample) is not ObservedTrainingSample:
            raise TypeError("training_sampleはexact ObservedTrainingSampleが必要です。")
        training_sample = indexed_observation.training_sample
        if not isinstance(training_sample.input_features, Tensor) or not isinstance(
            training_sample.observed_class_labels, Tensor
        ):
            raise TypeError("標本payloadはTensorが必要です。")
        if (
            training_sample.input_features.ndim != 2
            or training_sample.input_features.shape[0] != 1
            or training_sample.input_features.shape[1] < 1
        ):
            raise ValueError("特徴はshape[1,F]が必要です。")
        if training_sample.observed_class_labels.shape != (1, 1):
            raise ValueError("ラベルはshape[1,1]が必要です。")
    pending_assignment_snapshot = pending_training_assignment_buffer.get_state_snapshot()
    if (
        tuple(
            indexed_observation.sample_index for indexed_observation in pending_sample_observations
        )
        != pending_assignment_snapshot.pending_sample_indices
    ):
        raise ValueError("保留位置と観測位置は同じ順・件数で完全一致する必要があります。")
    current_training_model_id = current_training_model_assignment.current_training_model_id
    classifier = held_model_training_state_registry.get_held_model_training_state(
        model_id=current_training_model_id
    ).classifier
    buffered_change_interval_partition = (
        pending_training_assignment_buffer.get_change_interval_partition(
            estimated_change_span_sample_count=estimated_change_span_sample_count
        )
    )
    earlier_sample_count = len(buffered_change_interval_partition.earlier_sample_indices)
    earlier_observations = pending_sample_observations[:earlier_sample_count]
    change_interval_observations = pending_sample_observations[earlier_sample_count:]
    # 取込み（損失の計算と統計の更新）が受け付ける入力であることを、状態の更新より前に、
    # 順伝播なしで確かめる。損失は、取込みが1回だけ計算する。
    for indexed_observation in earlier_observations:
        validate_classifier_bounded_loss_inputs(
            classifier=classifier,
            input_features=indexed_observation.training_sample.input_features,
            observed_class_labels=indexed_observation.training_sample.observed_class_labels,
        )
    if earlier_observations:
        model_evaluation_sample_store.sample_and_append_model_evaluation_samples(
            model_id=current_training_model_id,
            evaluation_samples=tuple(
                ObservedEvaluationSample(
                    input_features=indexed_observation.training_sample.input_features,
                    observed_class_labels=indexed_observation.training_sample.observed_class_labels,
                )
                for indexed_observation in earlier_observations
            ),
            python_random_generator=python_random_generator,
        )
        absorb_assigned_training_samples_into_held_model(
            model_id=current_training_model_id,
            assigned_training_samples=tuple(
                indexed_observation.training_sample for indexed_observation in earlier_observations
            ),
            assigned_sample_concept_ids=tuple(
                indexed_observation.observed_concept_id
                for indexed_observation in earlier_observations
            ),
            held_model_training_state_registry=held_model_training_state_registry,
            training_sample_store=training_sample_store,
            model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
            loss_statistics_store=loss_statistics_store,
        )
    return PreparedAlarmTrainingIntervals(
        earlier_observations=earlier_observations,
        change_interval_observations=change_interval_observations,
        change_interval_start_sample_index=buffered_change_interval_partition.change_interval_start_sample_index,
        earlier_interval_absorbed_model_id=current_training_model_id
        if earlier_observations
        else None,
    )
