"""採用が決まった候補について、採番・登録・計数・標本・学習帰属の切替えを組み立てる。"""

from torch import Tensor

from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
    TrainingModelAssignmentChange,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
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
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.learning.training.temporary_model_id_allocation import (
    TemporaryModelIdAllocator,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import (
    PendingModelUploadState,
)
from federated_learning_experiments.runtime.adopted_candidate_initial_local_registration import (
    register_adopted_candidate_as_temporary_held_model,
)


def _validate_adoption_inputs(
    *,
    temporary_model_id_allocator: TemporaryModelIdAllocator,
    candidate_trained_sample_count: int,
    candidate_parameter_update_step_count: int,
    pending_assignment_training_samples: tuple[ObservedTrainingSample, ...],
    training_sample_store: ModelTrainingSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    current_training_model_assignment: CurrentTrainingModelAssignment,
) -> None:
    for state_owner, expected_owner_type, owner_name in (
        (temporary_model_id_allocator, TemporaryModelIdAllocator, "temporary_model_id_allocator"),
        (training_sample_store, ModelTrainingSampleStore, "training_sample_store"),
        (
            model_training_and_assignment_counts_store,
            ModelTrainingAndAssignmentCountsStore,
            "model_training_and_assignment_counts_store",
        ),
        (
            current_training_model_assignment,
            CurrentTrainingModelAssignment,
            "current_training_model_assignment",
        ),
    ):
        if type(state_owner) is not expected_owner_type:
            raise TypeError(f"{owner_name}はexact {expected_owner_type.__name__}が必要です。")
    for count_name, count_value in (
        ("candidate_trained_sample_count", candidate_trained_sample_count),
        ("candidate_parameter_update_step_count", candidate_parameter_update_step_count),
    ):
        if type(count_value) is not int:
            raise TypeError(f"{count_name}はbool・派生型以外のbuiltin intが必要です。")
        if count_value < 0:
            raise ValueError(f"{count_name}は非負が必要です。")
    if type(pending_assignment_training_samples) is not tuple:
        raise TypeError("pending_assignment_training_samplesはexact tupleが必要です。")
    for training_sample in pending_assignment_training_samples:
        if type(training_sample) is not ObservedTrainingSample:
            raise TypeError(
                "pending_assignment_training_samplesの各要素は"
                "exact ObservedTrainingSampleが必要です。"
            )


def _reject_temporary_model_id_already_in_use_for_adoption(
    *,
    temporary_model_id: int,
    training_sample_store: ModelTrainingSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    current_training_model_assignment: CurrentTrainingModelAssignment,
) -> None:
    """一覧・統計・送信保留での使用は、登録が状態変更より前に確認する。"""
    for (
        model_training_sample_collection
    ) in training_sample_store.snapshot_ordered_model_training_samples():
        if model_training_sample_collection.model_id == temporary_model_id:
            raise ValueError("次の一時IDは学習標本storeで使用済みです。")
    counts_snapshot = (
        model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts()
    )
    if (
        temporary_model_id in counts_snapshot.trained_sample_counts_by_model_id
        or temporary_model_id in counts_snapshot.parameter_update_step_counts_by_model_id
        or temporary_model_id in counts_snapshot.assigned_sample_counts_by_model_and_concept_id
    ):
        raise ValueError("次の一時IDは学習/割当計数で使用済みです。")
    if current_training_model_assignment.current_training_model_id == temporary_model_id:
        raise ValueError("次の一時IDは現在の学習帰属IDと同じです。")


def adopt_candidate_as_current_training_model(
    *,
    temporary_model_id_allocator: TemporaryModelIdAllocator,
    adopted_candidate_classifier: ResidualAdapterClassifier,
    candidate_concept_specific_parameter_optimizer_state: ParameterOptimizerState,
    initial_statistics_input_features: Tensor,
    initial_statistics_observed_class_labels: Tensor,
    upload_delay_round_count: int,
    candidate_trained_sample_count: int,
    candidate_parameter_update_step_count: int,
    pending_assignment_training_samples: tuple[ObservedTrainingSample, ...],
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    training_sample_store: ModelTrainingSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    current_training_model_assignment: CurrentTrainingModelAssignment,
    pending_model_upload_state: PendingModelUploadState,
) -> TrainingModelAssignmentChange:
    """登録の完了後に採番を確定し、計数→標本→現在の学習帰属IDを順次更新する。"""
    _validate_adoption_inputs(
        temporary_model_id_allocator=temporary_model_id_allocator,
        candidate_trained_sample_count=candidate_trained_sample_count,
        candidate_parameter_update_step_count=candidate_parameter_update_step_count,
        pending_assignment_training_samples=pending_assignment_training_samples,
        training_sample_store=training_sample_store,
        model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
        current_training_model_assignment=current_training_model_assignment,
    )
    temporary_model_id = temporary_model_id_allocator.next_temporary_model_id
    _reject_temporary_model_id_already_in_use_for_adoption(
        temporary_model_id=temporary_model_id,
        training_sample_store=training_sample_store,
        model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
        current_training_model_assignment=current_training_model_assignment,
    )
    register_adopted_candidate_as_temporary_held_model(
        temporary_model_id=temporary_model_id,
        adopted_candidate_classifier=adopted_candidate_classifier,
        candidate_concept_specific_parameter_optimizer_state=candidate_concept_specific_parameter_optimizer_state,
        initial_statistics_input_features=initial_statistics_input_features,
        initial_statistics_observed_class_labels=initial_statistics_observed_class_labels,
        upload_delay_round_count=upload_delay_round_count,
        held_model_training_state_registry=held_model_training_state_registry,
        loss_statistics_store=loss_statistics_store,
        current_training_model_assignment=current_training_model_assignment,
        pending_model_upload_state=pending_model_upload_state,
    )
    # 登録の検証が通るまで採番を消費しない。確定する値は上で読んだ値と同じ。
    temporary_model_id_allocator.allocate_temporary_model_id()
    model_training_and_assignment_counts_store.record_completed_model_training(
        model_id=temporary_model_id,
        trained_sample_count=candidate_trained_sample_count,
        parameter_update_step_count=candidate_parameter_update_step_count,
    )
    training_sample_store.append_model_training_samples(
        model_id=temporary_model_id, training_samples=pending_assignment_training_samples
    )
    assignment_change = current_training_model_assignment.assign_model_for_training(
        model_id=temporary_model_id
    )
    # 現在IDが一時IDと異なることを確認済みなので、必ず実変更の記録が返る。
    assert assignment_change is not None
    return assignment_change
