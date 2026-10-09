"""学習要求を記録し、保留中の学習要求のぶんだけ保有モデルを共同学習して、完了した学習量を計数へ反映する。"""

from random import Random

from torch.optim import Optimizer

from federated_learning_experiments.learning.models.shared_feature_extractor import (
    SharedFeatureExtractor,
)
from federated_learning_experiments.learning.training.held_model_joint_training_iterations import (
    perform_held_model_joint_training_iterations,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
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
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)


def _validate_training_request_owners(
    *,
    local_training_request_schedule: LocalTrainingRequestSchedule,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    training_sample_store: ModelTrainingSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
) -> None:
    if type(local_training_request_schedule) is not LocalTrainingRequestSchedule:
        raise TypeError(
            "local_training_request_schedule must be exact LocalTrainingRequestSchedule"
        )
    if type(held_model_training_state_registry) is not HeldModelTrainingStateRegistry:
        raise TypeError(
            "held_model_training_state_registry must be exact HeldModelTrainingStateRegistry"
        )
    if type(training_sample_store) is not ModelTrainingSampleStore:
        raise TypeError("training_sample_store must be exact ModelTrainingSampleStore")
    if (
        type(model_training_and_assignment_counts_store)
        is not ModelTrainingAndAssignmentCountsStore
    ):
        raise TypeError(
            "model_training_and_assignment_counts_store must be exact "
            "ModelTrainingAndAssignmentCountsStore"
        )


def train_held_models_for_pending_training_requests(
    *,
    local_training_request_schedule: LocalTrainingRequestSchedule,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    training_sample_store: ModelTrainingSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    batch_sample_count: int,
    python_random_generator: Random,
    local_training_settings: LocalTrainingSettings,
    shared_feature_extractor: SharedFeatureExtractor,
    shared_parameter_optimizer: Optimizer | None,
) -> tuple[float, ...]:
    """保留中の全学習要求のぶんを共同学習し、共同更新の完了ごとに計数へ足して、最後に保留を消化する。"""
    _validate_training_request_owners(
        local_training_request_schedule=local_training_request_schedule,
        held_model_training_state_registry=held_model_training_state_registry,
        training_sample_store=training_sample_store,
        model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
    )
    pending_training_request_count = local_training_request_schedule.pending_training_request_count
    if pending_training_request_count == 0:
        return ()
    held_model_training_bindings = (
        held_model_training_state_registry.snapshot_ordered_held_model_training_bindings()
    )
    ordered_model_training_samples = training_sample_store.snapshot_ordered_model_training_samples()
    held_model_ids = frozenset(
        training_binding.model_id for training_binding in held_model_training_bindings
    )
    completed_joint_update_losses: list[float] = []
    for _joint_update_iteration_index in range(
        local_training_request_schedule.calculate_pending_joint_update_iteration_count()
    ):
        joint_update_losses = perform_held_model_joint_training_iterations(
            requested_joint_update_iteration_count=1,
            held_model_training_bindings=held_model_training_bindings,
            ordered_model_training_samples=ordered_model_training_samples,
            batch_sample_count=batch_sample_count,
            python_random_generator=python_random_generator,
            local_training_settings=local_training_settings,
            shared_feature_extractor=shared_feature_extractor,
            shared_parameter_optimizer=shared_parameter_optimizer,
            update_shared_features=True,
        )
        if not joint_update_losses:
            continue
        completed_joint_update_losses.extend(joint_update_losses)
        # 共同更新が完了した回だけ、参加したモデル（保有していて、標本がbatchの件数以上ある
        # モデル。標本列の順）の学習量を足す。batchの件数は反復の検査を通っている。
        for model_training_samples in ordered_model_training_samples:
            if (
                model_training_samples.model_id in held_model_ids
                and len(model_training_samples.training_samples) >= batch_sample_count
            ):
                model_training_and_assignment_counts_store.record_completed_model_training(
                    model_id=model_training_samples.model_id,
                    trained_sample_count=batch_sample_count,
                    parameter_update_step_count=1,
                )
    local_training_request_schedule.acknowledge_completed_training_requests(
        completed_training_request_count=pending_training_request_count
    )
    return tuple(completed_joint_update_losses)


def record_training_request_and_train_held_models_when_due(
    *,
    local_training_request_schedule: LocalTrainingRequestSchedule,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    training_sample_store: ModelTrainingSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    batch_sample_count: int,
    python_random_generator: Random,
    local_training_settings: LocalTrainingSettings,
    shared_feature_extractor: SharedFeatureExtractor,
    shared_parameter_optimizer: Optimizer | None,
) -> tuple[float, ...]:
    """学習要求を1件記録し、保留が更新間隔に達していれば保留中の全要求のぶんを学習する。"""
    _validate_training_request_owners(
        local_training_request_schedule=local_training_request_schedule,
        held_model_training_state_registry=held_model_training_state_registry,
        training_sample_store=training_sample_store,
        model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
    )
    local_training_request_schedule.record_training_request()
    if not local_training_request_schedule.has_pending_requests_reaching_update_interval():
        return ()
    return train_held_models_for_pending_training_requests(
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
