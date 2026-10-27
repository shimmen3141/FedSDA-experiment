"""FedDriftの一roundの保有モデル学習を接続する。"""

from random import Random

from torch import cat

from federated_learning_experiments.learning.training.held_model_training_batch_sampling import (
    _validate_observed_training_sample,
    sample_training_batches_for_held_models,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingState,
    HeldModelTrainingStateRegistry,
    _validate_held_model_training_state_inputs,
)
from federated_learning_experiments.learning.training.independent_model_parameter_update import (
    update_independent_model_parameters,
    validate_independent_model_parameter_update_inputs,
)
from federated_learning_experiments.learning.training.model_training_and_assignment_counts import (
    ModelTrainingAndAssignmentCountsStore,
)
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)


def train_feddrift_held_models_for_round(
    *,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    training_sample_store: ModelTrainingSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    python_random_generator: Random,
    batch_sample_count: int,
    updates_per_sample: int,
    k_steps: int,
    eligible_model_ids: frozenset[int] | None = None,
) -> tuple[float, ...]:
    """全参加列を検査してから、store順に各モデルの予算を続けて更新する。"""
    if eligible_model_ids is not None and (
        type(eligible_model_ids) is not frozenset
        or any(type(mid) is not int or mid < 0 for mid in eligible_model_ids)
    ):
        raise ValueError("学習対象は非負builtin intのfrozensetが必要です。")
    for owner, expected in (
        (held_model_training_state_registry, HeldModelTrainingStateRegistry),
        (training_sample_store, ModelTrainingSampleStore),
        (model_training_and_assignment_counts_store, ModelTrainingAndAssignmentCountsStore),
        (python_random_generator, Random),
    ):
        if type(owner) is not expected:
            raise TypeError(f"ownerはexact {expected.__name__}が必要です。")
    for value, minimum in ((batch_sample_count, 1), (updates_per_sample, 0), (k_steps, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError("件数と予算は範囲内のbuiltin intが必要です。")
    update_count = updates_per_sample * k_steps
    if update_count == 0:
        return ()
    states = {
        state.model_id: state
        for state in held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    }
    participants = tuple(
        collection
        for collection in training_sample_store.snapshot_ordered_model_training_samples()
        if collection.model_id in states
        and len(collection.training_samples) >= batch_sample_count
        and (eligible_model_ids is None or collection.model_id in eligible_model_ids)
    )
    # 乱数/先行モデルを更新する前に、全参加populationを検査する。
    for collection in participants:
        state = states[collection.model_id]
        if type(state) is not HeldModelTrainingState:
            raise TypeError("保有記録はexact HeldModelTrainingStateが必要です。")
        _validate_held_model_training_state_inputs(
            model_id=state.model_id,
            classifier=state.classifier,
            concept_specific_parameter_optimizer_state=state.concept_specific_parameter_optimizer_state,
        )
        for sample in collection.training_samples:
            _validate_observed_training_sample(
                training_sample=sample,
                expected_input_feature_count=state.classifier.input_feature_count,
                tensor_name="training_sample",
            )
        validate_independent_model_parameter_update_inputs(
            classifier=state.classifier,
            parameter_optimizer=state.concept_specific_parameter_optimizer_state.parameter_optimizer,
            input_features=cat([s.input_features for s in collection.training_samples]),
            observed_class_labels=cat(
                [s.observed_class_labels for s in collection.training_samples]
            ),
        )
    completed_losses: list[float] = []
    for collection in participants:
        state = states[collection.model_id]
        for _ in range(update_count):
            (batch,) = sample_training_batches_for_held_models(
                held_model_ids=frozenset((state.model_id,)),
                ordered_model_training_samples=(collection,),
                batch_sample_count=batch_sample_count,
                python_random_generator=python_random_generator,
            )
            loss = update_independent_model_parameters(
                classifier=state.classifier,
                parameter_optimizer=state.concept_specific_parameter_optimizer_state.parameter_optimizer,
                input_features=batch.input_features,
                observed_class_labels=batch.observed_class_labels,
            )
            model_training_and_assignment_counts_store.record_completed_model_training(
                model_id=state.model_id,
                trained_sample_count=batch_sample_count,
                parameter_update_step_count=1,
            )
            completed_losses.append(loss)
    return tuple(completed_losses)
