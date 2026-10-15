"""毎回新しく抽出したbatchで保有モデルの共同更新を繰り返す。"""

from random import Random

from torch.optim import Optimizer

from federated_learning_experiments.learning.models.shared_feature_extractor import (
    SharedFeatureExtractor,
)

from .held_model_training_batch_sampling import sample_training_batches_for_held_models
from .held_model_training_binding import HeldModelTrainingBinding
from .joint_model_parameter_update import perform_joint_model_parameter_update
from .local_training_settings import LocalTrainingSettings
from .model_training_sample_records import ModelTrainingSampleCollection
from .participating_model_training_batch import ParticipatingModelTrainingBatch


def _index_held_model_training_bindings(
    *, held_model_training_bindings: tuple[HeldModelTrainingBinding, ...]
) -> dict[int, HeldModelTrainingBinding]:
    if type(held_model_training_bindings) is not tuple:
        raise ValueError("held_model_training_bindingsはexact tupleが必要です。")
    bindings_by_model_id: dict[int, HeldModelTrainingBinding] = {}
    for binding_index, training_binding in enumerate(held_model_training_bindings):
        if type(training_binding) is not HeldModelTrainingBinding:
            raise ValueError(
                f"held_model_training_bindings[{binding_index}]は"
                "exact HeldModelTrainingBindingが必要です。"
            )
        if type(training_binding.model_id) is not int:
            raise ValueError(
                f"held_model_training_bindings[{binding_index}].model_idは"
                "bool以外のbuiltin intが必要です。"
            )
        if training_binding.model_id in bindings_by_model_id:
            raise ValueError(
                f"held_model_training_bindings[{binding_index}].model_id="
                f"{training_binding.model_id}が重複しています。"
            )
        bindings_by_model_id[training_binding.model_id] = training_binding
    return bindings_by_model_id


def perform_held_model_joint_training_iterations(
    *,
    requested_joint_update_iteration_count: int,
    held_model_training_bindings: tuple[HeldModelTrainingBinding, ...],
    ordered_model_training_samples: tuple[ModelTrainingSampleCollection, ...],
    batch_sample_count: int,
    python_random_generator: Random,
    local_training_settings: LocalTrainingSettings,
    shared_feature_extractor: SharedFeatureExtractor,
    shared_parameter_optimizer: Optimizer | None,
    update_shared_features: bool,
) -> tuple[float, ...]:
    """完了した共同更新の損失を順に返し、借用した状態を巻き戻さない。"""
    if (
        type(requested_joint_update_iteration_count) is not int
        or requested_joint_update_iteration_count < 0
    ):
        raise ValueError(
            "requested_joint_update_iteration_countはbool以外の非負のbuiltin intが必要です。"
        )
    if requested_joint_update_iteration_count == 0:
        return ()
    bindings_by_model_id = _index_held_model_training_bindings(
        held_model_training_bindings=held_model_training_bindings
    )
    completed_joint_update_losses: list[float] = []
    for _joint_update_iteration_index in range(requested_joint_update_iteration_count):
        sampled_training_batches = sample_training_batches_for_held_models(
            ordered_model_training_samples=ordered_model_training_samples,
            held_model_ids=frozenset(bindings_by_model_id),
            batch_sample_count=batch_sample_count,
            python_random_generator=python_random_generator,
        )
        if not sampled_training_batches:
            continue
        participating_training_batches = tuple(
            ParticipatingModelTrainingBatch(
                classifier=bindings_by_model_id[training_batch.model_id].classifier,
                concept_specific_parameter_optimizer=bindings_by_model_id[
                    training_batch.model_id
                ].concept_specific_parameter_optimizer,
                input_features=training_batch.input_features,
                observed_class_labels=training_batch.observed_class_labels,
            )
            for training_batch in sampled_training_batches
        )
        joint_update_loss = perform_joint_model_parameter_update(
            participating_training_batches=participating_training_batches,
            local_training_settings=local_training_settings,
            shared_feature_extractor=shared_feature_extractor,
            shared_parameter_optimizer=shared_parameter_optimizer,
            update_shared_features=update_shared_features,
        )
        if joint_update_loss is None:
            raise RuntimeError("非空の参加batchに対して共同更新が損失を返しませんでした。")
        completed_joint_update_losses.append(joint_update_loss)
    return tuple(completed_joint_update_losses)
