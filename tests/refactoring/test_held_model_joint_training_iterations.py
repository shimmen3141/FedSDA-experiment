"""実旧の共同学習反復と新しい借用状態の接続を照合する。"""

import random
from unittest.mock import Mock

import pytest
import torch
from test_joint_model_parameter_update import (
    assert_joint_update_states_equal,
    build_joint_update_oracle_pair,
)

from federated_drift_experiment.clients.shared_backbone import (
    _SharedRepresentationFedSDAClientMixin,
)
from federated_learning_experiments.learning.training import (
    held_model_joint_training_iterations as training_iteration_module,
)
from federated_learning_experiments.learning.training.held_model_joint_training_iterations import (
    perform_held_model_joint_training_iterations,
)
from federated_learning_experiments.learning.training.held_model_training_binding import (
    HeldModelTrainingBinding,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ModelTrainingSampleCollection,
    ObservedTrainingSample,
)


def build_training_iteration_oracle_pair(*, class_count, optimizer_variant, monkeypatch):
    participating_training_batches, shared_parameter_optimizer, legacy_client = (
        build_joint_update_oracle_pair(
            class_count=class_count,
            batch_sample_counts=(9, 100),
            optimizer_variant=optimizer_variant,
            monkeypatch=monkeypatch,
        )
    )
    ordered_model_training_samples = tuple(
        ModelTrainingSampleCollection(
            model_id=model_id,
            training_samples=tuple(
                ObservedTrainingSample(
                    input_features=training_batch.input_features[
                        training_sample_index : training_sample_index + 1
                    ],
                    observed_class_labels=training_batch.observed_class_labels[
                        training_sample_index : training_sample_index + 1
                    ],
                )
                for training_sample_index in range(len(training_batch.input_features))
            ),
        )
        for model_id, training_batch in zip((4, -7), participating_training_batches)
    )
    legacy_client.train_data_store = {
        model_training_samples.model_id: [
            (training_sample.input_features, training_sample.observed_class_labels, 0)
            for training_sample in model_training_samples.training_samples
        ]
        for model_training_samples in ordered_model_training_samples
    }
    held_model_training_bindings = tuple(
        reversed(
            tuple(
                HeldModelTrainingBinding(
                    model_id=model_id,
                    classifier=training_batch.classifier,
                    concept_specific_parameter_optimizer=training_batch.concept_specific_parameter_optimizer,
                )
                for model_id, training_batch in zip((4, -7), participating_training_batches)
            )
        )
    )
    return (
        participating_training_batches,
        shared_parameter_optimizer,
        legacy_client,
        ordered_model_training_samples,
        held_model_training_bindings,
    )


def run_legacy_training_iterations(
    *, legacy_client, iteration_count, update_shared_features, initial_random_state
):
    sampled_batch_history = []
    legacy_weighted_loss_history = []

    def sample_legacy_training_batches():
        legacy_training_batches = _SharedRepresentationFedSDAClientMixin._sample_training_batches(
            legacy_client
        )
        sampled_batch_history.append(
            tuple(
                (model_id, input_features.clone(), observed_class_labels.clone())
                for model_id, input_features, observed_class_labels in legacy_training_batches
            )
        )
        return legacy_training_batches

    legacy_client._sample_training_batches = Mock(side_effect=sample_legacy_training_batches)
    loss_hooks = [
        classifier.loss_fn.register_forward_hook(
            lambda classifier, training_inputs, joint_update_loss: (
                legacy_weighted_loss_history.append(
                    joint_update_loss.detach().clone() * len(training_inputs[1])
                )
            )
        )
        for classifier in legacy_client.models.values()
    ]
    global_python_random_state = random.getstate()
    try:
        random.setstate(initial_random_state)
        _SharedRepresentationFedSDAClientMixin._train_heads_together(
            legacy_client, count_multiplier=iteration_count, update_backbone=update_shared_features
        )
        expected_random_state = random.getstate()
    finally:
        random.setstate(global_python_random_state)
        for loss_hook in loss_hooks:
            loss_hook.remove()
    expected_losses = []
    loss_offset = 0
    for legacy_training_batches in sampled_batch_history:
        participating_model_count = len(legacy_training_batches)
        if not participating_model_count:
            continue
        total_batch_sample_count = sum(
            len(input_features) for _, input_features, _ in legacy_training_batches
        )
        expected_losses.append(
            float(
                (
                    sum(
                        legacy_weighted_loss_history[
                            loss_offset : loss_offset + participating_model_count
                        ]
                    )
                    / total_batch_sample_count
                ).item()
            )
        )
        loss_offset += participating_model_count
    return tuple(expected_losses), expected_random_state, sampled_batch_history


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("update_shared_features", [True, False])
@pytest.mark.parametrize("iteration_count", [0, 1, 4])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_joint_training_iterations_match_legacy(
    class_count, optimizer_variant, update_shared_features, iteration_count, batch_size, monkeypatch
):
    global_python_random_state = random.getstate()
    global_torch_random_state = torch.get_rng_state()
    try:
        (
            participating_training_batches,
            shared_parameter_optimizer,
            legacy_client,
            ordered_model_training_samples,
            held_model_training_bindings,
        ) = build_training_iteration_oracle_pair(
            class_count=class_count, optimizer_variant=optimizer_variant, monkeypatch=monkeypatch
        )
        legacy_client.batch_size = batch_size
        python_random_generator = random.Random(731)
        expected_losses, expected_random_state, sampled_batch_history = (
            run_legacy_training_iterations(
                legacy_client=legacy_client,
                iteration_count=iteration_count,
                update_shared_features=update_shared_features,
                initial_random_state=python_random_generator.getstate(),
            )
        )
        original_sampling_function = (
            training_iteration_module.sample_training_batches_for_held_models
        )
        new_sampled_batch_history = []
        monkeypatch.setattr(
            training_iteration_module,
            "sample_training_batches_for_held_models",
            Mock(
                side_effect=lambda **sampling_request: (
                    new_sampled_batch_history.append(original_sampling_function(**sampling_request))
                    or new_sampled_batch_history[-1]
                )
            ),
        )
        actual_losses = perform_held_model_joint_training_iterations(
            requested_joint_update_iteration_count=iteration_count,
            held_model_training_bindings=held_model_training_bindings,
            ordered_model_training_samples=ordered_model_training_samples,
            batch_sample_count=batch_size,
            python_random_generator=python_random_generator,
            local_training_settings=LocalTrainingSettings(
                local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
                shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
            ),
            shared_feature_extractor=participating_training_batches[0].classifier.feature_extractor,
            shared_parameter_optimizer=shared_parameter_optimizer,
            update_shared_features=update_shared_features,
        )
        assert actual_losses == expected_losses
        assert python_random_generator.getstate() == expected_random_state
        assert len(new_sampled_batch_history) == len(sampled_batch_history) == iteration_count
        for sampled_training_batches, legacy_training_batches in zip(
            new_sampled_batch_history, sampled_batch_history
        ):
            assert tuple(batch.model_id for batch in sampled_training_batches) == (4, -7)
            for sampled_training_batch, (model_id, input_features, observed_class_labels) in zip(
                sampled_training_batches, legacy_training_batches
            ):
                assert sampled_training_batch.model_id == model_id
                assert torch.equal(sampled_training_batch.input_features, input_features)
                assert torch.equal(
                    sampled_training_batch.observed_class_labels, observed_class_labels
                )
        assert_joint_update_states_equal(
            participating_training_batches=participating_training_batches,
            shared_parameter_optimizer=shared_parameter_optimizer,
            legacy_client=legacy_client,
        )
        assert random.getstate() == global_python_random_state
    finally:
        random.setstate(global_python_random_state)
        torch.set_rng_state(global_torch_random_state)
