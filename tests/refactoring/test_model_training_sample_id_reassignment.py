"""登録確認の学習標本付替えと借用参照を旧処理へ照合する。"""

import random
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_held_model_joint_training_iterations import (
    build_training_iteration_oracle_pair,
    run_legacy_training_iterations,
)
from test_held_model_training_batch_sampling import assert_sampled_batches_equal
from test_joint_model_parameter_update import assert_joint_update_states_equal
from test_loss_statistics_model_id_reassignment import build_legacy_registration_client
from test_model_training_sample_storage import (
    append_legacy_training_samples,
    assert_training_sample_storage_matches_legacy,
    build_training_sample_storage_oracle,
    snapshot_training_sample_reference_ids,
)

from federated_drift_experiment.clients.base import BaseClient
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.training.held_model_joint_training_iterations import (
    perform_held_model_joint_training_iterations,
)
from federated_learning_experiments.learning.training.held_model_training_batch_sampling import (
    sample_training_batches_for_held_models,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)


class DerivedModelId(int):
    """拒否用int派生型。"""


@pytest.mark.parametrize("initial_model_ids", [(), (4,), (-7,), (-7, 4, 9), (4, -7, 9), (4, 9, -7)])
@pytest.mark.parametrize("reassigned_model_id", [4, 12, -7])
@pytest.mark.parametrize("source_sample_count", [0, 3])
def test_training_samples_id_reassignment_matches_legacy_registration(
    initial_model_ids, reassigned_model_id, source_sample_count
):
    sample_store, legacy_client, training_samples = build_training_sample_storage_oracle(
        storage_case="empty"
    )
    legacy_client.current_model_id = -7
    legacy_client.pending_model_params = None
    legacy_client.pending_model_stats = None
    legacy_client.pending_model_ready = True
    for model_id in initial_model_ids:
        if model_id == -7:
            training_sample_collection = (
                training_samples[0],
                training_samples[2],
                training_samples[0],
            )[:source_sample_count]
        else:
            training_sample_collection = (training_samples[1], training_samples[3])
        sample_store.append_model_training_samples(
            model_id=model_id, training_samples=training_sample_collection
        )
        append_legacy_training_samples(
            legacy_client=legacy_client,
            model_id=model_id,
            training_samples=training_sample_collection,
        )
    previous_snapshot = sample_store.snapshot_ordered_model_training_samples()
    previous_reference_ids = snapshot_training_sample_reference_ids(sample_store)
    assert (
        sample_store.reassign_model_training_samples_id(
            original_model_id=-7, reassigned_model_id=reassigned_model_id
        )
        is None
    )
    BaseClient.confirm_model_registration(legacy_client, reassigned_model_id)
    assert_training_sample_storage_matches_legacy(
        sample_store=sample_store, legacy_client=legacy_client
    )
    assert tuple(collection.model_id for collection in previous_snapshot) == initial_model_ids
    for collection, reference_ids in zip(previous_snapshot, previous_reference_ids):
        assert (
            tuple(
                (id(sample), id(sample.input_features), id(sample.observed_class_labels))
                for sample in collection.training_samples
            )
            == reference_ids[1]
        )


@pytest.mark.parametrize(
    "original_model_id,reassigned_model_id,expected_model_ids",
    [(0, -2, (19, -2)), (3, -4, (19, -4)), (3, 3, (19, 3)), (-5, -5, (19, -5))],
)
def test_training_samples_id_reassignment_accepts_signed_ids(
    original_model_id, reassigned_model_id, expected_model_ids
):
    sample_store, _, training_samples = build_training_sample_storage_oracle(storage_case="empty")
    for model_id in (original_model_id, 19):
        sample_store.append_model_training_samples(
            model_id=model_id, training_samples=training_samples
        )
    sample_store.reassign_model_training_samples_id(
        original_model_id=original_model_id, reassigned_model_id=reassigned_model_id
    )
    assert (
        tuple(
            collection.model_id
            for collection in sample_store.snapshot_ordered_model_training_samples()
        )
        == expected_model_ids
    )
    assert (
        sample_store.snapshot_ordered_model_training_samples()[-1].training_samples
        == training_samples
    )


@pytest.mark.parametrize("invalid_model_id", [True, 1.0, "4", None, DerivedModelId(4), np.int64(4)])
@pytest.mark.parametrize("invalid_parameter_name", ["original_model_id", "reassigned_model_id"])
@pytest.mark.parametrize("source_present", [False, True])
@pytest.mark.parametrize("destination_present", [False, True])
def test_training_samples_id_reassignment_rejects_invalid_ids_without_mutation(
    invalid_model_id, invalid_parameter_name, source_present, destination_present
):
    sample_store, _, training_samples = build_training_sample_storage_oracle(storage_case="empty")
    for model_id in (*((-7,) if source_present else ()), *((4,) if destination_present else ())):
        sample_store.append_model_training_samples(
            model_id=model_id, training_samples=training_samples
        )
    previous_reference_ids = snapshot_training_sample_reference_ids(sample_store)
    previous_parameter_values = tuple(
        (sample.input_features.clone(), sample.observed_class_labels.clone())
        for sample in training_samples
    )
    with pytest.raises(TypeError, match=invalid_parameter_name):
        sample_store.reassign_model_training_samples_id(
            original_model_id=(
                invalid_model_id if invalid_parameter_name == "original_model_id" else -7
            ),
            reassigned_model_id=(
                invalid_model_id if invalid_parameter_name == "reassigned_model_id" else 4
            ),
        )
    assert snapshot_training_sample_reference_ids(sample_store) == previous_reference_ids
    for sample, (input_features, observed_class_labels) in zip(
        training_samples, previous_parameter_values
    ):
        assert torch.equal(sample.input_features, input_features)
        assert torch.equal(sample.observed_class_labels, observed_class_labels)


def test_reassigned_training_samples_preserve_snapshots_and_support_append():
    sample_store, _, training_samples = build_training_sample_storage_oracle(storage_case="empty")
    sample_store.append_model_training_samples(
        model_id=-7, training_samples=(training_samples[0],) * 2
    )
    sample_store.append_model_training_samples(model_id=4, training_samples=(training_samples[1],))
    previous_snapshot = sample_store.snapshot_ordered_model_training_samples()
    sample_store.reassign_model_training_samples_id(original_model_id=-7, reassigned_model_id=4)
    assert tuple(collection.model_id for collection in previous_snapshot) == (-7, 4)
    assert previous_snapshot[1].training_samples[0] is training_samples[1]
    sample_store.append_model_training_samples(model_id=4, training_samples=(training_samples[2],))
    training_sample_collection = sample_store.snapshot_ordered_model_training_samples()[0]
    assert training_sample_collection.model_id == 4
    assert len(training_sample_collection.training_samples) == 3
    assert training_sample_collection.training_samples[0] is training_samples[0]
    assert training_sample_collection.training_samples[1] is training_samples[0]
    assert training_sample_collection.training_samples[2] is training_samples[2]
    assert len(previous_snapshot[0].training_samples) == 2
    training_samples[0].input_features.add_(1)
    assert torch.equal(
        previous_snapshot[0].training_samples[0].input_features,
        training_sample_collection.training_samples[0].input_features,
    )


@pytest.mark.parametrize("payload_case", ["opaque", "float64", "meta"])
def test_training_samples_id_reassignment_does_not_inspect_payloads(payload_case):
    input_features = (
        object()
        if payload_case == "opaque"
        else torch.ones(
            (1, 2),
            dtype=torch.float64 if payload_case == "float64" else torch.float32,
            device="meta" if payload_case == "meta" else "cpu",
        )
    )
    observed_class_labels = object()
    training_sample = ObservedTrainingSample(
        input_features=input_features, observed_class_labels=observed_class_labels
    )
    sample_store = ModelTrainingSampleStore()
    sample_store.append_model_training_samples(model_id=-7, training_samples=(training_sample,))
    sample_store.reassign_model_training_samples_id(original_model_id=-7, reassigned_model_id=4)
    training_sample_collection = sample_store.snapshot_ordered_model_training_samples()[0]
    assert training_sample_collection.model_id == 4
    assert training_sample_collection.training_samples[0] is training_sample
    assert training_sample.input_features is input_features
    assert training_sample.observed_class_labels is observed_class_labels


@pytest.mark.parametrize("class_count", [2, 4])
@pytest.mark.parametrize("optimizer_variant", ["standard", "amsgrad", "sgd"])
@pytest.mark.parametrize("update_shared_features", [True, False])
def test_reassigned_training_samples_continue_sampling_and_joint_updates(
    class_count, optimizer_variant, update_shared_features, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        (
            training_batches,
            shared_optimizer,
            legacy_client,
            ordered_training_samples,
            training_bindings,
        ) = build_training_iteration_oracle_pair(
            class_count=class_count, optimizer_variant=optimizer_variant, monkeypatch=monkeypatch
        )
        sample_store = ModelTrainingSampleStore()
        for training_sample_collection in ordered_training_samples:
            sample_store.append_model_training_samples(
                model_id=training_sample_collection.model_id,
                training_samples=training_sample_collection.training_samples,
            )
        legacy_registration_client = build_legacy_registration_client(
            loss_statistics_store=ModelAndClassLossStatisticsStore()
        )
        legacy_registration_client.models = legacy_client.models
        legacy_registration_client.train_data_store = legacy_client.train_data_store
        legacy_registration_client.model_training_examples = legacy_client.model_training_examples
        legacy_registration_client.model_optimizer_steps = legacy_client.model_optimizer_steps
        legacy_client.batch_size = 3
        python_random_generator = random.Random(731)
        previous_random_states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        previous_numeric_environment = (
            torch.get_default_dtype(),
            torch.get_default_device(),
            torch.is_grad_enabled(),
        )
        for step_index in range(3):
            if step_index == 1:
                previous_snapshot = sample_store.snapshot_ordered_model_training_samples()
                sample_store.reassign_model_training_samples_id(
                    original_model_id=-7, reassigned_model_id=12
                )
                legacy_registration_client.confirm_model_registration(12)
                # 上位がbinding IDを明示対応する。旧binding自体は変更しない。
                training_bindings = tuple(
                    replace(
                        training_binding,
                        model_id=12
                        if training_binding.model_id == -7
                        else training_binding.model_id,
                    )
                    for training_binding in training_bindings
                )
                assert tuple(collection.model_id for collection in previous_snapshot) == (4, -7)
            assert_training_sample_storage_matches_legacy(
                sample_store=sample_store, legacy_client=legacy_client
            )
            expected_losses, expected_random_state, sampled_batch_history = (
                run_legacy_training_iterations(
                    legacy_client=legacy_client,
                    iteration_count=1,
                    update_shared_features=update_shared_features,
                    initial_random_state=python_random_generator.getstate(),
                )
            )
            # previewは独立Randomを使い、実更新へ渡すRandomは消費しない。
            sampled_batches = sample_training_batches_for_held_models(
                held_model_ids=frozenset(binding.model_id for binding in training_bindings),
                ordered_model_training_samples=sample_store.snapshot_ordered_model_training_samples(),
                batch_sample_count=3,
                python_random_generator=deepcopy(python_random_generator),
            )
            assert len(sampled_batch_history) == 1
            assert_sampled_batches_equal(
                sampled_batches=sampled_batches, legacy_batches=sampled_batch_history[0]
            )
            actual_losses = perform_held_model_joint_training_iterations(
                requested_joint_update_iteration_count=1,
                held_model_training_bindings=training_bindings,
                ordered_model_training_samples=sample_store.snapshot_ordered_model_training_samples(),
                batch_sample_count=3,
                python_random_generator=python_random_generator,
                local_training_settings=LocalTrainingSettings(
                    local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
                    shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
                ),
                shared_feature_extractor=training_batches[0].classifier.feature_extractor,
                shared_parameter_optimizer=shared_optimizer,
                update_shared_features=update_shared_features,
            )
            assert actual_losses == expected_losses
            assert python_random_generator.getstate() == expected_random_state
            assert_joint_update_states_equal(
                participating_training_batches=training_batches,
                shared_parameter_optimizer=shared_optimizer,
                legacy_client=legacy_client,
            )
        assert tuple(
            collection.model_id
            for collection in sample_store.snapshot_ordered_model_training_samples()
        ) == (4, 12)
        assert random.getstate() == previous_random_states[0]
        assert np.random.get_state()[0] == previous_random_states[1][0]
        assert np.array_equal(np.random.get_state()[1], previous_random_states[1][1])
        assert np.random.get_state()[2:] == previous_random_states[1][2:]
        assert torch.equal(torch.get_rng_state(), previous_random_states[2])
        assert previous_numeric_environment == (
            torch.get_default_dtype(),
            torch.get_default_device(),
            torch.is_grad_enabled(),
        )
