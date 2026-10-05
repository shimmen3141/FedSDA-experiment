"""学習標本保持を実旧追加・サーバ対応へ直接照合する。"""

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from federated_drift_experiment.clients.base import BaseClient
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ModelTrainingSampleCollection,
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)


class ModelIdIntSubclass(int):
    pass


class TrainingSamplesTupleSubclass(tuple):
    pass


class ModelIdMappingDictSubclass(dict):
    pass


class ObservedTrainingSampleSubclass(ObservedTrainingSample):
    pass


def build_training_sample_storage_oracle(*, storage_case="mixed"):
    sample_store = ModelTrainingSampleStore()
    legacy_client = SimpleNamespace(
        current_model_id=999,
        model_stats={},
        stored_data={},
        train_data_store=defaultdict(list),
        model_training_examples={},
        model_optimizer_steps={},
        model_concept_counts={},
        models={},
        stored_data_limit=1,
        _after_models_rebuilt=Mock(),
        _record_model_concept=Mock(),
        _record_model_compute=Mock(),
        _update_model_stats=Mock(),
    )
    training_samples = tuple(
        ObservedTrainingSample(
            input_features=torch.tensor([[float(sample_index), -1.0]], device="cpu"),
            observed_class_labels=torch.tensor([[float(sample_index % 2)]], device="cpu"),
        )
        for sample_index in range(5)
    )
    if storage_case == "mixed":
        for model_id, sample_count in ((8, 4), (-7, 3), (2, 0), (4, 1)):
            sample_store.append_model_training_samples(
                model_id=model_id, training_samples=training_samples[:sample_count]
            )
            append_legacy_training_samples(
                legacy_client=legacy_client,
                model_id=model_id,
                training_samples=training_samples[:sample_count],
            )
        sample_store.append_model_training_samples(
            model_id=8, training_samples=(training_samples[0], training_samples[4])
        )
        append_legacy_training_samples(
            legacy_client=legacy_client,
            model_id=8,
            training_samples=(training_samples[0], training_samples[4]),
        )
    return sample_store, legacy_client, training_samples


def append_legacy_training_samples(*, legacy_client, model_id, training_samples):
    if not training_samples:
        # 候補標本の直接extendと同じ空列作成。空absorbとは区別する。
        legacy_client.train_data_store[model_id].extend(())
        return
    legacy_model = SimpleNamespace(get_absolute_error=Mock(return_value=0.25))
    legacy_client.models[model_id] = legacy_model
    BaseClient._absorb_into_store(
        legacy_client,
        model_id,
        [
            (training_sample.input_features, training_sample.observed_class_labels, 77)
            for training_sample in training_samples
        ],
    )


def remap_legacy_training_samples(*, legacy_client, model_id_mapping):
    BaseClient.apply_server_mapping(legacy_client, model_id_mapping, {})


def assert_training_sample_storage_matches_legacy(*, sample_store, legacy_client):
    previous_snapshot = sample_store.snapshot_ordered_model_training_samples()
    assert type(previous_snapshot) is tuple
    assert tuple(
        model_training_samples.model_id for model_training_samples in previous_snapshot
    ) == (tuple(legacy_client.train_data_store))
    for model_training_samples in previous_snapshot:
        assert type(model_training_samples) is ModelTrainingSampleCollection
        assert type(model_training_samples.training_samples) is tuple
        assert len(model_training_samples.training_samples) == len(
            legacy_client.train_data_store[model_training_samples.model_id]
        )
        for training_sample, legacy_training_sample in zip(
            model_training_samples.training_samples,
            legacy_client.train_data_store[model_training_samples.model_id],
        ):
            assert training_sample.input_features is legacy_training_sample[0]
            assert training_sample.observed_class_labels is legacy_training_sample[1]


def snapshot_training_sample_reference_ids(sample_store):
    return tuple(
        (
            model_training_samples.model_id,
            tuple(
                (
                    id(training_sample),
                    id(training_sample.input_features),
                    id(training_sample.observed_class_labels),
                )
                for training_sample in model_training_samples.training_samples
            ),
        )
        for model_training_samples in sample_store.snapshot_ordered_model_training_samples()
    )


@pytest.mark.parametrize("storage_case", ["empty", "mixed"])
@pytest.mark.parametrize(
    "model_id_mapping",
    [
        {},
        {8: 8},
        {8: 9},
        {8: 4, -7: 4},
        {8: -7, -7: 2},
        {8: -7, -7: 8},
        {8: 2, -7: 2, 4: 2},
        {300: -99},
    ],
)
def test_storage_remapping_matches_actual_legacy(storage_case, model_id_mapping):
    sample_store, legacy_client, training_samples = build_training_sample_storage_oracle(
        storage_case=storage_case
    )
    assert_training_sample_storage_matches_legacy(
        sample_store=sample_store, legacy_client=legacy_client
    )
    previous_snapshot = sample_store.snapshot_ordered_model_training_samples()
    sample_store.remap_model_training_sample_collections(model_id_mapping=model_id_mapping)
    remap_legacy_training_samples(legacy_client=legacy_client, model_id_mapping=model_id_mapping)
    assert_training_sample_storage_matches_legacy(
        sample_store=sample_store, legacy_client=legacy_client
    )
    if storage_case == "mixed":
        assert tuple(
            model_training_samples.model_id for model_training_samples in previous_snapshot
        ) == (8, -7, 2, 4)
        assert previous_snapshot[0].training_samples[0] is training_samples[0]
        assert len(previous_snapshot[0].training_samples) == 6


def test_empty_read_does_not_create_models_and_empty_append_preserves_first_seen_order():
    sample_store = ModelTrainingSampleStore()
    assert sample_store.snapshot_ordered_model_training_samples() == ()
    assert sample_store.snapshot_ordered_model_training_samples() == ()
    for model_id in (-7, 8, -7, 2):
        sample_store.append_model_training_samples(model_id=model_id, training_samples=())
    assert tuple(
        model_training_samples.model_id
        for model_training_samples in sample_store.snapshot_ordered_model_training_samples()
    ) == (-7, 8, 2)


def test_snapshot_structure_is_independent_but_sample_payload_is_borrowed():
    sample_store, legacy_client, training_samples = build_training_sample_storage_oracle()
    previous_snapshot = sample_store.snapshot_ordered_model_training_samples()
    sample_store.append_model_training_samples(model_id=8, training_samples=(training_samples[2],))
    assert len(previous_snapshot[0].training_samples) == 6
    assert len(sample_store.snapshot_ordered_model_training_samples()[0].training_samples) == 7
    assert previous_snapshot[0].training_samples[0] is training_samples[0]
    assert previous_snapshot[0].training_samples[4] is training_samples[0]
    training_samples[0].input_features.add_(10)
    assert (
        previous_snapshot[0].training_samples[0].input_features
        is training_samples[0].input_features
    )
    assert previous_snapshot[0].training_samples[0].input_features[0, 0].item() == 10
    with pytest.raises(AttributeError):
        previous_snapshot[0].model_id = 20


@pytest.mark.parametrize("invalid_value", [True, False, 1.0, "8", None, ModelIdIntSubclass(8)])
def test_invalid_append_model_id_preserves_state(invalid_value):
    sample_store, legacy_client, training_samples = build_training_sample_storage_oracle()
    previous_reference_ids = snapshot_training_sample_reference_ids(sample_store)
    with pytest.raises(TypeError):
        sample_store.append_model_training_samples(
            model_id=invalid_value, training_samples=training_samples
        )
    assert snapshot_training_sample_reference_ids(sample_store) == previous_reference_ids


@pytest.mark.parametrize("invalid_value", [[], None, {}, TrainingSamplesTupleSubclass(())])
def test_invalid_training_sample_container_preserves_state(invalid_value):
    sample_store, legacy_client, training_samples = build_training_sample_storage_oracle()
    previous_reference_ids = snapshot_training_sample_reference_ids(sample_store)
    with pytest.raises(TypeError):
        sample_store.append_model_training_samples(model_id=100, training_samples=invalid_value)
    assert snapshot_training_sample_reference_ids(sample_store) == previous_reference_ids


@pytest.mark.parametrize("model_id", [8, 100])
@pytest.mark.parametrize("invalid_value", [None, (), object()])
def test_invalid_tail_sample_does_not_partially_append(model_id, invalid_value):
    sample_store, legacy_client, training_samples = build_training_sample_storage_oracle()
    previous_reference_ids = snapshot_training_sample_reference_ids(sample_store)
    with pytest.raises(TypeError):
        sample_store.append_model_training_samples(
            model_id=model_id, training_samples=(training_samples[0], invalid_value)
        )
    assert snapshot_training_sample_reference_ids(sample_store) == previous_reference_ids


def test_sample_subclass_is_rejected_without_state_change():
    sample_store, legacy_client, training_samples = build_training_sample_storage_oracle()
    previous_reference_ids = snapshot_training_sample_reference_ids(sample_store)
    with pytest.raises(TypeError):
        sample_store.append_model_training_samples(
            model_id=8,
            training_samples=(
                ObservedTrainingSampleSubclass(
                    input_features=training_samples[0].input_features,
                    observed_class_labels=training_samples[0].observed_class_labels,
                ),
            ),
        )
    assert snapshot_training_sample_reference_ids(sample_store) == previous_reference_ids


@pytest.mark.parametrize(
    "invalid_value",
    [
        None,
        [],
        (),
        ModelIdMappingDictSubclass({}),
        {8: 4, "unused": 5},
        {8: 4, 400: True},
        {8: 4, 400: 2.0},
        {8: 4, ModelIdIntSubclass(400): 2},
        {8: 4, 400: ModelIdIntSubclass(2)},
    ],
)
def test_invalid_mapping_preserves_all_samples_and_order(invalid_value):
    sample_store, legacy_client, training_samples = build_training_sample_storage_oracle()
    previous_reference_ids = snapshot_training_sample_reference_ids(sample_store)
    with pytest.raises(TypeError):
        sample_store.remap_model_training_sample_collections(model_id_mapping=invalid_value)
    assert snapshot_training_sample_reference_ids(sample_store) == previous_reference_ids


def test_model_ids_have_no_range_limit_and_mapping_preserves_duplicate_references():
    sample_store, legacy_client, training_samples = build_training_sample_storage_oracle(
        storage_case="empty"
    )
    for model_id in (10**100, -(10**100), 0):
        sample_store.append_model_training_samples(
            model_id=model_id, training_samples=(training_samples[0],) * 20
        )
    sample_store.remap_model_training_sample_collections(
        model_id_mapping={10**100: 0, -(10**100): 0}
    )
    previous_snapshot = sample_store.snapshot_ordered_model_training_samples()
    assert len(previous_snapshot) == 1
    assert previous_snapshot[0].model_id == 0
    assert len(previous_snapshot[0].training_samples) == 60
    assert all(
        training_sample is training_samples[0]
        for training_sample in previous_snapshot[0].training_samples
    )
