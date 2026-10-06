"""登録確認の学習標本付替えと借用参照を旧処理へ照合する。"""

import numpy as np
import pytest
import torch
from test_model_training_sample_storage import (
    append_legacy_training_samples,
    assert_training_sample_storage_matches_legacy,
    build_training_sample_storage_oracle,
    snapshot_training_sample_reference_ids,
)

from federated_drift_experiment.clients.base import BaseClient
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
