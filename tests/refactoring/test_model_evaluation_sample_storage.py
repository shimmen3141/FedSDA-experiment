"""評価標本の保持/抽出/ID対応を実旧処理へ照合する。"""

import random
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from test_classifier_bounded_loss_evaluation import build_bounded_loss_oracle_pair

from federated_drift_experiment import config
from federated_drift_experiment.clients.base import BaseClient
from federated_learning_experiments.evaluation.model_evaluation_sample_records import (
    ModelEvaluationSampleCollection,
    ObservedEvaluationSample,
)
from federated_learning_experiments.evaluation.model_evaluation_sample_store import (
    ModelEvaluationSampleStore,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
)


class ModelIdIntSubclass(int):
    pass


class EvaluationSamplesTupleSubclass(tuple):
    pass


class ModelIdMappingDictSubclass(dict):
    pass


class ObservedEvaluationSampleSubclass(ObservedEvaluationSample):
    pass


class RandomSubclass(random.Random):
    pass


def build_evaluation_storage_oracle(*, capacity=7, append_sample_count=20):
    sample_store = ModelEvaluationSampleStore(
        maximum_stored_sample_count_per_model=capacity, added_batch_sample_count=append_sample_count
    )
    legacy_client = SimpleNamespace(
        stored_data=defaultdict(list),
        stored_data_limit=capacity,
        current_model_id=9999,
        model_stats={},
        train_data_store=defaultdict(list),
        model_training_examples={},
        model_optimizer_steps={},
        model_concept_counts={},
        models={},
        _after_models_rebuilt=Mock(),
        pending_model_params=None,
        pending_model_stats=None,
        pending_model_ready=True,
        mapping_change_positions=[],
        processed_samples=0,
        _record_adaptation_event=Mock(),
    )
    evaluation_samples = tuple(
        ObservedEvaluationSample(
            input_features=torch.tensor([[float(sample_count), -1.0]], device="cpu"),
            observed_class_labels=torch.tensor([[float(sample_count % 2)]], device="cpu"),
        )
        for sample_count in range(8)
    )
    return sample_store, legacy_client, evaluation_samples


def run_legacy_evaluation_operation(
    *, legacy_client, operation_name, operation_arguments, initial_random_state
):
    previous_random_states = random.getstate()
    try:
        random.setstate(initial_random_state)
        if operation_name == "append":
            BaseClient._store_evaluation_data(legacy_client, **operation_arguments)
        elif operation_name == "reassign":
            legacy_client.current_model_id = operation_arguments["original_model_id"]
            BaseClient.confirm_model_registration(
                legacy_client, operation_arguments["reassigned_model_id"]
            )
        else:
            BaseClient.apply_server_mapping(
                legacy_client, operation_arguments["model_id_mapping"], {}
            )
        return random.getstate()
    finally:
        random.setstate(previous_random_states)


def assert_evaluation_storage_matches_legacy(*, sample_store, legacy_client):
    previous_snapshot = sample_store.snapshot_ordered_model_evaluation_samples()
    assert tuple(collection.model_id for collection in previous_snapshot) == tuple(
        legacy_client.stored_data
    )
    for collection in previous_snapshot:
        assert type(collection) is ModelEvaluationSampleCollection
        assert type(collection.evaluation_samples) is tuple
        assert len(collection.evaluation_samples) == len(
            legacy_client.stored_data[collection.model_id]
        )
        for evaluation_sample, sampled_records in zip(
            collection.evaluation_samples, legacy_client.stored_data[collection.model_id]
        ):
            assert evaluation_sample.input_features is sampled_records[0]
            assert evaluation_sample.observed_class_labels is sampled_records[1]


def snapshot_evaluation_reference_ids(sample_store):
    return tuple(
        (
            collection.model_id,
            tuple(
                (
                    id(evaluation_sample),
                    id(evaluation_sample.input_features),
                    id(evaluation_sample.observed_class_labels),
                )
                for evaluation_sample in collection.evaluation_samples
            ),
        )
        for collection in sample_store.snapshot_ordered_model_evaluation_samples()
    )


@pytest.mark.parametrize("capacity", [1, 3, 7])
@pytest.mark.parametrize("append_sample_count", [0, 1, 4, 20])
@pytest.mark.parametrize("model_id", [-7, 0, 4])
@pytest.mark.parametrize("sample_count", [0, 1, 4, 8])
def test_evaluation_sample_append_matches_legacy(
    capacity, append_sample_count, model_id, sample_count, monkeypatch
):
    sample_store, legacy_client, evaluation_samples = build_evaluation_storage_oracle(
        capacity=capacity, append_sample_count=append_sample_count
    )
    monkeypatch.setattr(config, "EVAL_STORE_SAMPLE_SIZE", append_sample_count)
    python_random_generator = random.Random(731)
    # 初出順、反復追加、重複payloadと全件抽出の消費順も旧処理へ照合する。
    operation_arguments = (
        (8, evaluation_samples[:2]),
        (model_id, evaluation_samples[:sample_count]),
        (model_id, (evaluation_samples[0],) * sample_count),
    )
    for model_id, sampled_records in operation_arguments:
        expected_random_state = run_legacy_evaluation_operation(
            legacy_client=legacy_client,
            operation_name="append",
            operation_arguments={
                "model_id": model_id,
                "data_list": [
                    (evaluation_sample.input_features, evaluation_sample.observed_class_labels)
                    for evaluation_sample in sampled_records
                ],
            },
            initial_random_state=python_random_generator.getstate(),
        )
        sample_store.sample_and_append_model_evaluation_samples(
            model_id=model_id,
            evaluation_samples=sampled_records,
            python_random_generator=python_random_generator,
        )
        assert python_random_generator.getstate() == expected_random_state
        assert_evaluation_storage_matches_legacy(
            sample_store=sample_store, legacy_client=legacy_client
        )


@pytest.mark.parametrize("reassigned_model_id", [-7, 4, 12])
@pytest.mark.parametrize("sample_count", [0, 3])
@pytest.mark.parametrize("storage_case", ["present", "absent", "source_first"])
def test_evaluation_sample_single_id_reassignment_matches_legacy(
    reassigned_model_id, sample_count, storage_case, monkeypatch
):
    sample_store, legacy_client, evaluation_samples = build_evaluation_storage_oracle()
    monkeypatch.setattr(config, "EVAL_STORE_SAMPLE_SIZE", 20)
    python_random_generator = random.Random(731)
    model_ids = (8, 4, 2) if storage_case == "source_first" else (4, 8, 2)
    for model_id in model_ids:
        sampled_records = (
            evaluation_samples[:sample_count] if model_id == 8 else evaluation_samples[:2]
        )
        sample_store.sample_and_append_model_evaluation_samples(
            model_id=model_id,
            evaluation_samples=sampled_records,
            python_random_generator=python_random_generator,
        )
    sample_store.remap_model_evaluation_sample_collections(
        model_id_mapping={8: -7 if storage_case != "absent" else 9},
        python_random_generator=python_random_generator,
    )
    for collection in sample_store.snapshot_ordered_model_evaluation_samples():
        legacy_client.stored_data[collection.model_id] = [
            (evaluation_sample.input_features, evaluation_sample.observed_class_labels)
            for evaluation_sample in collection.evaluation_samples
        ]
    previous_snapshot = sample_store.snapshot_ordered_model_evaluation_samples()
    previous_reference_ids = snapshot_evaluation_reference_ids(sample_store)
    expected_random_state = run_legacy_evaluation_operation(
        legacy_client=legacy_client,
        operation_name="reassign",
        operation_arguments={"original_model_id": -7, "reassigned_model_id": reassigned_model_id},
        initial_random_state=python_random_generator.getstate(),
    )
    sample_store.reassign_model_evaluation_samples_id(
        original_model_id=-7, reassigned_model_id=reassigned_model_id
    )
    assert python_random_generator.getstate() == expected_random_state
    assert_evaluation_storage_matches_legacy(sample_store=sample_store, legacy_client=legacy_client)
    assert (
        tuple(
            (
                collection.model_id,
                tuple(
                    (
                        id(evaluation_sample),
                        id(evaluation_sample.input_features),
                        id(evaluation_sample.observed_class_labels),
                    )
                    for evaluation_sample in collection.evaluation_samples
                ),
            )
            for collection in previous_snapshot
        )
        == previous_reference_ids
    )


@pytest.mark.parametrize("capacity", [1, 3, 7])
@pytest.mark.parametrize("sample_count", [0, 3, 6])
@pytest.mark.parametrize(
    "model_id_mapping",
    [
        {},
        {4: 12},
        {4: 8, 8: 2},
        {4: 8, 8: 4},
        {4: 12, 8: 12, 2: 12},
        {8: -7, 4: -7},
        {999: 12},
        {4: 4, 8: 8},
    ],
)
def test_evaluation_sample_mapping_matches_legacy(
    capacity, sample_count, model_id_mapping, monkeypatch
):
    sample_store, legacy_client, evaluation_samples = build_evaluation_storage_oracle(
        capacity=capacity
    )
    monkeypatch.setattr(config, "EVAL_STORE_SAMPLE_SIZE", 20)
    python_random_generator = random.Random(731)
    for model_id, sampled_records in (
        (4, evaluation_samples[:sample_count]),
        (8, evaluation_samples[:sample_count]),
        (2, ()),
    ):
        expected_random_state = run_legacy_evaluation_operation(
            legacy_client=legacy_client,
            operation_name="append",
            operation_arguments={
                "model_id": model_id,
                "data_list": [
                    (evaluation_sample.input_features, evaluation_sample.observed_class_labels)
                    for evaluation_sample in sampled_records
                ],
            },
            initial_random_state=python_random_generator.getstate(),
        )
        sample_store.sample_and_append_model_evaluation_samples(
            model_id=model_id,
            evaluation_samples=sampled_records,
            python_random_generator=python_random_generator,
        )
        assert python_random_generator.getstate() == expected_random_state
    for _ in range(2):
        expected_random_state = run_legacy_evaluation_operation(
            legacy_client=legacy_client,
            operation_name="remap",
            operation_arguments={"model_id_mapping": model_id_mapping},
            initial_random_state=python_random_generator.getstate(),
        )
        sample_store.remap_model_evaluation_sample_collections(
            model_id_mapping=model_id_mapping, python_random_generator=python_random_generator
        )
        assert python_random_generator.getstate() == expected_random_state
        assert_evaluation_storage_matches_legacy(
            sample_store=sample_store, legacy_client=legacy_client
        )


@pytest.mark.parametrize(
    "invalid_parameter_name", ["maximum_stored_sample_count_per_model", "added_batch_sample_count"]
)
@pytest.mark.parametrize("invalid_value", [True, -1, 1.0, None, np.int64(2), ModelIdIntSubclass(2)])
def test_evaluation_storage_rejects_invalid_fixed_conditions(invalid_parameter_name, invalid_value):
    operation_arguments = {
        "maximum_stored_sample_count_per_model": 3,
        "added_batch_sample_count": 2,
    }
    operation_arguments[invalid_parameter_name] = invalid_value
    with pytest.raises(ValueError, match=invalid_parameter_name):
        ModelEvaluationSampleStore(**operation_arguments)


def test_evaluation_storage_rejects_zero_capacity():
    with pytest.raises(ValueError, match="maximum_stored_sample_count_per_model"):
        ModelEvaluationSampleStore(
            maximum_stored_sample_count_per_model=0, added_batch_sample_count=0
        )


@pytest.mark.parametrize("model_id", [-7, 4])
@pytest.mark.parametrize(
    "operation_name,invalid_parameter_name,invalid_value",
    [
        ("sample_and_append_model_evaluation_samples", "model_id", True),
        ("sample_and_append_model_evaluation_samples", "model_id", ModelIdIntSubclass(4)),
        ("sample_and_append_model_evaluation_samples", "evaluation_samples", []),
        (
            "sample_and_append_model_evaluation_samples",
            "evaluation_samples",
            EvaluationSamplesTupleSubclass(),
        ),
        ("sample_and_append_model_evaluation_samples", "evaluation_samples", (None,)),
        (
            "sample_and_append_model_evaluation_samples",
            "evaluation_samples",
            (ObservedEvaluationSampleSubclass(input_features=None, observed_class_labels=None),),
        ),
        ("sample_and_append_model_evaluation_samples", "python_random_generator", None),
        (
            "sample_and_append_model_evaluation_samples",
            "python_random_generator",
            RandomSubclass(731),
        ),
        ("reassign_model_evaluation_samples_id", "original_model_id", True),
        ("reassign_model_evaluation_samples_id", "reassigned_model_id", np.int64(4)),
        ("remap_model_evaluation_sample_collections", "model_id_mapping", []),
        (
            "remap_model_evaluation_sample_collections",
            "model_id_mapping",
            ModelIdMappingDictSubclass(),
        ),
        ("remap_model_evaluation_sample_collections", "model_id_mapping", {4: 8, 12: True}),
        ("remap_model_evaluation_sample_collections", "model_id_mapping", {4: 8, "bad": 12}),
        ("remap_model_evaluation_sample_collections", "python_random_generator", None),
        (
            "remap_model_evaluation_sample_collections",
            "python_random_generator",
            RandomSubclass(731),
        ),
    ],
)
def test_evaluation_storage_rejects_before_state_or_random_changes(
    model_id, operation_name, invalid_parameter_name, invalid_value
):
    sample_store, _, evaluation_samples = build_evaluation_storage_oracle()
    python_random_generator = random.Random(731)
    sample_store.sample_and_append_model_evaluation_samples(
        model_id=4,
        evaluation_samples=evaluation_samples,
        python_random_generator=python_random_generator,
    )
    previous_reference_ids = snapshot_evaluation_reference_ids(sample_store)
    previous_random_states = python_random_generator.getstate()
    parameter_snapshot = tuple(
        (evaluation_sample.input_features.clone(), evaluation_sample.observed_class_labels.clone())
        for evaluation_sample in evaluation_samples
    )
    if operation_name == "sample_and_append_model_evaluation_samples":
        operation_arguments = dict(
            model_id=model_id,
            evaluation_samples=evaluation_samples,
            python_random_generator=python_random_generator,
        )
    elif operation_name == "reassign_model_evaluation_samples_id":
        operation_arguments = dict(original_model_id=model_id, reassigned_model_id=12)
    else:
        operation_arguments = dict(
            model_id_mapping={4: 12}, python_random_generator=python_random_generator
        )
    operation_arguments[invalid_parameter_name] = invalid_value
    with pytest.raises(TypeError, match=invalid_parameter_name):
        getattr(sample_store, operation_name)(**operation_arguments)
    assert snapshot_evaluation_reference_ids(sample_store) == previous_reference_ids
    assert python_random_generator.getstate() == previous_random_states
    for evaluation_sample, sampled_records in zip(evaluation_samples, parameter_snapshot):
        assert torch.equal(evaluation_sample.input_features, sampled_records[0])
        assert torch.equal(evaluation_sample.observed_class_labels, sampled_records[1])


@pytest.mark.parametrize("input_case", ["opaque", "float64", "meta"])
def test_evaluation_storage_borrows_uninspected_payloads_and_preserves_snapshots(input_case):
    input_features = (
        object()
        if input_case == "opaque"
        else torch.ones(
            (1, 2), dtype=torch.float64, device="meta" if input_case == "meta" else "cpu"
        )
    )
    evaluation_sample = ObservedEvaluationSample(
        input_features=input_features, observed_class_labels=object()
    )
    sample_store = ModelEvaluationSampleStore(
        maximum_stored_sample_count_per_model=2, added_batch_sample_count=2
    )
    python_random_generator = random.Random(731)
    sample_store.sample_and_append_model_evaluation_samples(
        model_id=4,
        evaluation_samples=(evaluation_sample,) * 2,
        python_random_generator=python_random_generator,
    )
    previous_snapshot = sample_store.snapshot_ordered_model_evaluation_samples()
    sample_store.reassign_model_evaluation_samples_id(original_model_id=4, reassigned_model_id=-7)
    sample_store.remap_model_evaluation_sample_collections(
        model_id_mapping={-7: 12}, python_random_generator=python_random_generator
    )
    sample_store.sample_and_append_model_evaluation_samples(
        model_id=12,
        evaluation_samples=(evaluation_sample,),
        python_random_generator=python_random_generator,
    )
    collection = sample_store.snapshot_ordered_model_evaluation_samples()[0]
    assert collection.model_id == 12 and len(collection.evaluation_samples) == 2
    assert previous_snapshot[0].model_id == 4 and len(previous_snapshot[0].evaluation_samples) == 2
    assert all(
        sampled_records is evaluation_sample for sampled_records in collection.evaluation_samples
    )
    assert all(
        sampled_records.input_features is input_features
        for sampled_records in previous_snapshot[0].evaluation_samples
    )
    with pytest.raises(AttributeError):
        previous_snapshot[0].model_id = 99


@pytest.mark.parametrize("append_sample_count", [0, 2])
@pytest.mark.parametrize("model_id", [-7, 4])
@pytest.mark.parametrize("operation_name", ["append", "remap"])
def test_evaluation_storage_checks_random_on_empty_noop_paths(
    append_sample_count, model_id, operation_name
):
    sample_store, _, _ = build_evaluation_storage_oracle(append_sample_count=append_sample_count)
    previous_snapshot = sample_store.snapshot_ordered_model_evaluation_samples()
    with pytest.raises(TypeError, match="python_random_generator"):
        if operation_name == "append":
            sample_store.sample_and_append_model_evaluation_samples(
                model_id=model_id, evaluation_samples=(), python_random_generator=None
            )
        else:
            sample_store.remap_model_evaluation_sample_collections(
                model_id_mapping={}, python_random_generator=None
            )
    assert sample_store.snapshot_ordered_model_evaluation_samples() == previous_snapshot


@pytest.mark.parametrize("class_count", [2, 4])
def test_evaluation_storage_connects_reassignment_and_multiple_overflows_to_losses(
    class_count, monkeypatch
):
    with torch.random.fork_rng(devices=[]):
        classifier, legacy_classifier, sampled_records = build_bounded_loss_oracle_pair(
            class_count=class_count, sample_count=6, monkeypatch=monkeypatch
        )
        evaluation_samples = tuple(
            ObservedEvaluationSample(
                input_features=sampled_records.input_features[sample_count : sample_count + 1],
                observed_class_labels=sampled_records.observed_class_labels[
                    sample_count : sample_count + 1
                ],
            )
            for sample_count in range(6)
        )
        sample_store, legacy_client, _ = build_evaluation_storage_oracle(
            capacity=4, append_sample_count=4
        )
        monkeypatch.setattr(config, "EVAL_STORE_SAMPLE_SIZE", 4)
        python_random_generator = random.Random(731)
        previous_random_states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        previous_numeric_environment = (
            torch.get_default_dtype(),
            torch.get_default_device(),
            torch.is_grad_enabled(),
        )
        for model_id in (4, 8, 2, 6):
            expected_random_state = run_legacy_evaluation_operation(
                legacy_client=legacy_client,
                operation_name="append",
                operation_arguments={
                    "model_id": model_id,
                    "data_list": [
                        (evaluation_sample.input_features, evaluation_sample.observed_class_labels)
                        for evaluation_sample in evaluation_samples
                    ],
                },
                initial_random_state=python_random_generator.getstate(),
            )
            sample_store.sample_and_append_model_evaluation_samples(
                model_id=model_id,
                evaluation_samples=evaluation_samples,
                python_random_generator=python_random_generator,
            )
            assert python_random_generator.getstate() == expected_random_state
        previous_snapshot = sample_store.snapshot_ordered_model_evaluation_samples()
        # 上位で旧仮ID状態を対応し、正式確認のpop上書き経路を使う。
        expected_random_state = run_legacy_evaluation_operation(
            legacy_client=legacy_client,
            operation_name="remap",
            operation_arguments={"model_id_mapping": {4: -7}},
            initial_random_state=python_random_generator.getstate(),
        )
        sample_store.remap_model_evaluation_sample_collections(
            model_id_mapping={4: -7}, python_random_generator=python_random_generator
        )
        assert python_random_generator.getstate() == expected_random_state
        expected_random_state = run_legacy_evaluation_operation(
            legacy_client=legacy_client,
            operation_name="reassign",
            operation_arguments={"original_model_id": -7, "reassigned_model_id": 12},
            initial_random_state=python_random_generator.getstate(),
        )
        sample_store.reassign_model_evaluation_samples_id(
            original_model_id=-7, reassigned_model_id=12
        )
        assert python_random_generator.getstate() == expected_random_state
        expected_random_state = run_legacy_evaluation_operation(
            legacy_client=legacy_client,
            operation_name="remap",
            operation_arguments={"model_id_mapping": {8: 9, 2: 15, 6: 15, 12: 9}},
            initial_random_state=python_random_generator.getstate(),
        )
        sample_store.remap_model_evaluation_sample_collections(
            model_id_mapping={8: 9, 2: 15, 6: 15, 12: 9},
            python_random_generator=python_random_generator,
        )
        assert python_random_generator.getstate() == expected_random_state
        assert_evaluation_storage_matches_legacy(
            sample_store=sample_store, legacy_client=legacy_client
        )
        assert tuple(collection.model_id for collection in previous_snapshot) == (4, 8, 2, 6)
        assert tuple(
            collection.model_id
            for collection in sample_store.snapshot_ordered_model_evaluation_samples()
        ) == (9, 15)
        for collection in sample_store.snapshot_ordered_model_evaluation_samples():
            assert len(collection.evaluation_samples) == 4
            input_features = torch.cat(
                [
                    evaluation_sample.input_features
                    for evaluation_sample in collection.evaluation_samples
                ]
            )
            observed_class_labels = torch.cat(
                [
                    evaluation_sample.observed_class_labels
                    for evaluation_sample in collection.evaluation_samples
                ]
            )
            with torch.no_grad():
                expected_losses = legacy_classifier.per_sample_error(
                    input_features, observed_class_labels
                )
            actual_losses = evaluate_classifier_per_sample_bounded_losses(
                classifier=classifier,
                input_features=input_features,
                observed_class_labels=observed_class_labels,
            )
            assert torch.equal(actual_losses, expected_losses)
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
