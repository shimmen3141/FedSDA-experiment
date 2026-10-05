"""実旧samplerへ参加順・抽出Tensor・終端乱数状態を直接照合する。"""

import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from federated_drift_experiment.clients.shared_backbone import _SharedRepresentationFedSDAClientMixin
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample, ModelTrainingSampleCollection, SampledModelTrainingBatch,
)
from federated_learning_experiments.learning.training.held_model_training_batch_sampling import sample_training_batches_for_held_models


def build_sampling_oracle_inputs(*, batch_sample_count=2, input_contract_case="mixed"):
    ordered_model_training_samples = []
    legacy_training_samples = {}
    for model_id, sample_count in ((8, 5), (55, 5), (-7, 100), (2, 0), (4, batch_sample_count - 1)):
        training_samples = tuple(ObservedTrainingSample(
            input_features=torch.tensor([[float(training_sample_index), float(model_id)]], dtype=torch.float32, device="cpu"),
            observed_class_labels=torch.tensor([[float(training_sample_index % 2)]], dtype=torch.float32, device="cpu"))
            for training_sample_index in range(sample_count))
        if input_contract_case == "same_reference" and training_samples:
            training_samples = (training_samples[0],) * sample_count
        ordered_model_training_samples.append(ModelTrainingSampleCollection(model_id=model_id, training_samples=training_samples))
        # 真の概念情報は旧oracleだけに付け、新sampleの契約へ持ち込まない。
        legacy_training_samples[model_id] = [(training_sample.input_features, training_sample.observed_class_labels, 0)
            for training_sample in training_samples]
    held_model_ids = frozenset((8, -7, 2, 4, 999))
    if input_contract_case == "empty":
        ordered_model_training_samples = []
        legacy_training_samples = {}
    elif input_contract_case == "no_eligible":
        held_model_ids = frozenset((2, 4, 999))
    legacy_client = SimpleNamespace(models={model_id: None for model_id in held_model_ids},
        train_data_store=legacy_training_samples, batch_size=batch_sample_count)
    return held_model_ids, tuple(ordered_model_training_samples), random.Random(137), legacy_client


def run_legacy_training_batch_sampling(*, legacy_client, python_random_generator):
    global_python_random_state = random.getstate()
    try:
        random.setstate(python_random_generator.getstate())
        legacy_batches = _SharedRepresentationFedSDAClientMixin._sample_training_batches(legacy_client)
        return legacy_batches, random.getstate()
    finally:
        random.setstate(global_python_random_state)


def assert_sampled_batches_equal(*, sampled_batches, legacy_batches):
    assert type(sampled_batches) is tuple
    assert len(sampled_batches) == len(legacy_batches)
    for sampled_batch, (model_id, input_features, observed_class_labels) in zip(sampled_batches, legacy_batches):
        assert type(sampled_batch) is SampledModelTrainingBatch
        assert sampled_batch.model_id == model_id
        assert torch.equal(sampled_batch.input_features, input_features)
        assert torch.equal(sampled_batch.observed_class_labels, observed_class_labels)


@pytest.mark.parametrize("input_contract_case,batch_sample_count", [
    ("mixed", 2), ("mixed", 1), ("mixed", 5), ("same_reference", 5), ("empty", 2), ("no_eligible", 2),
])
def test_training_batch_sampling_matches_legacy(input_contract_case, batch_sample_count):
    held_model_ids, ordered_model_training_samples, python_random_generator, legacy_client = build_sampling_oracle_inputs(
        batch_sample_count=batch_sample_count, input_contract_case=input_contract_case)
    global_python_random_state = random.getstate()
    for _ in range(3):
        legacy_batches, expected_random_state = run_legacy_training_batch_sampling(
            legacy_client=legacy_client, python_random_generator=python_random_generator)
        actual_random_state = python_random_generator.getstate()
        sampled_batches = sample_training_batches_for_held_models(held_model_ids=held_model_ids,
            ordered_model_training_samples=ordered_model_training_samples, batch_sample_count=batch_sample_count,
            python_random_generator=python_random_generator)
        assert_sampled_batches_equal(sampled_batches=sampled_batches, legacy_batches=legacy_batches)
        assert python_random_generator.getstate() == expected_random_state
        assert random.getstate() == global_python_random_state
        assert tuple(sampled_batch.model_id for sampled_batch in sampled_batches) == (
            () if input_contract_case in ("empty", "no_eligible") else (8, -7))
        if not sampled_batches:
            assert python_random_generator.getstate() == actual_random_state
        else:
            for sampled_batch in sampled_batches:
                assert sampled_batch.input_features.shape == (batch_sample_count, 2)
                assert sampled_batch.observed_class_labels.shape == (batch_sample_count, 1)
                if input_contract_case != "same_reference":
                    assert len(set(sampled_batch.input_features[:, 0].tolist())) == batch_sample_count
                else:
                    assert torch.equal(sampled_batch.input_features, sampled_batch.input_features[0:1].expand(batch_sample_count, -1))


def test_training_batch_sampling_observes_draw_order(monkeypatch):
    held_model_ids, ordered_model_training_samples, python_random_generator, legacy_client = build_sampling_oracle_inputs()
    monkeypatch.setattr(python_random_generator, "sample", Mock(wraps=python_random_generator.sample))
    sampled_batches = sample_training_batches_for_held_models(held_model_ids=held_model_ids,
        ordered_model_training_samples=ordered_model_training_samples, batch_sample_count=2,
        python_random_generator=python_random_generator)
    assert python_random_generator.sample.call_count == 2
    eligible_model_training_samples = (ordered_model_training_samples[0], ordered_model_training_samples[2])
    for sample_call, model_training_samples in zip(python_random_generator.sample.call_args_list, eligible_model_training_samples):
        assert sample_call.args[0] is model_training_samples.training_samples
        assert sample_call.args[1] == 2
    assert tuple(sampled_batch.model_id for sampled_batch in sampled_batches) == (8, -7)
