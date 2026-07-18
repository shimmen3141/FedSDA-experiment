"""FedDrift論文準拠データセットと多クラスモデルのテスト。"""
import gzip
import struct

import numpy as np
import pytest
import torch

from federated_drift_experiment import config
from federated_drift_experiment import data
from federated_drift_experiment.data.schedules import (
    FEDDRIFT_2CONCEPT_PATTERN,
    FEDDRIFT_4CONCEPT_PATTERN,
    expand_feddrift_pattern,
)
from federated_drift_experiment.data.mnist import (
    _read_images,
    _read_labels,
    apply_mnist_concept,
)
from federated_drift_experiment.data import streams
from federated_drift_experiment.models import SimpleMLP
from federated_drift_experiment.experiment import run_random_drift_experiment


@pytest.fixture
def restore_dataset():
    original = config.DATASET
    original_schedule = config.CONCEPT_SCHEDULE
    yield
    config.DATASET = original
    config.CONCEPT_SCHEDULE = original_schedule


def test_feddrift_two_concept_schedule_matches_reference_pattern():
    schedules = expand_feddrift_pattern(FEDDRIFT_2CONCEPT_PATTERN, 10, 5000)
    assert len(schedules) == 10
    assert all(len(schedule) == 5000 for schedule in schedules)
    for time_id in range(10):
        offset = time_id * 500
        assert [schedule[offset] for schedule in schedules] == list(
            FEDDRIFT_2CONCEPT_PATTERN[time_id]
        )


def test_feddrift_four_concept_schedule_handles_non_divisible_length():
    schedules = expand_feddrift_pattern(FEDDRIFT_4CONCEPT_PATTERN, 3, 23)
    assert [len(schedule) for schedule in schedules] == [23, 23, 23]
    assert set().union(*(set(schedule) for schedule in schedules)) <= {0, 1, 2, 3}


@pytest.mark.parametrize("dataset", list(config._FEATURE_DIMS))
def test_every_dataset_supports_random_and_feddrift_fixed(dataset, restore_dataset):
    random_schedule = data.make_concept_schedules(
        2, 20, min_stable_period=0, drift_prob=0.0,
        schedule_type="random", dataset=dataset,
    )
    assert random_schedule == [[0] * 20, [0] * 20]

    fixed = data.make_concept_schedules(
        2, 20, schedule_type="feddrift_fixed", dataset=dataset
    )
    pattern = (FEDDRIFT_2CONCEPT_PATTERN
               if config.num_concepts(dataset) == 2
               else FEDDRIFT_4CONCEPT_PATTERN)
    assert fixed == expand_feddrift_pattern(pattern, 2, 20)


def test_default_schedule_remains_random_for_existing_results(restore_dataset):
    config.DATASET = "sea2"
    config.CONCEPT_SCHEDULE = "random"
    schedules = data.make_concept_schedules(
        1, 20, min_stable_period=0, drift_prob=0.0
    )
    assert schedules == [[0] * 20]


def test_mnist_concepts_swap_expected_label_pairs():
    labels = np.arange(10)
    assert np.array_equal(apply_mnist_concept(labels, 0), labels)
    assert list(apply_mnist_concept(labels, 1)[1:3]) == [2, 1]
    assert list(apply_mnist_concept(labels, 2)[3:5]) == [4, 3]
    assert list(apply_mnist_concept(labels, 3)[5:7]) == [6, 5]


def test_idx_reader_parses_mnist_files(tmp_path):
    images_path = tmp_path / "images.gz"
    labels_path = tmp_path / "labels.gz"
    pixels = np.array([0, 255, 64, 128], dtype=np.uint8)
    with gzip.open(images_path, "wb") as stream:
        stream.write(struct.pack(">IIII", 2051, 1, 2, 2))
        stream.write(pixels.tobytes())
    with gzip.open(labels_path, "wb") as stream:
        stream.write(struct.pack(">II", 2049, 1))
        stream.write(bytes([7]))

    images = _read_images(images_path)
    labels = _read_labels(labels_path)
    assert images.shape == (1, 4)
    assert images[0, 1] == pytest.approx(1.0)
    assert labels.tolist() == [7]


def test_generate_mnist_data_uses_concept_transform(monkeypatch):
    monkeypatch.setitem(
        streams._GENERATORS, "mnist2",
        lambda concept_id, n: (
            np.zeros((n, 784), dtype=np.float32),
            apply_mnist_concept(np.full(n, 1), concept_id),
        ),
    )
    x, y = data.generate_data(1, n_samples=2, dataset="mnist2")
    assert x.shape == (2, 784)
    assert y.shape == (2, 1)
    assert y.tolist() == [[2.0], [2.0]]


def test_multiclass_model_exposes_bounded_error_and_updates(restore_dataset):
    config.DATASET = "mnist4"
    model = SimpleMLP()
    x = torch.zeros((2, 784))
    y = torch.tensor([[1.0], [2.0]])

    errors = model.per_sample_error(x, y)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = model.update(x, y)

    assert model.predict(x).shape == (2, 1)
    assert errors.shape == (2,)
    assert torch.all((0.0 <= errors) & (errors <= 1.0))
    assert loss > 0.0
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_mnist_multiclass_path_runs_through_feddrift(monkeypatch):
    def fake_mnist(concept_id, n_samples):
        labels = np.arange(n_samples, dtype=np.int64) % 10
        features = np.zeros((n_samples, 784), dtype=np.float32)
        features[np.arange(n_samples), labels] = 1.0
        return features, apply_mnist_concept(labels, concept_id)

    monkeypatch.setitem(streams._GENERATORS, "mnist2", fake_mnist)
    for name, value in {
        "DATASET": "mnist2",
        "CONCEPT_SCHEDULE": "feddrift_fixed",
        "N_CLIENTS": 1,
        "TOTAL_DATA_POINTS": 10,
        "PRETRAIN_SAMPLES": 10,
        "PRETRAIN_EPOCHS": 1,
        "PRETRAIN_BATCH_SIZE": 5,
        "CLIENT_BATCH_SIZE": 5,
        "NEW_MODEL_EPOCHS": 1,
        "FEDDRIFT_DETECT_BATCH": 5,
    }.items():
        monkeypatch.setattr(config, name, value)

    result = run_random_drift_experiment(
        mode="FedDrift_v2", random_seed=0, verbose=False, show_plot=False
    )
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["total_true"] == 1
