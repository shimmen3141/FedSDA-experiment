import torch

from federated_drift_experiment.provisional_model import (
    has_consistent_validation_advantage,
    temporal_holdout,
)


def test_temporal_holdout_reserves_latest_samples_for_validation():
    x = torch.arange(10, dtype=torch.float32).view(-1, 1)
    y = torch.arange(10).view(-1, 1)

    holdout = temporal_holdout(x, y, validation_fraction=0.3)

    assert holdout is not None
    assert holdout.training_x.view(-1).tolist() == list(range(7))
    assert holdout.validation_x.view(-1).tolist() == [7, 8, 9]
    assert holdout.validation_y.view(-1).tolist() == [7, 8, 9]


def test_temporal_holdout_rejects_too_short_interval():
    x = torch.zeros((2, 1))
    y = torch.zeros((2, 1))

    assert temporal_holdout(x, y, validation_fraction=0.2) is None


def test_validation_advantage_must_hold_for_full_and_recent_intervals():
    reference = torch.tensor([0.8, 0.8, 0.8, 0.8])
    consistently_better = torch.tensor([0.6, 0.6, 0.6, 0.6])
    recently_worse = torch.tensor([0.2, 0.2, 0.9, 0.9])

    assert has_consistent_validation_advantage(
        consistently_better, reference, min_delta=0.01
    )
    assert not has_consistent_validation_advantage(
        recently_worse, reference, min_delta=0.01
    )
