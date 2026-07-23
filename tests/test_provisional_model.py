import torch

from federated_drift_experiment.provisional_model import (
    ProvisionalModelDecision,
    has_consistent_validation_advantage,
    temporal_holdout,
    validation_rejection_reason,
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


def test_validation_rejection_reason_identifies_failed_time_range():
    reference = torch.tensor([0.8, 0.8, 0.8, 0.8])

    assert validation_rejection_reason(
        torch.tensor([0.6, 0.6, 0.9, 0.9]), reference, 0.01
    ) == "recent_interval"
    assert validation_rejection_reason(
        torch.tensor([0.9, 0.9, 0.9, 0.9]), reference, 0.01
    ) == "full_and_recent"


def test_provisional_decision_exposes_candidate_advantage_margins():
    decision = ProvisionalModelDecision(
        position=100,
        detector="e-SR",
        accepted=True,
        reason="accepted",
        interval_count=30,
        training_count=24,
        validation_count=6,
        reference_model_id=2,
        candidate_mean_loss=0.2,
        reference_mean_loss=0.5,
        candidate_recent_loss=0.3,
        reference_recent_loss=0.4,
    )

    assert abs(decision.full_margin - 0.3) < 1e-12
    assert abs(decision.recent_margin - 0.1) < 1e-12
