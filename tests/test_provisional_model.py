import torch

from federated_drift_experiment.provisional_model import (
    ForwardValidationSession,
    ProvisionalModelDecision,
    disjoint_validation_rejection_reason,
    has_consistent_validation_advantage,
    has_disjoint_validation_advantage,
    sequential_tournament_winner,
    select_forward_fitting_reference,
    temporal_holdout,
    validation_rejection_reason,
)


def test_forward_validation_session_waits_for_target_count():
    session = ForwardValidationSession(
        proposal_position=10,
        estimated_change_point=8,
        episode_id=None,
        old_model_id=0,
        detector="e-SR",
        candidate=object(),
        training_x=torch.zeros((3, 1)),
        training_y=torch.zeros((3, 1)),
        held_data=[],
        reference_models={0: object()},
        target_count=3,
    )

    session.append_losses(0.2, {0: 0.4})
    session.append_losses(0.3, {0: 0.5})
    assert not session.ready

    session.append_losses(0.1, {0: 0.6})
    assert session.ready
    assert session.validation_count == 3


def test_sequential_session_has_no_fixed_sample_target():
    session = ForwardValidationSession(
        proposal_position=10,
        estimated_change_point=8,
        episode_id=None,
        old_model_id=0,
        detector="e-SR",
        candidate=object(),
        training_x=torch.zeros((3, 1)),
        training_y=torch.zeros((3, 1)),
        held_data=[],
        reference_models={0: object()},
        target_count=None,
    )

    for _ in range(100):
        session.append_losses(0.0, {0: 1.0})

    assert not session.ready
    assert session.validation_count == 100


def test_sequential_tournament_certifies_candidate_against_all_references():
    winner = sequential_tournament_winner(
        candidate_losses=[0.0] * 20,
        reference_losses={0: [1.0] * 20, 1: [0.8] * 20},
        alpha=0.05,
    )

    assert winner == "candidate"


def test_sequential_tournament_can_select_existing_reference():
    winner = sequential_tournament_winner(
        candidate_losses=[0.9] * 20,
        reference_losses={0: [0.8] * 20, 1: [0.0] * 20},
        alpha=0.05,
    )

    assert winner == 1


def test_sequential_tournament_waits_when_no_candidate_dominates():
    winner = sequential_tournament_winner(
        candidate_losses=[0.5] * 200,
        reference_losses={0: [0.5] * 200},
        alpha=0.05,
    )

    assert winner is None


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


def test_disjoint_validation_advantage_requires_both_halves():
    reference = torch.tensor([0.8, 0.8, 0.8, 0.8])

    assert has_disjoint_validation_advantage(
        torch.tensor([0.6, 0.6, 0.7, 0.7]), reference, min_delta=0.01
    )
    assert not has_disjoint_validation_advantage(
        torch.tensor([0.9, 0.9, 0.2, 0.2]), reference, min_delta=0.01
    )
    assert disjoint_validation_rejection_reason(
        torch.tensor([0.9, 0.9, 0.2, 0.2]), reference, min_delta=0.01
    ) == "first_interval"


def test_forward_requalification_selects_best_still_fitting_reference():
    selected = select_forward_fitting_reference(
        reference_losses={
            0: [0.25, 0.25],
            1: [0.16, 0.18],
            2: [0.40, 0.40],
        },
        reference_historical_means={0: 0.10, 1: 0.10, 2: 0.10},
        distance_threshold=0.10,
    )

    assert selected == 1


def test_forward_requalification_returns_none_when_all_references_mismatch():
    selected = select_forward_fitting_reference(
        reference_losses={0: [0.25, 0.25], 1: [0.31, 0.29]},
        reference_historical_means={0: 0.10, 1: 0.10},
        distance_threshold=0.10,
    )

    assert selected is None


def test_forward_requalification_prefers_current_model_when_it_still_fits():
    selected = select_forward_fitting_reference(
        reference_losses={
            0: [0.18, 0.18],
            1: [0.12, 0.12],
        },
        reference_historical_means={0: 0.10, 1: 0.10},
        distance_threshold=0.10,
        preferred_model_id=0,
    )

    assert selected == 0


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
