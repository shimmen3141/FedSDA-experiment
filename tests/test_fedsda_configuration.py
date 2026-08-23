import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from federated_drift_experiment import config
from federated_drift_experiment.clients import ADWINFedSDAClient
from federated_drift_experiment.models import SimpleMLP


def _make_client():
    model = SimpleMLP()
    return ADWINFedSDAClient(
        client_id=0,
        initial_models={0: model},
        initial_stats={0: {'n': 10, 'mean': 0.1, 'M2': 0.0}},
        distance_threshold=0.1,
        verbose=False,
    )


def test_new_model_upload_delay_is_counted_in_rounds(monkeypatch):
    monkeypatch.setattr(config, "FEDSDA_MODEL_UPLOAD_DELAY_ROUNDS", 2)
    monkeypatch.setattr(config, "NEW_MODEL_EPOCHS", 1)
    client = _make_client()
    bx = torch.zeros((config.CLIENT_BATCH_SIZE, config.input_dim()))
    by = torch.zeros((config.CLIENT_BATCH_SIZE, 1))

    client._spawn_new_model(bx, by)
    assert not client.has_pending_model()

    client.promote_pending_to_ready()
    assert not client.has_pending_model()

    client.promote_pending_to_ready()
    assert client.has_pending_model()


def test_new_model_training_none_keeps_copied_parameters(monkeypatch):
    monkeypatch.setattr(config, "NEW_MODEL_TRAINING", "none")
    client = _make_client()
    source_params = client.models[0].get_params()
    bx = torch.randn((config.CLIENT_BATCH_SIZE, config.input_dim()))
    by = torch.zeros((config.CLIENT_BATCH_SIZE, 1))

    temporary_id, _ = client._spawn_new_model(bx, by)

    created_params = client.models[temporary_id].get_params()
    assert all(torch.equal(source_params[name], created_params[name])
               for name in source_params)
    assert client.compute_counters["optimizer_steps"] == 0


def test_model_concept_counts_follow_assigned_training_data():
    client = _make_client()
    x = torch.zeros((1, config.input_dim()))
    y = torch.zeros((1, 1))

    client._absorb_into_store(0, [(x, y, 3), (x, y, 3), (x, y, 1)])

    assert client.get_model_concept_counts(0) == {3: 2, 1: 1}


def test_new_model_can_copy_selected_existing_model(monkeypatch):
    monkeypatch.setattr(config, "NEW_MODEL_TRAINING", "none")
    client = _make_client()
    selected_model = SimpleMLP()
    selected_params = selected_model.get_params()
    selected_params = {
        name: torch.full_like(value, 0.25)
        for name, value in selected_params.items()
    }
    selected_model.set_params(selected_params)
    client.models[1] = selected_model
    client.model_stats[1] = {'n': 10, 'mean': 0.1, 'M2': 0.0}
    bx = torch.randn((config.CLIENT_BATCH_SIZE, config.input_dim()))
    by = torch.zeros((config.CLIENT_BATCH_SIZE, 1))

    temporary_id, _ = client._spawn_new_model(
        bx, by, initialization_params=selected_params
    )

    created_params = client.models[temporary_id].get_params()
    assert all(
        torch.equal(selected_params[name], created_params[name])
        for name in selected_params
    )


def test_new_model_initializer_is_lowest_loss_evaluated_model(monkeypatch):
    monkeypatch.setattr(config, "NEW_MODEL_INITIALIZATION", "best_candidate")
    client = _make_client()
    client.models[1] = SimpleMLP()

    selected = client._select_initialization_params([
        (0, 0.8),
        (1, 0.3),
    ])

    expected = client.models[1].get_params()
    assert all(torch.equal(selected[name], expected[name]) for name in expected)


def test_new_model_initializer_can_use_current_model(monkeypatch):
    monkeypatch.setattr(config, "NEW_MODEL_INITIALIZATION", "current")
    client = _make_client()
    client.current_model_id = 0

    selected = client._select_initialization_params([(0, 0.1), (1, 0.2)])

    expected = client.models[0].get_params()
    assert all(torch.equal(selected[name], expected[name]) for name in expected)


def test_new_model_initializer_can_average_existing_models(monkeypatch):
    monkeypatch.setattr(config, "NEW_MODEL_INITIALIZATION", "average")
    client = _make_client()
    first = client.models[0].get_params()
    first = {name: torch.zeros_like(value) for name, value in first.items()}
    client.models[0].set_params(first)
    second_model = SimpleMLP()
    second = {
        name: torch.full_like(value, 0.5)
        for name, value in second_model.get_params().items()
    }
    second_model.set_params(second)
    client.models[1] = second_model

    selected = client._select_initialization_params([])

    assert all(
        torch.equal(selected[name], torch.full_like(selected[name], 0.25))
        for name in selected
    )


def test_model_reuse_selects_best_fitting_model():
    client = _make_client()
    candidates = [(0, 0.30), (1, 0.10)]

    assert client._select_reuse_candidate(candidates) == (1, 0.10)


def test_new_model_training_early_stopping_uses_at_most_max_epochs(monkeypatch):
    monkeypatch.setattr(config, "NEW_MODEL_TRAINING", "early_stopping")
    monkeypatch.setattr(config, "NEW_MODEL_EPOCHS", 8)
    monkeypatch.setattr(config, "NEW_MODEL_EARLY_STOPPING_PATIENCE", 2)
    client = _make_client()
    bx = torch.randn((config.CLIENT_BATCH_SIZE, config.input_dim()))
    by = torch.zeros((config.CLIENT_BATCH_SIZE, 1))

    client._spawn_new_model(bx, by)

    # 学習部分は1ミニバッチなので、更新回数は最大エポック数以下になる。
    assert 1 <= client.compute_counters["optimizer_steps"] <= config.NEW_MODEL_EPOCHS


def test_forward_validation_accepts_candidate_after_future_samples(monkeypatch):
    monkeypatch.setattr(config, "NEW_MODEL_TRAINING", "none")
    monkeypatch.setattr(config, "NEW_MODEL_FORWARD_VALIDATION_SAMPLES", 3)
    monkeypatch.setattr(config, "NEW_MODEL_CREATION_POLICY", "forward_validated")
    client = _make_client()
    bx = torch.randn((4, config.input_dim()))
    by = torch.zeros((4, 1))
    held_data = [
        (bx[index:index + 1], by[index:index + 1])
        for index in range(len(bx))
    ]

    client._begin_forward_validation(
        bx,
        by,
        held_data,
        client.models[0].get_params(),
        sample_idx=100,
        estimated_start=97,
        episode_id=None,
    )
    session = client._forward_validation
    for _ in range(3):
        session.append_losses(0.1, {0: 0.5})

    drift_type = client._finalize_forward_validation(sample_idx=103)

    assert drift_type == 2
    assert client.current_model_id < 0
    assert client.local_switch_positions == [103]
    assert client.provisional_model_decisions[0].accepted
    assert client.provisional_model_decisions[0].validation_source == "forward"
    assert client.provisional_model_decisions[0].resolution_delay == 3
    assert len(client.train_data_store[client.current_model_id]) == len(held_data)


def test_forward_persistent_rejects_advantage_limited_to_second_half(monkeypatch):
    monkeypatch.setattr(config, "NEW_MODEL_TRAINING", "none")
    monkeypatch.setattr(config, "NEW_MODEL_FORWARD_VALIDATION_SAMPLES", 4)
    monkeypatch.setattr(config, "NEW_MODEL_CREATION_POLICY", "forward_persistent")
    client = _make_client()
    bx = torch.randn((4, config.input_dim()))
    by = torch.zeros((4, 1))
    held_data = [
        (bx[index:index + 1], by[index:index + 1])
        for index in range(len(bx))
    ]

    client._begin_forward_validation(
        bx,
        by,
        held_data,
        client.models[0].get_params(),
        sample_idx=100,
        estimated_start=97,
        episode_id=None,
    )
    session = client._forward_validation
    for candidate_loss in (0.9, 0.9, 0.2, 0.2):
        session.append_losses(candidate_loss, {0: 0.8})

    drift_type = client._finalize_forward_validation(sample_idx=104)

    decision = client.provisional_model_decisions[0]
    assert drift_type == 0
    assert client.current_model_id == 0
    assert not decision.accepted
    assert decision.reason == "first_interval"


def test_shadow_tournament_trains_candidate_and_all_reference_shadows(monkeypatch):
    monkeypatch.setattr(config, "NEW_MODEL_TRAINING", "none")
    monkeypatch.setattr(config, "NEW_MODEL_FORWARD_VALIDATION_SAMPLES", 3)
    monkeypatch.setattr(config, "NEW_MODEL_CREATION_POLICY", "shadow_tournament")
    client = _make_client()
    client.models[1] = SimpleMLP()
    client.model_stats[1] = {"n": 10, "mean": 0.1, "M2": 0.0}
    bx = torch.randn((4, config.input_dim()))
    by = torch.zeros((4, 1))
    calls = []

    def record_training(model, training_x, training_y):
        calls.append((model, training_x, training_y))

    monkeypatch.setattr(client, "_train_new_model", record_training)
    client._begin_forward_validation(
        bx,
        by,
        [],
        client.models[0].get_params(),
        sample_idx=100,
        estimated_start=97,
        episode_id=None,
    )

    # 候補1個と既存モデル2個へ、同一の検知区間を渡す。
    assert len(calls) == 3
    assert all(training_x is bx and training_y is by
               for _, training_x, training_y in calls)

    steps_before = client.compute_counters["optimizer_steps"]
    client._observe_forward_validation(
        torch.zeros((1, config.input_dim())),
        torch.zeros((1, 1)),
        sample_idx=101,
    )
    # forward損失を記録した後も、全shadowを同じ1回ずつ更新する。
    assert client.compute_counters["optimizer_steps"] - steps_before == 3


def test_shadow_tournament_adopts_winning_reference_without_new_id(monkeypatch):
    monkeypatch.setattr(config, "NEW_MODEL_TRAINING", "none")
    monkeypatch.setattr(config, "NEW_MODEL_FORWARD_VALIDATION_SAMPLES", 3)
    monkeypatch.setattr(config, "NEW_MODEL_CREATION_POLICY", "shadow_tournament")
    client = _make_client()
    bx = torch.randn((4, config.input_dim()))
    by = torch.zeros((4, 1))
    held_data = [
        (bx[index:index + 1], by[index:index + 1])
        for index in range(len(bx))
    ]
    client._begin_forward_validation(
        bx,
        by,
        held_data,
        client.models[0].get_params(),
        sample_idx=100,
        estimated_start=97,
        episode_id=None,
    )
    session = client._forward_validation
    winning_params = {
        name: torch.full_like(value, 0.125)
        for name, value in session.reference_models[0].get_params().items()
    }
    session.reference_models[0].set_params(winning_params)
    for _ in range(3):
        session.append_losses(0.4, {0: 0.2})

    drift_type = client._finalize_forward_validation(sample_idx=103)

    assert drift_type == 0
    assert client.current_model_id == 0
    assert set(client.models) == {0}
    assert client.provisional_model_decisions[0].reason == "reference_won"
    adopted_params = client.models[0].get_params()
    assert all(
        torch.equal(adopted_params[name], winning_params[name])
        for name in winning_params
    )


def test_forward_requalification_reuses_fitting_existing_model(monkeypatch):
    monkeypatch.setattr(config, "NEW_MODEL_TRAINING", "none")
    monkeypatch.setattr(config, "NEW_MODEL_FORWARD_VALIDATION_SAMPLES", 3)
    monkeypatch.setattr(config, "NEW_MODEL_CREATION_POLICY", "forward_requalified")
    client = _make_client()
    client.models[1] = SimpleMLP()
    client.model_stats[1] = {"n": 10, "mean": 0.1, "M2": 0.0}
    bx = torch.randn((4, config.input_dim()))
    by = torch.zeros((4, 1))
    held_data = [
        (bx[index:index + 1], by[index:index + 1])
        for index in range(len(bx))
    ]

    client._begin_forward_validation(
        bx,
        by,
        held_data,
        client.models[0].get_params(),
        sample_idx=100,
        estimated_start=97,
        episode_id=None,
    )
    session = client._forward_validation
    for _ in range(3):
        session.append_losses(0.05, {0: 0.5, 1: 0.15})

    drift_type = client._finalize_forward_validation(sample_idx=103)

    decision = client.provisional_model_decisions[0]
    assert drift_type == 1
    assert client.current_model_id == 1
    assert not decision.accepted
    assert decision.reason == "reference_refit"
    assert decision.reference_model_id == 1
    assert abs(decision.reference_excess - 0.05) < 1e-6
    assert len(client.train_data_store[1]) == len(held_data)


def test_forward_requalification_keeps_fitting_current_model(monkeypatch):
    monkeypatch.setattr(config, "NEW_MODEL_TRAINING", "none")
    monkeypatch.setattr(config, "NEW_MODEL_FORWARD_VALIDATION_SAMPLES", 3)
    monkeypatch.setattr(
        config,
        "NEW_MODEL_CREATION_POLICY",
        "forward_requalified_current_first",
    )
    client = _make_client()
    client.models[1] = SimpleMLP()
    client.model_stats[1] = {"n": 10, "mean": 0.1, "M2": 0.0}
    bx = torch.randn((4, config.input_dim()))
    by = torch.zeros((4, 1))
    held_data = [
        (bx[index:index + 1], by[index:index + 1])
        for index in range(len(bx))
    ]

    client._begin_forward_validation(
        bx,
        by,
        held_data,
        client.models[0].get_params(),
        sample_idx=100,
        estimated_start=97,
        episode_id=None,
    )
    session = client._forward_validation
    for _ in range(3):
        session.append_losses(0.05, {0: 0.18, 1: 0.12})

    drift_type = client._finalize_forward_validation(sample_idx=103)

    decision = client.provisional_model_decisions[0]
    assert drift_type == 0
    assert client.current_model_id == 0
    assert client.local_switch_positions == []
    assert not decision.accepted
    assert decision.reason == "current_reference_refit"
    assert decision.reference_model_id == 0
    assert len(client.train_data_store[0]) == len(held_data)


def test_incomplete_forward_validation_is_rejected_at_end(monkeypatch):
    monkeypatch.setattr(config, "NEW_MODEL_TRAINING", "none")
    monkeypatch.setattr(config, "NEW_MODEL_FORWARD_VALIDATION_SAMPLES", 3)
    client = _make_client()
    bx = torch.randn((4, config.input_dim()))
    by = torch.zeros((4, 1))
    held_data = [
        (bx[index:index + 1], by[index:index + 1])
        for index in range(len(bx))
    ]
    client._begin_forward_validation(
        bx,
        by,
        held_data,
        client.models[0].get_params(),
        sample_idx=100,
        estimated_start=97,
        episode_id=None,
    )
    client._forward_validation.append_losses(0.1, {0: 0.5})
    client.processed_samples = 102

    client.finalize_incomplete_forward_validation()

    decision = client.provisional_model_decisions[0]
    assert not decision.accepted
    assert decision.reason == "insufficient_forward_data"
    assert decision.validation_count == 1
    assert client._forward_validation is None
    assert len(client.train_data_store[0]) == len(held_data)
