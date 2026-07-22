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
