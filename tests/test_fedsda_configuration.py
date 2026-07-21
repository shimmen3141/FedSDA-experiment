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
