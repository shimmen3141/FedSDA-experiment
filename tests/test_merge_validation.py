import torch

from federated_drift_experiment import config
from federated_drift_experiment.servers import FedSDANoCachedServer


class _PairEvaluationClient:
    def __init__(self, difference_by_model):
        self.difference_by_model = difference_by_model

    def get_held_model_ids(self):
        return set(self.difference_by_model)

    def evaluate_model_pair(
        self, candidate_params, reference_params, target_model_id
    ):
        difference = self.difference_by_model[target_model_id]
        return 10, 10 * difference, 10 * difference * difference


def _server(monkeypatch, differences):
    monkeypatch.setattr(config, "FEDSDA_MERGE_VALIDATION", "candidate_loss")
    server = FedSDANoCachedServer(distance_threshold=0.1, verbose=False)
    server.global_models = {
        0: {"weight": torch.tensor([0.0])},
        1: {"weight": torch.tensor([1.0])},
    }
    server.clients = [_PairEvaluationClient(differences)]
    return server


def test_merge_validation_accepts_candidate_with_no_loss_increase(monkeypatch):
    server = _server(monkeypatch, {0: -0.1, 1: 0.0})

    clusters = server._validate_merge_candidates([[0, 1]], {0: 1, 1: 1})

    assert clusters == [[0, 1]]
    assert server.merge_validation_proposal_count == 1
    assert server.merge_validation_accept_count == 1
    assert server.merge_validation_reject_count == 0
    assert server.comm_models_down == 2


def test_merge_validation_rejects_candidate_with_higher_loss(monkeypatch):
    server = _server(monkeypatch, {0: 0.1, 1: 0.2})

    clusters = server._validate_merge_candidates([[0, 1]], {0: 1, 1: 1})

    assert clusters == [[0], [1]]
    assert server.merge_validation_accept_count == 0
    assert server.merge_validation_reject_count == 1


def test_merge_validation_is_noop_by_default(monkeypatch):
    monkeypatch.setattr(config, "FEDSDA_MERGE_VALIDATION", "none")
    server = FedSDANoCachedServer(distance_threshold=0.1, verbose=False)

    assert server._validate_merge_candidates([[0, 1]], {}) == [[0, 1]]
    assert server.merge_validation_proposal_count == 0
