import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from federated_drift_experiment import config
from federated_drift_experiment.clients import FedDriftV2Client
from federated_drift_experiment.clustering import cluster_models
from federated_drift_experiment.experiment import _run_feddrift_v2_timestep
from federated_drift_experiment.models import SimpleMLP
from federated_drift_experiment.server import FedDriftV2Server


def test_linkage_strategies_distinguish_chain_merges():
    distances = {(0, 1): 0.05, (0, 2): 0.20, (1, 2): 0.05}

    assert cluster_models([0, 1, 2], distances, 0.1, "connected") == [[0, 1, 2]]
    assert cluster_models([0, 1, 2], distances, 0.1, "complete") == [[0, 1], [2]]


def test_new_model_is_mature_after_configured_isolation_window():
    initial_model = SimpleMLP()
    initial_stats = {0: {'n': 10, 'mean': 0.1, 'M2': 0.0}}
    client = FedDriftV2Client(
        client_id=0,
        initial_models={0: initial_model},
        initial_stats=initial_stats,
        distance_threshold=0.1,
        verbose=False,
    )
    server = FedDriftV2Server(
        distance_threshold=0.1,
        isolation_timesteps=1,
        linkage="complete",
        verbose=False,
    )
    server.register_model_params(0, initial_model.get_params())
    server.register_model_stats(0, initial_stats[0])
    server.register_client(client)

    bx = torch.zeros((config.CLIENT_BATCH_SIZE, config.input_dim()))
    by = torch.zeros((config.CLIENT_BATCH_SIZE, 1))
    temp_id, _ = client._spawn_new_model(bx, by, pending_ready=True)
    client.current_model_id = temp_id
    client.train_data_store[temp_id].extend([(bx[i:i + 1], by[i:i + 1]) for i in range(len(bx))])

    server.prepare_timestep(3)
    new_id = client.current_model_id
    assert new_id >= 0
    assert new_id not in server.mature_model_ids(3)

    server.run_training_round(3)
    assert new_id in server.global_models
    assert new_id in server.mature_model_ids(4)


def test_timestep_uses_exactly_configured_fedavg_rounds(monkeypatch):
    monkeypatch.setattr(config, "FEDDRIFT_DETECT_BATCH", 2)
    monkeypatch.setattr(config, "FEDDRIFT_ROUNDS", 2)

    initial_model = SimpleMLP()
    initial_stats = {0: {'n': 10, 'mean': 0.1, 'M2': 0.0}}
    clients = [
        FedDriftV2Client(
            client_id=client_id,
            initial_models={0: initial_model},
            initial_stats=initial_stats,
            distance_threshold=0.1,
            verbose=False,
        )
        for client_id in range(2)
    ]
    server = FedDriftV2Server(
        distance_threshold=0.1,
        isolation_timesteps=1,
        linkage="complete",
        verbose=False,
    )
    server.register_model_params(0, initial_model.get_params())
    server.register_model_stats(0, initial_stats[0])
    for client in clients:
        server.register_client(client)

    data = []
    concepts = []
    for _ in clients:
        xs = torch.zeros((2, config.input_dim()))
        ys = torch.zeros((2, 1))
        data.append([(xs[i], ys[i]) for i in range(2)])
        concepts.append([0, 0])

    _run_feddrift_v2_timestep(
        clients, server, data, concepts, t=0, use_server=True, verbose=False
    )

    # 2クライアント × 1モデル × 2ラウンド。クラスタリング用の余分なFedAvgはない。
    assert server.comm_models_up == 4
    assert server.comm_models_down == 4
