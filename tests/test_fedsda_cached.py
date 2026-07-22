import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from federated_drift_experiment import config
from federated_drift_experiment.clients import ADWINFedSDAClient
from federated_drift_experiment.experiment import _run_per_sample_timestep
from federated_drift_experiment.models import SimpleMLP
from federated_drift_experiment.servers import FedSDACachedServer


def _make_client_and_server():
    model = SimpleMLP()
    stats = {0: {'n': 10, 'mean': 0.1, 'M2': 0.0}}
    client = ADWINFedSDAClient(
        client_id=0,
        initial_models={0: model},
        initial_stats=stats,
        distance_threshold=0.1,
        verbose=False,
    )
    server = FedSDACachedServer(distance_threshold=0.1, verbose=False)
    server.register_model_params(0, model.get_params())
    server.register_model_stats(0, stats[0])
    server.register_client(client)
    return client, server


def test_every_round_policy_clusters_without_new_models(monkeypatch):
    monkeypatch.setattr(config, "FEDSDA_CLUSTERING_POLICY", "every_round")
    client, server = _make_client_and_server()

    second_model = SimpleMLP()
    client.models[1] = second_model
    client.cached_global_model_params[1] = second_model.get_params()
    server.register_model_params(1, second_model.get_params())
    server.register_model_stats(1, {'n': 10, 'mean': 0.2, 'M2': 0.0})

    calls = []
    monkeypatch.setattr(
        server,
        "_cross_evaluate",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {},
    )
    monkeypatch.setattr(
        server,
        "perform_hierarchical_clustering",
        lambda model_ids, stats_matrix: [[model_id] for model_id in model_ids],
    )

    server.run_round(t=0)
    server.run_round(t=1)

    assert len(calls) == 2
    assert all(call[1] == {
        "send_model_params": False,
        "use_client_cache": True,
    } for call in calls)


def test_cached_server_rejects_unknown_clustering_policy(monkeypatch):
    monkeypatch.setattr(config, "FEDSDA_CLUSTERING_POLICY", "unknown")
    try:
        FedSDACachedServer(distance_threshold=0.1, verbose=False)
    except ValueError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("Unknown clustering policy must be rejected")


def test_no_cached_every_round_policy_enables_clustering_without_new_models(monkeypatch):
    monkeypatch.setattr(config, "FEDSDA_CLUSTERING_POLICY", "every_round")

    class Client:
        def process_one_step(self, *args):
            pass

        def flush_pending_updates(self):
            pass

        def has_pending_model(self):
            return False

        def promote_pending_to_ready(self):
            pass

    class Server:
        def record_client_state_summaries(self):
            pass

        def run_round(self, t, clustering_enabled):
            self.clustering_enabled = clustering_enabled

    server = Server()
    sample = (torch.zeros((1, config.input_dim())), torch.zeros((1, 1)))
    _run_per_sample_timestep(
        [Client()], server, [[sample]], [[0]], t=0, use_server=True, verbose=False,
    )

    assert server.clustering_enabled is True


def test_evaluation_cache_is_not_changed_by_local_training():
    client, _ = _make_client_and_server()
    cached_before = client.cached_global_model_params[0]

    live_params = client.models[0].get_params()
    changed_params = {name: value + 1.0 for name, value in live_params.items()}
    client.models[0].set_params(changed_params)

    for name in cached_before:
        assert torch.equal(client.cached_global_model_params[0][name], cached_before[name])
        assert not torch.equal(client.models[0].get_params()[name], cached_before[name])


def test_new_model_is_clustered_only_after_first_broadcast(monkeypatch):
    monkeypatch.setattr(config, "FEDSDA_MODEL_UPLOAD_DELAY_ROUNDS", 1)
    monkeypatch.setattr(config, "NEW_MODEL_EPOCHS", 1)
    client, server = _make_client_and_server()

    bx = torch.zeros((config.CLIENT_BATCH_SIZE, config.input_dim()))
    by = torch.zeros((config.CLIENT_BATCH_SIZE, 1))
    temp_id, _ = client._spawn_new_model(bx, by)
    client.current_model_id = temp_id
    client.train_data_store[temp_id].extend(
        (bx[index:index + 1], by[index:index + 1]) for index in range(len(bx))
    )
    client.promote_pending_to_ready()

    cross_evaluation_calls = []
    original_cross_evaluate = server._cross_evaluate

    def record_cross_evaluation(*args, **kwargs):
        cross_evaluation_calls.append((args, kwargs))
        return original_cross_evaluate(*args, **kwargs)

    monkeypatch.setattr(server, "_cross_evaluate", record_cross_evaluation)

    server.run_round(t=0)
    new_model_id = client.current_model_id
    assert new_model_id >= 0
    assert cross_evaluation_calls == []
    assert new_model_id in client.cached_global_model_params
    assert new_model_id in server.models_pending_clustering

    models_down_before = server.comm_models_down
    server.run_round(t=1)

    assert len(cross_evaluation_calls) == 1
    assert cross_evaluation_calls[0][1] == {
        "send_model_params": False,
        "use_client_cache": True,
    }
    # クロス評価ではモデルを再送せず、通常ブロードキャスト分だけ増える。
    assert server.comm_models_down - models_down_before == len(server.global_models)
    assert server.models_pending_clustering == set()


def test_cached_merge_is_applied_before_fedavg(monkeypatch):
    monkeypatch.setattr(config, "FEDSDA_MODEL_UPLOAD_DELAY_ROUNDS", 1)
    monkeypatch.setattr(config, "NEW_MODEL_EPOCHS", 1)
    client, server = _make_client_and_server()

    bx = torch.zeros((config.CLIENT_BATCH_SIZE, config.input_dim()))
    by = torch.zeros((config.CLIENT_BATCH_SIZE, 1))
    temp_id, _ = client._spawn_new_model(bx, by)
    client.current_model_id = temp_id
    client.train_data_store[temp_id].extend(
        (bx[index:index + 1], by[index:index + 1]) for index in range(len(bx))
    )
    client.promote_pending_to_ready()
    server.run_round(t=0)
    new_model_id = client.current_model_id

    monkeypatch.setattr(
        server,
        "perform_hierarchical_clustering",
        lambda model_ids, stats_matrix: [[0, new_model_id]],
    )
    server.run_round(t=1)

    assert sorted(server.global_models) == [0]
    assert sorted(model_id for model_id in client.models if model_id >= 0) == [0]
    assert client.current_model_id == 0


def test_finalize_protocol_clusters_only_distributed_pending_models(monkeypatch):
    monkeypatch.setattr(config, "FEDSDA_MODEL_UPLOAD_DELAY_ROUNDS", 1)
    monkeypatch.setattr(config, "NEW_MODEL_EPOCHS", 1)
    client, server = _make_client_and_server()

    bx = torch.zeros((config.CLIENT_BATCH_SIZE, config.input_dim()))
    by = torch.zeros((config.CLIENT_BATCH_SIZE, 1))
    temp_id, _ = client._spawn_new_model(bx, by)
    client.current_model_id = temp_id
    client.train_data_store[temp_id].extend(
        (bx[index:index + 1], by[index:index + 1]) for index in range(len(bx))
    )
    client.promote_pending_to_ready()
    server.run_round(t=0)
    new_model_id = client.current_model_id
    assert new_model_id in server.models_pending_clustering

    monkeypatch.setattr(
        server,
        "perform_hierarchical_clustering",
        lambda model_ids, stats_matrix: [[0, new_model_id]],
    )
    models_up_before = server.comm_models_up
    models_down_before = server.comm_models_down
    messages_before = server.comm_messages_up + server.comm_messages_down

    server.finalize_protocol(t=1)

    assert sorted(server.global_models) == [0]
    assert server.models_pending_clustering == set()
    # 終端処理は配布済みキャッシュを使い、モデル本体の追加通信や学習を行わない。
    assert server.comm_models_up == models_up_before
    assert server.comm_models_down == models_down_before
    assert server.comm_messages_up + server.comm_messages_down > messages_before
