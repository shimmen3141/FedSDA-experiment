import torch

from federated_drift_experiment import config, run_random_drift_experiment
from federated_drift_experiment.clients import (
    SharedBackboneRestartingSoftRoutingFedSDAClient,
)
from federated_drift_experiment.experiment import MODE_SPECS
from federated_drift_experiment.models import (
    SharedBackboneMLP,
    parameter_payload_size,
)
from federated_drift_experiment.servers import SharedBackboneFedSDANoCachedServer


def _two_head_client():
    first = SharedBackboneMLP()
    second = SharedBackboneMLP(backbone=first.backbone)
    return SharedBackboneRestartingSoftRoutingFedSDAClient(
        client_id=0,
        initial_models={0: first, 1: second},
        initial_stats={
            0: {'n': 10, 'mean': 0.1, 'M2': 0.0},
            1: {'n': 10, 'mean': 0.2, 'M2': 0.0},
        },
        verbose=False,
    )


def test_shared_backbone_mode_has_dedicated_client_server_and_model():
    spec = MODE_SPECS[
        "FedSDA_NoCached_SharedBackbone_ClassESR_RestartingSoftRouting"
    ]

    assert spec.client_cls is SharedBackboneRestartingSoftRoutingFedSDAClient
    assert spec.server_cls is SharedBackboneFedSDANoCachedServer
    assert spec.model_cls is SharedBackboneMLP


def test_shared_models_update_one_backbone_but_keep_separate_heads():
    first = SharedBackboneMLP()
    second = SharedBackboneMLP(backbone=first.backbone)
    before_backbone = {
        name: value.clone() for name, value in first.backbone.state_dict().items()
    }
    before_second_head = {
        name: value.clone() for name, value in second.head.state_dict().items()
    }
    x = torch.randn(8, config.dataset_spec().input_dim)
    y = torch.randint(0, 2, (8, 1)).float()

    first.update(x, y)

    assert first.backbone is second.backbone
    assert any(
        not torch.equal(value, before_backbone[name])
        for name, value in first.backbone.state_dict().items()
    )
    assert all(
        torch.equal(value, before_second_head[name])
        for name, value in second.head.state_dict().items()
    )
    assert first.backbone.optimizer is second.backbone.optimizer


def test_shared_soft_routing_extracts_features_once_for_two_heads():
    client = _two_head_client()
    x = torch.randn(1, config.dataset_spec().input_dim)
    y = torch.tensor([[1.0]])

    client._record_prediction(x, y, concept_id=0)

    assert client.compute_counters["prediction_examples"] == 2
    assert client.compute_counters["backbone_examples"] == 1
    assert client.compute_counters["head_examples"] == 2


def test_shared_server_counts_backbone_once_per_client_transfer():
    client = _two_head_client()
    sample = (
        torch.randn(1, config.dataset_spec().input_dim),
        torch.tensor([[1.0]]),
    )
    client.train_data_store[0].append(sample)
    client.train_data_store[1].append(sample)
    server = SharedBackboneFedSDANoCachedServer(verbose=False)
    server.register_client(client)
    server.register_model_params(0, client.models[0].get_params())
    server.register_model_params(1, client.models[1].get_params())
    full_values, _ = parameter_payload_size(client.models[0].get_params())

    server.update_global_models([0, 1])
    server.broadcast_models()

    # 論理モデル転送はupload 2 + download 2だが、各方向の共有部は1回だけ送る。
    assert server.comm_models_up == 2
    assert server.comm_models_down == 2
    assert server.comm_parameter_values_up < 2 * full_values
    assert server.comm_parameter_values_down < 2 * full_values
    first_backbone, _ = SharedBackboneMLP.split_params(server.global_models[0])
    second_backbone, _ = SharedBackboneMLP.split_params(server.global_models[1])
    assert all(
        torch.equal(value, second_backbone[name])
        for name, value in first_backbone.items()
    )


def test_shared_backbone_experiment_reports_component_metrics(monkeypatch):
    monkeypatch.setattr(config, "DATASET", "circle2")
    monkeypatch.setattr(config, "N_CLIENTS", 2)
    monkeypatch.setattr(config, "TOTAL_DATA_POINTS", 100)
    monkeypatch.setattr(config, "PRETRAIN_SAMPLES", 30)
    monkeypatch.setattr(config, "PRETRAIN_EPOCHS", 1)
    monkeypatch.setattr(config, "AGGREGATION_INTERVAL", 50)

    results = run_random_drift_experiment(
        mode="FedSDA_NoCached_SharedBackbone_ClassESR_RestartingSoftRouting",
        random_seed=0,
        verbose=False,
        show_plot=False,
    )

    assert results["comm_parameter_values_total"] > 0
    assert results["comm_bytes_total"] > 0
    assert results["compute_backbone_examples_total"] > 0
    assert results["compute_head_examples_total"] > 0
    assert results["final_parameter_values"] > 0
