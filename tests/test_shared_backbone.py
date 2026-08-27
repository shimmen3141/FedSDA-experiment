import numpy as np
import torch

from federated_drift_experiment import config, run_random_drift_experiment
from federated_drift_experiment.clients import (
    ResidualAdapterClassADWINRestartingSoftRoutingFedSDAClient,
    ResidualAdapterClassConditionalESRFedSDAClient,
    ResidualAdapterRestartingSoftRoutingFedSDAClient,
    SharedBackboneRestartingSoftRoutingFedSDAClient,
)
from federated_drift_experiment.experiment import (
    MODE_SPECS,
    _routing_window_accuracies,
)
from federated_drift_experiment.models import (
    ResidualAdapterMLP,
    SharedBackboneMLP,
    SharedClassifierResidualAdapterMLP,
    parameter_payload_size,
)
from federated_drift_experiment.servers import SharedBackboneFedSDANoCachedServer


def test_routing_window_accuracies_separates_recovery_and_stable_samples():
    result = _routing_window_accuracies(
        [[1, 0, 1, 1, 0]], [[1]], window=2,
    )

    assert result["recovery_accuracy"] == 0.5
    assert result["stable_accuracy"] == 2 / 3


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


def _populate_training_store(client):
    input_dim = config.dataset_spec().input_dim
    for model_id in (0, 1):
        for index in range(client.batch_size):
            client.train_data_store[model_id].append((
                torch.full((1, input_dim), float(index + model_id) / 10),
                torch.tensor([[float((index + model_id) % 2)]]),
            ))


def test_shared_backbone_mode_has_dedicated_client_server_and_model():
    spec = MODE_SPECS[
        "FedSDA_NoCached_SharedBackbone_ClassESR_RestartingSoftRouting"
    ]

    assert spec.client_cls is SharedBackboneRestartingSoftRoutingFedSDAClient
    assert spec.server_cls is SharedBackboneFedSDANoCachedServer
    assert spec.model_cls is SharedBackboneMLP


def test_residual_adapter_mode_has_dedicated_client_and_model():
    spec = MODE_SPECS[
        "FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting"
    ]

    assert spec.client_cls is ResidualAdapterRestartingSoftRoutingFedSDAClient
    assert spec.server_cls is SharedBackboneFedSDANoCachedServer
    assert spec.model_cls is ResidualAdapterMLP


def test_shared_classifier_residual_mode_reuses_fedsda_flow():
    spec = MODE_SPECS[
        "FedSDA_NoCached_ResidualAdapter_SharedClassifier_"
        "ClassESR_RestartingSoftRouting"
    ]

    assert spec.client_cls is ResidualAdapterRestartingSoftRoutingFedSDAClient
    assert spec.server_cls is SharedBackboneFedSDANoCachedServer
    assert spec.model_cls is SharedClassifierResidualAdapterMLP


def test_residual_adapter_class_adwin_mode_reuses_routing_architecture():
    spec = MODE_SPECS[
        "FedSDA_NoCached_ResidualAdapter_ClassADWIN_RestartingSoftRouting"
    ]

    assert (
        spec.client_cls
        is ResidualAdapterClassADWINRestartingSoftRoutingFedSDAClient
    )
    assert spec.server_cls is SharedBackboneFedSDANoCachedServer
    assert spec.model_cls is ResidualAdapterMLP


def _two_residual_adapter_client():
    first = ResidualAdapterMLP()
    second = ResidualAdapterMLP(backbone=first.backbone)
    second.set_params(first.get_params())
    return ResidualAdapterRestartingSoftRoutingFedSDAClient(
        client_id=0,
        initial_models={0: first, 1: second},
        initial_stats={
            0: {'n': 10, 'mean': 0.1, 'M2': 0.0},
            1: {'n': 10, 'mean': 0.1, 'M2': 0.0},
        },
        verbose=False,
    )


def test_periodic_routing_active_set_skips_inactive_model_forward(monkeypatch):
    monkeypatch.setattr(config, "DATASET", "circle2")
    monkeypatch.setattr(
        config, "ROUTING_ACTIVE_SET_POLICY", "periodic_forward_probe"
    )
    monkeypatch.setattr(config, "NEW_MODEL_FORWARD_VALIDATION_SAMPLES", 2)
    monkeypatch.setattr(config, "SOFT_ROUTING_CONTEXT", "global")
    client = _two_residual_adapter_client()
    x = torch.zeros(1, config.dataset_spec().input_dim)
    y = torch.zeros(1, 1)

    client.processed_samples = 1
    before = client.compute_counters["prediction_forward_calls"]
    client._record_prediction(x, y, concept_id=0)
    client.processed_samples = 2
    client._record_prediction(x, y, concept_id=0)
    after_probe = client.compute_counters["prediction_forward_calls"]
    evidence_after_probe = dict(client.expert_router.cumulative_losses)
    switching_after_probe = dict(client.switching_expert_router.weights)

    client.processed_samples = 3
    client._record_prediction(x, y, concept_id=0)
    after_apply = client.compute_counters["prediction_forward_calls"]

    assert after_probe - before == 4
    assert after_apply - after_probe == 1
    assert client.routing_active_set.probe_sample_count == 2
    assert client.routing_active_set.apply_retained_global_model_count_sum == 1
    assert client.expert_router.cumulative_losses == evidence_after_probe
    assert client.switching_expert_router.weights == switching_after_probe

    client._set_local_current_model(1)
    client.processed_samples = 4
    before_restart_probe = client.compute_counters["prediction_forward_calls"]
    client._record_prediction(x, y, concept_id=1)

    assert (
        client.compute_counters["prediction_forward_calls"]
        - before_restart_probe
    ) == 2


def test_residual_adapter_hard_routing_mode_has_dedicated_client_and_model():
    spec = MODE_SPECS["FedSDA_NoCached_ResidualAdapter_ClassESR"]

    assert spec.client_cls is ResidualAdapterClassConditionalESRFedSDAClient
    assert spec.server_cls is SharedBackboneFedSDANoCachedServer
    assert spec.model_cls is ResidualAdapterMLP
    assert not issubclass(
        spec.client_cls, ResidualAdapterRestartingSoftRoutingFedSDAClient
    )


def test_residual_adapter_starts_with_same_function_as_full_sharing(monkeypatch):
    monkeypatch.setattr(config, "DATASET", "circle2")
    full = SharedBackboneMLP()
    residual = ResidualAdapterMLP(backbone=full.backbone)
    residual.head.load_state_dict(full.head.state_dict())
    x = torch.randn(8, config.dataset_spec().input_dim)

    assert residual.adapter.rank == min(
        config.SHARED_ADAPTER_RANK, full.backbone.output_dim
    )
    assert torch.count_nonzero(residual.adapter.up.weight) == 0
    assert torch.count_nonzero(residual.adapter.up.bias) == 0
    assert torch.equal(full(x), residual(x))


def test_residual_adapter_is_personalized_and_trainable(monkeypatch):
    monkeypatch.setattr(config, "DATASET", "circle2")
    model = ResidualAdapterMLP()
    x = torch.randn(16, config.dataset_spec().input_dim)
    y = torch.randint(0, 2, (16, 1)).float()

    model.update(x, y)
    _, personalized = model.split_params(model.get_params())

    assert torch.count_nonzero(model.adapter.up.weight) > 0
    assert any(name.startswith("adapter.") for name in personalized)


def test_shared_classifier_is_pooled_while_expert_residuals_stay_separate(
    monkeypatch,
):
    monkeypatch.setattr(config, "DATASET", "circle2")
    first = SharedClassifierResidualAdapterMLP()
    second = SharedClassifierResidualAdapterMLP(backbone=first.backbone)
    x = torch.randn(16, config.dataset_spec().input_dim)
    y = torch.randint(0, 2, (16, 1)).float()
    before_shared = {
        name: value.clone()
        for name, value in first.backbone.shared_classifier.state_dict().items()
    }
    before_second_residual = {
        name: value.clone() for name, value in second.head.state_dict().items()
    }

    assert torch.equal(first(x), second(x))
    first.update(x, y)
    shared, personalized = first.split_params(first.get_params())

    assert first.backbone is second.backbone
    assert any(
        not torch.equal(value, before_shared[name])
        for name, value in first.backbone.shared_classifier.state_dict().items()
    )
    assert all(
        torch.equal(value, before_second_residual[name])
        for name, value in second.head.state_dict().items()
    )
    assert any(name.startswith("backbone.shared_classifier.") for name in shared)
    assert any(name.startswith("adapter.") for name in personalized)
    assert any(name.startswith("head.") for name in personalized)


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


def test_joint_training_updates_backbone_once_and_both_heads(monkeypatch):
    monkeypatch.setattr(config, "SHARED_BACKBONE_TRAINING", "joint")
    monkeypatch.setattr(config, "SHARED_BACKBONE_GRADIENT_STRATEGY", "mean")
    client = _two_head_client()
    _populate_training_store(client)
    backbone_before = {
        name: value.clone() for name, value in client._shared_backbone().state_dict().items()
    }
    heads_before = {
        model_id: {
            name: value.clone() for name, value in model.head.state_dict().items()
        }
        for model_id, model in client.models.items()
    }

    client.train_all_held_models()

    assert client.compute_counters["backbone_optimizer_steps"] == 1
    assert client.compute_counters["head_optimizer_steps"] == 2
    assert client.compute_counters["optimizer_steps"] == 2
    assert client.backbone_gradient_diagnostics["pair_count"] == 1
    assert client.backbone_gradient_diagnostics["applied_pair_count"] == 1
    assert client.backbone_gradient_diagnostics["update_comparison_count"] == 1
    assert (
        client.backbone_gradient_diagnostics["applied_conflict_count"]
        == client.backbone_gradient_diagnostics["conflict_count"]
    )
    assert abs(
        client.backbone_gradient_diagnostics["update_cosine_sum"] - 1.0
    ) < 1e-6
    assert abs(
        client.backbone_gradient_diagnostics["update_norm_ratio_sum"] - 1.0
    ) < 1e-6
    assert client.backbone_gradient_diagnostics["update_delta_ratio_sum"] == 0.0
    assert any(
        not torch.equal(value, backbone_before[name])
        for name, value in client._shared_backbone().state_dict().items()
    )
    for model_id, model in client.models.items():
        assert any(
            not torch.equal(value, heads_before[model_id][name])
            for name, value in model.head.state_dict().items()
        )


def test_pcgrad_joint_training_records_diagnostics_and_updates(monkeypatch):
    monkeypatch.setattr(config, "SHARED_BACKBONE_TRAINING", "joint")
    monkeypatch.setattr(config, "SHARED_BACKBONE_GRADIENT_STRATEGY", "pcgrad")
    client = _two_head_client()
    _populate_training_store(client)

    client.train_all_held_models()

    assert client.compute_counters["backbone_optimizer_steps"] == 1
    assert client.backbone_gradient_diagnostics["pair_count"] == 1
    assert client.backbone_gradient_diagnostics["applied_pair_count"] == 1
    assert client.backbone_gradient_diagnostics["update_comparison_count"] == 1
    assert (
        client.backbone_gradient_diagnostics["applied_conflict_count"]
        <= client.backbone_gradient_diagnostics["conflict_count"]
    )


def test_frozen_training_keeps_backbone_and_updates_both_heads(monkeypatch):
    monkeypatch.setattr(config, "SHARED_BACKBONE_TRAINING", "frozen")
    client = _two_head_client()
    _populate_training_store(client)
    backbone_before = {
        name: value.clone() for name, value in client._shared_backbone().state_dict().items()
    }
    heads_before = {
        model_id: {
            name: value.clone() for name, value in model.head.state_dict().items()
        }
        for model_id, model in client.models.items()
    }

    client.train_all_held_models()

    assert client.compute_counters["backbone_optimizer_steps"] == 0
    assert client.compute_counters["head_optimizer_steps"] == 2
    assert all(
        torch.equal(value, backbone_before[name])
        for name, value in client._shared_backbone().state_dict().items()
    )
    for model_id, model in client.models.items():
        assert any(
            not torch.equal(value, heads_before[model_id][name])
            for name, value in model.head.state_dict().items()
        )


def test_sequential_training_remains_the_default(monkeypatch):
    monkeypatch.setattr(config, "SHARED_BACKBONE_TRAINING", "sequential")
    client = _two_head_client()
    _populate_training_store(client)

    client.train_all_held_models()

    assert client.compute_counters["backbone_optimizer_steps"] == 2
    assert client.compute_counters["head_optimizer_steps"] == 2
    assert client.compute_counters["optimizer_steps"] == 2


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


def test_shared_cross_evaluation_deduplicates_backbone_and_heads():
    first_client = _two_head_client()
    second_client = _two_head_client()
    server = SharedBackboneFedSDANoCachedServer(verbose=False)
    first_params = first_client.models[0].get_params()
    second_params = first_client.models[1].get_params()
    backbone, first_head = SharedBackboneMLP.split_params(first_params)
    _, second_head = SharedBackboneMLP.split_params(second_params)
    backbone_values, _ = parameter_payload_size(backbone)
    first_head_values, _ = parameter_payload_size(first_head)
    second_head_values, _ = parameter_payload_size(second_head)

    server._begin_cross_evaluation_model_transfers()
    server._record_cross_evaluation_model_transfer(
        0, first_params, [first_client, second_client]
    )
    server._record_cross_evaluation_model_transfer(
        0, first_params, [first_client]
    )
    server._record_cross_evaluation_model_transfer(
        1, second_params, [first_client]
    )

    assert server.comm_models_down == 3
    assert server.comm_parameter_values_down == (
        2 * backbone_values + 2 * first_head_values + second_head_values
    )


def test_aggregation_restart_recalibrates_router_after_round(monkeypatch):
    monkeypatch.setattr(
        config,
        "SHARED_BACKBONE_ROUTING_RECALIBRATION",
        "aggregation_restart",
    )
    client = _two_head_client()
    _populate_training_store(client)
    probabilities = client.expert_router.probabilities([0, 1])
    client.expert_router.update({0: 0.0, 1: 1.0}, probabilities)
    server = SharedBackboneFedSDANoCachedServer(verbose=False)
    server.register_client(client)
    server.register_model_params(0, client.models[0].get_params())
    server.register_model_params(1, client.models[1].get_params())

    server.run_round(1, clustering_enabled=False)

    assert client.expert_router.probabilities([0, 1]) == {0: 0.5, 1: 0.5}
    assert client.expert_router.aggregation_restart_count == 1


def test_fifo_replay_rebuilds_router_from_post_aggregation_predictions(monkeypatch):
    monkeypatch.setattr(
        config,
        "SHARED_BACKBONE_ROUTING_RECALIBRATION",
        "fifo_replay",
    )
    client = _two_head_client()
    with torch.no_grad():
        client.models[0].head.weight.zero_()
        client.models[0].head.bias.fill_(5.0)
        client.models[1].head.weight.zero_()
        client.models[1].head.bias.fill_(-5.0)
    for _ in range(4):
        client.buffer.append((
            torch.zeros(1, config.dataset_spec().input_dim),
            torch.tensor([[0.0]]),
            0,
        ))
    probabilities = client.expert_router.probabilities([0, 1])
    client.expert_router.update({0: 0.0, 1: 1.0}, probabilities)

    client.recalibrate_routing_after_aggregation()

    replayed = client.expert_router.probabilities([0, 1])
    assert replayed[1] > replayed[0]
    assert client.expert_router.aggregation_recalibration_count == 1
    assert client.expert_router.aggregation_recalibration_sample_count == 4
    assert client.compute_counters["routing_recalibration_examples"] == 8
    assert client.compute_counters["backbone_examples"] == 4
    assert client.compute_counters["head_examples"] == 8


def test_leader_change_replay_preserves_evidence_for_same_fifo_leader(monkeypatch):
    monkeypatch.setattr(
        config,
        "SHARED_BACKBONE_ROUTING_RECALIBRATION",
        "leader_change_replay",
    )
    client = _two_head_client()
    with torch.no_grad():
        client.models[0].head.weight.zero_()
        client.models[0].head.bias.fill_(-5.0)
        client.models[1].head.weight.zero_()
        client.models[1].head.bias.fill_(5.0)
    for _ in range(4):
        client.buffer.append((
            torch.zeros(1, config.dataset_spec().input_dim),
            torch.tensor([[0.0]]),
        ))
    probabilities = client.expert_router.probabilities([0, 1])
    client.expert_router.update({0: 0.0, 1: 1.0}, probabilities)
    previous_losses = dict(client.expert_router.cumulative_losses)

    client.recalibrate_routing_after_aggregation()

    assert client.expert_router.cumulative_losses == previous_losses
    assert client.expert_router.aggregation_recalibration_check_count == 1
    assert client.expert_router.aggregation_recalibration_skip_count == 1
    assert client.expert_router.aggregation_recalibration_count == 0
    # replayを省略しても、leader判定のためのFIFO評価計算は発生する。
    assert client.compute_counters["routing_recalibration_examples"] == 8


def test_persistent_leader_replay_is_available_to_shared_client(monkeypatch):
    monkeypatch.setattr(
        config,
        "SHARED_BACKBONE_ROUTING_RECALIBRATION",
        "persistent_leader_change_replay",
    )
    client = _two_head_client()
    with torch.no_grad():
        client.models[0].head.weight.zero_()
        client.models[0].head.bias.fill_(5.0)
        client.models[1].head.weight.zero_()
        client.models[1].head.bias.fill_(-5.0)
    for _ in range(4):
        client.buffer.append((
            torch.zeros(1, config.dataset_spec().input_dim),
            torch.tensor([[0.0]]),
        ))
    probabilities = client.expert_router.probabilities([0, 1])
    client.expert_router.update({0: 0.0, 1: 1.0}, probabilities)

    client.recalibrate_routing_after_aggregation()

    assert client.expert_router.probabilities([0, 1])[1] > 0.5
    assert client.expert_router.aggregation_recalibration_count == 1


def test_shared_backbone_experiment_reports_component_metrics(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(config, "DATASET", "circle2")
    monkeypatch.setattr(config, "N_CLIENTS", 2)
    monkeypatch.setattr(config, "TOTAL_DATA_POINTS", 100)
    monkeypatch.setattr(config, "PRETRAIN_SAMPLES", 30)
    monkeypatch.setattr(config, "PRETRAIN_EPOCHS", 1)
    monkeypatch.setattr(config, "AGGREGATION_INTERVAL", 50)
    monkeypatch.setattr(config, "SOFT_ROUTING_CONTEXT", "predicted_class")

    raw_path = tmp_path / "shared-routing.npz"
    results = run_random_drift_experiment(
        mode="FedSDA_NoCached_SharedBackbone_ClassESR_RestartingSoftRouting",
        random_seed=0,
        verbose=False,
        show_plot=False,
        raw_path=str(raw_path),
    )

    assert results["comm_parameter_values_total"] > 0
    assert results["comm_bytes_total"] > 0
    assert results["compute_backbone_examples_total"] > 0
    assert results["compute_head_examples_total"] > 0
    assert results["compute_backbone_optimizer_steps_total"] > 0
    assert results["compute_head_optimizer_steps_total"] > 0
    assert results["final_parameter_values"] > 0
    assert results["routing_class_macro_oracle_accuracy"] > 0
    assert results["routing_class_macro_mixture_accuracy"] > 0
    assert results["routing_confidence_leader_accuracy"] > 0
    assert results["routing_class_macro_confidence_leader_accuracy"] > 0
    assert results["routing_class_oracle_gap_std"] >= 0
    assert results["routing_meta_accuracy"] > 0
    assert results["routing_meta_global_accuracy"] >= 0
    assert results["routing_meta_context_mixture_accuracy"] >= 0
    assert results["routing_meta_context_leader_accuracy"] >= 0
    assert results["routing_meta_best_candidate_gain_rate"] <= 1
    assert results["routing_meta_context_leader_weight_mean"] >= 0
    assert results["routing_switching_accuracy"] > 0
    assert -1 <= results["routing_switching_gain_rate"] <= 1
    assert -1 <= results["routing_switching_global_gain_rate"] <= 1
    assert 0 <= results["routing_switching_stable_accuracy"] <= 1
    assert -1 <= results["routing_switching_stable_gain_rate"] <= 1
    assert 0 <= results["routing_switching_recovery_accuracy"] <= 1
    assert -1 <= results["routing_switching_recovery_gain_rate"] <= 1
    assert results["routing_switching_effective_experts_mean"] >= 1
    assert results["routing_meta_switching_accuracy"] > 0
    assert -1 <= results["routing_meta_switching_meta_gain_rate"] <= 1
    assert -1 <= results[
        "routing_meta_switching_switching_gain_rate"
    ] <= 1
    assert 0 <= results[
        "routing_meta_switching_selected_switching_rate"
    ] <= 1
    assert results["routing_loo_evaluation_count"] >= 0
    assert -1 <= results["routing_loo_bounded_delta_mean"] <= 1
    assert -1 <= results["routing_loo_zero_one_delta_mean"] <= 1
    assert 0 <= results["routing_loo_positive_rate"] <= 1
    assert (
        results["routing_loo_active_unassigned_nonpositive_model_count"]
        <= results["routing_loo_active_unassigned_evaluable_model_count"]
    )
    assert (
        results["routing_loo_active_joint_nonpositive_model_count"]
        <= results["routing_loo_active_evaluable_model_count"]
    )
    assert 0 <= results[
        "routing_archive_shadow_retained_global_model_rate"
    ] <= 1
    assert -1 <= results["routing_archive_shadow_accuracy_delta"] <= 1
    assert 0 <= results["routing_active_set_retained_global_model_rate"] <= 1
    assert 0 <= results[
        "routing_active_set_apply_retained_global_model_rate"
    ] <= 1
    with np.load(raw_path) as raw:
        assert raw["routing_class_client_ids"].shape == (
            len(raw["routing_class_ids"]),
        )
        assert raw["routing_class_sample_counts"].sum() == results[
            "routing_sample_count"
        ]
        assert "routing_class_confidence_leader_correct_counts" in raw
        assert raw["history_routing_meta_correct"].shape == (2, 100)
        assert raw["history_routing_meta_global_correct"].shape == (2, 100)
        assert raw["history_routing_switching_correct"].shape == (2, 100)
        assert raw["history_routing_switching_leader_id"].shape == (2, 100)
        assert raw[
            "history_routing_switching_effective_experts"
        ].shape == (2, 100)
        assert raw[
            "history_routing_meta_switching_correct"
        ].shape == (2, 100)
        assert raw[
            "history_routing_meta_switching_selected_switching"
        ].shape == (2, 100)
        assert raw[
            "routing_class_meta_sample_counts"
        ].sum() == results["routing_sample_count"]
        assert raw["routing_loo_sample_counts"].sum() == results[
            "routing_loo_evaluation_count"
        ]
        assert len(raw["routing_loo_client_ids"]) == len(
            raw["routing_loo_model_ids"]
        )
        assert len(raw["routing_loo_pool_epochs"]) == len(
            raw["routing_loo_block_indices"]
        )
        assert "routing_loo_is_active_final" in raw
        assert "routing_loo_is_assigned_final" in raw
        assert "routing_loo_final_active_model_ids" in raw
        assert "routing_loo_final_assigned_model_ids" in raw
        assert set(raw["routing_loo_final_assigned_model_ids"]) <= set(
            raw["routing_loo_final_active_model_ids"]
        )
        assigned_record_ids = set(
            raw["routing_loo_model_ids"][
                raw["routing_loo_is_assigned_final"]
            ]
        )
        assert assigned_record_ids <= set(
            raw["routing_loo_final_assigned_model_ids"]
        )


def test_meta_context_actual_accuracy_matches_shadow_prediction(
    monkeypatch,
):
    monkeypatch.setattr(config, "DATASET", "circle2")
    monkeypatch.setattr(config, "N_CLIENTS", 2)
    monkeypatch.setattr(config, "TOTAL_DATA_POINTS", 100)
    monkeypatch.setattr(config, "PRETRAIN_SAMPLES", 30)
    monkeypatch.setattr(config, "PRETRAIN_EPOCHS", 1)
    monkeypatch.setattr(config, "AGGREGATION_INTERVAL", 50)

    monkeypatch.setattr(config, "SOFT_ROUTING_CONTEXT", "predicted_class")
    shadow = run_random_drift_experiment(
        mode="FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting",
        random_seed=0,
        verbose=False,
        show_plot=False,
    )
    monkeypatch.setattr(
        config, "SOFT_ROUTING_CONTEXT", "meta_predicted_class"
    )
    actual = run_random_drift_experiment(
        mode="FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting",
        random_seed=0,
        verbose=False,
        show_plot=False,
    )

    assert actual["accuracy"] == shadow["routing_meta_accuracy"]
    assert actual["routing_meta_accuracy"] == actual["accuracy"]
    assert actual["comm_models_total"] == shadow["comm_models_total"]
    assert actual["compute_model_examples_total"] == shadow[
        "compute_model_examples_total"
    ]


def test_meta_switching_actual_accuracy_matches_shadow_prediction(
    monkeypatch,
):
    monkeypatch.setattr(config, "DATASET", "circle2")
    monkeypatch.setattr(config, "N_CLIENTS", 2)
    monkeypatch.setattr(config, "TOTAL_DATA_POINTS", 100)
    monkeypatch.setattr(config, "PRETRAIN_SAMPLES", 30)
    monkeypatch.setattr(config, "PRETRAIN_EPOCHS", 1)
    monkeypatch.setattr(config, "AGGREGATION_INTERVAL", 50)

    monkeypatch.setattr(
        config, "SOFT_ROUTING_CONTEXT", "meta_predicted_class"
    )
    shadow = run_random_drift_experiment(
        mode="FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting",
        random_seed=0,
        verbose=False,
        show_plot=False,
    )
    monkeypatch.setattr(config, "SOFT_ROUTING_CONTEXT", "meta_switching")
    actual = run_random_drift_experiment(
        mode="FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting",
        random_seed=0,
        verbose=False,
        show_plot=False,
    )

    assert actual["accuracy"] == shadow["routing_meta_switching_accuracy"]
    assert actual["routing_meta_switching_accuracy"] == actual["accuracy"]
    assert actual["comm_models_total"] == shadow["comm_models_total"]
    assert actual["compute_model_examples_total"] == shadow[
        "compute_model_examples_total"
    ]


def test_residual_adapter_experiment_reports_component_metrics(monkeypatch):
    monkeypatch.setattr(config, "DATASET", "circle2")
    monkeypatch.setattr(config, "N_CLIENTS", 2)
    monkeypatch.setattr(config, "TOTAL_DATA_POINTS", 100)
    monkeypatch.setattr(config, "PRETRAIN_SAMPLES", 30)
    monkeypatch.setattr(config, "PRETRAIN_EPOCHS", 1)
    monkeypatch.setattr(config, "AGGREGATION_INTERVAL", 50)

    results = run_random_drift_experiment(
        mode="FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting",
        random_seed=0,
        verbose=False,
        show_plot=False,
    )

    assert results["comm_parameter_values_total"] > 0
    assert results["compute_backbone_examples_total"] > 0
    assert results["compute_head_examples_total"] > 0
    assert results["final_parameter_values"] > 0


def test_residual_adapter_hard_routing_experiment_runs_without_router(monkeypatch):
    monkeypatch.setattr(config, "DATASET", "circle2")
    monkeypatch.setattr(config, "N_CLIENTS", 2)
    monkeypatch.setattr(config, "TOTAL_DATA_POINTS", 100)
    monkeypatch.setattr(config, "PRETRAIN_SAMPLES", 30)
    monkeypatch.setattr(config, "PRETRAIN_EPOCHS", 1)
    monkeypatch.setattr(config, "AGGREGATION_INTERVAL", 50)

    results = run_random_drift_experiment(
        mode="FedSDA_NoCached_ResidualAdapter_ClassESR",
        random_seed=0,
        verbose=False,
        show_plot=False,
    )

    assert results["comm_parameter_values_total"] > 0
    assert results["compute_backbone_examples_total"] > 0
    assert results["compute_head_examples_total"] > 0
    assert results.get("routing_mixture_predictions_total", 0) == 0
