import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from federated_drift_experiment import config
from federated_drift_experiment.clients import (
    ADWINFedSDAClient,
    ClassConditionalESRFedSDAClient,
    ESRFedSDAClient,
    ProtectedSoftRoutingClassConditionalESRFedSDAClient,
    RestartingSoftRoutingClassConditionalESRFedSDAClient,
)
from federated_drift_experiment.drift_detectors import BoundedMeanEDetector
from federated_drift_experiment.experiment import MODE_SPECS
from federated_drift_experiment.models import SimpleMLP
from federated_drift_experiment.servers import FedSDACachedServer, FedSDANoCachedServer


def test_bounded_mean_e_detector_detects_upward_shift_and_returns_split():
    detector = BoundedMeanEDetector(baseline=0.2, alpha=0.001)

    for _ in range(200):
        detector.update(0.1)
        assert not detector.drift_detected

    for _ in range(20):
        detector.update(0.8)
        if detector.drift_detected:
            break

    assert detector.drift_detected
    assert detector.e_value >= 1.0 / detector.alpha
    assert 1 <= detector.width <= 20


def test_bounded_mean_e_detector_stays_quiet_below_baseline():
    detector = BoundedMeanEDetector(baseline=0.2, alpha=0.001, max_candidates=500)
    for _ in range(400):
        detector.update(0.1)
    assert not detector.drift_detected


def test_e_detector_modes_reuse_server_flows_without_changing_existing_modes():
    assert MODE_SPECS["FedSDA_NoCached_ADWIN"].client_cls is ADWINFedSDAClient
    assert MODE_SPECS["FedSDA_Cached_ADWIN"].client_cls is ADWINFedSDAClient
    assert MODE_SPECS["FedSDA_NoCached_ESR"].client_cls is ESRFedSDAClient
    assert MODE_SPECS["FedSDA_Cached_ESR"].client_cls is ESRFedSDAClient
    assert MODE_SPECS["FedSDA_NoCached_ESR"].server_cls is FedSDANoCachedServer
    assert MODE_SPECS["FedSDA_Cached_ESR"].server_cls is FedSDACachedServer


def test_e_detector_client_disables_uncontrolled_forced_check():
    client = ESRFedSDAClient(
        client_id=0,
        initial_models={0: SimpleMLP()},
        initial_stats={0: {"n": 100, "mean": 0.2, "M2": 1.0}},
        verbose=False,
    )
    assert not hasattr(client, "adwin")
    assert not client._forced_drift_check(100)


def test_e_detector_candidate_start_is_recorded_without_changing_fifo_split():
    client = ESRFedSDAClient(
        client_id=0,
        initial_models={0: SimpleMLP()},
        initial_stats={0: {"n": 100, "mean": 0.2, "M2": 1.0}},
        verbose=False,
    )
    client.e_detector.width = 80
    client.buffer.extend([(None, None)] * 30)

    assert client._estimated_drift_start(100) == 71
    assert client._detector_candidate_start(100) == 21


def test_forced_check_can_be_disabled_without_changing_default(monkeypatch):
    client = ADWINFedSDAClient(
        client_id=0,
        initial_models={0: SimpleMLP()},
        initial_stats={0: {"n": 100, "mean": 0.2, "M2": 1.0}},
        verbose=False,
    )
    monkeypatch.setattr(config, "FEDSDA_ENABLE_FORCED_DRIFT_CHECK", False)
    assert not client._forced_drift_check(100)


def test_class_conditional_e_detector_finds_class_local_increase():
    client = ClassConditionalESRFedSDAClient(
        client_id=0,
        initial_models={0: SimpleMLP()},
        initial_stats={0: {"n": 100, "mean": 0.6, "M2": 1.0}},
        verbose=False,
    )

    detected = None
    for sample_idx in range(800):
        class_id = sample_idx % 2
        if sample_idx < 400:
            error = 0.1 if class_id == 0 else 0.5
        else:
            error = 1.0 if class_id == 0 else 0.0
        y = torch.tensor([[float(class_id)]])
        if client._update_drift_detectors(error, y, sample_idx):
            detected = sample_idx
            break

    assert detected is not None
    assert detected >= 400
    assert client._class_drift_start is not None
    assert client._class_drift_start >= 400


def test_class_conditional_e_detector_modes_reuse_protocol_servers():
    assert MODE_SPECS["FedSDA_NoCached_ClassESR"].client_cls is ClassConditionalESRFedSDAClient
    assert MODE_SPECS["FedSDA_Cached_ClassESR"].client_cls is ClassConditionalESRFedSDAClient
    assert MODE_SPECS["FedSDA_NoCached_ClassESR"].server_cls is FedSDANoCachedServer
    assert MODE_SPECS["FedSDA_Cached_ClassESR"].server_cls is FedSDACachedServer


def test_class_esr_component_weights_keep_existing_equal_mixture():
    client = ClassConditionalESRFedSDAClient(
        client_id=0,
        initial_models={0: SimpleMLP()},
        initial_stats={0: {"n": 100, "mean": 0.2, "M2": 1.0}},
        verbose=False,
    )

    assert client.overall_component_weight == pytest.approx(1.0 / 3.0)
    assert client.class_component_weight == pytest.approx(1.0 / 3.0)


def test_restarting_soft_routing_restarts_only_after_model_change():
    spec = MODE_SPECS[
        "FedSDA_NoCached_ClassESR_RestartingSoftRouting"
    ]
    client = spec.client_cls(
        client_id=0,
        initial_models={0: SimpleMLP(), 1: SimpleMLP()},
        initial_stats={
            0: {"n": 100, "mean": 0.2, "M2": 1.0},
            1: {"n": 100, "mean": 0.2, "M2": 1.0},
        },
        verbose=False,
    )
    client.expert_router.probabilities([0, 1])

    client._set_local_current_model(0)
    assert client.expert_router.concept_restart_count == 0

    client._set_local_current_model(1)
    assert client.expert_router.concept_restart_count == 1
    assert client.expert_router.probabilities([0, 1]) == {0: 0.5, 1: 0.5}
    assert spec.client_cls is RestartingSoftRoutingClassConditionalESRFedSDAClient


def test_restarting_soft_routing_records_oracle_recovery_diagnostics():
    spec = MODE_SPECS[
        "FedSDA_NoCached_ClassESR_RestartingSoftRouting"
    ]
    client = spec.client_cls(
        client_id=0,
        initial_models={0: SimpleMLP(), 1: SimpleMLP()},
        initial_stats={
            0: {"n": 100, "mean": 0.2, "M2": 1.0},
            1: {"n": 100, "mean": 0.2, "M2": 1.0},
        },
        verbose=False,
    )
    # 等重み混合は0.5となって誤答するが、モデル1単体なら正答する。
    client.models[0].forward = lambda x: torch.full((len(x), 1), 0.2)
    client.models[1].forward = lambda x: torch.full((len(x), 1), 0.9)
    client.models[0].per_sample_error = lambda x, y: torch.full((len(x),), 0.8)
    client.models[1].per_sample_error = lambda x, y: torch.full((len(x),), 0.1)
    client._prediction_probabilities = lambda _: {0: 0.75, 1: 0.25}

    client._record_prediction(
        torch.zeros((1, config.dataset_spec().input_dim)), torch.ones((1, 1)), 0
    )

    assert client.routing_diagnostics == {
        "sample_count": 1,
        "oracle_correct_count": 1,
        "mixture_correct_count": 0,
        "leader_correct_count": 0,
        "confidence_leader_correct_count": 1,
        "missed_oracle_count": 1,
        "confidence_leader_missed_oracle_count": 0,
    }
    assert client.history_routing_oracle_correct == [1]
    assert dict(client.routing_class_diagnostics[1]) == {
        "sample_count": 1,
        "oracle_correct_count": 1,
        "mixture_correct_count": 0,
        "leader_correct_count": 0,
        "confidence_leader_correct_count": 1,
        "missed_oracle_count": 1,
        "confidence_leader_missed_oracle_count": 0,
    }


def test_soft_routing_reuses_prediction_forward_for_expert_loss():
    spec = MODE_SPECS[
        "FedSDA_NoCached_ClassESR_RestartingSoftRouting"
    ]
    client = spec.client_cls(
        client_id=0,
        initial_models={0: SimpleMLP(), 1: SimpleMLP()},
        initial_stats={
            0: {"n": 100, "mean": 0.2, "M2": 1.0},
            1: {"n": 100, "mean": 0.2, "M2": 1.0},
        },
        verbose=False,
    )
    forward_calls = {0: 0, 1: 0}

    for model_id, score in ((0, 0.1), (1, 0.9)):
        def forward(x, model_id=model_id, score=score):
            forward_calls[model_id] += 1
            return torch.full((len(x), 1), score)

        client.models[model_id].forward = forward
        # 損失計算で二度目のforwardを行う旧経路へ戻った場合は失敗させる。
        client.models[model_id].per_sample_error = lambda *_: pytest.fail(
            "per_sample_error must not issue another forward"
        )

    client._record_prediction(
        torch.zeros((1, config.dataset_spec().input_dim)), torch.ones((1, 1)), 0
    )

    assert forward_calls == {0: 1, 1: 1}


def test_soft_routing_records_leave_one_out_contribution_without_forward():
    spec = MODE_SPECS[
        "FedSDA_NoCached_ClassESR_RestartingSoftRouting"
    ]
    client = spec.client_cls(
        client_id=0,
        initial_models={0: SimpleMLP(), 1: SimpleMLP()},
        initial_stats={
            0: {"n": 100, "mean": 0.2, "M2": 1.0},
            1: {"n": 100, "mean": 0.2, "M2": 1.0},
        },
        verbose=False,
    )
    forward_calls = {0: 0, 1: 0}
    for model_id, score in ((0, 0.1), (1, 0.9)):
        def forward(x, model_id=model_id, score=score):
            forward_calls[model_id] += 1
            return torch.full((len(x), 1), score)

        client.models[model_id].forward = forward

    client._record_prediction(
        torch.zeros((1, config.dataset_spec().input_dim)),
        torch.ones((1, 1)),
        concept_id=0,
    )

    records = {
        model_id: aggregate
        for _, _, model_id, aggregate in (
            client.routing_leave_one_out_diagnostics.iter_records()
        )
    }
    assert forward_calls == {0: 1, 1: 1}
    assert records[0].bounded_delta_sum == pytest.approx(-0.4)
    assert records[1].bounded_delta_sum == pytest.approx(0.4)
    assert records[0].zero_one_delta_sum == -1.0
    assert records[1].zero_one_delta_sum == 0.0
    assert records[0].hard_assignment_count == 1
    assert records[1].hard_assignment_count == 0
    assert records[0].fallback_count == 0
    assert client.routing_leave_one_out_diagnostics.pool_epoch == 0


def test_predicted_class_context_keeps_separate_online_evidence(monkeypatch):
    monkeypatch.setattr(config, "SOFT_ROUTING_CONTEXT", "predicted_class")
    spec = MODE_SPECS[
        "FedSDA_NoCached_ClassESR_RestartingSoftRouting"
    ]
    client = spec.client_cls(
        client_id=0,
        initial_models={0: SimpleMLP(), 1: SimpleMLP()},
        initial_stats={
            0: {"n": 100, "mean": 0.2, "M2": 1.0},
            1: {"n": 100, "mean": 0.2, "M2": 1.0},
        },
        verbose=False,
    )
    client.models[0].forward = lambda x: torch.full((len(x), 1), 0.2)
    client.models[1].forward = lambda x: torch.full((len(x), 1), 0.9)

    client._record_prediction(
        torch.zeros((1, config.dataset_spec().input_dim)), torch.ones((1, 1)), 0
    )

    assert set(client.context_expert_routers) == {1}
    assert client.context_expert_routers[1].cumulative_losses == {
        0: pytest.approx(0.8),
        1: pytest.approx(0.1),
    }
    client._on_local_model_change(0, 1)
    assert client.context_expert_routers[1].cumulative_losses == {}


def test_predicted_class_records_shadow_meta_router_without_extra_forward(
    monkeypatch,
):
    monkeypatch.setattr(config, "SOFT_ROUTING_CONTEXT", "predicted_class")
    monkeypatch.setattr(config, "SOFT_ROUTING_META_LOSS", "bounded_score")
    spec = MODE_SPECS[
        "FedSDA_NoCached_ClassESR_RestartingSoftRouting"
    ]
    client = spec.client_cls(
        client_id=0,
        initial_models={0: SimpleMLP(), 1: SimpleMLP()},
        initial_stats={
            0: {"n": 100, "mean": 0.2, "M2": 1.0},
            1: {"n": 100, "mean": 0.2, "M2": 1.0},
        },
        verbose=False,
    )
    forward_calls = {0: 0, 1: 0}

    for model_id, score in ((0, 0.4), (1, 0.9)):
        def forward(x, model_id=model_id, score=score):
            forward_calls[model_id] += 1
            return torch.full((len(x), 1), score)

        client.models[model_id].forward = forward

    # 大域ルータはmodel 0、予測クラス0の文脈ルータはmodel 1を選ぶ状態にする。
    client.expert_router.cumulative_losses = {0: 0.0, 1: 2.0}
    client.expert_router.mixability_gap = 1.0
    context_router = client.context_expert_routers[0]
    context_router.cumulative_losses = {0: 2.0, 1: 0.0}
    context_router.mixability_gap = 1.0

    client._record_prediction(
        torch.zeros((1, config.dataset_spec().input_dim)),
        torch.ones((1, 1)),
        0,
    )

    assert forward_calls == {0: 1, 1: 1}
    assert client.routing_meta_diagnostics == {
        "sample_count": 1,
        "correct_count": 1,
        "actual_correct_count": 1,
        "global_correct_count": 0,
        "context_mixture_correct_count": 1,
        "context_leader_correct_count": 1,
        "context_leader_weight_sum": pytest.approx(0.5),
        "context_leader_preferred_count": 0,
    }
    assert client.history_routing_meta_correct == [1]
    assert client.history_routing_meta_global_correct == [0]
    assert client.history_routing_meta_context_mixture_correct == [1]
    assert client.history_routing_meta_context_leader_correct == [1]
    assert client.history_routing_meta_context_leader_weight == [
        pytest.approx(0.5)
    ]
    assert client.shadow_meta_routers[0].cumulative_losses == {
        "global_mixture": pytest.approx(0.5),
        "context_leader": pytest.approx(0.1),
    }
    assert dict(client.routing_class_diagnostics[1])[
        "meta_correct_count"
    ] == 1

    client._on_local_model_change(0, 1)
    assert client.shadow_meta_routers[0].cumulative_losses == {}


def test_meta_predicted_class_uses_meta_scores_as_actual_prediction(
    monkeypatch,
):
    monkeypatch.setattr(
        config, "SOFT_ROUTING_CONTEXT", "meta_predicted_class"
    )
    spec = MODE_SPECS[
        "FedSDA_NoCached_ClassESR_RestartingSoftRouting"
    ]
    client = spec.client_cls(
        client_id=0,
        initial_models={0: SimpleMLP(), 1: SimpleMLP(), 2: SimpleMLP()},
        initial_stats={
            model_id: {"n": 100, "mean": 0.2, "M2": 1.0}
            for model_id in range(3)
        },
        verbose=False,
    )
    for model_id, score in ((0, 0.1), (1, 0.2), (2, 0.9)):
        client.models[model_id].forward = (
            lambda x, score=score: torch.full((len(x), 1), score)
        )

    # 大域候補と文脈leaderは正解するが、文脈mixtureだけは0.45となり誤答する。
    client.expert_router.cumulative_losses = {0: 2.0, 1: 2.0, 2: 0.0}
    context_router = client.context_expert_routers[1]
    context_router.probabilities = lambda _: {0: 0.3, 1: 0.3, 2: 0.4}

    client._record_prediction(
        torch.zeros((1, config.dataset_spec().input_dim)),
        torch.ones((1, 1)),
        0,
    )

    assert client.history_accuracy == [1.0]
    assert client.routing_meta_diagnostics["correct_count"] == 1
    assert client.routing_meta_diagnostics["actual_correct_count"] == 1
    assert client.routing_meta_diagnostics[
        "context_mixture_correct_count"
    ] == 0


def test_meta_switching_uses_the_selected_top_level_candidate(monkeypatch):
    monkeypatch.setattr(config, "SOFT_ROUTING_CONTEXT", "meta_switching")
    monkeypatch.setattr(config, "SOFT_ROUTING_TOP_COMBINATION", "leader")
    spec = MODE_SPECS[
        "FedSDA_NoCached_ClassESR_RestartingSoftRouting"
    ]
    client = spec.client_cls(
        client_id=0,
        initial_models={0: SimpleMLP(), 1: SimpleMLP(), 2: SimpleMLP()},
        initial_stats={
            model_id: {"n": 100, "mean": 0.2, "M2": 1.0}
            for model_id in range(3)
        },
        verbose=False,
    )
    for model_id, score in ((0, 0.1), (1, 0.2), (2, 0.9)):
        client.models[model_id].forward = (
            lambda x, score=score: torch.full((len(x), 1), score)
        )

    # 現行Metaは正解し、モデル追従型mixtureは誤答する状態を作る。
    client.expert_router.cumulative_losses = {0: 2.0, 1: 2.0, 2: 0.0}
    context_router = client.context_expert_routers[1]
    context_router.probabilities = lambda _: {0: 0.3, 1: 0.3, 2: 0.4}
    client.switching_expert_router.weights = {0: 0.9, 1: 0.05, 2: 0.05}
    client.meta_switching_router.weights = {"meta": 0.1, "switching": 0.9}

    client._record_prediction(
        torch.zeros((1, config.dataset_spec().input_dim)),
        torch.ones((1, 1)),
        0,
    )

    assert client.routing_meta_diagnostics["correct_count"] == 1
    assert client.routing_meta_switching_diagnostics[
        "selected_switching_count"
    ] == 1
    assert client.routing_meta_switching_diagnostics["correct_count"] == 0
    assert client.history_accuracy == [0.0]


def test_meta_switching_can_mix_top_level_candidates(monkeypatch):
    monkeypatch.setattr(config, "SOFT_ROUTING_CONTEXT", "meta_switching")
    monkeypatch.setattr(config, "SOFT_ROUTING_TOP_COMBINATION", "mixture")
    spec = MODE_SPECS[
        "FedSDA_NoCached_ClassESR_RestartingSoftRouting"
    ]
    client = spec.client_cls(
        client_id=0,
        initial_models={0: SimpleMLP(), 1: SimpleMLP(), 2: SimpleMLP()},
        initial_stats={
            model_id: {"n": 100, "mean": 0.2, "M2": 1.0}
            for model_id in range(3)
        },
        verbose=False,
    )
    for model_id, score in ((0, 0.1), (1, 0.2), (2, 0.9)):
        client.models[model_id].forward = (
            lambda x, score=score: torch.full((len(x), 1), score)
        )

    client.expert_router.cumulative_losses = {0: 2.0, 1: 2.0, 2: 0.0}
    context_router = client.context_expert_routers[1]
    context_router.probabilities = lambda _: {0: 0.3, 1: 0.3, 2: 0.4}
    client.switching_expert_router.weights = {0: 0.9, 1: 0.05, 2: 0.05}
    # switching側が僅かにleaderでも、Meta側の強い予測を捨てずに混合する。
    client.meta_switching_router.weights = {"meta": 0.49, "switching": 0.51}

    client._record_prediction(
        torch.zeros((1, config.dataset_spec().input_dim)),
        torch.ones((1, 1)),
        0,
    )

    assert client.routing_meta_switching_diagnostics[
        "selected_switching_count"
    ] == 1
    assert client.routing_meta_switching_diagnostics["correct_count"] == 1
    assert client.history_accuracy == [1.0]


def test_meta_router_can_update_with_zero_one_loss(monkeypatch):
    monkeypatch.setattr(config, "SOFT_ROUTING_CONTEXT", "predicted_class")
    monkeypatch.setattr(config, "SOFT_ROUTING_META_LOSS", "zero_one")
    spec = MODE_SPECS[
        "FedSDA_NoCached_ClassESR_RestartingSoftRouting"
    ]
    client = spec.client_cls(
        client_id=0,
        initial_models={0: SimpleMLP(), 1: SimpleMLP()},
        initial_stats={
            0: {"n": 100, "mean": 0.2, "M2": 1.0},
            1: {"n": 100, "mean": 0.2, "M2": 1.0},
        },
        verbose=False,
    )
    client.models[0].forward = lambda x: torch.full((len(x), 1), 0.4)
    client.models[1].forward = lambda x: torch.full((len(x), 1), 0.9)
    client.expert_router.cumulative_losses = {0: 0.0, 1: 2.0}
    client.expert_router.mixability_gap = 1.0
    context_router = client.context_expert_routers[0]
    context_router.cumulative_losses = {0: 2.0, 1: 0.0}
    context_router.mixability_gap = 1.0

    client._record_prediction(
        torch.zeros((1, config.dataset_spec().input_dim)),
        torch.ones((1, 1)),
        0,
    )

    assert client.shadow_meta_routers[0].cumulative_losses == {
        "global_mixture": pytest.approx(1.0),
        "context_leader": pytest.approx(0.0),
    }


def test_protected_soft_routing_keeps_incumbent_until_proposal_is_better():
    spec = MODE_SPECS["FedSDA_NoCached_ClassESR_ProtectedSoftRouting"]
    client = spec.client_cls(
        client_id=0,
        initial_models={0: SimpleMLP(), 1: SimpleMLP()},
        initial_stats={
            0: {"n": 100, "mean": 0.2, "M2": 1.0},
            1: {"n": 100, "mean": 0.2, "M2": 1.0},
        },
        verbose=False,
    )
    proposal = client.expert_router.probabilities([0, 1])

    protected = client._prediction_probabilities(proposal)

    assert protected == {0: 1.0, 1: 0.0}
    assert client.history_routing_gate_open == [False]

    client.expert_router.cumulative_losses = {0: 2.0, 1: 0.0}
    proposal = {0: 0.25, 1: 0.75}

    released = client._prediction_probabilities(proposal)

    assert released == proposal
    assert client.history_routing_gate_open[-1]
    assert spec.client_cls is ProtectedSoftRoutingClassConditionalESRFedSDAClient
