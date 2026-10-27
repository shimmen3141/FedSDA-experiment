"""公式との整合に必要な設定と、旧設定の境界を確かめる。"""

from dataclasses import replace

import pytest
import torch

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.methods.feddrift.detection_batch_assessment import (
    assess_feddrift_detection_batch,
)
from federated_learning_experiments.methods.feddrift.loss_distance_model_clustering import (
    _loss_increase_distances,
)
from federated_learning_experiments.runtime.feddrift_default_run_settings import (
    build_default_feddrift_run_settings,
)
from federated_learning_experiments.runtime.feddrift_detection_batch_processing import (
    FeddriftDetectionBatchProcessor,
)
from federated_learning_experiments.sweep.experiment_sweep_plan import (
    ExperimentSweepPlan,
    build_planned_experiment_runs,
)
from tests.refactoring.test_feddrift_detection_batch_assessment import build_models
from tests.refactoring.test_feddrift_detection_batch_processing import (
    build_processing_oracle,
    make_samples,
)
from tests.refactoring.test_feddrift_run_participant_factory import make_inputs
from tests.refactoring.test_feddrift_server_synchronization import (
    seed_multiple_cached_models,
    server_pair,
)


@pytest.mark.parametrize(
    "field,value",
    (
        ("model_comparison_criterion", "unknown"),
        ("model_comparison_criterion", 1),
        ("new_model_parameter_source", "unknown"),
        ("detection_reference_after_drift", None),
    ),
)
def test_invalid_client_policy_is_rejected(monkeypatch, field, value):
    settings, _ = make_inputs(monkeypatch)
    with pytest.raises((TypeError, ValueError)):
        replace(settings.run_client_settings, **{field: value})


@pytest.mark.parametrize(
    "field,value",
    (
        ("model_comparison_criterion", "unknown"),
        ("model_comparison_criterion", True),
        ("training_model_eligibility", "unknown"),
        ("training_model_eligibility", None),
        ("consolidation_weight_timing", "unknown"),
        ("consolidation_weight_timing", 1),
    ),
)
def test_invalid_server_policy_is_rejected(monkeypatch, field, value):
    settings, _ = make_inputs(monkeypatch)
    with pytest.raises((TypeError, ValueError)):
        replace(settings.server_settings, **{field: value})


def official_settings(settings):
    return replace(
        settings,
        run_client_settings=replace(
            settings.run_client_settings,
            model_comparison_criterion="classification_error_rate",
            new_model_parameter_source="minimum_criterion_held_model",
            detection_reference_after_drift="minimum_held_model_batch_value",
        ),
        server_settings=replace(
            settings.server_settings,
            model_comparison_criterion="classification_error_rate",
            training_model_eligibility="models_assigned_in_current_timestep",
            consolidation_weight_timing="weights_including_current_timestep_assignments",
            new_model_isolation_timestep_count=1,
        ),
    )


def test_official_policies_require_matching_criterion_and_one_timestep(monkeypatch):
    settings, _ = make_inputs(monkeypatch)
    new = official_settings(settings)
    with pytest.raises(ValueError):
        replace(new.server_settings, new_model_isolation_timestep_count=2)
    with pytest.raises(ValueError):
        replace(new, server_settings=settings.server_settings)
    with pytest.raises(ValueError):
        replace(
            settings,
            run_client_settings=replace(
                settings.run_client_settings,
                new_model_parameter_source="minimum_criterion_held_model",
            ),
        )


def set_constant_prediction(classifier, probability):
    with torch.no_grad():
        for parameter in classifier.parameters():
            parameter.zero_()
        tuple(classifier.parameters())[-1].fill_(torch.logit(torch.tensor(probability)))


@pytest.mark.parametrize("probability", (0.51, 0.5))
def test_confidence_change_uses_prediction_error_and_first_tie(monkeypatch, probability):
    _, models = build_models(class_count=2, monkeypatch=monkeypatch)
    for _, model in models:
        set_constant_prediction(model, probability)
    arguments = dict(
        held_models=models,
        input_features=torch.zeros(3, 3),
        observed_class_labels=torch.ones(3, 1),
        previous_minimum_mean_bounded_loss=0.01,
        loss_increase_threshold=0.1,
    )
    old = assess_feddrift_detection_batch(**arguments)
    new = assess_feddrift_detection_batch(
        **arguments, model_comparison_criterion="classification_error_rate"
    )
    assert old.drift_detected
    assert new.drift_detected == (probability == 0.5)
    assert new.minimum_mean_bounded_loss == (1.0 if probability == 0.5 else 0.0)
    assert new.minimum_loss_model_id == models[0][0]


def test_drift_copies_best_existing_model_and_saves_its_error(monkeypatch):
    _, _, owners = build_processing_oracle(
        class_count=2, previous_loss=0.0, threshold=0.1, monkeypatch=monkeypatch
    )
    states = owners[
        "held_model_training_state_registry"
    ].snapshot_ordered_held_model_training_states()
    for state, probability in zip(states, (0.9, 0.1, 0.8), strict=True):
        set_constant_prediction(state.classifier, probability)
    processor = FeddriftDetectionBatchProcessor(
        **owners,
        model_comparison_criterion="classification_error_rate",
        new_model_parameter_source="minimum_criterion_held_model",
        detection_reference_after_drift="minimum_held_model_batch_value",
    )
    # ラベル0が2件、1が1件なので、現在の9は誤り2/3、-104が最小の1/3。
    result = processor.process_detection_batch(
        training_samples=make_samples(sample_count=3, class_count=2), processed_sample_count=3
    )
    assert result.adaptation_kind == "new_model_created"
    assert result.next_previous_minimum_mean_bounded_loss == pytest.approx(1 / 3)
    created = owners["held_model_training_state_registry"].get_held_model_training_state(
        model_id=result.assigned_model_id
    )
    for actual, expected in zip(
        created.classifier.parameters(), states[1].classifier.parameters(), strict=True
    ):
        assert torch.equal(actual, expected)


def test_error_distance_is_clamped_at_zero():
    sums = {
        (0, 0): (10, 5.0, 5.0),
        (0, 1): (10, 2.0, 2.0),
        (1, 0): (10, 1.0, 1.0),
        (1, 1): (10, 4.0, 4.0),
    }
    args = dict(model_ids=(0, 1), loss_sums_by_model_pair=sums, minimum_evaluated_sample_count=5)
    assert _loss_increase_distances(**args)[0, 1] == pytest.approx(-0.3)
    assert (
        _loss_increase_distances(**args, model_comparison_criterion="classification_error_rate")[
            0, 1
        ]
        == 0.0
    )


def prepared_server(monkeypatch):
    old, server, clients = server_pair(
        monkeypatch, 2, "sgd", evaluation_capacity=50, evaluation_add=20
    )
    seed_multiple_cached_models(old, server, clients)
    server._settings = replace(
        server._settings,
        model_comparison_criterion="classification_error_rate",
        training_model_eligibility="models_assigned_in_current_timestep",
        consolidation_weight_timing="weights_including_current_timestep_assignments",
        new_model_isolation_timestep_count=1,
        maximum_cross_evaluation_client_count=2,
        maximum_client_evaluation_sample_count=50,
        local_training_round_count=1,
    )
    # 上流helperは旧との対照用に標本・計数も入れる。ここでは保有構造だけを借りる。
    for client in clients:
        client._settings = replace(
            client._settings,
            model_comparison_criterion="classification_error_rate",
            new_model_parameter_source="minimum_criterion_held_model",
            detection_reference_after_drift="minimum_held_model_batch_value",
        )
        client.detection_batch_processor._criterion = "classification_error_rate"
        client.detection_batch_processor._parameter_source = "minimum_criterion_held_model"
        client.detection_batch_processor._reference_after_drift = "minimum_held_model_batch_value"
        client.owners.training_sample_store._training_samples_by_model_id.clear()
        client.owners.evaluation_sample_store._evaluation_samples_by_model_id.clear()
        client.owners.model_training_and_assignment_counts_store._parameter_update_step_counts_by_model_id.clear()
        client.owners.model_training_and_assignment_counts_store._trained_sample_counts_by_model_id.clear()
    return server, clients


def test_inactive_model_is_preserved_without_training_or_parameter_transfer(monkeypatch):
    server, clients = prepared_server(monkeypatch)
    # 全モデルを隔離し、この確認では統合を起こさない。全clientの現在の割当は2。
    server._isolation_until = {0: 10, 2: 10, 3: 10}
    snapshots = []
    for client in clients:
        for mid in (0, 2, 3):
            client.owners.training_sample_store.append_model_training_samples(
                model_id=mid, training_samples=make_samples(sample_count=4, class_count=2)
            )
        snapshots.append(
            client.owners.held_model_training_state_registry.get_held_model_training_state(
                model_id=0
            )
        )
    transfers = []
    original = server._record_parameter_transfer
    monkeypatch.setattr(
        server,
        "_record_parameter_transfer",
        lambda direction, parameters, count: (
            transfers.append((direction, count)),
            original(direction, parameters, count),
        ),
    )
    server.synchronize_timestep(timestep_index=0)
    assert transfers == [("upload", 1), ("upload", 1), ("download", 2)]
    communication = server.communication_volume_record_store.get_state_snapshot()
    assert communication.uploaded_message_count == len(clients)
    assert communication.downloaded_message_count == len(clients)
    for client, previous in zip(clients, snapshots, strict=True):
        registry = client.owners.held_model_training_state_registry
        assert registry.get_held_model_training_state(model_id=0) is previous
        assert client.get_held_model_ids() == {0, 2, 3}
        counts = client.owners.model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts()
        assert set(counts.parameter_update_step_counts_by_model_id) == {2}


def test_current_assignments_are_shared_across_clients_for_training(monkeypatch):
    server, clients = prepared_server(monkeypatch)
    server._isolation_until = {0: 10, 2: 10, 3: 10}
    clients[0].owners.current_training_model_assignment.assign_model_for_training(model_id=0)
    for client in clients:
        for mid in (0, 2, 3):
            client.owners.training_sample_store.append_model_training_samples(
                model_id=mid, training_samples=make_samples(sample_count=4, class_count=2)
            )
    server.synchronize_timestep(timestep_index=0)
    for client in clients:
        counts = client.owners.model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts()
        assert set(counts.parameter_update_step_counts_by_model_id) == {0, 2}


def test_current_assignment_counts_weight_merge_and_are_not_counted_twice(monkeypatch):
    server, clients = prepared_server(monkeypatch)
    # clientごとA=5、B=10、全体はA=10、B=20。保存済みの重みは同数。
    server._model_weights.update({0: 10, 2: 10, 3: 7})
    for client in clients:
        for mid, count in ((0, 5), (2, 10)):
            client.owners.training_sample_store.append_model_training_samples(
                model_id=mid, training_samples=make_samples(sample_count=count, class_count=2)
            )
    before = {
        mid: server.global_model_repository.get_global_model_parameters(model_id=mid)
        for mid in (0, 2)
    }
    monkeypatch.setattr(
        server,
        "_cross_evaluate",
        lambda *args: {(i, j): (10, 0.0, 0.0) for i in (0, 2) for j in (0, 2)},
    )
    server.prepare_timestep(timestep_index=0)
    assert server.snapshot_model_weights() == {0: 30, 3: 7}
    for name, actual in server.global_model_repository.get_global_model_parameters(
        model_id=0
    ).items():
        torch.testing.assert_close(actual, (before[0][name] * 10 + before[2][name] * 20) / 30)
    server.aggregate_and_distribute_training_round(timestep_index=0)
    assert server.snapshot_model_weights()[0] == 30


def test_cross_evaluation_uses_errors_and_current_batch_samples(monkeypatch):
    server, clients = prepared_server(monkeypatch)
    for client in clients:
        # 評価標本の追加を含む本番のバッチ処理を通す。
        for state in client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states():
            set_constant_prediction(state.classifier, 0.9)
        client.detection_batch_processor.process_detection_batch(
            training_samples=make_samples(sample_count=12, class_count=2), processed_sample_count=12
        )
    reference = (
        clients[0]
        .owners.held_model_training_state_registry.get_held_model_training_state(model_id=0)
        .classifier
    )
    server.global_model_repository.set_global_model_parameters(
        model_id=0,
        parameter_snapshot={
            name: value.detach().clone() for name, value in reference.state_dict().items()
        },
    )
    statistics = server._cross_evaluate(0, (0,))
    # 各clientの現在の塊12件、正解6件。誤りの二乗和も誤り数と同じ。
    assert statistics[0, 0] == (24, 12.0, 12.0)


def test_one_timestep_isolation_expires_and_other_clients_can_select(monkeypatch):
    server, clients = prepared_server(monkeypatch)
    creating = clients[0].detection_batch_processor
    creating._previous_loss = 0.0
    creating._threshold = 0.1
    creating._criterion = "classification_error_rate"
    creating._parameter_source = "minimum_criterion_held_model"
    creating._reference_after_drift = "minimum_held_model_batch_value"
    for client in clients:
        for state in client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states():
            set_constant_prediction(state.classifier, 0.9)
    transition = creating.process_detection_batch(
        training_samples=make_samples(sample_count=12, class_count=2), processed_sample_count=12
    )
    assert transition.adaptation_kind == "new_model_created"
    server.synchronize_timestep(timestep_index=0)
    assert server.snapshot_isolation_until()[4] == 1
    other = clients[1]
    states = other.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    assert 4 in {state.model_id for state in states}
    for state in states:
        set_constant_prediction(state.classifier, 0.1 if state.model_id == 4 else 0.9)
    selected = assess_feddrift_detection_batch(
        held_models=tuple((s.model_id, s.classifier) for s in states),
        input_features=torch.zeros(6, 3),
        observed_class_labels=torch.zeros(6, 1),
        previous_minimum_mean_bounded_loss=None,
        loss_increase_threshold=0.1,
        model_comparison_criterion="classification_error_rate",
    )
    assert selected.minimum_loss_model_id == 4
    candidates = []
    original = server._cross_evaluate

    def observe(timestep, ids):
        candidates.append(ids)
        return original(timestep, ids)

    monkeypatch.setattr(server, "_cross_evaluate", observe)
    server.prepare_timestep(timestep_index=1)
    assert 4 in candidates[0]


@pytest.mark.parametrize(
    "dataset,threshold",
    (
        ("sea2", 0.04),
        ("sea4", 0.04),
        ("sine2", 0.2),
        ("circle2", 0.1),
        ("mnist2", 0.1),
        ("mnist4", 0.1),
    ),
)
def test_default_policies_and_dataset_threshold_with_explicit_override(dataset, threshold):
    conditions = ExperimentRunConditions(
        dataset_name=dataset,
        random_seed=0,
        client_count=7,
        per_client_sample_count=12,
        server_aggregation_interval_per_client_samples=3,
    )
    for override, expected in ((None, threshold), (0.07, 0.07)):
        kwargs = {} if override is None else {"loss_increase_threshold": override}
        settings = build_default_feddrift_run_settings(
            experiment_run_conditions=conditions, **kwargs
        ).run_participant_settings
        assert settings == official_settings(settings)
        assert settings.run_client_settings.loss_increase_threshold == expected
        assert settings.server_settings.maximum_same_cluster_loss_increase == expected
        assert settings.server_settings.maximum_cross_evaluation_client_count == 7


@pytest.mark.parametrize("axis", (None, (0.07, 0.09)))
def test_sweep_threshold_overrides_dataset_defaults(axis):
    runs = build_planned_experiment_runs(
        sweep_plan=ExperimentSweepPlan(
            configuration_names=("feddrift",),
            dataset_names=("sea2", "sine2"),
            random_seeds=(0,),
            client_count=7,
            per_client_sample_count=12,
            server_aggregation_intervals_per_client_samples=(3,),
            feddrift_loss_increase_thresholds=axis,
        )
    )
    assert [
        r.run_settings.run_participant_settings.run_client_settings.loss_increase_threshold
        for r in runs
    ] == (list(axis) * 2 if axis else [0.04, 0.2])
    for run in runs:
        participant = run.run_settings.run_participant_settings
        assert (
            participant.run_client_settings.loss_increase_threshold
            == participant.server_settings.maximum_same_cluster_loss_increase
        )


def test_historical_feddrift_settings_and_unchanged_fedsda_fedavg_settings():
    import behavior_golden_cases as golden

    for version in range(1, 14):
        for name in ("feddrift", "feddrift-drift-tail"):
            participant = golden.build_case_settings(name, version).run_participant_settings
            assert participant.run_client_settings.model_comparison_criterion == "bounded_loss"
            assert participant.run_client_settings.loss_increase_threshold == (
                0.03 if name == "feddrift-drift-tail" else 0.1
            )
            assert participant.server_settings.maximum_cross_evaluation_client_count == 3
    for name in golden.CASE_NAMES_BY_BASELINE_VERSION[13]:
        old, new = golden.build_case_settings(name, 13), golden.build_case_settings(name, 14)
        if name in ("feddrift", "feddrift-drift-tail"):
            assert new.run_participant_settings == official_settings(new.run_participant_settings)
        else:
            assert old == new
