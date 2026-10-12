"""FedDriftモデル診断の記録と全体接続を実旧へ照合する。"""

from copy import deepcopy
from dataclasses import replace
from math import isnan

import pytest

from federated_learning_experiments.evaluation.feddrift_model_diagnostic_records import (
    FeddriftSynchronizedTimestepModelCounts,
)
from federated_learning_experiments.runtime.feddrift_model_diagnostic_metric_derivation import (
    derive_feddrift_model_diagnostic_metrics,
)
from federated_learning_experiments.runtime.feddrift_run_participant_factory import (
    FeddriftRunParticipantFactory,
)
from federated_learning_experiments.runtime.feddrift_stream_run_execution import (
    execute_feddrift_stream_protocol_run,
)
from tests.refactoring.test_feddrift_run_client import assert_rng_equal, rng_snapshot
from tests.refactoring.test_feddrift_server_synchronization import (
    assert_server_equal,
    run_pair,
    seed_multiple_cached_models,
    server_pair,
)
from tests.refactoring.test_feddrift_stream_finalization import (
    assert_terminal_values_equal,
    observe_pair,
    terminal_snapshot,
)
from tests.refactoring.test_feddrift_stream_run_execution import run_inputs, run_legacy


@pytest.mark.parametrize("rounds", [1, 3])
def test_model_counts_are_recorded_only_after_successful_normal_synchronization(
    monkeypatch, rounds
):
    old, server, clients = server_pair(
        monkeypatch, local_training_round_count=rounds, evaluation_capacity=12, evaluation_add=6
    )
    seed_multiple_cached_models(old, server, clients)
    assert server.snapshot_synchronized_timestep_model_counts() == ()
    expected = []
    for timestep in (5, 10):

        def old_timestep(timestep=timestep):
            old.prepare_timestep(timestep)
            for _ in range(rounds):
                for c in old.clients:
                    c.local_train(k_steps=3)
                old.run_training_round(timestep)

        run_pair(
            old_timestep,
            lambda timestep=timestep: server.synchronize_timestep(timestep_index=timestep),
            clients,
        )
        assert_server_equal(old, server)
        expected.append(
            FeddriftSynchronizedTimestepModelCounts(
                timestep_index=timestep,
                global_model_count=len(old.global_models),
                client_held_model_counts=tuple(len(c.models) for c in old.clients),
            )
        )
        assert server.snapshot_synchronized_timestep_model_counts() == tuple(expected)


@pytest.mark.parametrize(
    "operation",
    ["prepare_timestep", "aggregate_and_distribute_training_round", "finalize_observed_stream"],
)
def test_non_synchronization_operations_do_not_add_timestep_counts(monkeypatch, operation):
    old, server, clients = server_pair(monkeypatch)
    getattr(server, operation)(timestep_index=0)
    assert server.snapshot_synchronized_timestep_model_counts() == ()


def test_failed_last_training_round_does_not_append_model_counts(monkeypatch):
    old, server, clients = server_pair(monkeypatch, local_training_round_count=3)
    operation = server.aggregate_and_distribute_training_round
    calls = []
    cause = RuntimeError("last aggregation failure")

    def fail_last(**kwargs):
        calls.append(kwargs)
        if len(calls) == 3:
            raise cause
        return operation(**kwargs)

    monkeypatch.setattr(server, "aggregate_and_distribute_training_round", fail_last)
    with pytest.raises(RuntimeError) as caught:
        server.synchronize_timestep(timestep_index=0)
    assert caught.value is cause
    assert len(calls) == 3
    assert server.snapshot_synchronized_timestep_model_counts() == ()


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("timestep_index", True, TypeError),
        ("timestep_index", -1, ValueError),
        ("global_model_count", True, TypeError),
        ("global_model_count", 0, ValueError),
        ("client_held_model_counts", [], TypeError),
        ("client_held_model_counts", (), ValueError),
        ("client_held_model_counts", (True,), TypeError),
        ("client_held_model_counts", (0,), ValueError),
    ],
)
def test_model_count_record_rejects_invalid_values(field, value, error):
    values = dict(timestep_index=0, global_model_count=1, client_held_model_counts=(1, 1))
    values[field] = value
    with pytest.raises(error):
        FeddriftSynchronizedTimestepModelCounts(**values)


def test_synchronization_counts_real_registry_including_delayed_temporary_model(monkeypatch):
    from tests.refactoring.test_feddrift_stream_finalization import delay_pending

    old, server, clients = server_pair(monkeypatch)
    observe_pair(old, server, clients, count=3, force_drift=True)
    for previous, client in zip(old.clients, clients, strict=True):
        delay_pending(previous, client)
    server.synchronize_timestep(timestep_index=0)
    record = server.snapshot_synchronized_timestep_model_counts()[0]
    assert record.global_model_count == 1
    assert record.client_held_model_counts == (2, 2)
    assert any(
        state.model_id < 0
        for state in clients[
            0
        ].owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    )


@pytest.mark.parametrize(
    "operation",
    [
        "prepare_timestep",
        "synchronize_timestep",
        "aggregate_and_distribute_training_round",
        "finalize_observed_stream",
    ],
)
def test_bad_existing_record_is_refused_before_operation(monkeypatch, operation):
    old, server, clients = server_pair(monkeypatch)
    before = deepcopy(terminal_snapshot(server))
    record = FeddriftSynchronizedTimestepModelCounts(
        timestep_index=0, global_model_count=1, client_held_model_counts=(1, 1)
    )
    object.__setattr__(record, "global_model_count", True)
    server._synchronized_timestep_model_counts.append(record)
    with pytest.raises(TypeError):
        getattr(server, operation)(timestep_index=0)
    server._synchronized_timestep_model_counts.clear()
    assert_terminal_values_equal(terminal_snapshot(server), before)


def assert_optional_auc_equal(value, old):
    if isnan(old):
        assert value is None
    else:
        assert value == old


def assert_diagnostic_matches_legacy(metrics, old):
    for prefix, summary in (
        ("model_assigned_samples", metrics.stored_training_sample_count_summary),
        ("model_training_examples", metrics.trained_sample_count_summary),
        ("model_optimizer_steps", metrics.parameter_update_step_count_summary),
    ):
        for field, suffix in (
            ("total_value", "total"),
            ("mean_value", "mean"),
            ("minimum_value", "min"),
            ("coefficient_of_variation", "cv"),
        ):
            assert getattr(summary, field) == old[prefix + "_" + suffix]
    for field, key in (
        ("mean_global_model_count_after_synchronization", "mean_model_count"),
        ("maximum_global_model_count_after_synchronization", "max_model_count"),
        ("total_global_model_count_over_synchronized_timesteps", "model_count_auc"),
        ("true_concept_known_pair_count", "clustering_oracle_pair_count"),
        ("same_true_concept_pair_count", "clustering_oracle_same_pair_count"),
    ):
        assert getattr(metrics, field) == old[key]
    agreement = metrics.cluster_assignment_agreement
    for field, key in (
        ("same_cluster_same_concept_pair_count", "tp"),
        ("same_cluster_different_concept_pair_count", "fp"),
        ("different_cluster_same_concept_pair_count", "fn"),
        ("different_cluster_different_concept_pair_count", "tn"),
        ("same_cluster_precision", "precision"),
        ("same_cluster_recall", "recall"),
        ("same_cluster_f1", "f1"),
    ):
        assert getattr(agreement, field) == old["clustering_oracle_merge_" + key]
    assert_optional_auc_equal(
        metrics.loss_increase_distance_same_concept_auc, old["clustering_oracle_loss_distance_auc"]
    )
    assert_optional_auc_equal(
        metrics.concept_specific_parameter_distance_same_concept_auc,
        old["clustering_oracle_parameter_distance_auc"],
    )


@pytest.mark.parametrize(
    "dataset,optimizer,count,threshold",
    [
        ("sine2", "sgd", 2, 0.03),
        ("sine2", "sgd", 11, 1.0),
        ("sine2", "adam", 17, 0.0),
        ("sea4", "sgd", 11, 0.0),
        ("mnist2", "adam", 11, 1.0),
    ],
)
@pytest.mark.parametrize("rounds", [1, 3])
def test_diagnostics_match_actual_legacy_whole_run_and_do_not_mutate(
    monkeypatch, dataset, optimizer, count, threshold, rounds
):
    settings, run_settings = run_inputs(monkeypatch, dataset, optimizer, count, threshold, rounds)
    old = run_legacy(monkeypatch, run_settings, threshold)
    factory = FeddriftRunParticipantFactory(run_participant_settings=settings)
    execute_feddrift_stream_protocol_run(
        execution_settings=run_settings, participant_factory=factory
    )
    participants = factory.prepared_run_participants
    before = deepcopy(terminal_snapshot(participants.run_server))
    rng = rng_snapshot()
    metrics = derive_feddrift_model_diagnostic_metrics(
        run_clients=participants.run_clients, run_server=participants.run_server
    )
    assert_diagnostic_matches_legacy(metrics, old["metrics"])
    amounts = metrics.client_held_model_training_amounts
    expected = tuple(
        (
            c.client_id,
            mid,
            len(c.train_data_store.get(mid, ())),
            c.model_training_examples.get(mid, 0),
            c.model_optimizer_steps.get(mid, 0),
        )
        for c in old["participants"][1]
        for mid in c.models
    )
    assert (
        tuple(
            (
                a.client_id,
                a.model_id,
                a.stored_training_sample_count,
                a.trained_sample_count,
                a.parameter_update_step_count,
            )
            for a in amounts
        )
        == expected
    )
    assert_terminal_values_equal(terminal_snapshot(participants.run_server), before)
    assert_rng_equal(rng)


@pytest.mark.parametrize("threshold", [-1.0, 1.0])
def test_model_amounts_and_cluster_summary_match_real_legacy_after_consolidation(
    monkeypatch, threshold
):
    from federated_drift_experiment import experiment

    old, server, clients = server_pair(
        monkeypatch,
        evaluation_capacity=12,
        evaluation_add=6,
        maximum_same_cluster_loss_increase=threshold,
    )
    seed_multiple_cached_models(old, server, clients)
    run_pair(
        lambda: old.prepare_timestep(10),
        lambda: server.prepare_timestep(timestep_index=10),
        clients,
    )
    assert_server_equal(old, server)
    expected = {}
    experiment._add_model_diagnostic_results(expected, old.clients, old, [[], []])
    experiment._add_telemetry_results(expected, old.clients, experiment._new_round_telemetry())
    metrics = derive_feddrift_model_diagnostic_metrics(run_clients=clients, run_server=server)
    assert_diagnostic_matches_legacy(metrics, expected)
    assert metrics.true_concept_known_pair_count > 0


def test_missing_counters_and_statistics_only_ids_are_not_held_model_amounts(monkeypatch):
    old, server, clients = server_pair(monkeypatch)
    clients[0].owners.training_sample_store.append_model_training_samples(
        model_id=99, training_samples=()
    )
    clients[0].owners.model_training_and_assignment_counts_store.record_completed_model_training(
        model_id=99, trained_sample_count=500, parameter_update_step_count=300
    )
    metrics = derive_feddrift_model_diagnostic_metrics(run_clients=clients, run_server=server)
    assert [(a.client_id, a.model_id) for a in metrics.client_held_model_training_amounts] == [
        (0, 0),
        (1, 0),
    ]
    assert metrics.trained_sample_count_summary.total_value == 0
    assert metrics.parameter_update_step_count_summary.total_value == 0


@pytest.mark.parametrize(
    "invalid",
    [
        "client_tuple",
        "client_order",
        "client_id",
        "count_bool",
        "count_negative",
        "count_id",
        "registry_owner",
        "training_owner",
        "counter_owner",
        "pair_owner",
        "pair_type",
        "pair_flag",
        "pair_nan",
    ],
)
def test_diagnostic_refusals_are_read_only(monkeypatch, invalid):
    from federated_learning_experiments.evaluation.model_clustering_record_store import (
        ModelPairClusteringObservation,
    )

    old, server, clients = server_pair(monkeypatch)
    chosen = clients
    if invalid == "client_tuple":
        chosen = list(clients)
    elif invalid == "client_order":
        chosen = clients[::-1]
    elif invalid == "client_id":
        clients[1]._client_id = clients[0].client_id
    elif invalid.startswith("count_"):
        values = clients[
            0
        ].owners.model_training_and_assignment_counts_store._trained_sample_counts_by_model_id
        values[True if invalid == "count_id" else 0] = (
            True if invalid == "count_bool" else -1 if invalid == "count_negative" else 1
        )
    elif invalid in ("registry_owner", "training_owner", "counter_owner"):
        field = {
            "registry_owner": "held_model_training_state_registry",
            "training_owner": "training_sample_store",
            "counter_owner": "model_training_and_assignment_counts_store",
        }[invalid]
        owner = getattr(clients[0].owners, field)
        object.__setattr__(clients[0].owners, field, object())
    elif invalid == "pair_owner":
        owner = server.model_clustering_record_store
        server.model_clustering_record_store = object()
    else:
        pair = ModelPairClusteringObservation(
            round_index=0,
            lower_model_id=0,
            higher_model_id=1,
            loss_increase_distance=1.0,
            decision_score=1.0,
            assigned_to_same_cluster=False,
            true_concepts_match=None,
            concept_specific_parameter_distance=1.0,
        )
        if invalid == "pair_flag":
            object.__setattr__(pair, "true_concepts_match", 1)
        if invalid == "pair_nan":
            object.__setattr__(pair, "loss_increase_distance", float("nan"))
        server.model_clustering_record_store._pair_clustering_observations.append(
            None if invalid == "pair_type" else pair
        )
    rng = rng_snapshot()
    # 不正owner自体のsnapshotは取れないため、その参照を検査後に戻して全stateを比較する。
    if invalid in ("registry_owner", "training_owner", "counter_owner"):
        corrupt = getattr(clients[0].owners, field)
        object.__setattr__(clients[0].owners, field, owner)
    elif invalid == "pair_owner":
        corrupt = server.model_clustering_record_store
        server.model_clustering_record_store = owner
    before = deepcopy(terminal_snapshot(server))
    if invalid in ("registry_owner", "training_owner", "counter_owner"):
        object.__setattr__(clients[0].owners, field, corrupt)
    elif invalid == "pair_owner":
        server.model_clustering_record_store = corrupt
    with pytest.raises((TypeError, ValueError)):
        derive_feddrift_model_diagnostic_metrics(run_clients=chosen, run_server=server)
    if invalid in ("registry_owner", "training_owner", "counter_owner"):
        assert getattr(clients[0].owners, field) is corrupt
        object.__setattr__(clients[0].owners, field, owner)
    elif invalid == "pair_owner":
        assert server.model_clustering_record_store is corrupt
        server.model_clustering_record_store = owner
    assert_terminal_values_equal(terminal_snapshot(server), before)
    assert_rng_equal(rng)


def test_whole_run_metrics_connect_model_diagnostics(monkeypatch):
    from federated_learning_experiments.runtime.feddrift_run_metric_derivation import (
        derive_feddrift_run_metrics,
    )
    from tests.refactoring.test_feddrift_run_metric_derivation import METRIC_SETTINGS

    settings, run_settings = run_inputs(monkeypatch, count=11)
    factory = FeddriftRunParticipantFactory(run_participant_settings=settings)
    result = execute_feddrift_stream_protocol_run(
        execution_settings=run_settings, participant_factory=factory
    )
    participants = factory.prepared_run_participants
    metrics = derive_feddrift_run_metrics(
        run_result=result, participants=participants, run_metric_settings=METRIC_SETTINGS
    )
    standalone = derive_feddrift_model_diagnostic_metrics(
        run_clients=participants.run_clients, run_server=participants.run_server
    )
    assert metrics.model_diagnostic_metrics == standalone
    assert len(metrics.model_diagnostic_metrics.synchronized_timestep_model_counts) == 3


@pytest.mark.parametrize("invalid", ["length", "position", "client_length"])
def test_whole_metrics_reject_inconsistent_synchronization_records(monkeypatch, invalid):
    from federated_learning_experiments.runtime.feddrift_run_metric_derivation import (
        derive_feddrift_run_metrics,
    )
    from tests.refactoring.test_feddrift_run_metric_derivation import METRIC_SETTINGS

    settings, run_settings = run_inputs(monkeypatch, count=5)
    factory = FeddriftRunParticipantFactory(run_participant_settings=settings)
    result = execute_feddrift_stream_protocol_run(
        execution_settings=run_settings, participant_factory=factory
    )
    participants = factory.prepared_run_participants
    server = participants.run_server
    before = deepcopy(terminal_snapshot(server))
    original = server._synchronized_timestep_model_counts
    corrupted = original.copy()
    if invalid == "length":
        corrupted.clear()
    elif invalid == "position":
        corrupted[0] = replace(corrupted[0], timestep_index=5)
    else:
        corrupted[0] = replace(corrupted[0], client_held_model_counts=(1,))
    server._synchronized_timestep_model_counts = corrupted
    rng = rng_snapshot()
    with pytest.raises(ValueError):
        derive_feddrift_run_metrics(
            run_result=result, participants=participants, run_metric_settings=METRIC_SETTINGS
        )
    assert server._synchronized_timestep_model_counts is corrupted
    server._synchronized_timestep_model_counts = original
    assert_terminal_values_equal(terminal_snapshot(server), before)
    assert_rng_equal(rng)


@pytest.mark.parametrize("distances", [(0.2, 0.2, 0.8), (float("inf"), 0.2, 0.8)])
def test_auc_connection_uses_separate_distances_and_excludes_unknown_pairs(monkeypatch, distances):
    old, server, clients = server_pair(monkeypatch, evaluation_capacity=12, evaluation_add=6)
    seed_multiple_cached_models(old, server, clients)
    run_pair(
        lambda: old.prepare_timestep(10),
        lambda: server.prepare_timestep(timestep_index=10),
        clients,
    )
    old_records = old.model_lineage.clustering_pair_observations
    new_records = server.model_clustering_record_store._pair_clustering_observations
    assert new_records
    # 実旧が生成した対の型を使い、異なる観測時刻の既知/未知の対を組み立てる。
    old_records[:] = [replace(old_records[0], round_index=10 + i) for i in range(3)]
    new_records[:] = [replace(new_records[0], round_index=10 + i) for i in range(3)]
    for i, (distance, flag) in enumerate(zip(distances, (True, False, None), strict=True)):
        old_records[i] = replace(
            old_records[i],
            oracle_same_concept=flag,
            distance=distance,
            personalized_parameter_distance=0.9 - i * 0.4,
        )
        new_records[i] = replace(
            new_records[i],
            true_concepts_match=flag,
            loss_increase_distance=distance,
            concept_specific_parameter_distance=0.9 - i * 0.4,
        )
    expected = old.clustering_oracle_diagnostic_summary()
    metrics = derive_feddrift_model_diagnostic_metrics(run_clients=clients, run_server=server)
    assert metrics.true_concept_known_pair_count == 2
    assert metrics.same_true_concept_pair_count == 1
    assert_optional_auc_equal(
        metrics.loss_increase_distance_same_concept_auc,
        expected["clustering_oracle_loss_distance_auc"],
    )
    assert_optional_auc_equal(
        metrics.concept_specific_parameter_distance_same_concept_auc,
        expected["clustering_oracle_parameter_distance_auc"],
    )
    assert metrics.concept_specific_parameter_distance_same_concept_auc == 0.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("client_id", True),
        ("model_id", True),
        ("stored_training_sample_count", True),
        ("trained_sample_count", True),
        ("parameter_update_step_count", True),
        ("stored_training_sample_count", -1),
        ("trained_sample_count", -1),
        ("parameter_update_step_count", -1),
    ],
)
def test_training_amount_record_rejects_invalid_integer(field, value):
    from federated_learning_experiments.evaluation.feddrift_model_diagnostic_records import (
        FeddriftClientHeldModelTrainingAmount,
    )

    values = dict(
        client_id=0,
        model_id=-1,
        stored_training_sample_count=0,
        trained_sample_count=0,
        parameter_update_step_count=0,
    )
    values[field] = value
    with pytest.raises(TypeError if value is True else ValueError):
        FeddriftClientHeldModelTrainingAmount(**values)
