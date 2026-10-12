"""保有数と実パラメータ演算を独立算式・実旧へ照合する。"""

from copy import deepcopy
from dataclasses import fields
from math import isnan

import pytest

from federated_drift_experiment import config
from federated_learning_experiments.methods.feddrift import (
    cached_model_cluster_consolidation as cached,
)
from federated_learning_experiments.runtime import feddrift_run_server as server_module
from federated_learning_experiments.runtime.feddrift_measured_run_execution import (
    execute_feddrift_stream_protocol_run_with_computation_measurement,
)
from federated_learning_experiments.runtime.feddrift_run_metric_derivation import (
    derive_feddrift_run_metrics,
)
from tests.refactoring.test_feddrift_run_client import (
    assert_rng_equal,
    observation_block,
    rng_snapshot,
)
from tests.refactoring.test_feddrift_run_metric_derivation import (
    METRIC_SETTINGS,
    basic_metric_values,
)
from tests.refactoring.test_feddrift_server_synchronization import (
    assert_server_equal,
    run_pair,
    seed_multiple_cached_models,
    server_pair,
)
from tests.refactoring.test_feddrift_stream_finalization import (
    assert_terminal_values_equal,
    finish_legacy,
    observe_pair,
    terminal_snapshot,
)
from tests.refactoring.test_feddrift_stream_run_execution import run_inputs


@pytest.mark.parametrize("cached", [False, True])
def test_held_counts_record_prediction_time_models_across_creation_and_consolidation(
    monkeypatch, cached
):
    old, server, clients = server_pair(monkeypatch)
    if cached:
        seed_multiple_cached_models(old, server, clients)
    before = [
        len(
            c.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        )
        for c in clients
    ]
    assert [c.detection_batch_processor.held_model_count for c in clients] == before
    observe_pair(old, server, clients, count=3, force_drift=True)
    for c, expected in zip(clients, before, strict=True):
        assert (
            c.stream_processor.held_model_count_record_store.snapshot_held_model_counts()
            == (expected,) * 3
        )
        assert c.detection_batch_processor.held_model_count == expected + 1
    run_pair(
        lambda: old.prepare_timestep(5), lambda: server.prepare_timestep(timestep_index=5), clients
    )
    assert_server_equal(old, server)
    next_counts = [
        len(
            c.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        )
        for c in clients
    ]
    observe_pair(old, server, clients, count=3, start=3)
    for c, initial, next_count in zip(clients, before, next_counts, strict=True):
        assert (
            c.stream_processor.held_model_count_record_store.snapshot_held_model_counts()
            == (initial,) * 3 + (next_count,) * 3
        )
        assert len(c.stream_processor.sample_predictions) == 6


@pytest.mark.parametrize("failure", ["prediction", "flush", "invalid_count"])
def test_failed_sample_does_not_append_held_count(monkeypatch, failure):
    old, server, clients = server_pair(monkeypatch)
    client = clients[0]
    cause = RuntimeError("injected sample failure")

    def fail(*args, **kwargs):
        raise cause

    if failure == "prediction":
        monkeypatch.setattr(
            client.detection_batch_processor.current_training_model_state.classifier,
            "forward",
            fail,
        )
    elif failure == "flush":
        monkeypatch.setattr(client.stream_processor, "flush_detection_buffer", fail)
    else:
        monkeypatch.setattr(
            type(client.detection_batch_processor), "held_model_count", property(lambda self: True)
        )
    with pytest.raises(TypeError if failure == "invalid_count" else RuntimeError) as caught:
        client.process_observed_block(
            observed_samples=observation_block(start=0, count=3, class_count=2),
            evaluation_concept_ids=(0, 0, 0),
        )
    if failure != "invalid_count":
        assert caught.value is cause
    records = client.stream_processor.held_model_count_record_store.snapshot_held_model_counts()
    assert records == ((1, 1) if failure == "flush" else ())
    assert len(records) == len(client.stream_processor.sample_predictions)
    if failure == "invalid_count":
        assert client.stream_processor.processed_sample_count == 0


@pytest.mark.parametrize("weights", ["positive", "mixed_zero", "all_zero"])
@pytest.mark.parametrize("threshold", [-1.0, 1.0])
def test_server_and_client_consolidation_count_actual_average_inputs(
    monkeypatch, weights, threshold
):
    old, server, clients = server_pair(
        monkeypatch,
        evaluation_capacity=12,
        evaluation_add=6,
        maximum_same_cluster_loss_increase=threshold,
    )
    seed_multiple_cached_models(old, server, clients)
    if weights != "positive":
        for mid in (0, 2, 3):
            value = 0 if weights == "all_zero" or mid == 2 else 7
            old.model_weights[mid] = server._model_weights[mid] = value
    expected = {"server": 0, "client": 0}

    def observe(kind, operation):
        def wrapped(**kwargs):
            result = operation(**kwargs)
            expected[kind] += sum(
                p.numel()
                for parameters in kwargs["model_parameters_by_model_id"].values()
                for p in parameters.values()
            )
            return result

        return wrapped

    monkeypatch.setattr(
        cached,
        "average_feddrift_cached_cluster_parameters",
        observe("server", cached.average_feddrift_cached_cluster_parameters),
    )
    monkeypatch.setattr(
        server_module,
        "average_feddrift_cached_cluster_parameters",
        observe("client", server_module.average_feddrift_cached_cluster_parameters),
    )
    initial = server.snapshot_server_computation_counts()
    assert all(getattr(initial, f.name) == 0 for f in fields(initial))
    pair_count = 0
    parameter_count = sum(
        p.numel()
        for p in server.global_model_repository.get_global_model_parameters(model_id=0).values()
    )
    for timestep in (5, 10):
        run_pair(
            lambda timestep=timestep: old.prepare_timestep(timestep),
            lambda timestep=timestep: server.prepare_timestep(timestep_index=timestep),
            clients,
        )
        assert_server_equal(old, server)
        counts = server.snapshot_server_computation_counts()
        pair_count = len(
            server.model_clustering_record_store.snapshot_pair_clustering_observations()
        )
        assert counts.consolidation_parameter_multiply_accumulate_count == expected["server"]
        assert server.client_consolidation_parameter_multiply_accumulate_count == expected["client"]
        assert (
            counts.diagnostic_parameter_distance_multiply_accumulate_count
            == 3 * parameter_count * pair_count
        )
        assert counts.aggregation_parameter_multiply_accumulate_count == 0
    assert pair_count > 0
    if threshold > 0:
        assert expected["server"] > 0
        assert expected["client"] == 2 * expected["server"]
    else:
        assert expected == {"server": 0, "client": 0}


@pytest.mark.parametrize("rounds", [1, 3])
def test_aggregation_count_matches_positive_weight_inputs_across_r_and_zero_weight(
    monkeypatch, rounds
):
    old, server, clients = server_pair(monkeypatch, local_training_round_count=rounds)
    observe_pair(old, server, clients, count=3)
    # client1は標本重み0として除外し、client0だけが加重和に参加する。
    old.clients[1].train_data_store[0] = []
    clients[1].owners.training_sample_store._training_samples_by_model_id[0] = []
    expected = sum(
        p.numel()
        for p in server.global_model_repository.get_global_model_parameters(model_id=0).values()
    )

    def old_timestep():
        old.prepare_timestep(0)
        for _ in range(rounds):
            for client in old.clients:
                client.local_train(k_steps=3)
            old.run_training_round(0)

    run_pair(old_timestep, lambda: server.synchronize_timestep(timestep_index=0), clients)
    assert_server_equal(old, server)
    counts = server.snapshot_server_computation_counts()
    assert counts.aggregation_parameter_multiply_accumulate_count == expected * rounds
    assert counts.consolidation_parameter_multiply_accumulate_count == 0


def test_terminal_ready_aggregation_is_counted_and_repeat_empty_terminal_adds_nothing(monkeypatch):
    old, server, clients = server_pair(monkeypatch)
    monkeypatch.setattr(
        config, "PRETRAIN_SAMPLES", server._settings.initial_model_aggregation_weight
    )
    observe_pair(old, server, clients, count=2, force_drift=True)
    before = server.snapshot_server_computation_counts()
    assert before.aggregation_parameter_multiply_accumulate_count == 0
    run_pair(
        lambda: finish_legacy(old),
        lambda: server.finalize_observed_stream(timestep_index=0),
        clients,
    )
    assert_server_equal(old, server)
    counts = server.snapshot_server_computation_counts()
    assert counts.aggregation_parameter_multiply_accumulate_count > 0
    server.finalize_observed_stream(timestep_index=0)
    assert server.snapshot_server_computation_counts() == counts
    assert all(
        len(c.stream_processor.held_model_count_record_store.snapshot_held_model_counts()) == 2
        for c in clients
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
def test_bad_timestep_does_not_change_computation_counts_or_state(monkeypatch, operation):
    old, server, clients = server_pair(monkeypatch)
    before = deepcopy(terminal_snapshot(server))
    counts = server.snapshot_server_computation_counts()
    client_count = server.client_consolidation_parameter_multiply_accumulate_count
    with pytest.raises(TypeError):
        getattr(server, operation)(timestep_index=True)
    assert_terminal_values_equal(terminal_snapshot(server), before)
    assert server.snapshot_server_computation_counts() == counts
    assert server.client_consolidation_parameter_multiply_accumulate_count == client_count


@pytest.mark.parametrize("count", [2, 11])
def test_cost_summary_uses_measured_run_records_and_preserves_basic_metrics(monkeypatch, count):
    settings, run_settings = run_inputs(monkeypatch, count=count)
    measured = execute_feddrift_stream_protocol_run_with_computation_measurement(
        execution_settings=run_settings, run_participant_settings=settings
    )
    participants = measured.participants
    result = measured.run_result
    before = deepcopy(terminal_snapshot(participants.run_server))
    rng = rng_snapshot()
    basic = derive_feddrift_run_metrics(
        run_result=result, participants=participants, run_metric_settings=METRIC_SETTINGS
    )
    metrics = derive_feddrift_run_metrics(
        run_result=result,
        participants=participants,
        run_metric_settings=METRIC_SETTINGS,
        model_computation_counts=measured.model_computation_counts,
    )
    assert_terminal_values_equal(basic_metric_values(metrics), basic_metric_values(basic))
    assert basic.model_computation_counts is basic.computation_cost_summary is None
    assert metrics.model_computation_counts == measured.model_computation_counts
    expected_processed = result.processed_sample_count_per_client * len(participants.run_clients)
    expected_held = sum(
        sum(c.stream_processor.held_model_count_record_store.snapshot_held_model_counts())
        for c in participants.run_clients
    )
    assert metrics.processed_sample_count == expected_processed
    assert metrics.held_model_sample_count == expected_held
    assert (
        metrics.client_consolidation_parameter_multiply_accumulate_count
        == participants.run_server.client_consolidation_parameter_multiply_accumulate_count
    )
    assert (
        metrics.server_computation_counts
        == participants.run_server.snapshot_server_computation_counts()
    )
    summary = metrics.computation_cost_summary
    assert summary is not None
    assert summary.processed_sample_count == expected_processed
    assert summary.held_model_sample_count == expected_held
    linear = measured.model_computation_counts
    assert summary.client_forward_multiply_accumulate_count == (
        linear.concept_specific_part_training_forward_multiply_accumulate_count
        + linear.concept_specific_part_inference_forward_multiply_accumulate_count
    )
    assert (
        summary.client_estimated_backward_multiply_accumulate_count
        == linear.concept_specific_part_estimated_backward_multiply_accumulate_count
    )
    assert summary.client_shared_part_forward_multiply_accumulate_count == 0
    server_counts = participants.run_server.snapshot_server_computation_counts()
    assert summary.server_multiply_accumulate_count == (
        server_counts.aggregation_parameter_multiply_accumulate_count
        + server_counts.consolidation_parameter_multiply_accumulate_count
    )
    assert (
        summary.server_diagnostic_multiply_accumulate_count
        == server_counts.diagnostic_parameter_distance_multiply_accumulate_count
    )
    if count == 2:
        assert expected_processed == expected_held == 0
        assert isnan(summary.mean_held_model_count)
    else:
        assert expected_processed == 18
        assert expected_held >= expected_processed
    assert_terminal_values_equal(terminal_snapshot(participants.run_server), before)
    assert_rng_equal(rng)


@pytest.mark.parametrize(
    "invalid",
    [
        "held_length",
        "held_bool",
        "held_zero",
        "held_negative",
        "held_type",
        "model_type",
        "model_bool",
        "model_negative",
    ],
)
def test_summary_rejects_bad_records_without_updates(monkeypatch, invalid):
    settings, run_settings = run_inputs(monkeypatch, count=5)
    measured = execute_feddrift_stream_protocol_run_with_computation_measurement(
        execution_settings=run_settings, run_participant_settings=settings
    )
    participants = measured.participants
    store = participants.run_clients[0].stream_processor.held_model_count_record_store
    counts = measured.model_computation_counts
    if invalid == "held_length":
        store._held_model_counts.pop()
    elif invalid.startswith("held_") and invalid != "held_type":
        store._held_model_counts[0] = {"held_bool": True, "held_zero": 0, "held_negative": -1}[
            invalid
        ]
    elif invalid == "held_type":
        participants.run_clients[0].stream_processor.held_model_count_record_store = object()
    elif invalid == "model_type":
        counts = object()
    else:
        object.__setattr__(counts, fields(counts)[0].name, True if invalid == "model_bool" else -1)
    before = deepcopy(terminal_snapshot(participants.run_server))
    rng = rng_snapshot()
    with pytest.raises((ValueError, TypeError)):
        derive_feddrift_run_metrics(
            run_result=measured.run_result,
            participants=participants,
            run_metric_settings=METRIC_SETTINGS,
            model_computation_counts=counts,
        )
    assert_terminal_values_equal(terminal_snapshot(participants.run_server), before)
    assert_rng_equal(rng)


@pytest.mark.parametrize(
    "counter",
    [
        "_aggregation_parameter_multiply_accumulate_count",
        "_consolidation_parameter_multiply_accumulate_count",
        "_diagnostic_parameter_distance_multiply_accumulate_count",
        "_client_consolidation_parameter_multiply_accumulate_count",
    ],
)
@pytest.mark.parametrize("value", [True, -1])
@pytest.mark.parametrize(
    "operation",
    [
        "prepare_timestep",
        "synchronize_timestep",
        "aggregate_and_distribute_training_round",
        "finalize_observed_stream",
    ],
)
def test_invalid_existing_counter_is_rejected_before_operation(
    monkeypatch, counter, value, operation
):
    old, server, clients = server_pair(monkeypatch)
    before = deepcopy(terminal_snapshot(server))
    setattr(server, counter, value)
    with pytest.raises(TypeError if value is True else ValueError):
        getattr(server, operation)(timestep_index=0)
    assert getattr(server, counter) is value
    setattr(server, counter, 0)
    assert_terminal_values_equal(terminal_snapshot(server), before)


def test_zero_weight_aggregation_and_empty_terminal_do_not_add_counts(monkeypatch):
    old, server, clients = server_pair(monkeypatch)
    run_pair(
        lambda: old.run_training_round(0),
        lambda: server.aggregate_and_distribute_training_round(timestep_index=0),
        clients,
    )
    assert_server_equal(old, server)
    server.finalize_observed_stream(timestep_index=0)
    assert all(
        getattr(server.snapshot_server_computation_counts(), f.name) == 0
        for f in fields(server.snapshot_server_computation_counts())
    )
    assert server.client_consolidation_parameter_multiply_accumulate_count == 0
