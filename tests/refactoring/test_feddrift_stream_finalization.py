"""観測済みbufferから最終配布までを実旧の終端操作へ照合する。"""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import fields, is_dataclass
from math import isnan

import pytest
import torch

from federated_drift_experiment import config, experiment
from tests.refactoring.test_feddrift_run_client import (
    assert_rng_equal,
    observation_block,
    rng_snapshot,
    snapshot_client,
)
from tests.refactoring.test_feddrift_run_participant_factory import prepare_pair
from tests.refactoring.test_feddrift_server_synchronization import (
    assert_server_equal,
    run_pair,
    seed_multiple_cached_models,
    server_pair,
)


def assert_terminal_values_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_terminal_values_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_terminal_values_equal(a, b)
    elif is_dataclass(actual):
        for field in fields(actual):
            assert_terminal_values_equal(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(actual, float) and isnan(actual):
        assert isinstance(expected, float) and isnan(expected)
    else:
        assert actual == expected


def finish_legacy(server, timestep=0):
    for client in server.clients:
        client.flush()
    if any(client.has_pending_model() for client in server.clients):
        server.run_round(timestep, clustering_enabled=True)
        for client in server.clients:
            client.promote_pending_to_ready()
    server.finalize_protocol(timestep + 1)


def observe_pair(old, new, clients, *, count, force_drift=False, start=0, class_count=2):
    block = observation_block(start=start, count=count, class_count=class_count)
    concepts = tuple(i % 2 for i in range(start, start + count))
    data = [
        (torch.tensor((s.feature_values,)), torch.tensor([[float(s.class_label)]])) for s in block
    ]
    for old_client, client in zip(old.clients, clients, strict=True):
        if force_drift:
            old_client.last_min_loss = 0.0
            old_client.distance_threshold = 0.0
            client.detection_batch_processor._previous_loss = 0.0
            client.detection_batch_processor._threshold = 0.0
        run_pair(
            lambda old_client=old_client: old_client.process_batch(data, concepts),
            lambda client=client: client.process_observed_block(
                observed_samples=block, evaluation_concept_ids=concepts
            ),
            clients,
        )
    assert_server_equal(old, new)


def delay_pending(old_client, client):
    old_client.pending_model_ready = False
    state = client.owners.pending_model_upload_state
    pending = state.get_pending_model_upload()
    state.queue_model_upload(
        model_id=pending.model_id,
        parameter_snapshot=pending.parameter_snapshot,
        upload_delay_round_count=2,
    )


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("optimizer_name", ("sgd", "adam"))
@pytest.mark.parametrize(
    "case",
    (
        "empty",
        "partial_retained",
        "partial_drift",
        "ready",
        "delayed",
        "mixed",
        "cached",
        "cached_zero",
        "ready_partial_retained",
    ),
)
@pytest.mark.parametrize("disable_grad", (False, True))
def test_finalization_matches_real_terminal_flow(
    monkeypatch, class_count, optimizer_name, case, disable_grad
):
    old, new, clients = server_pair(monkeypatch, class_count, optimizer_name)
    monkeypatch.setattr(config, "PRETRAIN_SAMPLES", new._settings.initial_model_aggregation_weight)
    if case in ("cached", "cached_zero"):
        seed_multiple_cached_models(old, new, clients)
    if case != "empty":
        observe_pair(
            old,
            new,
            clients,
            count=2 if case.startswith("partial") else 3,
            force_drift=case != "partial_retained",
            class_count=class_count,
        )
    if case == "ready_partial_retained":
        for old_client, client in zip(old.clients, clients, strict=True):
            old_client.distance_threshold = 1.0
            client.detection_batch_processor._threshold = 1.0
        observe_pair(old, new, clients, count=2, start=3, class_count=class_count)
    if case == "delayed":
        for old_client, client in zip(old.clients, clients, strict=True):
            delay_pending(old_client, client)
    elif case == "mixed":
        delay_pending(old.clients[1], clients[1])
    elif case == "cached_zero":
        for old_client, client in zip(old.clients, clients, strict=True):
            old_client.train_data_store.clear()
            client.owners.training_sample_store._training_samples_by_model_id.clear()
    predictions = tuple(c.stream_processor.sample_predictions for c in clients)
    counts = tuple(c.stream_processor.processed_sample_count for c in clients)
    with torch.no_grad() if disable_grad else nullcontext():
        run_pair(
            lambda: finish_legacy(old),
            lambda: new.finalize_observed_stream(timestep_index=0),
            clients,
        )
    assert_server_equal(old, new)
    assert predictions == tuple(c.stream_processor.sample_predictions for c in clients)
    assert counts == tuple(c.stream_processor.processed_sample_count for c in clients)
    assert all(not c.stream_processor.buffered_training_samples for c in clients)
    if case == "delayed":
        assert all(
            c.owners.pending_model_upload_state.remaining_upload_delay_round_count == 2
            for c in clients
        )
    elif case == "mixed":
        assert clients[1].has_model_ready_for_server_registration()
    # 再呼出しも旧と一致し、mixedでは前回昇格したモデルを回収する。
    run_pair(
        lambda: finish_legacy(old, 1),
        lambda: new.finalize_observed_stream(timestep_index=1),
        clients,
    )
    assert_server_equal(old, new)


def terminal_snapshot(server):
    return (
        tuple(snapshot_client(c, c.python_random_generator) for c in server.run_clients),
        tuple(
            (
                c.stream_processor.processed_sample_count,
                c.stream_processor.buffered_training_samples,
                c.stream_processor.sample_predictions,
                c.stream_processor.completed_detection_batch_transitions,
                (
                    c.stream_processor.held_model_count_record_store.snapshot_held_model_counts()
                    if hasattr(
                        c.stream_processor.held_model_count_record_store,
                        "snapshot_held_model_counts",
                    )
                    else type(c.stream_processor.held_model_count_record_store)
                ),
            )
            for c in server.run_clients
        ),
        tuple(
            (mid, server.global_model_repository.get_global_model_parameters(model_id=mid))
            for mid in server.global_model_repository.global_model_ids
        ),
        server.global_model_repository.snapshot_global_model_loss_statistics(),
        server.global_model_repository.snapshot_model_registration_records(),
        server.global_model_repository.next_global_model_id,
        server.snapshot_model_weights(),
        server.snapshot_isolation_until(),
        server.snapshot_server_computation_counts(),
        server.client_consolidation_parameter_multiply_accumulate_count,
        server.communication_volume_record_store.get_state_snapshot(),
        server.cross_evaluation_record_store.snapshot_client_cross_evaluation_records(),
        server.model_clustering_record_store.snapshot_pair_clustering_observations(),
        server.model_clustering_record_store.snapshot_model_clustering_observations(),
    )


@pytest.mark.parametrize(
    "case",
    (
        "bool_time",
        "negative_time",
        "bad_settings",
        "bad_weight",
        "bad_count",
        "bad_buffer",
        "bad_previous_loss",
        "bad_threshold",
        "bad_allocator",
        "bad_delay",
        "bad_pending_shape",
        "bad_training_label",
        "bad_second_id",
    ),
)
def test_finalization_rejects_before_any_client_flush(monkeypatch, case):
    old, new, clients = server_pair(monkeypatch)
    observe_pair(old, new, clients, count=2, force_drift=True)
    second = clients[1]
    timestep = 0
    if case == "bool_time":
        timestep = True
    elif case == "negative_time":
        timestep = -1
    elif case == "bad_settings":
        object.__setattr__(new._settings, "local_training_round_count", False)
    elif case == "bad_weight":
        new._model_weights[0] = True
    elif case == "bad_count":
        second.stream_processor._processed_count = True
    elif case == "bad_buffer":
        second.stream_processor._buffer[-1].input_features.fill_(float("nan"))
    elif case == "bad_previous_loss":
        second.detection_batch_processor._previous_loss = float("nan")
    elif case == "bad_threshold":
        second.detection_batch_processor._threshold = float("nan")
    elif case == "bad_allocator":
        second.owners.temporary_model_id_allocator._next_temporary_model_id = True
    elif case == "bad_delay":
        second.owners.pending_model_upload_state._remaining_upload_delay_round_count = True
    elif case == "bad_second_id":
        # 次ID1は空いていても、その次2が既存clientモデルと衝突する。
        state = second.owners.held_model_training_state_registry.get_held_model_training_state(
            model_id=0
        )
        classifier, optimizer_state = deepcopy(
            (state.classifier, state.concept_specific_parameter_optimizer_state)
        )
        second.owners.held_model_training_state_registry.register_held_model_training_state(
            model_id=2,
            classifier=classifier,
            concept_specific_parameter_optimizer_state=optimizer_state,
        )
    else:
        # bufferを確定してreadyを作り、その後にpayload/sample契約を壊す。
        for client in clients:
            client.flush_detection_buffer()
        pending = second.owners.pending_model_upload_state.get_pending_model_upload()
        if case == "bad_pending_shape":
            name = next(iter(pending.parameter_snapshot))
            pending.parameter_snapshot[name] = torch.zeros(1)
        else:
            collection = (
                second.owners.training_sample_store.snapshot_ordered_model_training_samples()[-1]
            )
            collection.training_samples[0].observed_class_labels.fill_(3.0)
    before = deepcopy(terminal_snapshot(new))
    global_before = rng_snapshot()
    borrowed_before = second.python_random_generator.getstate()
    with pytest.raises((TypeError, ValueError)):
        new.finalize_observed_stream(timestep_index=timestep)
    assert_terminal_values_equal(before, terminal_snapshot(new))
    assert_rng_equal(global_before)
    assert second.python_random_generator.getstate() == borrowed_before


@pytest.mark.parametrize("dataset", ("sine2", "mnist2"))
@pytest.mark.parametrize("optimizer", ("sgd", "adam"))
def test_factory_observation_sync_finalization_flow_matches_real_legacy(
    monkeypatch, dataset, optimizer
):
    old, participants, _, inputs = prepare_pair(monkeypatch, dataset, optimizer)
    new, clients = participants.run_server, participants.run_clients
    generator = inputs["sample_generator"]
    for t in range(2):
        blocks = [
            tuple(generator.generate_sample(concept_id=t % 2) for _ in range(3)) for _ in clients
        ]

        def old_step(t=t, blocks=blocks):
            data = [
                [
                    (torch.tensor(s.feature_values), torch.tensor([float(s.class_label)]))
                    for s in block
                ]
                for block in blocks
            ]
            experiment._run_feddrift_timestep(
                old.clients, old, data, [[t % 2] * 3 for _ in blocks], t, True, False
            )

        def new_step(t=t, blocks=blocks):
            for client, block in zip(clients, blocks, strict=True):
                client.process_observed_block(
                    observed_samples=block, evaluation_concept_ids=(t % 2,) * 3
                )
            new.synchronize_timestep(timestep_index=t)

        run_pair(old_step, new_step, clients)
        assert_server_equal(old, new)
    # 通常同期後、観測済みの部分batchから最終モデル登録へ接続する。
    for old_client, client in zip(old.clients, clients, strict=True):
        old_client.last_min_loss = 0.0
        old_client.distance_threshold = 0.0
        client.detection_batch_processor._previous_loss = 0.0
        client.detection_batch_processor._threshold = 0.0
        block = tuple(generator.generate_sample(concept_id=1) for _ in range(2))
        data = [
            (torch.tensor((s.feature_values,)), torch.tensor([[float(s.class_label)]]))
            for s in block
        ]
        run_pair(
            lambda old_client=old_client, data=data: old_client.process_batch(data, (1, 1)),
            lambda client=client, block=block: client.process_observed_block(
                observed_samples=block, evaluation_concept_ids=(1, 1)
            ),
            clients,
        )
    with torch.no_grad():
        run_pair(
            lambda: finish_legacy(old, 1),
            lambda: new.finalize_observed_stream(timestep_index=1),
            clients,
        )
    assert_server_equal(old, new)
    assert all(c.stream_processor.processed_sample_count == 8 for c in clients)
    assert not any(c.has_model_ready_for_server_registration() for c in clients)
