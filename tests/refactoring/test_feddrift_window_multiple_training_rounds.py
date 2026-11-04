"""窓の反復学習と、CFLの時刻途中の分割を実体のサーバ経路で守る。"""

from dataclasses import replace
from hashlib import sha256
from json import dumps

import numpy as np
import pytest
import torch

from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.runtime.feddrift_experiment_run import (
    execute_feddrift_experiment_run,
)
from federated_learning_experiments.runtime.feddrift_run_client import FeddriftRunClient
from federated_learning_experiments.runtime.feddrift_run_record_tables import (
    build_feddrift_run_record_tables,
)
from federated_learning_experiments.runtime.feddrift_run_server import FeddriftRunServer
from federated_learning_experiments.sweep import experiment_sweep_command as command
from federated_learning_experiments.sweep.experiment_configurations import (
    EXPERIMENT_CONFIGURATION_NAMES,
)
from federated_learning_experiments.sweep.experiment_sweep_plan import (
    ExperimentSweepPlan,
    build_planned_experiment_runs,
    make_run_identifier,
)
from tests.refactoring.test_feddrift_oracle_concept_assignment_configuration import (
    SMALL_FULL_RUN_CLIENT_COUNT,
    SMALL_FULL_RUN_SAMPLE_COUNT,
    SMALL_FULL_RUN_TIMESTEP_COUNT,
    settings,
    small_full_run_settings,
)

WINDOWS = ("feddrift-window", "feddrift-cfl-window")


def with_rounds(base, rounds):
    participant = base.run_participant_settings
    return replace(
        base,
        run_participant_settings=replace(
            participant,
            server_settings=replace(participant.server_settings, local_training_round_count=rounds),
        ),
    )


def plan(names=WINDOWS, rounds=None):
    return ExperimentSweepPlan(
        configuration_names=names,
        dataset_names=("sine2",),
        random_seeds=(0,),
        client_count=10,
        per_client_sample_count=2500,
        server_aggregation_intervals_per_client_samples=(50,),
        feddrift_local_training_round_counts=rounds,
    )


def test_all_default_settings_and_identifiers_match_before_change():
    values = [
        (
            n,
            convert_settings_to_plain_mapping(settings(configuration=n)),
            make_run_identifier(configuration_name=n, run_settings=settings(configuration=n)),
        )
        for n in EXPERIMENT_CONFIGURATION_NAMES[:27]
    ]
    # d734a8b2の27構成。保存設定の全項目と識別子を実装前に採取した。
    assert len(values) == 27
    assert sha256(dumps(values, sort_keys=True).encode()).hexdigest() == (
        "889de5ea90e55bbac4870cc56290fe10323f9a88952cf4ecf900315cde094352"
    )


def test_sweep_cli_defaults_and_distinct_identifiers():
    args = command._build_argument_parser().parse_args(
        [
            "--configurations",
            *WINDOWS,
            "--datasets",
            "sine2",
            "--seeds",
            "0",
            "--aggregation-intervals",
            "50",
            "--feddrift-local-training-round-counts",
            "1",
            "3",
        ]
    )
    parsed = command._build_sweep_plan(args)
    assert parsed.feddrift_local_training_round_counts == (1, 3)
    runs = build_planned_experiment_runs(sweep_plan=parsed)
    assert len(runs) == 4
    assert [
        r.run_settings.run_participant_settings.server_settings.local_training_round_count
        for r in runs
    ] == [1, 3, 1, 3]
    assert len({r.run_identifier for r in runs}) == 4
    assert build_planned_experiment_runs(sweep_plan=plan()) == (
        build_planned_experiment_runs(sweep_plan=plan(rounds=(1,)))
    )
    for run in build_planned_experiment_runs(sweep_plan=plan(rounds=(1,))):
        assert run.run_settings == settings(configuration=run.configuration_name)


@pytest.mark.parametrize(
    "name",
    [
        n
        for n in EXPERIMENT_CONFIGURATION_NAMES
        if n not in WINDOWS and not n.startswith("feddrift-fedem-")
    ],
)
def test_sweep_rejects_nonwindow_rounds(name):
    with pytest.raises(ValueError, match="feddrift_local_training_round_counts"):
        plan((name,), (3,))
    with pytest.raises(ValueError, match="feddrift_local_training_round_counts"):
        plan((WINDOWS[0], name), (1, 3))
    assert build_planned_experiment_runs(sweep_plan=plan((name,), (1,)))


@pytest.mark.parametrize("values", [(), (0,), (-1,), (1, 1), (True,), (1.0,), [1, 3]])
def test_round_sweep_requires_distinct_positive_builtin_integers(values):
    with pytest.raises((TypeError, ValueError)):
        plan(rounds=values)


def test_assembly_keeps_ifca_single_round_and_allows_both_windows():
    for name in WINDOWS:
        assert with_rounds(settings(configuration=name), 3)
    with pytest.raises(ValueError, match="IFCA\\+Window.*1時刻1round"):
        with_rounds(settings(configuration="feddrift-ifca-window"), 2)


@pytest.mark.parametrize("name", ("feddrift-adaptive-fedavg-window", "feddrift-driftsurf"))
def test_assembly_rejects_rounds_for_other_window_methods(name):
    with pytest.raises(ValueError, match="1時刻1round"):
        with_rounds(settings(configuration=name), 3)


def test_round_axis_declarations_and_artifact_version():
    from federated_drift_experiment.experiment_spec.options import CHOICE_CONSTRAINTS, option
    from federated_drift_experiment.experiment_spec.parameters import parameter
    from federated_drift_experiment.experiment_spec.sweep import (
        FEDDRIFT_LOCAL_TRAINING_ROUND_COUNT_AXIS,
    )
    from federated_learning_experiments.runtime.feddrift_run_artifact_files import (
        RUN_ARTIFACT_FORMAT_VERSION,
    )
    from federated_learning_experiments.sweep.experiment_sweep_records import (
        SWEEP_RECORD_FORMAT_VERSION,
    )

    name = "feddrift_local_training_round_count"
    assert (
        option(name).cli_name == parameter(name).cli_name == "feddrift-local-training-round-counts"
    )
    assert parameter(name).code_name == "server_settings.local_training_round_count"
    assert option(name).active_when
    assert any(c.option_id == name and c.requires_selections for c in CHOICE_CONSTRAINTS)
    assert FEDDRIFT_LOCAL_TRAINING_ROUND_COUNT_AXIS.parameter_id == name
    assert FEDDRIFT_LOCAL_TRAINING_ROUND_COUNT_AXIS.values == (1,)
    assert RUN_ARTIFACT_FORMAT_VERSION == SWEEP_RECORD_FORMAT_VERSION == 9


def test_cfl_reassignment_destination_is_checked_before_state_updates(monkeypatch):
    from copy import deepcopy

    from federated_learning_experiments.runtime.cfl_training_round import split_cfl_training_round

    checked = []

    def inspect(server, *, timestep_index):
        client = server.run_clients[0]
        original_samples = sample_collections(client)
        client.owners.training_sample_store.append_model_training_samples(
            model_id=1, training_samples=()
        )
        state = deepcopy(server._cfl_state)
        records = list(server._cfl_records)
        next_id = server.global_model_repository.next_global_model_id
        with pytest.raises(ValueError, match="新しいモデルID"):
            split_cfl_training_round(
                run_server=server, timestep_index=timestep_index, active_ids=(0,)
            )
        assert server._cfl_state == state
        assert server._cfl_records == records
        assert not server._cfl_record_round_indices
        assert server.global_model_repository.next_global_model_id == next_id
        client.owners.training_sample_store.clear_model_training_samples()
        for mid, samples in original_samples.items():
            client.owners.training_sample_store.append_model_training_samples(
                model_id=mid, training_samples=samples
            )
        checked.append(True)

    monkeypatch.setattr(FeddriftRunServer, "aggregate_and_distribute_training_round", inspect)
    base = with_rounds(settings(configuration=WINDOWS[1]), 3)
    base = replace(
        base,
        execution_settings=replace(
            base.execution_settings,
            experiment_run_conditions=replace(
                base.execution_settings.experiment_run_conditions,
                client_count=2,
                per_client_sample_count=50,
            ),
        ),
    )
    execute_feddrift_experiment_run(run_settings=base)
    assert checked == [True] * 3


def sample_collections(client):
    return {
        c.model_id: c.training_samples
        for c in client.owners.training_sample_store.snapshot_ordered_model_training_samples()
    }


def counts(client):
    return client.owners.model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts()


@pytest.mark.slow
def test_fedavg_window_three_rounds_reuse_window_and_triple_parameter_communication(monkeypatch):
    original = FeddriftRunClient.train_local_round
    seen = {}
    calls = []

    def inspect(client, *, eligible_model_ids=None):
        end = client.stream_processor.processed_sample_count
        samples = sample_collections(client)[0]
        assert len(samples) == 50 and eligible_model_ids == frozenset({0})
        key = (client.client_id, end)
        if key in seen:
            assert all(a is b for a, b in zip(seen[key], samples, strict=True))
        seen[key] = samples
        calls.append(key)
        return original(client, eligible_model_ids=eligible_model_ids)

    monkeypatch.setattr(FeddriftRunClient, "train_local_round", inspect)
    base = small_full_run_settings(configuration=WINDOWS[0])
    once = execute_feddrift_experiment_run(run_settings=base)
    calls.clear()
    seen.clear()
    three = execute_feddrift_experiment_run(run_settings=with_rounds(base, 3))
    assert len(calls) == SMALL_FULL_RUN_TIMESTEP_COUNT * SMALL_FULL_RUN_CLIENT_COUNT * 3
    assert all(calls.count(key) == 3 for key in seen)
    streams = three.measured_run.run_result.observed_client_streams
    for (cid, end), samples in seen.items():
        for offset, sample in enumerate(samples):
            expected = streams[cid].observed_samples[end - 50 + offset]
            torch.testing.assert_close(
                sample.input_features,
                torch.tensor((expected.feature_values,), dtype=torch.float32),
                rtol=0,
                atol=0,
            )
            assert sample.observed_class_labels.item() == expected.class_label
    one_server = once.measured_run.participants.run_server
    server = three.measured_run.participants.run_server
    assert server.global_model_repository.global_model_ids == (0,)
    one = one_server.communication_volume_record_store.get_state_snapshot()
    actual = server.communication_volume_record_store.get_state_snapshot()
    for field in (
        "uploaded_model_count",
        "downloaded_model_count",
        "uploaded_byte_count",
        "downloaded_byte_count",
        "uploaded_parameter_value_count",
        "downloaded_parameter_value_count",
    ):
        assert getattr(actual, field) == 3 * getattr(one, field)
    assert actual.downloaded_message_count == one.downloaded_message_count
    print("FedAvg+Window R=3: upload/download parameter communication = 3.0x")


def test_cfl_split_first_round_reassigns_samples_then_aggregates_new_model(monkeypatch):
    train = FeddriftRunClient.train_local_round
    aggregate = FeddriftRunServer.aggregate_and_distribute_training_round
    round_calls = {}
    local_parameters = {}
    previous_samples = {}
    previous_counts = {}
    split_clients = set()
    messages = []
    round_index = 0

    def inspect(client, *, eligible_model_ids=None):
        key = (client.client_id, client.stream_processor.processed_sample_count)
        r = round_calls.get(key, 0)
        round_calls[key] = r + 1
        collections = sample_collections(client)
        selected = client.owners.current_training_model_assignment.current_training_model_id
        # 同じ時刻内の不均等な件数で重みつき平均を検査する。
        if key[1] == 50 and r == 0:
            store = client.owners.training_sample_store
            samples = collections[selected][: 50 - client.client_id]
            store.clear_model_training_samples()
            store.append_model_training_samples(model_id=selected, training_samples=samples)
            collections = sample_collections(client)
        assert eligible_model_ids == frozenset({0} if key[1] == 50 and r == 0 else {0, 1})
        assert len(collections[selected]) == (50 - client.client_id if key[1] == 50 else 50)
        assert all(not samples for mid, samples in collections.items() if mid != selected)
        if key[1] == 50:
            if r == 0:
                previous_samples[client.client_id] = collections[0]
            else:
                assert all(
                    a is b
                    for a, b in zip(
                        previous_samples[client.client_id], collections[selected], strict=True
                    )
                )
        else:
            assert not any(
                sample is previous
                for sample in collections[selected]
                for previous in previous_samples[client.client_id]
            )
        before = counts(client)
        result = train(client, eligible_model_ids=eligible_model_ids)
        after = counts(client)
        assert (
            after.assigned_sample_counts_by_model_and_concept_id
            == before.assigned_sample_counts_by_model_and_concept_id
        )
        for field in (
            "trained_sample_counts_by_model_id",
            "parameter_update_step_counts_by_model_id",
        ):
            for mid, value in getattr(before, field).items():
                if mid != selected:
                    assert getattr(after, field)[mid] == value
        assert after.trained_sample_counts_by_model_id.get(
            selected, 0
        ) > before.trained_sample_counts_by_model_id.get(selected, 0)
        assert after.parameter_update_step_counts_by_model_id.get(
            selected, 0
        ) > before.parameter_update_step_counts_by_model_id.get(selected, 0)
        previous_counts[client.client_id] = after
        classifier = client.owners.held_model_training_state_registry.get_held_model_training_state(
            model_id=selected
        ).classifier
        local_parameters[client.client_id] = {
            k: v.clone() for k, v in classifier.state_dict().items()
        }
        return result

    def inspect_aggregate(server, *, timestep_index):
        nonlocal round_index
        r = round_index % 3
        round_index += 1
        if timestep_index == 0 and r == 0:
            server._cfl_state.cfl_norm = 100.0
            server._cfl_state.eps1 = 10.0
            server._cfl_state.eps2 = 60.0
            for client in server.run_clients:
                classifier = (
                    client.owners.held_model_training_state_registry.get_held_model_training_state(
                        model_id=0
                    ).classifier
                )
                classifier.load_state_dict(
                    {
                        k: v + (100 if client.client_id % 2 else -100)
                        for k, v in server._cfl_initial_parameters.items()
                    }
                )
        before_mac = server.snapshot_server_computation_counts().aggregation_parameter_multiply_accumulate_count
        aggregate(server, timestep_index=timestep_index)
        messages.append(
            server.communication_volume_record_store.get_state_snapshot().downloaded_message_count
        )
        if timestep_index == 0 and r == 0:
            assert (
                server.snapshot_server_computation_counts().aggregation_parameter_multiply_accumulate_count
                == before_mac
            )
            for client in server.run_clients:
                selected = client.owners.current_training_model_assignment.current_training_model_id
                if selected == 1:
                    split_clients.add(client.client_id)
                assert counts(client) == previous_counts[client.client_id]
                for value, initial in zip(
                    server.global_model_repository.get_global_model_parameters(
                        model_id=selected
                    ).values(),
                    server._cfl_initial_parameters.values(),
                    strict=True,
                ):
                    assert torch.equal(value, initial)
                for mid in (0, 1):
                    classifier = client.owners.held_model_training_state_registry.get_held_model_training_state(
                        model_id=mid
                    ).classifier
                    for name, value in classifier.state_dict().items():
                        assert torch.equal(value, server._cfl_initial_parameters[name])
        else:
            for mid in (0, 1):
                members = [
                    c
                    for c in server.run_clients
                    if c.owners.current_training_model_assignment.current_training_model_id == mid
                ]
                total = sum(len(sample_collections(c)[mid]) for c in members)
                expected = {
                    k: sum(
                        local_parameters[c.client_id][k] * (len(sample_collections(c)[mid]) / total)
                        for c in members
                    )
                    for k in local_parameters[members[0].client_id]
                }
                actual = server.global_model_repository.get_global_model_parameters(model_id=mid)
                for k in expected:
                    torch.testing.assert_close(actual[k], expected[k])

    monkeypatch.setattr(FeddriftRunClient, "train_local_round", inspect)
    monkeypatch.setattr(
        FeddriftRunServer, "aggregate_and_distribute_training_round", inspect_aggregate
    )
    base = with_rounds(settings(configuration=WINDOWS[1]), 3)
    base = replace(
        base,
        execution_settings=replace(
            base.execution_settings,
            experiment_run_conditions=replace(
                base.execution_settings.experiment_run_conditions,
                client_count=4,
                per_client_sample_count=100,
            ),
        ),
    )
    run = execute_feddrift_experiment_run(run_settings=base)
    tables = build_feddrift_run_record_tables(experiment_run=run)
    assert split_clients and len(split_clients) == 2
    assert tables["cfl_split_check.round_index"].tolist() == [0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
    assert tables["cfl_split_check.did_split"].sum() == 1
    # 最初の通知4通に加え、対象集合が変わるround 1の直前だけ4通増える。
    assert messages == [4, 8, 8, 12, 12, 12]
    for client in run.measured_run.participants.run_clients:
        selected = int(client.client_id in split_clients)
        np.testing.assert_array_equal(
            tables["sample_prediction.prediction_model_id"][client.client_id, 50:], selected
        )
        assert tables["cfl_assignment.model_id"].reshape(2, 4)[1, client.client_id] == selected


@pytest.mark.slow
@pytest.mark.parametrize("dataset,limit", [("sine2", 2), ("mnist4", 4)])
def test_cfl_three_round_small_full_run(monkeypatch, dataset, limit):
    # 既存の実体照合と同じ標本・判定規則を、各roundで照合する。
    train = FeddriftRunClient.train_local_round
    windows = []

    def inspect(client, *, eligible_model_ids=None):
        end = client.stream_processor.processed_sample_count
        selected = client.owners.current_training_model_assignment.current_training_model_id
        for mid, samples in sample_collections(client).items():
            assert len(samples) == (50 if mid == selected else 0)
            windows.extend((client.client_id, end - 50 + i, s) for i, s in enumerate(samples))
        return train(client, eligible_model_ids=eligible_model_ids)

    monkeypatch.setattr(FeddriftRunClient, "train_local_round", inspect)
    run = execute_feddrift_experiment_run(
        run_settings=with_rounds(small_full_run_settings(dataset, WINDOWS[1]), 3)
    )
    clients, timesteps = SMALL_FULL_RUN_CLIENT_COUNT, SMALL_FULL_RUN_TIMESTEP_COUNT
    tables = build_feddrift_run_record_tables(experiment_run=run)
    assigned = tables["cfl_assignment.model_id"].reshape(timesteps, clients)
    np.testing.assert_array_equal(
        tables["sample_prediction.prediction_model_id"], np.repeat(assigned.T, 50, axis=1)
    )
    expected = dict.fromkeys(range(clients), 0)
    next_id = 1
    for t in range(timesteps):
        assert assigned[t].tolist() == list(expected.values())
        indices = np.flatnonzero(tables["cfl_split_check.timestep_index"] == t)
        assert set(tables["cfl_split_check.round_index"][indices]) == {0, 1, 2}
        for i in indices:

            def get(name, i=i):
                return tables[f"cfl_split_check.{name}"][i]

            mean, maximum = get("mean_norm"), get("max_norm")
            checked = (
                mean <= get("cfl_norm_before")
                and mean < get("eps1_before")
                and maximum > get("eps2_before")
            )
            assert get("checked_bipartition") == checked
            split = checked and ((1 - get("alpha_cross")) / 2) ** 0.5 > 0.1 and next_id < limit
            assert get("did_split") == split
            eps1 = np.float32(mean) / 10.0
            assert get("eps1_after") == (
                float(eps1) if mean > get("cfl_norm_before") else get("eps1_before")
            )
            if split:
                assert get("new_model_id") == next_id
                expected.update({int(cid): next_id for cid in get("cl2_client_ids").split(",")})
                next_id += 1
    assert len(windows) == clients * SMALL_FULL_RUN_SAMPLE_COUNT * 3
    streams = run.measured_run.run_result.observed_client_streams
    for cid, index, sample in windows:
        observed = streams[cid].observed_samples[index]
        torch.testing.assert_close(
            sample.input_features,
            torch.tensor((observed.feature_values,), dtype=torch.float32),
            rtol=0,
            atol=0,
        )
        assert sample.observed_class_labels.item() == observed.class_label
    server = run.measured_run.participants.run_server
    assert len(server.global_model_repository.global_model_ids) <= limit
    assert len(tables["model_clustering.model_id"]) == len(tables["detection_batch.client_id"]) == 0
    split = tables["cfl_split_check.did_split"]
    print(
        f"CFL R=3 {dataset}: splits={int(split.sum())}, timesteps={tables['cfl_split_check.timestep_index'][split].tolist()}, rounds={tables['cfl_split_check.round_index'][split].tolist()}"
    )
