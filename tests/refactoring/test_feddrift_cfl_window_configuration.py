"""公式CFLの判定・ラベルと、実際の学習・配布の経路を照合する。"""

from dataclasses import replace

import numpy as np
import pytest
import torch
from sklearn.cluster import AgglomerativeClustering

from federated_learning_experiments.methods.feddrift.cfl_cluster_splitting import (
    CflClusterSplittingState,
    bipartition_complete_linkage,
)
from federated_learning_experiments.runtime.feddrift_experiment_run import (
    execute_feddrift_experiment_run,
)
from federated_learning_experiments.runtime.feddrift_run_client import FeddriftRunClient
from federated_learning_experiments.runtime.feddrift_run_record_tables import (
    build_feddrift_run_record_tables,
)
from federated_learning_experiments.runtime.feddrift_run_server import FeddriftRunServer
from tests.refactoring.test_feddrift_oracle_concept_assignment_configuration import (
    SMALL_FULL_RUN_CLIENT_COUNT,
    SMALL_FULL_RUN_SAMPLE_COUNT,
    SMALL_FULL_RUN_TIMESTEP_COUNT,
    settings,
    small_full_run_settings,
)

CONFIGURATION = "feddrift-cfl-window"


def reference_check(state, updates, gamma, limit, *, models=None, initial_parameters=None):
    """写しのcluster_cfl/cfl_util_*。公式の再初期化だけを両モデルのθ0へ変える。"""
    norm, eps1, eps2, next_id = state
    records = []
    for mid, reports in sorted(updates.items()):
        clients = tuple(reports)
        weight_updates = list(reports.values())
        if not weight_updates:
            continue
        max_norm = np.max([np.linalg.norm(dW) for dW in weight_updates])
        mean_norm = np.linalg.norm(np.mean(np.stack(weight_updates), axis=0))
        before = (norm, eps1, eps2)
        checked, split, alpha, new_id, cl1, cl2 = False, False, float("nan"), -1, (), ()
        if mean_norm > norm:
            norm = mean_norm
            eps1 = norm / 10.0
            eps2 = 6 * eps1
        elif mean_norm < eps1 and max_norm > eps2:
            checked = True
            S = np.zeros((len(weight_updates), len(weight_updates)))
            for i, s1 in enumerate(weight_updates):
                for j, s2 in enumerate(weight_updates):
                    S[i, j] = np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2) + 1e-12)
            labels = (
                AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="complete")
                .fit(-S)
                .labels_
            )
            one, two = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
            cl1, cl2 = tuple(clients[i] for i in one), tuple(clients[i] for i in two)
            alpha = max(max(S[i, j] for j in two) for i in one)
            if ((1 - alpha) / 2.0) ** 0.5 > gamma and next_id < limit:
                split, new_id = True, next_id
                next_id += 1
                if models is not None:
                    assert initial_parameters is not None
                    for reset in (mid, new_id):
                        models[reset] = {k: v.clone() for k, v in initial_parameters.items()}
        records.append(
            (
                mid,
                clients,
                float(mean_norm),
                float(max_norm),
                before,
                (float(norm), float(eps1), float(eps2)),
                checked,
                float(alpha),
                split,
                new_id,
                cl1,
                cl2,
            )
        )
    return (float(norm), float(eps1), float(eps2), next_id), records


@pytest.mark.parametrize("seed", range(30))
def test_bipartition_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    S = rng.uniform(-1, 1, (10, 10))
    S = (S + S.T) / 2
    expected = (
        AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="complete")
        .fit(-S)
        .labels_
    )
    np.testing.assert_array_equal(bipartition_complete_linkage(S), expected)


@pytest.mark.parametrize(
    "S",
    [
        np.ones((10, 10)),
        np.eye(10),
        np.array([[1, 1, -1, -1], [1, 1, -1, -1], [-1, -1, 1, 1], [-1, -1, 1, 1]]),
    ],
)
def test_bipartition_ties_match_sklearn(S):
    expected = (
        AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="complete")
        .fit(-S)
        .labels_
    )
    np.testing.assert_array_equal(bipartition_complete_linkage(S), expected)


def test_official_checks_state_order_cap_and_random_updates():
    rng = np.random.default_rng(29)
    sequences = [
        {0: {i: np.array([10.0, 0.0]) for i in range(4)}},
        {0: {i: np.array([20.0, 0.0]) for i in range(4)}},
        {0: {i: np.array([(-1.0) ** i * 20, 0]) for i in range(4)}},
        {
            1: {2: np.array([0.0, 20]), 3: np.array([0.0, -20])},
            0: {0: np.array([60.0, 0]), 1: np.array([60.0, 0])},
        },
        {0: {}, 1: {2: np.array([1.0, 0.0])}},
    ] + [{0: {i: rng.normal(size=20).astype(np.float32) for i in range(10)}} for _ in range(20)]
    for limit in (1, 2, 4):
        actual = CflClusterSplittingState()
        state = (0.0, 0.0, 10000.0, 1)
        for updates in sequences:
            state, expected = reference_check(state, updates, 0.1, limit)
            records = actual.check(updates_by_model=updates, gamma=0.1, maximum_model_count=limit)
            assert (actual.cfl_norm, actual.eps1, actual.eps2, actual.next_model_id) == state
            for a, b in zip(records, expected, strict=True):
                for x, y in zip(a, b, strict=True):
                    if isinstance(y, float) and np.isnan(y):
                        assert np.isnan(x)
                    else:
                        assert x == y


def test_configuration_defaults_and_unchanged_identifier():
    from federated_learning_experiments.core.settings_serialization import (
        convert_settings_to_plain_mapping,
    )
    from federated_learning_experiments.sweep.experiment_sweep_plan import make_run_identifier

    actual = settings(configuration=CONFIGURATION)
    client = actual.run_participant_settings.run_client_settings
    assert client.model_assignment_policy == "cfl_bipartition_assignment"
    assert client.training_sample_window == "current_timestep"
    assert client.cfl_split_gamma == 0.1
    default = settings(configuration="feddrift")
    assert (
        "cfl_split_gamma"
        not in convert_settings_to_plain_mapping(default)["run_participant_settings"][
            "run_client_settings"
        ]
    )
    assert (
        make_run_identifier(configuration_name="feddrift", run_settings=default)
        == "sine2_interval50_seed0_1fbf192c89f9"
    )
    with pytest.raises(ValueError):
        replace(client, cfl_split_gamma=float("nan"))


def test_high_cross_similarity_prevents_split_and_cluster_order_changes_decision():
    angles = np.arange(64) * (2 * np.pi / 64)
    updates = {i: np.array([np.cos(a), np.sin(a)]) for i, a in enumerate(angles)}
    state = CflClusterSplittingState(cfl_norm=1.0, eps1=0.1, eps2=0.6)
    result = state.check(updates_by_model={0: updates}, gamma=0.1, maximum_model_count=4)[0]
    assert result[6] and result[7] > 0.98 and not result[8]
    _, expected = reference_check((1.0, 0.1, 0.6, 1), {0: updates}, 0.1, 4)
    np.testing.assert_equal([result], expected)
    low = {0: np.array([20.0, 0]), 1: np.array([-20.0, 0])}
    high = {2: np.array([60.0, 0]), 3: np.array([60.0, 0])}
    ascending = CflClusterSplittingState(cfl_norm=20.0, eps1=2.0, eps2=12.0)
    rows = ascending.check(updates_by_model={1: low, 0: high}, gamma=0.1, maximum_model_count=4)
    assert [r[0] for r in rows] == [0, 1]
    assert rows[1][4] == (60.0, 6.0, 36.0) and not rows[1][8]
    reverse = CflClusterSplittingState(cfl_norm=20.0, eps1=2.0, eps2=12.0)
    assert reverse.check(updates_by_model={1: low}, gamma=0.1, maximum_model_count=4)[0][8]


def test_threshold_updates_keep_official_numpy_float32_arithmetic():
    update = np.array([1.1, 2.2], dtype=np.float32)
    state = CflClusterSplittingState()
    rows = state.check(updates_by_model={0: {0: update}}, gamma=0.1, maximum_model_count=2)
    norm = np.linalg.norm(update)
    assert state.eps1 == float(norm / 10.0)
    assert state.eps2 == float(6 * (norm / 10.0))
    assert rows[0][5] == (float(norm), float(norm / 10.0), float(6 * (norm / 10.0)))


def test_server_split_skips_every_model_aggregation(monkeypatch):
    from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
        snapshot_classifier_parameters,
    )
    from federated_learning_experiments.runtime.feddrift_detection_batch_processing import (
        FeddriftDetectionBatchProcessor,
    )

    original = FeddriftRunServer.aggregate_and_distribute_training_round
    observed = []
    expected_assignment = {}
    checked_empty = []
    current_state = FeddriftDetectionBatchProcessor.current_training_model_state.fget

    def inspect_current(processor):
        if expected_assignment:
            cid = next(
                (
                    c.client_id
                    for c in participants[0].run_clients
                    if c.detection_batch_processor is processor
                ),
                None,
            )
            client = participants[0].run_clients[cid]
            if client.stream_processor.processed_sample_count == 100:
                assert all(
                    not c.training_samples
                    for c in processor._training_samples.snapshot_ordered_model_training_samples()
                )
                assert processor._assignment.current_training_model_id == expected_assignment[cid]
                checked_empty.append(cid)
        return current_state(processor)

    monkeypatch.setattr(
        FeddriftDetectionBatchProcessor, "current_training_model_state", property(inspect_current)
    )
    participants = []

    def force(server, *, timestep_index):
        if timestep_index == 1:
            server._cfl_state.cfl_norm = 100.0
            server._cfl_state.eps1 = 10.0
            server._cfl_state.eps2 = 60.0
            before = server.global_model_repository.get_global_model_parameters(model_id=1)
            reference_models = {
                mid: server.global_model_repository.get_global_model_parameters(model_id=mid)
                for mid in server.global_model_repository.global_model_ids
            }
            updates = {0: {}, 1: {}}
            for i, client in enumerate(server.run_clients):
                mid = client.owners.current_training_model_assignment.current_training_model_id
                classifier = (
                    client.owners.held_model_training_state_registry.get_held_model_training_state(
                        model_id=mid
                    ).classifier
                )
                base = server.global_model_repository.get_global_model_parameters(model_id=mid)
                classifier.load_state_dict(
                    {
                        k: v + ((100 if i % 2 else -100) if mid == 0 else 0.01)
                        for k, v in base.items()
                    }
                )
                local = snapshot_classifier_parameters(classifier=classifier)
                updates[mid][i] = np.concatenate(
                    [(local[k] - base[k]).numpy().flatten() for k in base]
                )
            _, reference = reference_check(
                (100.0, 10.0, 60.0, 2),
                updates,
                0.1,
                3,
                models=reference_models,
                initial_parameters=server._cfl_initial_parameters,
            )
        original(server, timestep_index=timestep_index)
        if timestep_index == 0:
            # 2つ目の既存クラスタも用意し、別クラスタの集約まで省かれることを確かめる。
            server._maximum_cfl_model_count = 3
            server._cfl_state.next_model_id = 2
            assert server.global_model_repository.allocate_global_model_id() == 1
            server.global_model_repository.set_global_model_parameters(
                model_id=1,
                parameter_snapshot={k: v + 7 for k, v in server._cfl_initial_parameters.items()},
            )
            server.global_model_repository.set_global_model_loss_statistics(
                model_id=1, loss_statistics=server._cfl_initial_statistics
            )
            server.global_model_repository.record_model_registration(
                model_id=1, registered_round_index=0, registering_client_id=None
            )
            server._model_weights[1] = 1
            server._aggregate_and_distribute_models(active_ids=(), update_model_weights=False)
            for client in server.run_clients[2:]:
                client.owners.current_training_model_assignment.assign_model_for_training(
                    model_id=1
                )
        if timestep_index == 1:
            actual = [row for t, row in server._cfl_records if t == 1]
            np.testing.assert_equal(actual, reference)
            assert actual[0][8] and not actual[1][8]
            assert server.global_model_repository.global_model_ids == (0, 1, 2)
            for mid in (0, 1, 2):
                target = before if mid == 1 else server._cfl_initial_parameters
                for k, v in target.items():
                    assert torch.equal(v, reference_models[mid][k])
                for k, v in server.global_model_repository.get_global_model_parameters(
                    model_id=mid
                ).items():
                    assert torch.equal(v, target[k])
                for client in server.run_clients:
                    classifier = client.owners.held_model_training_state_registry.get_held_model_training_state(
                        model_id=mid
                    ).classifier
                    for k, v in classifier.state_dict().items():
                        assert torch.equal(v, target[k])
            for client in server.run_clients:
                expected_assignment[client.client_id] = (
                    2
                    if client.client_id in actual[0][11]
                    else 0
                    if client.client_id in actual[0][10]
                    else 1
                )
                assert (
                    client.owners.current_training_model_assignment.current_training_model_id
                    == expected_assignment[client.client_id]
                )
            participants.append(server)
            observed.append(True)

    monkeypatch.setattr(FeddriftRunServer, "aggregate_and_distribute_training_round", force)
    base = settings(configuration=CONFIGURATION)
    run = execute_feddrift_experiment_run(
        run_settings=replace(
            base,
            execution_settings=replace(
                base.execution_settings,
                experiment_run_conditions=replace(
                    base.execution_settings.experiment_run_conditions,
                    client_count=4,
                    per_client_sample_count=150,
                ),
            ),
        )
    )
    tables = build_feddrift_run_record_tables(experiment_run=run)
    assert observed == [True]
    assert set(checked_empty) == set(range(4))
    for cid, mid in expected_assignment.items():
        np.testing.assert_array_equal(
            tables["sample_prediction.prediction_model_id"][cid, 100:], mid
        )
    assert tables["cfl_split_check.did_split"].sum() == 1


@pytest.mark.slow
@pytest.mark.parametrize("dataset,limit", [("sine2", 2), ("mnist4", 4)])
def test_small_full_run_assignment_window_prediction_and_checks(monkeypatch, dataset, limit):
    windows = []
    original = FeddriftRunClient.train_local_round

    def inspect(client, *, eligible_model_ids=None):
        end = client.stream_processor.processed_sample_count
        selected = client.owners.current_training_model_assignment.current_training_model_id
        for (
            collection
        ) in client.owners.training_sample_store.snapshot_ordered_model_training_samples():
            assert len(collection.training_samples) == (
                50 if collection.model_id == selected else 0
            )
            for offset, sample in enumerate(collection.training_samples):
                windows.append((client.client_id, end - 50 + offset, sample))
        return original(client, eligible_model_ids=eligible_model_ids)

    monkeypatch.setattr(FeddriftRunClient, "train_local_round", inspect)
    client_count, timesteps = SMALL_FULL_RUN_CLIENT_COUNT, SMALL_FULL_RUN_TIMESTEP_COUNT
    run = execute_feddrift_experiment_run(
        run_settings=small_full_run_settings(dataset, CONFIGURATION)
    )
    server = run.measured_run.participants.run_server
    tables = build_feddrift_run_record_tables(experiment_run=run)
    assigned = tables["cfl_assignment.model_id"].reshape(timesteps, client_count).T
    np.testing.assert_array_equal(assigned[:, 0], 0)
    np.testing.assert_array_equal(
        tables["sample_prediction.prediction_model_id"], np.repeat(assigned, 50, axis=1)
    )
    expected = dict.fromkeys(range(client_count), 0)
    next_model_id = 1
    for t in range(timesteps):
        assert assigned[:, t].tolist() == list(expected.values())
        for timestep, row in server._cfl_records:
            if timestep != t:
                continue
            mid, clients, mean, maximum, before, after, checked, alpha, split, new, one, two = row
            expected_split = checked and ((1 - alpha) / 2) ** 0.5 > 0.1 and next_model_id < limit
            assert split == expected_split
            assert new == (next_model_id if expected_split else -1)
            assert checked == (mean <= before[0] and mean < before[1] and maximum > before[2])
            eps1 = np.float32(mean) / 10.0  # 実際のモデルのstate_dictはfloat32。
            assert after == ((mean, float(eps1), float(6 * eps1)) if mean > before[0] else before)
            if split:
                next_model_id += 1
                expected.update(dict.fromkeys(one, mid))
                expected.update(dict.fromkeys(two, new))
    streams = run.measured_run.run_result.observed_client_streams
    assert len(windows) == client_count * SMALL_FULL_RUN_SAMPLE_COUNT
    for cid, index, sample in windows:
        expected_sample = streams[cid].observed_samples[index]
        torch.testing.assert_close(
            sample.input_features,
            torch.tensor((expected_sample.feature_values,), dtype=torch.float32),
            rtol=0,
            atol=0,
        )
        assert sample.observed_class_labels.item() == expected_sample.class_label
    assert len(server.global_model_repository.global_model_ids) <= limit
    assert len(tables["model_clustering.model_id"]) == 0
    assert len(tables["detection_batch.client_id"]) == 0
    changes = np.count_nonzero(np.diff(tables["sample_prediction.observed_concept_id"], axis=1))
    splits = tables["cfl_split_check.timestep_index"][tables["cfl_split_check.did_split"]]
    print(
        f"{dataset}: changes={changes}, splits={splits.tolist()}, models={len(server.global_model_repository.global_model_ids)}, windows={len(windows)}"
    )
