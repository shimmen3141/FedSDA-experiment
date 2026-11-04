"""FedEMの公式996faddの式と、ストリームへ接続した4構成を照合する。"""

from dataclasses import replace

import numpy as np
import pytest
import torch

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.runtime import feddrift_client_round_training as training
from federated_learning_experiments.runtime.feddrift_experiment_run import (
    execute_feddrift_experiment_run,
)
from federated_learning_experiments.runtime.feddrift_run_client import (
    FeddriftRunClient,
    assemble_feddrift_run_client,
)
from federated_learning_experiments.runtime.feddrift_run_record_tables import (
    build_feddrift_run_record_tables,
)
from federated_learning_experiments.runtime.feddrift_run_server import FeddriftRunServer
from federated_learning_experiments.runtime.fedem_mixture import (
    FedemClientMixture,
    _fedem_responsibilities,
    _fedem_weighted_loss,
)
from federated_learning_experiments.sweep.experiment_configurations import (
    build_experiment_run_settings,
)
from federated_learning_experiments.sweep.experiment_sweep_plan import build_planned_experiment_runs
from tests.refactoring.test_feddrift_oracle_concept_assignment_configuration import (
    observe,
    settings,
)
from tests.refactoring.test_feddrift_run_client import build_assembly_inputs
from tests.refactoring.test_feddrift_window_multiple_training_rounds import plan, with_rounds

CONFIGURATIONS = tuple(
    f"feddrift-fedem-{scope}{suffix}"
    for scope in ("window", "history")
    for suffix in ("", "-pretrained-component0")
)


@pytest.mark.parametrize("scope", ("window", "history"))
def test_tiny_three_round_prediction_timing_and_transfers(scope, monkeypatch):
    base = with_rounds(
        build_experiment_run_settings(
            configuration_name=f"feddrift-fedem-{scope}",
            experiment_run_conditions=ExperimentRunConditions(
                dataset_name="sine2",
                random_seed=0,
                client_count=2,
                per_client_sample_count=12,
                server_aggregation_interval_per_client_samples=3,
            ),
        ),
        3,
    )
    participant = base.run_participant_settings
    base = replace(
        base,
        run_participant_settings=replace(
            participant,
            run_client_settings=replace(
                participant.run_client_settings, local_training_batch_sample_count=3
            ),
        ),
    )
    predict = FedemClientMixture.predict
    counters = {}

    def inspect(fedem, features):
        index = counters.get(id(fedem), 0)
        assert all(i < index for i in fedem.sample_indices)
        t = index // 3
        pi = (
            torch.full((2,), 0.5)
            if t == 0
            else torch.tensor([r[2] for r in fedem.mixture_records if r[0] == t - 1])
        )
        torch.testing.assert_close(fedem.mixture_weights, pi, atol=0, rtol=0)
        outputs = []
        handles = [
            s.classifier.register_forward_hook(
                lambda module, inputs, output: outputs.append(output.detach().clone())
            )
            for s in fedem._registry.snapshot_ordered_held_model_training_states()
        ]
        try:
            actual = predict(fedem, features)
        finally:
            for handle in handles:
                handle.remove()
        expected = torch.einsum("m,mbc->bc", pi, torch.stack(outputs))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
        counters[id(fedem)] = index + 1
        return actual

    monkeypatch.setattr(FedemClientMixture, "predict", inspect)
    run = execute_feddrift_experiment_run(run_settings=base)
    server = run.measured_run.participants.run_server
    transfers = server.communication_volume_record_store.get_state_snapshot()
    assert transfers.uploaded_model_count == transfers.downloaded_model_count == 4 * 3 * 2 * 2
    for c in server.run_clients:
        count = c.owners.model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts()
        assert count.parameter_update_step_counts_by_model_id == {0: 36, 1: 36}
    tables = build_feddrift_run_record_tables(experiment_run=run)
    assert len(tables["fedem_sample_responsibility.sample_index"]) == 24
    assert set(counters.values()) == {12}
    for t, counts in enumerate(run.measured_run.timestep_model_computation_counts):
        assert (
            counts.sample_processing_model_computation_counts.concept_specific_part_inference_example_count
            == 12
        )
        expected = 36 if scope == "window" else 36 * (t + 1)
        assert (
            counts.synchronization_model_computation_counts.concept_specific_part_inference_example_count
            == expected
        )


def test_official_estep_and_weighted_loss():
    # omarfoq/fedem 996fadd client.py MixtureClient.update_sample_weights /
    # update_learners_weights、learners/learner.py Learner.fit_batch。
    # results/_sol_prompts/fedem-official-excerpt.pyと同じ式（外部ファイル不要）。
    losses = torch.tensor([[0.2, 1.4], [2.1, 0.1], [0.7, 0.8]])
    pi = torch.tensor([0.3, 0.7])
    expected = torch.softmax(torch.log(pi) - losses, dim=1)
    actual = _fedem_responsibilities(losses, pi)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(actual.mean(0), expected.T.mean(1), atol=1e-6, rtol=0)
    indices = torch.tensor([2, 0])
    loss = torch.tensor([1.3, 0.4], requires_grad=True)
    result = _fedem_weighted_loss(loss, actual[indices, 1])
    # 公式は2次元のloss_vecに.Tを取る。1次元では転置は恒等なので省く（警告を避ける）。
    torch.testing.assert_close(result, (loss @ expected[indices, 1]).mean() / 2)
    result.backward()
    torch.testing.assert_close(loss.grad, expected[indices, 1] / 2)
    # 極端な損失と0重みもNaNを作らない。
    assert torch.isfinite(_fedem_responsibilities(losses * 1e5, torch.tensor([0.0, 1.0]))).all()


@pytest.mark.parametrize("class_count", (2, 4))
def test_real_estep_precedes_pi_update_and_weighted_mstep(class_count, monkeypatch):
    _, inputs = build_assembly_inputs(monkeypatch=monkeypatch, class_count=class_count)
    client = assemble_feddrift_run_client(
        **{
            **inputs,
            "initial_model_count": 2,
            "run_client_settings": replace(
                inputs["run_client_settings"],
                model_assignment_policy="fedem_responsibility_assignment",
                training_sample_window="all_observed_history",
                cluster_model_initialization="all_random",
            ),
        }
    )
    observe(client, (0, 1, 0))
    fedem = client.stream_processor._fedem
    with torch.no_grad():
        states = client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        features = torch.cat([s.input_features for s in fedem._samples])
        labels = torch.cat([s.observed_class_labels for s in fedem._samples])
        loss_vectors = [
            torch.nn.functional.binary_cross_entropy(
                s.classifier(features), labels, reduction="none"
            ).view(-1)
            if class_count == 2
            else torch.nn.functional.cross_entropy(
                s.classifier(features), labels.view(-1).long(), reduction="none"
            )
            for s in states
        ]
        expected = torch.softmax(torch.log(fedem.mixture_weights) - torch.stack(loss_vectors, 1), 1)
    # 学習のstoreの各成分の標本列が、qを計算した標本列と同じ順に並ぶ（trainerは同じ位置でqと標本を取り出す）。
    for collection in client.owners.training_sample_store.snapshot_ordered_model_training_samples():
        assert len(collection.training_samples) == len(fedem._samples)
        for stored, observed in zip(collection.training_samples, fedem._samples, strict=True):
            assert torch.equal(stored.input_features, observed.input_features)
            assert torch.equal(stored.observed_class_labels, observed.observed_class_labels)
    weighted_loss = training._fedem_weighted_loss
    seen = []

    def inspect_mstep(losses, weights):
        # この時点でE-stepとπ更新が完了し、学習中のqは固定されている。
        torch.testing.assert_close(fedem.responsibilities, expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(fedem.mixture_weights, expected.mean(0), atol=1e-6, rtol=0)
        torch.testing.assert_close(weights.sort().values, expected[:, len(seen) // 6].sort().values)
        seen.append(1)
        return weighted_loss(losses, weights)

    monkeypatch.setattr(training, "_fedem_weighted_loss", inspect_mstep)
    assert len(client.train_local_round()) == 12  # 成分ごと2 updates/sample × 3標本。
    assert len(seen) == 12


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_configuration_initialization_and_round_axis(configuration, monkeypatch):
    actual = settings(configuration=configuration).run_participant_settings.run_client_settings
    assert actual.model_assignment_policy == "fedem_responsibility_assignment"
    assert actual.training_sample_window == (
        "current_timestep" if "-window" in configuration else "all_observed_history"
    )
    assert actual.cluster_model_initialization == (
        "pretrained_cluster0" if configuration.endswith("-component0") else "all_random"
    )
    _, inputs = build_assembly_inputs(monkeypatch=monkeypatch)
    inputs["run_client_settings"] = replace(
        inputs["run_client_settings"],
        model_assignment_policy=actual.model_assignment_policy,
        training_sample_window=actual.training_sample_window,
        cluster_model_initialization=actual.cluster_model_initialization,
    )
    before = torch.get_rng_state().clone()
    python_before = inputs["python_random_generator"].getstate()
    numpy_before = np.random.get_state()
    clients = [
        assemble_feddrift_run_client(**{**inputs, "client_id": cid}, initial_model_count=4)
        for cid in (0, 1)
    ]
    assert torch.equal(before, torch.get_rng_state())
    assert inputs["python_random_generator"].getstate() == python_before
    for a, b in zip(numpy_before, np.random.get_state(), strict=True):
        np.testing.assert_equal(a, b)
    states = [
        c.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        for c in clients
    ]
    for client in clients:
        torch.testing.assert_close(
            client.stream_processor._fedem.mixture_weights, torch.full((4,), 0.25)
        )
    for a, b in zip(*states, strict=True):
        for p, q in zip(a.classifier.parameters(), b.classifier.parameters(), strict=True):
            assert torch.equal(p, q)
    values = [
        torch.cat([p.detach().flatten() for p in s.classifier.parameters()]) for s in states[0]
    ]
    assert all(not torch.equal(a, b) for i, a in enumerate(values) for b in values[i + 1 :])
    original = torch.cat([p.detach().flatten() for p in inputs["initial_classifier"].parameters()])
    assert torch.equal(values[0], original) == configuration.endswith("-component0")
    if configuration.endswith("-component0"):
        # ID 1〜M-1は、乱数の初期化の構成と同じ値（要求1.2）。
        random_inputs = {
            **inputs,
            "run_client_settings": replace(
                inputs["run_client_settings"], cluster_model_initialization="all_random"
            ),
        }
        random_states = assemble_feddrift_run_client(
            **random_inputs, initial_model_count=4
        ).owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        for state, random_state in zip(states[0][1:], random_states[1:], strict=True):
            for p, q in zip(
                state.classifier.parameters(), random_state.classifier.parameters(), strict=True
            ):
                assert torch.equal(p, q)
    for state in states[0][1:]:
        assert not state.concept_specific_parameter_optimizer_state.parameter_optimizer.state
    runs = build_planned_experiment_runs(sweep_plan=plan((configuration,), (1, 3)))
    assert len(runs) == 2
    assert len({run.run_identifier for run in runs}) == 2


# 小さい全体runの規模（client数、clientごとの標本数）。既定は他のFedDrift系の全体runと同じ`settings()`の
# 10 client×2,500標本。historyはE-stepと学習の対象が時刻とともに増え重いので、mnist4-historyは、検査が
# 全部働く最小の規模にする（3時刻で学習storeが増え、間隔50が学習のバッチ32標本以上、client 3つで集約）。
SMALL_FULL_RUN_SCALES = {("mnist4", "history", 1): (3, 150)}
SMALL_FULL_RUN_INTERVAL = 50


@pytest.mark.parametrize(
    "dataset,scope,rounds",
    (
        ("sine2", "window", 1),
        ("mnist4", "window", 1),
        ("sine2", "history", 1),
        ("mnist4", "history", 1),
        ("sine2", "window", 3),
        ("sine2", "history", 3),
    ),
)
@pytest.mark.slow
def test_small_full_run(dataset, scope, rounds, monkeypatch):
    configuration = f"feddrift-fedem-{scope}"
    clients, samples = SMALL_FULL_RUN_SCALES.get((dataset, scope, rounds), (10, 2500))
    interval = SMALL_FULL_RUN_INTERVAL
    # 既定の規模では、`settings()`と同じ条件になる。
    run_settings = build_experiment_run_settings(
        configuration_name=configuration,
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name=dataset,
            random_seed=0,
            client_count=clients,
            per_client_sample_count=samples,
            server_aggregation_interval_per_client_samples=interval,
        ),
    )
    base = with_rounds(run_settings, rounds)
    m = 2 if dataset == "sine2" else 4
    train = FeddriftRunClient.train_local_round
    aggregate = FeddriftRunServer.aggregate_and_distribute_training_round
    order = []
    calls = []
    prediction_counts = {}
    predict = FedemClientMixture.predict

    def inspect_prediction(fedem, features):
        index = prediction_counts.get(id(fedem), 0)
        # flushされるのは予測→ラベル観測後。現在の標本は学習storeにもE-stepにもまだない。
        assert all(i < index for i in fedem.sample_indices)
        t = index // interval
        assert fedem._timestep_index == t - 1
        expected_pi = (
            torch.full((m,), 1 / m)
            if t == 0
            else torch.tensor([row[2] for row in fedem.mixture_records if row[0] == t - 1])
        )
        torch.testing.assert_close(fedem.mixture_weights, expected_pi, atol=0, rtol=0)
        probabilities = []
        states = fedem._registry.snapshot_ordered_held_model_training_states()
        handles = []
        if index % interval == 0:
            handles = [
                s.classifier.register_forward_hook(
                    lambda module, inputs, output: probabilities.append(output.detach().clone())
                )
                for s in states
            ]
        try:
            result = predict(fedem, features)
        finally:
            for handle in handles:
                handle.remove()
        if index % interval == 0:
            with torch.no_grad():
                if states[0].classifier.class_count > 2:
                    probabilities = [p.softmax(1) for p in probabilities]
                # learners_ensemble.py evaluate_iteratorの確率の混合。
                expected = torch.einsum("m,mbc->bc", expected_pi, torch.stack(probabilities)).clamp(
                    0, 1
                )
                torch.testing.assert_close(result, expected, atol=1e-6, rtol=0)
        prediction_counts[id(fedem)] = index + 1
        return result

    def inspect_train(client, **kwargs):
        fedem = client.stream_processor._fedem
        end = client.stream_processor.processed_sample_count
        collections = client.owners.training_sample_store.snapshot_ordered_model_training_samples()
        assert {c.model_id for c in collections} == set(range(m))
        expected_count = interval if scope == "window" else end
        assert all(len(c.training_samples) == expected_count for c in collections)
        assert len(fedem.sample_indices) == expected_count
        assert fedem.sample_indices == list(range(end - expected_count, end))
        before = fedem.mixture_weights.clone()
        result = train(client, **kwargs)
        torch.testing.assert_close(
            fedem.mixture_weights, fedem.responsibilities[-interval:].mean(0)
        )
        calls.append((client.client_id, end, before, fedem.mixture_weights.clone()))
        order.append(client.client_id)
        return result

    def inspect_aggregate(server, **kwargs):
        assert order[-clients:] == list(range(clients))  # 全clientのE/π/Mの後で集約。
        participants = [
            c.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
            for c in server.run_clients
        ]
        counts = [
            len(
                c.owners.training_sample_store.snapshot_ordered_model_training_samples()[
                    0
                ].training_samples
            )
            for c in server.run_clients
        ]
        # aggregator.py clients_weights。qの和ではなく標本件数で同じ重み。
        weights = torch.tensor(counts, dtype=torch.float32)
        weights /= weights.sum()
        expected = [
            {
                k: sum(s[mid].classifier.state_dict()[k] * w for s, w in zip(participants, weights))
                for k in participants[0][mid].classifier.state_dict()
            }
            for mid in range(m)
        ]
        pis = [c.stream_processor._fedem.mixture_weights.clone() for c in server.run_clients]
        before_transfers = server.communication_volume_record_store.get_state_snapshot()
        aggregate(server, **kwargs)
        after_transfers = server.communication_volume_record_store.get_state_snapshot()
        assert (
            after_transfers.uploaded_model_count - before_transfers.uploaded_model_count
            == clients * m
        )
        assert (
            after_transfers.downloaded_model_count - before_transfers.downloaded_model_count
            == clients * m
        )
        for mid in range(m):
            for key, value in server.global_model_repository.get_global_model_parameters(
                model_id=mid
            ).items():
                torch.testing.assert_close(value, expected[mid][key], atol=1e-6, rtol=1e-5)
        for c, pi in zip(server.run_clients, pis):
            torch.testing.assert_close(
                c.stream_processor._fedem.mixture_weights, pi, atol=0, rtol=0
            )

    monkeypatch.setattr(FeddriftRunClient, "train_local_round", inspect_train)
    monkeypatch.setattr(
        FeddriftRunServer, "aggregate_and_distribute_training_round", inspect_aggregate
    )
    monkeypatch.setattr(FedemClientMixture, "predict", inspect_prediction)
    run = execute_feddrift_experiment_run(run_settings=base)
    tables = build_feddrift_run_record_tables(experiment_run=run)
    assert len(calls) == samples // interval * clients * rounds
    pi = tables["fedem_client_mixture.mixture_weight"].reshape(clients, samples // interval, m)
    np.testing.assert_allclose(pi.sum(2), 1, atol=1e-6)
    assert len(tables["fedem_sample_responsibility.sample_index"]) == clients * samples
    np.testing.assert_array_equal(
        tables["fedem_sample_responsibility.sample_index"].reshape(clients, samples),
        np.tile(np.arange(samples), (clients, 1)),
    )
    assert len(tables["model_clustering.model_id"]) == 0
    assert len(tables["model_registration.model_id"]) == m
    assert np.all(tables["server_timestep.global_model_count_after_synchronization"] == m)
    assert len(tables["detection_batch.client_id"]) == 0
    # E-stepでの最大成分と真概念の一致を成分ごと多数決で測る（IDの置換に依存しない純度）。
    qmax = tables["fedem_sample_responsibility.max_responsibility_model_id"]
    concepts = tables["sample_prediction.observed_concept_id"].reshape(-1)
    purity = sum(
        np.bincount(concepts[qmax == mid], minlength=m).max(initial=0) for mid in range(m)
    ) / len(qmax)
    print(
        f"{configuration}/{dataset}/R={rounds}: pi(client0)={pi[0, [0, pi.shape[1] // 2, -1]].tolist()}, purity={purity:.6f}"
    )


def test_small_full_runs_are_distributed_one_case_per_process(request):
    # 全pytestの並列実行（conftestの配り方）で、小さい全体runの各ケースが別の組になり、
    # このファイルのほかのtestはファイルの組のままであること。
    from tests.conftest import select_parallel_test_scope

    file_path = "tests/refactoring/test_feddrift_fedem_configuration.py"
    case_ids = [
        f"{dataset}-{scope}-{rounds}"
        for dataset, scope, rounds in (
            ("sine2", "window", 1),
            ("mnist4", "window", 1),
            ("sine2", "history", 1),
            ("mnist4", "history", 1),
            ("sine2", "window", 3),
            ("sine2", "history", 3),
        )
    ]
    scopes = {
        select_parallel_test_scope(test_node_id=f"{file_path}::test_small_full_run[{case_id}]")
        for case_id in case_ids
    }
    assert len(scopes) == len(case_ids)
    assert file_path not in scopes
    collected = [
        item
        for item in request.session.items
        if item.path.name == "test_feddrift_fedem_configuration.py"
    ]
    for item in collected:
        scope = select_parallel_test_scope(test_node_id=item.nodeid)
        if item.originalname == "test_small_full_run":
            assert scope == item.nodeid, item.nodeid
        else:
            assert scope == file_path, item.nodeid
