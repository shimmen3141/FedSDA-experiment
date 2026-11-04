"""FedCCFAの公式の式とストリームへの接続を確認する。"""

from dataclasses import replace
from random import Random

import numpy as np
import pytest
import torch

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.runtime.fedccfa_classifier_clustering import (
    FedccfaClientState,
    _alignment_loss,
    _alignment_start_round,
    _balanced_indices,
    _convert_binary_classifier,
    _entropy_weight,
    _label_groups,
    _madd,
    _weighted_mean,
)
from federated_learning_experiments.runtime.feddrift_experiment_run import (
    execute_feddrift_experiment_run,
)
from federated_learning_experiments.runtime.feddrift_run_record_tables import (
    build_feddrift_run_record_tables,
)
from federated_learning_experiments.runtime.feddrift_run_server import FeddriftRunServer
from federated_learning_experiments.sweep.experiment_configurations import (
    build_experiment_run_settings,
)
from federated_learning_experiments.sweep.experiment_sweep_plan import build_planned_experiment_runs
from tests.refactoring import fedccfa_official_excerpt as official
from tests.refactoring.test_feddrift_oracle_concept_assignment_configuration import (
    settings as oracle_settings,
)
from tests.refactoring.test_feddrift_run_client import build_assembly_inputs
from tests.refactoring.test_feddrift_window_multiple_training_rounds import plan, with_rounds

CONFIGURATIONS = tuple(
    f"feddrift-fedccfa-{scope}{suffix}"
    for scope in ("window", "window200")
    for suffix in ("", "-scaled-alignment")
)


def test_slow_runs_have_independent_parallel_scopes():
    from tests.conftest import select_parallel_test_scope

    path = "tests/refactoring/test_feddrift_fedccfa_configuration.py"
    node = f"{path}::test_original_mnist_behavior"
    assert select_parallel_test_scope(test_node_id=node) == node
    cases = (
        "sine2-window-1",
        "mnist4-window-1",
        "sine2-window200-scaled-alignment-1",
        "mnist4-window200-scaled-alignment-1",
        "sine2-window-3",
        "sine2-window200-scaled-alignment-3",
    )
    scopes = {
        select_parallel_test_scope(test_node_id=f"{path}::test_small_full_run[{case}]")
        for case in cases
    }
    assert len(scopes) == 6
    assert path not in scopes


def test_official_formulas():
    vectors = np.random.RandomState(3).normal(size=(5, 8)).astype(np.float32)
    np.testing.assert_allclose(_madd(vectors), official.madd(vectors), atol=1e-6)
    assert {frozenset(g) for g in _label_groups(vectors, 0.1)} == official.groups(vectors, 0.1)
    values = [torch.tensor([0.3, 1.0]), torch.tensor([1.2, -0.4]), torch.tensor([2.1, 3.0])]
    torch.testing.assert_close(_weighted_mean(values, [1, 1, 1]), official.mean(values))
    torch.testing.assert_close(
        _weighted_mean(values, [3, 2, 7]), official.representation(values, [3, 2, 7])
    )
    assert _entropy_weight([3, 2, 7], 20.0) == pytest.approx(official.weight([3, 2, 7], 20.0))
    features = torch.tensor([[0.2, 0.4], [0.6, -0.7], [1.0, 0.5]])
    labels = torch.tensor([0, 1, 2])
    torch.testing.assert_close(
        _alignment_loss(features, dict(enumerate(values)), labels, 0.1),
        official.alignment(features, values, labels, 0.1),
    )
    from random import Random

    indices = _balanced_indices(torch.tensor([0] * 7 + [1] * 3 + [2] * 5), Random(0))
    assert len(indices) == 3 * official.balanced_count([7, 3, 5])
    assert len(set(indices)) == len(indices)


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_configuration(configuration):
    settings = build_experiment_run_settings(
        configuration_name=configuration,
        experiment_run_conditions=oracle_settings().execution_settings.experiment_run_conditions,
    )
    client = settings.run_participant_settings.run_client_settings
    assert client.model_assignment_policy == "fedccfa_classifier_clustering"
    assert client.training_sample_window == (
        "recent_observed_samples" if "window200" in configuration else "current_timestep"
    )
    assert client.fedccfa_alignment_start == (
        "scaled_to_run_rounds" if configuration.endswith("scaled-alignment") else "fixed_round_20"
    )
    assert (
        client.fedccfa_dbscan_eps,
        client.fedccfa_alignment_gamma,
        client.fedccfa_alignment_temperature,
    ) == (0.1, 20.0, 0.1)


@pytest.mark.parametrize(
    "samples,interval,rounds,expected",
    ((5000, 50, 1, 10), (5000, 500, 1, 1), (5000, 500, 5, 5), (150, 50, 3, 1)),
)
def test_scaled_alignment_start(samples, interval, rounds, expected):
    assert _alignment_start_round("scaled_to_run_rounds", samples, interval, rounds) == expected
    assert _alignment_start_round("fixed_round_20", samples, interval, rounds) == 20


def test_binary_initial_probabilities_and_missing_classes(monkeypatch):
    _, inputs = build_assembly_inputs(monkeypatch=monkeypatch)
    classifier = inputs["initial_classifier"]
    x = torch.rand(40, classifier.input_feature_count)
    expected = classifier(x).detach()
    _convert_binary_classifier(classifier)
    assert classifier.classification_layer.out_features == 2
    torch.testing.assert_close(classifier(x), expected, atol=1e-6, rtol=0)
    state = FedccfaClientState(classifier=classifier, recent_window=True)
    state.append_timestep_samples(
        tuple(
            ObservedTrainingSample(
                input_features=v[None], observed_class_labels=torch.tensor([[1.0]])
            )
            for v in x
        ),
        40,
    )
    settings = replace(
        inputs["run_client_settings"],
        model_assignment_policy="fedccfa_classifier_clustering",
        training_sample_window="recent_observed_samples",
    )
    state.alignment_start_round = 0
    state.train_round(settings=settings, rng=Random(0), timestep_index=0, round_index=0)
    assert state.class_counts == [0, 40]
    assert len(state.balanced_sample_indices) == 5
    assert set(state.anchors) == {1}
    assert state.round_records[-1][2:4] == (0, 0.0)
    previous = state.anchors[1].clone()
    # 次の窓にはクラス0だけ。過去のクラス1のアンカーは保持する。
    state.samples[:] = [
        ObservedTrainingSample(input_features=v[None], observed_class_labels=torch.tensor([[0.0]]))
        for v in x
    ]
    state.train_round(settings=settings, rng=Random(1), timestep_index=1, round_index=0)
    torch.testing.assert_close(state.anchors[1], previous, atol=0, rtol=0)
    assert set(state.local_anchors) == {0}
    assert state.round_records[-1][2:4] == (0, 0.0)
    # 過去にも無いクラスは、ロジットから除かれ、残った列へラベルを写す。
    value = _alignment_loss(torch.rand(2, 3), {2: torch.rand(3)}, torch.tensor([2, 2]), 0.1)
    assert value == 0
    features = torch.rand(2, 3)
    anchors = {2: torch.rand(3), 3: torch.rand(3)}
    torch.testing.assert_close(
        _alignment_loss(features, anchors, torch.tensor([1, 2]), 0.1),
        official.alignment(features[1:], list(anchors.values()), torch.tensor([0]), 0.1),
    )


def test_sweep_eps_and_rounds():
    runs = build_planned_experiment_runs(
        sweep_plan=replace(plan(CONFIGURATIONS, (1, 3)), fedccfa_dbscan_eps=(0.05, 0.1))
    )
    assert len(runs) == 16
    assert len({r.run_identifier for r in runs}) == 16


def test_existing_31_settings_names_order_and_identifiers_are_unchanged():
    from hashlib import sha256
    from json import dumps

    from federated_learning_experiments.core.settings_serialization import (
        convert_settings_to_plain_mapping,
    )
    from federated_learning_experiments.sweep.experiment_configurations import (
        EXPERIMENT_CONFIGURATION_NAMES,
    )
    from federated_learning_experiments.sweep.experiment_sweep_plan import make_run_identifier

    conditions = oracle_settings().execution_settings.experiment_run_conditions
    values = []
    for name in EXPERIMENT_CONFIGURATION_NAMES[:31]:
        settings = build_experiment_run_settings(
            configuration_name=name, experiment_run_conditions=conditions
        )
        values.append(
            (
                name,
                convert_settings_to_plain_mapping(settings),
                make_run_identifier(configuration_name=name, run_settings=settings),
            )
        )
    # HEAD 743b6600の構成生成関数とも同じ入力で照合して採取した31構成の全保存設定・順・ID。
    assert (
        sha256(dumps(values, sort_keys=True).encode()).hexdigest()
        == "ec7f74c1415a33c41ed328e8e461c48e00bd75ccc6b7b0e787d14ecc48d177c0"
    )


@pytest.mark.parametrize(
    "field", ("fedccfa_dbscan_eps", "fedccfa_alignment_gamma", "fedccfa_alignment_temperature")
)
@pytest.mark.parametrize("value", (0.0, -1.0, float("nan"), float("inf"), True, 1))
def test_positive_finite_float_settings(field, value):
    settings = full_settings("sine2", "window", 1).run_participant_settings.run_client_settings
    with pytest.raises(ValueError):
        replace(settings, **{field: value})


def test_cli_axis_and_missing_class_aggregation(monkeypatch):
    from copy import deepcopy
    from types import SimpleNamespace

    from federated_learning_experiments.methods.fedsda.model_registration.global_model_repository import (
        GlobalModelRepository,
    )
    from federated_learning_experiments.runtime.fedccfa_training_round import (
        aggregate_fedccfa_training_round,
    )
    from federated_learning_experiments.sweep import experiment_sweep_command as command

    parsed = command._build_argument_parser().parse_args(
        [
            "--datasets",
            "sine2",
            "--seeds",
            "0",
            "--aggregation-intervals",
            "50",
            "--purpose",
            "test",
            "--fedccfa-dbscan-eps",
            "0.05",
            "0.1",
        ]
    )
    assert parsed.fedccfa_dbscan_eps == [0.05, 0.1]
    _, inputs = build_assembly_inputs(monkeypatch=monkeypatch, class_count=4)
    classifier = inputs["initial_classifier"]
    settings = replace(
        inputs["run_client_settings"],
        model_assignment_policy="fedccfa_classifier_clustering",
        training_sample_window="current_timestep",
    )
    states = [
        FedccfaClientState(classifier=deepcopy(classifier), recent_window=False) for _ in range(3)
    ]
    initial_anchors = {3: torch.full((classifier.classification_layer.in_features,), 9.0)}
    for state, labels in zip(states, ([0] * 40, [1] * 40, [0, 1] * 20)):
        state.anchors = deepcopy(initial_anchors)
        x = torch.rand(40, 3)
        state.append_timestep_samples(
            tuple(
                ObservedTrainingSample(
                    input_features=x[i : i + 1], observed_class_labels=torch.tensor([[float(y)]])
                )
                for i, y in enumerate(labels)
            ),
            40,
        )
        state.train_round(settings=settings, rng=Random(0), timestep_index=0, round_index=0)
        assert state.completed_stage_updates == (5, 2, 6)
    before = [deepcopy(s.classifier.classification_layer.state_dict()) for s in states]
    repo = GlobalModelRepository(
        initial_model_id=0,
        initial_parameter_snapshot=deepcopy(dict(classifier.state_dict())),
        initial_loss_statistics=inputs["initial_loss_statistics"],
    )
    adapter = SimpleNamespace(
        run_clients=tuple(
            SimpleNamespace(stream_processor=SimpleNamespace(_fedccfa=s), _settings=settings)
            for s in states
        ),
        global_model_repository=repo,
        _current_training_round_index=0,
        _fedccfa_communication_records=[],
        _aggregation_parameter_multiply_accumulate_count=0,
        _record_parameter_transfer=lambda *args: None,
    )
    aggregate_fedccfa_training_round(run_server=adapter, timestep_index=0)
    for state, original in zip(states, before):
        for label, count in enumerate(state.class_counts):
            row = next(r for r in state.cluster_records if r[2] == label)
            if not count:
                assert row[3] == -1
                for key in original:
                    torch.testing.assert_close(
                        state.classifier.classification_layer.state_dict()[key][label],
                        original[key][label],
                        atol=0,
                        rtol=0,
                    )
        torch.testing.assert_close(state.anchors[3], initial_anchors[3], atol=0, rtol=0)
    assert states[0].cluster_records[0][3] == states[2].cluster_records[0][3] == 0
    assert states[1].cluster_records[1][3] == states[2].cluster_records[1][3] == 0


def full_settings(dataset, scope, rounds, clients=3, samples=250):
    # 3 clientはMADDが定義できる最小、5時刻はwindow200が古い標本を捨てる最小。
    return with_rounds(
        build_experiment_run_settings(
            configuration_name=f"feddrift-fedccfa-{scope}",
            experiment_run_conditions=ExperimentRunConditions(
                dataset_name=dataset,
                random_seed=0,
                client_count=clients,
                per_client_sample_count=samples,
                server_aggregation_interval_per_client_samples=50,
            ),
        ),
        rounds,
    )


@pytest.mark.parametrize(
    "dataset,scope,rounds",
    (
        ("sine2", "window", 1),
        ("mnist4", "window", 1),
        ("sine2", "window200-scaled-alignment", 1),
        ("mnist4", "window200-scaled-alignment", 1),
        ("sine2", "window", 3),
        ("sine2", "window200-scaled-alignment", 3),
    ),
)
@pytest.mark.slow
def test_small_full_run(dataset, scope, rounds, monkeypatch):
    predictions = {}
    predict = FedccfaClientState.predict
    train = FedccfaClientState.train_round
    aggregate = FeddriftRunServer.aggregate_and_distribute_training_round
    windows = []
    snapshots = {}

    def inspect_predict(state, features):
        index = predictions.get(id(state), 0)
        assert all(i < index for i in state.sample_indices)
        expected_indices = (
            list(range(max(0, index - 200), index))
            if "window200" in scope
            else list(range(index - index % 50, index))
        )
        assert state.sample_indices == expected_indices
        # 集約の間はパラメータを変えない。
        current = [p.detach().clone() for p in state.classifier.parameters()]
        if index % 50:
            assert all(torch.equal(a, b) for a, b in zip(snapshots[id(state)], current))
        snapshots[id(state)] = current
        predictions[id(state)] = index + 1
        return predict(state, features)

    def inspect_train(state, **kwargs):
        end = predictions[id(state)]
        size = min(end, 200) if "window200" in scope else 50
        assert state.sample_indices == list(range(end - size, end))
        windows.append((end, state.sample_indices.copy()))
        result = train(state, **kwargs)
        assert state.completed_stage_updates == (5, 10, 50)
        assert state.stage_trace == [
            "label_distribution",
            "diagnostic",
            "personal_classifier",
            "representation",
            "local_anchors",
        ]
        y = torch.cat([s.observed_class_labels for s in state.samples]).view(-1).long()
        selected = y[state.balanced_sample_indices].bincount(minlength=state.classifier.class_count)
        assert all(
            selected[c] == min(5, min(n for n in state.class_counts if n))
            for c, n in enumerate(state.class_counts)
            if n
        )
        active = bool(state.round_records[-1][2])
        assert active == (
            state.round_index - 1 > 0 and state.round_index - 1 >= state.alignment_start_round
        )
        return result

    def inspect_aggregate(server, **kwargs):
        states = [c.stream_processor._fedccfa for c in server.run_clients]
        assert all(s.stage_trace[-1] == "local_anchors" for s in states)
        personal = [
            {k: v.clone() for k, v in s.classifier.classification_layer.state_dict().items()}
            for s in states
        ]
        previous_anchors = [{k: v.clone() for k, v in s.anchors.items()} for s in states]
        expected_rep = {
            k: _weighted_mean(
                [dict(s.classifier.named_parameters())[k].detach() for s in states],
                [len(s.samples) for s in states],
            )
            for k, _ in states[0].classifier.named_parameters()
            if not k.startswith("classification_layer.")
        }
        aggregate(server, **kwargs)
        for s in states:
            for k, v in expected_rep.items():
                torch.testing.assert_close(dict(s.classifier.named_parameters())[k], v)
        for i, s in enumerate(states):
            for row in s.cluster_records[-s.classifier.class_count :]:
                label, group, count = row[2:]
                assert count == s.class_counts[label]
                if count == 0:
                    assert group == -1
                if group == -1:
                    for k, v in personal[i].items():
                        torch.testing.assert_close(
                            s.classifier.classification_layer.state_dict()[k][label], v[label]
                        )
                    if label in previous_anchors[i]:
                        torch.testing.assert_close(s.anchors[label], previous_anchors[i][label])

    monkeypatch.setattr(FedccfaClientState, "predict", inspect_predict)
    monkeypatch.setattr(FedccfaClientState, "train_round", inspect_train)
    monkeypatch.setattr(
        FeddriftRunServer, "aggregate_and_distribute_training_round", inspect_aggregate
    )
    run = execute_feddrift_experiment_run(run_settings=full_settings(dataset, scope, rounds))
    tables = build_feddrift_run_record_tables(experiment_run=run)
    server = run.measured_run.participants.run_server
    classes = 2 if dataset == "sine2" else 10
    assert len(tables["fedccfa_round.client_id"]) == 3 * 5 * rounds
    assert len(tables["fedccfa_label_cluster.client_id"]) == 3 * 5 * rounds * classes
    assert set(predictions.values()) == {250}
    assert len(tables["detection_batch.client_id"]) == 0
    assert np.all(tables["server_timestep.global_model_count_after_synchronization"] == 1)
    for end, indices in windows:
        if end in (50, 100):
            assert indices == list(range(0 if "window200" in scope else end - 50, end))
    rows = server._fedccfa_communication_records
    assert server.communication_volume_record_store.get_state_snapshot().uploaded_message_count == 0
    assert (
        server.communication_volume_record_store.get_state_snapshot().downloaded_message_count == 0
    )
    for direction, field in (
        ("upload", "uploaded_parameter_value_count"),
        ("download", "downloaded_parameter_value_count"),
    ):
        assert sum(r[-1] for r in rows if r[3] == direction) == getattr(
            server.communication_volume_record_store.get_state_snapshot(), field
        )
    head = server.run_clients[0].stream_processor._fedccfa.initial_classifier_parameters
    for k, v in head.items():
        torch.testing.assert_close(
            server.global_model_repository.get_global_model_parameters(model_id=0)[
                "classification_layer." + k
            ],
            v,
        )
    # 各roundの部品ごとの個数。下りは実際の組に含まれた行とアンカーだけ。
    width = head["weight"].shape[1]
    head_size = sum(v.numel() for v in head.values())
    rep_size = sum(
        v.numel()
        for k, v in server.global_model_repository.get_global_model_parameters(model_id=0).items()
        if not k.startswith("classification_layer.")
    )
    for t in range(5):
        for r in range(rounds):
            records = [row for row in rows if row[:2] == (t, r)]
            for component, size in (
                ("representation", rep_size),
                ("diagnostic_classifier", head_size),
                ("personal_classifier", head_size),
            ):
                assert (
                    sum(row[-1] for row in records if row[2:4] == (component, "upload")) == 3 * size
                )
            assert (
                sum(row[-1] for row in records if row[2:4] == ("representation", "download"))
                == 3 * rep_size
            )
            grouped = sum(
                row[0] == t and row[1] == r and row[3] >= 0
                for c in server.run_clients
                for row in c.stream_processor._fedccfa.cluster_records
            )
            assert sum(
                row[-1] for row in records if row[2:4] == ("personal_classifier_row", "download")
            ) == grouped * (width + 1)
            assert (
                sum(row[-1] for row in records if row[2:4] == ("anchor", "download"))
                == grouped * width
            )
    counts = run.measured_run.model_computation_counts
    assert counts.concept_specific_parameter_optimizer_step_count == 3 * 5 * rounds * 65


@pytest.mark.slow
def test_original_mnist_behavior(monkeypatch):
    from federated_learning_experiments.data.observed_streams import ClientConceptTrace
    from federated_learning_experiments.runtime import feddrift_stream_run_execution as execution

    def traces(**kwargs):
        return tuple(
            ClientConceptTrace(
                client_id=i,
                concept_ids_by_sample_index=tuple(
                    int(i >= 5 and index >= 200) for index in range(600)
                ),
            )
            for i in range(10)
        )

    monkeypatch.setattr(execution, "generate_random_client_concept_traces", traces)
    run = execute_feddrift_experiment_run(
        run_settings=full_settings(
            "mnist4", "window200-scaled-alignment", 3, clients=10, samples=600
        )
    )
    tables = build_feddrift_run_record_tables(experiment_run=run)
    clusters = {}
    states = [c.stream_processor._fedccfa for c in run.measured_run.participants.run_clients]
    for t in (1, 2, 3, 9, 10, 11):
        clusters[t] = {}
        for label in range(10):
            groups = {}
            for i, s in enumerate(states):
                row = next(row for row in s.cluster_records if row[:3] == (t, 2, label))
                if row[3] >= 0:
                    groups.setdefault(row[3], []).append(i)
            clusters[t][label] = list(groups.values())
    labels = tables["sample_prediction.observed_class_id"].reshape(10, 600)
    correct = tables["sample_prediction.prediction_is_correct"].reshape(10, 600)

    def accuracy(start, end):
        mask = np.isin(labels[5:, start:end], [1, 2])
        return float(correct[5:, start:end][mask].mean() * 100)

    before, after = accuracy(200, 300), accuracy(450, 600)
    print(
        f"FedCCFA original behavior: clusters={clusters}; B labels1/2 early={before:.6f}%, late={after:.6f}%",
        flush=True,
    )
    failures = []

    def mixed(group):
        return any(i < 5 for i in group) and any(i >= 5 for i in group)

    for t in (9, 10, 11):
        for label in (1, 2):
            groups = clusters[t][label]
            if (
                any(mixed(g) for g in groups)
                or not any(all(i < 5 for i in g) for g in groups)
                or not any(all(i >= 5 for i in g) for g in groups)
            ):
                failures.append(f"t={t}, label={label}: 分離の条件を満たさない")
        if sum(any(mixed(g) for g in clusters[t][label]) for label in (0, 3, 4, 5, 6, 7, 8, 9)) < 5:
            failures.append(f"t={t}: 非交換クラスの混合が5未満")
    if after - before < 30:
        failures.append(f"回復の差が30ポイント未満: {after - before}")
    for t in (1, 2, 3):
        for label in (1, 2):
            if not any(mixed(g) for g in clusters[t][label]):
                failures.append(f"変化前t={t}, label={label}: 混合の組が無い")
    assert not failures, "\n".join(failures)
