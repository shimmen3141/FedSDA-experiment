"""DriftSurfの公式参照と、時刻の窓・構成・実runの契約。"""

import copy
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO

import numpy as np
import pytest
import torch

from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.learning.training.timestep_training_sample_store import (
    TimestepTrainingSampleStore,
)
from federated_learning_experiments.methods.feddrift.driftsurf_state import (
    DriftSurfSettings,
    DriftSurfState,
)
from federated_learning_experiments.runtime.feddrift_experiment_run import (
    execute_feddrift_experiment_run,
)
from federated_learning_experiments.runtime.feddrift_run_client import FeddriftRunClient
from federated_learning_experiments.runtime.feddrift_run_record_tables import (
    build_feddrift_run_record_tables,
)
from federated_learning_experiments.runtime.feddrift_run_server import FeddriftRunServer
from federated_learning_experiments.sweep.experiment_sweep_plan import (
    build_planned_experiment_runs,
    make_run_identifier,
)
from tests.refactoring.test_feddrift_experiment_sweep import plan
from tests.refactoring.test_feddrift_oracle_concept_assignment_configuration import settings

CONFIGURATION = "feddrift-driftsurf"


# 公式の__init__・_append_train_data・_reset・run_ds_algoを複写。
# _scoreのみ差し替え、反応終了時model_key=predの変更を1点加える。
class OfficialDriftSurfState:
    def __init__(self, delta=0.1, r=3, wl=10):
        self.reac_len = r
        self.delta = delta
        self.win_len = wl  # Number of batches to consider
        self.models = {"pred": None, "stab": None, "reac": None}
        self.train_data_dict = {"pred": [0], "stab": [0], "reac": None}
        self.train_keys = ["pred", "stab"]
        self.acc_best = 0
        self.acc_dict = None
        self.reac_ctr = None
        self.state = "stab"
        self.model_key = "pred"  # Model used for prediction

    def _score(self, model_key, test_data, device):
        return test_data[model_key]

    def _append_train_data(self, model_key, iter_id):
        self.train_data_dict[model_key].append(iter_id)
        # Limit the number of data batches based on window length
        if len(self.train_data_dict[model_key]) > self.win_len:
            self.train_data_dict[model_key].pop(0)

    def _reset(self, key):
        self.models[key] = None
        self.train_data_dict[key] = []

    def move_model_to_cpu(self):
        for key in self.models.keys():
            if self.models[key] is not None:
                self.models[key].cpu()

    def get_train_keys(self):
        return self.train_keys

    def get_train_data(self, key):
        return self.train_data_dict[key]

    def get_model_key(self):
        return self.model_key

    def set_model(self, key, model):
        self.models[key] = model

    def print_model(self):
        for key in self.models.keys():
            if self.models[key] is not None:
                print(f"Model {key}:")
                print(self.models[key].state_dict())

    def run_ds_algo(self, new_data, device, curr_iter):
        acc_pred = self._score("pred", new_data, device)
        print(f"DS Iteration {curr_iter}, acc: {acc_pred}")
        if acc_pred > self.acc_best:
            self.acc_best = acc_pred
        if self.state == "stab":
            if len(self.train_data_dict["stab"]) == 0:
                acc_stab = 0
            else:
                acc_stab = self._score("stab", new_data, device)
            if (acc_pred < self.acc_best - self.delta) or (acc_pred < acc_stab - self.delta / 2):
                # Enter reactive state
                self.state = "reac"
                self._reset("reac")
                self.reac_ctr = 0
                self.acc_dict = {"pred": np.zeros(self.reac_len), "reac": np.zeros(self.reac_len)}
            else:
                # Stay in stable state
                self._append_train_data("pred", curr_iter)
                self._append_train_data("stab", curr_iter)
                self.train_keys = ["pred", "stab"]
        if self.state == "reac":
            if self.reac_ctr > 0:
                acc_reac = self._score("reac", new_data, device)
                print(f"acc_reac = {acc_reac}")
                self.acc_dict["pred"][self.reac_ctr - 1] = acc_pred
                self.acc_dict["reac"][self.reac_ctr - 1] = acc_reac
                # Set key for next time step
                if acc_reac > acc_pred:
                    self.model_key = "reac"
                else:
                    self.model_key = "pred"
            self._append_train_data("pred", curr_iter)
            self._append_train_data("reac", curr_iter)
            self.train_keys = ["pred", "reac"]
            self.reac_ctr += 1
            if self.reac_ctr == self.reac_len:
                # Exit Reactive State
                self.state = "stab"
                self._reset("stab")
                if np.mean(self.acc_dict["pred"]) < np.mean(self.acc_dict["reac"]):
                    self.models["pred"] = self.models["reac"]
                    self.train_data_dict["pred"] = self.train_data_dict["reac"]
                    self.acc_best = np.amax(self.acc_dict["reac"])
                    self.model_key = "pred"
                self.model_key = "pred"  # design.mdの唯一の変更
                self.acc_dict = None
                self.reac_ctr = None
        # Debug
        print(self.state)
        print(self.train_data_dict)
        print(self.train_keys)
        print(self.acc_best)
        print(self.model_key)


@pytest.mark.parametrize("delta", (0.02, 0.10))
@pytest.mark.parametrize(
    "sequence", ("stable", "promote", "no_promote", "switch", "repeat", "random")
)
def test_state_matches_official_reference(delta, sequence):
    rows = np.random.RandomState(913).uniform(0, 1, (100, 3))
    if sequence == "stable":
        rows[:] = 0.8
    if sequence == "promote":
        rows[:] = ([0.9, 0.9, 0.9], [0.1, 0.1, 0.9], [0.1, 0.1, 0.9], [0.1, 0.1, 0.9]) * 25
    if sequence == "no_promote":
        rows[:] = ([0.6, 0.9, 0.1], [0.9, 0.9, 0.1], [0.6, 0.9, 0.8], [0.9, 0.9, 0.1]) * 25
    if sequence in ("switch", "repeat"):
        rows[:] = (
            [0.9, 0.9, 0.1],
            [0.1, 0.9, 0.9],
            [0.8, 0.8, 0.1],
            [0.1, 0.1, 0.9],
            [0.9, 0.9, 0.1],
        ) * 20
    reference = OfficialDriftSurfState(delta=delta)
    state = DriftSurfState(settings=DriftSurfSettings(accuracy_drop_threshold=delta))
    for t, row in enumerate(rows, 1):
        scores = dict(zip(("pred", "stab", "reac"), row.tolist(), strict=True))
        before = reference.state
        with redirect_stdout(StringIO()):
            reference.run_ds_algo(scores, None, t)
        result = state.advance(timestep_index=t, accuracies=scores)
        for name in ("state", "train_keys", "model_key", "train_data_dict", "acc_best", "reac_ctr"):
            assert getattr(state, name) == getattr(reference, name), (sequence, t, name)
        assert result.entered_reactive == (
            before == "stab" and reference.train_keys == ["pred", "reac"]
        )
        assert result.promoted == (
            reference.state == "stab"
            and reference.train_keys == ["pred", "reac"]
            and reference.models["pred"] == "reac"
        )
        assert result.reset_roles == (
            ("reac",)
            if result.entered_reactive
            else ("stab",)
            if before == "reac" and reference.state == "stab"
            else ()
        )
        for key in reference.train_keys:
            reference.models[key] = key


def test_timestep_store_retains_ten_and_orders_window():
    store = TimestepTrainingSampleStore()
    from federated_learning_experiments.learning.training.model_training_sample_records import (
        ObservedTrainingSample,
    )

    samples = tuple(
        ObservedTrainingSample(
            input_features=torch.tensor([[float(t)]]), observed_class_labels=torch.tensor([[0.0]])
        )
        for t in range(15)
    )
    for t in range(15):
        store.append_timestep(timestep_index=t, training_samples=(samples[t],))
    assert store.timestep_indices == tuple(range(5, 15))
    assert store.samples_for_window(timestep_indices=(14, 5, 10)) == (
        samples[5],
        samples[10],
        samples[14],
    )
    with pytest.raises(ValueError):
        store.samples_for_window(timestep_indices=(4,))


def test_configuration_and_default_serialization():
    actual = settings(configuration=CONFIGURATION)
    base = settings(configuration="feddrift")
    participant = base.run_participant_settings
    assert actual == replace(
        base,
        run_participant_settings=replace(
            participant,
            run_client_settings=replace(
                participant.run_client_settings,
                model_assignment_policy="driftsurf_global_role_assignment",
                training_sample_window="role_timestep_window",
                driftsurf_settings=DriftSurfSettings(),
            ),
        ),
    )
    saved = convert_settings_to_plain_mapping(actual)["run_participant_settings"][
        "run_client_settings"
    ]
    assert saved["driftsurf_settings"] == {"settings_type": "DriftSurfSettings"}
    assert (
        "driftsurf_settings"
        not in convert_settings_to_plain_mapping(base)["run_participant_settings"][
            "run_client_settings"
        ]
    )
    assert (
        make_run_identifier(configuration_name="feddrift", run_settings=base)
        == "sine2_interval50_seed0_1fbf192c89f9"
    )


@pytest.mark.parametrize("delta", (0.0, -0.1, float("nan"), float("inf")))
def test_invalid_threshold(delta):
    with pytest.raises(ValueError):
        DriftSurfSettings(accuracy_drop_threshold=delta)
    with pytest.raises(ValueError):
        replace(
            plan(),
            configuration_names=(CONFIGURATION,),
            driftsurf_accuracy_drop_thresholds=(delta,),
        )


def test_sweep_threshold_identity_and_inactive_rejection():
    with pytest.raises(ValueError):
        replace(plan(), driftsurf_accuracy_drop_thresholds=(0.1,))
    runs = build_planned_experiment_runs(
        sweep_plan=replace(
            plan(),
            configuration_names=(CONFIGURATION, "feddrift"),
            random_seeds=(0,),
            server_aggregation_intervals_per_client_samples=(50,),
            driftsurf_accuracy_drop_thresholds=(0.1, 0.02),
        )
    )
    assert len(runs) == 3
    assert len({r.run_identifier for r in runs}) == 3


@pytest.mark.slow
@pytest.mark.parametrize("dataset,delta", (("sine2", 0.1), ("mnist4", 0.1), ("sine2", 0.02)))
def test_small_full_run(monkeypatch, dataset, delta):
    reference = OfficialDriftSurfState(delta=delta)
    prepare = FeddriftRunServer.prepare_timestep
    train = FeddriftRunClient.train_local_round
    aggregate = FeddriftRunServer.aggregate_and_distribute_training_round
    initial = {}
    predictions = {}
    first_training_seen = set()
    checked = {"scores": 0, "windows": 0, "initial": 0, "entries": 0, "promotions": 0}

    def inspect_prepare(server, *, timestep_index):
        state = server._driftsurf_state
        roles = server._driftsurf_model_ids.copy()
        if timestep_index == 0:
            for client in server.run_clients:
                initial[client.client_id] = copy.deepcopy(
                    client.owners.held_model_training_state_registry.get_held_model_training_state(
                        model_id=0
                    ).classifier.state_dict()
                )
        predictions[timestep_index] = roles[state.model_key]
        scores = {}
        for role in state.scoring_roles if timestep_index else ():
            classifiers = []
            features = []
            labels = []
            for client in server.run_clients:
                classifier = (
                    client.owners.held_model_training_state_registry.get_held_model_training_state(
                        model_id=roles[role]
                    ).classifier
                )
                classifier.eval()
                classifiers.append(classifier)
                samples = client.stream_processor._driftsurf_pending_samples
                features.extend(s.input_features for s in samples)
                labels.extend(s.observed_class_labels for s in samples)
            for classifier in classifiers[1:]:
                for k, v in classifier.state_dict().items():
                    torch.testing.assert_close(v, classifiers[0].state_dict()[k], rtol=0, atol=0)
            with torch.no_grad():
                scores_tensor = classifiers[0](torch.cat(features))
                if classifiers[0].class_count == 2:
                    scores_tensor = torch.cat((1 - scores_tensor, scores_tensor), dim=-1)
                predicted = scores_tensor.argmax(-1).view(-1)
            scores[role] = float((predicted == torch.cat(labels).view(-1)).sum().item()) / len(
                features
            )
        for client in server.run_clients:
            collections = (
                client.owners.training_sample_store.snapshot_ordered_model_training_samples()
            )
            for collection in collections:
                assert not {id(sample) for sample in collection.training_samples} & {
                    id(sample) for sample in client.stream_processor._driftsurf_pending_samples
                }
        prepare(server, timestep_index=timestep_index)
        record = server._driftsurf_records[-1]
        for role, value in scores.items():
            assert record["accuracy_" + role] == value
            checked["scores"] += 1
        if timestep_index:
            with redirect_stdout(StringIO()):
                reference.run_ds_algo(scores, None, timestep_index)
            for name in (
                "state",
                "train_keys",
                "model_key",
                "train_data_dict",
                "acc_best",
                "reac_ctr",
            ):
                assert getattr(state, name) == getattr(reference, name)
        checked["entries"] += record["entered_reactive"]
        checked["promotions"] += record["promoted"]
        for role in reference.train_keys:
            reference.models[role] = role

    def inspect_aggregation(server, *, timestep_index):
        expected_weights = {}
        for client in server.run_clients:
            for (
                collection
            ) in client.owners.training_sample_store.snapshot_ordered_model_training_samples():
                expected_weights[collection.model_id] = expected_weights.get(
                    collection.model_id, 0
                ) + len(collection.training_samples)
        aggregate(server, timestep_index=timestep_index)
        for mid, weight in expected_weights.items():
            assert server.snapshot_model_weights()[mid] == weight
        # 古いIDはサーバと全clientの実体から除かれ、新IDは全clientで同じ。
        for client in server.run_clients:
            assert client.get_held_model_ids() == set(
                server.global_model_repository.global_model_ids
            )

    def inspect_train(client, *, eligible_model_ids=None):
        server_state = active_server[0]
        state = server_state._driftsurf_state
        roles = server_state._driftsurf_model_ids
        collections = client.owners.training_sample_store.snapshot_ordered_model_training_samples()
        assert {c.model_id for c in collections} == set(eligible_model_ids)
        for collection in collections:
            role = next(r for r, m in roles.items() if m == collection.model_id)
            expected = client.stream_processor._driftsurf_timestep_samples.samples_for_window(
                timestep_indices=tuple(state.train_data_dict[role])
            )
            assert collection.training_samples == expected
            assert len(expected) == 50 * len(state.train_data_dict[role])
            checked["windows"] += 1
            if (client.client_id, collection.model_id) not in first_training_seen:
                classifier = (
                    client.owners.held_model_training_state_registry.get_held_model_training_state(
                        model_id=collection.model_id
                    ).classifier
                )
                for k, v in classifier.state_dict().items():
                    torch.testing.assert_close(v, initial[client.client_id][k], rtol=0, atol=0)
                checked["initial"] += 1
                first_training_seen.add((client.client_id, collection.model_id))
        return train(client, eligible_model_ids=eligible_model_ids)

    active_server = []

    def capture_prepare(server, *, timestep_index):
        active_server[:] = [server]
        inspect_prepare(server, timestep_index=timestep_index)

    monkeypatch.setattr(FeddriftRunServer, "prepare_timestep", capture_prepare)
    monkeypatch.setattr(FeddriftRunClient, "train_local_round", inspect_train)
    monkeypatch.setattr(
        FeddriftRunServer, "aggregate_and_distribute_training_round", inspect_aggregation
    )
    configured = settings(dataset, CONFIGURATION)
    participant = configured.run_participant_settings
    configured = replace(
        configured,
        run_participant_settings=replace(
            participant,
            run_client_settings=replace(
                participant.run_client_settings,
                driftsurf_settings=DriftSurfSettings(accuracy_drop_threshold=delta),
            ),
        ),
    )
    run = execute_feddrift_experiment_run(run_settings=configured)
    tables = build_feddrift_run_record_tables(experiment_run=run)
    concepts = tables["sample_prediction.observed_concept_id"]
    changes = int(np.count_nonzero(concepts[:, 1:] != concepts[:, :-1]))
    assert changes > 0
    for t, mid in predictions.items():
        assert np.all(
            tables["sample_prediction.prediction_model_id"][:, t * 50 : (t + 1) * 50] == mid
        )
    assert np.all(tables["server_timestep.global_model_count_after_synchronization"] <= 3)
    assert len(tables["model_clustering.model_id"]) == 0
    assert len(tables["detection_batch.client_id"]) == 0
    assert len(tables["driftsurf.timestep_index"]) == 50
    streams = run.measured_run.run_result.observed_client_streams
    for client in active_server[0].run_clients:
        store = client.stream_processor._driftsurf_timestep_samples
        for t in store.timestep_indices:
            for offset, sample in enumerate(store.samples_for_window(timestep_indices=(t,))):
                observed = streams[client.client_id].observed_samples[t * 50 + offset]
                assert sample.input_features.tolist() == [list(observed.feature_values)]
                assert sample.observed_class_labels.item() == observed.class_label
    print(
        f"DRIFTSURF {dataset} delta={delta} changes={changes} checks={checked} models<=3 merges=0 prediction=pass"
    )
    # 固定3 runを合わせた受入はentries/promotionsから確認する。


def test_binary_correct_counts_expand_probabilities_and_tie_to_zero():
    from types import SimpleNamespace

    from federated_learning_experiments.learning.training.model_training_sample_records import (
        ObservedTrainingSample,
    )
    from federated_learning_experiments.runtime.driftsurf_timestep_preparation import (
        count_driftsurf_role_correct_predictions,
    )

    classifier = torch.nn.Identity()
    classifier.class_count = 2
    samples = [
        ObservedTrainingSample(
            input_features=torch.tensor([[p]]),
            observed_class_labels=torch.tensor([[label]], dtype=torch.float32),
        )
        for p, label in ((0.8, 1), (0.2, 0), (0.5, 0))
    ]
    client = SimpleNamespace(
        stream_processor=SimpleNamespace(_driftsurf_pending_samples=samples),
        owners=SimpleNamespace(
            held_model_training_state_registry=SimpleNamespace(
                get_held_model_training_state=lambda **kwargs: SimpleNamespace(
                    classifier=classifier
                )
            )
        ),
    )
    assert count_driftsurf_role_correct_predictions(run_client=client, model_id=0) == (3, 3)
    assert classifier.training


def test_explicit_default_threshold_is_same_run_and_invalid_combinations():
    configured = settings(configuration=CONFIGURATION)
    client = configured.run_participant_settings.run_client_settings
    for changes in (
        {"driftsurf_settings": None},
        {"training_sample_window": "current_timestep"},
        {"cluster_model_initialization": "all_random"},
    ):
        with pytest.raises(ValueError):
            replace(client, **changes)
    runs = build_planned_experiment_runs(
        sweep_plan=replace(
            plan(),
            configuration_names=(CONFIGURATION,),
            random_seeds=(0,),
            client_count=10,
            per_client_sample_count=2500,
            server_aggregation_intervals_per_client_samples=(50,),
            driftsurf_accuracy_drop_thresholds=(0.1,),
        )
    )
    assert runs[0].run_settings == configured
    assert runs[0].run_identifier == make_run_identifier(
        configuration_name=CONFIGURATION, run_settings=configured
    )


def test_promotion_uses_official_mean_rounding_at_float_boundary():
    state = DriftSurfState(settings=DriftSurfSettings())
    state.advance(timestep_index=1, accuracies={"pred": 0.1, "stab": 0.9})
    state.advance(
        timestep_index=2, accuracies={"pred": 0.01, "reac": float(np.nextafter(0.01, 1.0))}
    )
    result = state.advance(timestep_index=3, accuracies={"pred": 0.02, "reac": 0.02})
    # 和には差があるが、公式の長さ3の平均はどちらも0.01となる。
    assert not result.promoted


def test_sweep_command_connects_threshold_values_to_plan():
    from federated_learning_experiments.sweep.experiment_sweep_command import (
        _build_argument_parser,
        _build_sweep_plan,
    )

    arguments = _build_argument_parser().parse_args(
        [
            "--configurations",
            CONFIGURATION,
            "--datasets",
            "sine2",
            "--seeds",
            "0",
            "--aggregation-intervals",
            "50",
            "--driftsurf-accuracy-drop-thresholds",
            "0.02",
            "0.10",
        ]
    )
    sweep_plan = _build_sweep_plan(arguments)
    assert sweep_plan.driftsurf_accuracy_drop_thresholds == (0.02, 0.10)
    runs = build_planned_experiment_runs(sweep_plan=sweep_plan)
    assert [
        r.run_settings.run_participant_settings.run_client_settings.driftsurf_settings.accuracy_drop_threshold
        for r in runs
    ] == [0.02, 0.10]
    # parserの属性名はconfigurationsで、構成を外せば同じ値でも拒否する。
    arguments.configurations = ["feddrift"]
    with pytest.raises(ValueError):
        _build_sweep_plan(arguments)
