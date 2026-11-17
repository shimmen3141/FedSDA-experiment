"""構成`fedsda-soft-training-own-assigned-models`（柔らかい学習の責任度の対象を、そのclientが自分で固く割り当てた
標本を持つモデルと、いま割り当てる先に限った構成）を、設定・読取り・提供の関数・run・掃引の順に確かめる。"""

import json
from dataclasses import replace

import pytest
from test_fedsda_soft_training_assignment_count_aggregation_weights import (
    client_settings,
    make_store,
    observed_samples,
)

from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)

NAME = "fedsda-soft-training-own-assigned-models"
OWN = "models_with_own_assigned_samples"
ALL = "all_held_models"


# 設定と読取り（task 1）


def test_default_scope_is_all_held_models_and_is_not_saved():
    for name in ("fedsda", "fedsda-soft-training", "fedsda-training-mass-diagnostics"):
        settings = client_settings(name)
        assert settings.training_responsibility_model_scope == ALL
        assert "training_responsibility_model_scope" not in convert_settings_to_plain_mapping(
            settings
        )


def test_own_scope_is_accepted_with_soft_training_and_saved():
    changed = replace(
        client_settings("fedsda-soft-training"), training_responsibility_model_scope=OWN
    )
    assert changed.training_responsibility_model_scope == OWN
    assert convert_settings_to_plain_mapping(changed)["training_responsibility_model_scope"] == OWN


@pytest.mark.parametrize("value", ["unknown", True, None, 1])
def test_invalid_scope_values_are_rejected(value):
    with pytest.raises(RunSettingsValidationError):
        replace(client_settings("fedsda-soft-training"), training_responsibility_model_scope=value)


@pytest.mark.parametrize(
    "name",
    ["fedsda", "fedsda-training-mass-diagnostics", "fedsda-soft-training-single-model-prediction"],
)
def test_own_scope_is_rejected_without_soft_training_sharing_prediction_weights(name):
    with pytest.raises(RunSettingsValidationError):
        replace(client_settings(name), training_responsibility_model_scope=OWN)


def test_store_returns_only_model_ids_with_assigned_samples():
    store = make_store(None, with_responsibility_store=False)
    assert store.snapshot_model_ids_with_assigned_samples() == frozenset()
    store.append_model_training_samples(model_id=3, training_samples=())
    store.append_model_training_samples(model_id=5, training_samples=observed_samples(2))
    store.append_model_training_samples(model_id=-1, training_samples=observed_samples(1))
    ids = store.snapshot_model_ids_with_assigned_samples()
    assert ids == frozenset({5, -1})
    assert type(ids) is frozenset
    store.reassign_model_training_samples_id(original_model_id=-1, reassigned_model_id=7)
    assert store.snapshot_model_ids_with_assigned_samples() == frozenset({5, 7})
    store.remap_model_training_sample_collections(model_id_mapping={5: 7, 3: 9})
    assert store.snapshot_model_ids_with_assigned_samples() == frozenset({7})


# 責任度の提供の関数（task 2）


def build_three_model_provider_inputs(monkeypatch):
    """既存の`test_release_responsibilities_use_fresh_local_parameters_and_new_pool`と同じ組立てで、
    真のラベルの確率が0.8・0.2・0.5の3モデルを用意する。"""
    from copy import deepcopy
    from math import log
    from types import SimpleNamespace

    import torch
    from test_joint_model_parameter_update import build_joint_update_oracle_pair

    from federated_learning_experiments.learning.training.model_training_sample_records import (
        ObservedTrainingSample,
    )

    batches, _, _ = build_joint_update_oracle_pair(
        class_count=2,
        batch_sample_counts=(4, 4),
        optimizer_variant="sgd",
        monkeypatch=monkeypatch,
    )
    states = [SimpleNamespace(model_id=m, classifier=b.classifier) for m, b in enumerate(batches)]
    states.append(SimpleNamespace(model_id=2, classifier=deepcopy(states[1].classifier)))
    for state, probability in zip(states, (0.8, 0.2, 0.5), strict=True):
        with torch.no_grad():
            state.classifier.classification_layer.weight.zero_()
            state.classifier.classification_layer.bias.fill_(log(probability / (1 - probability)))
    registry = SimpleNamespace(snapshot_ordered_held_model_training_states=lambda: tuple(states))
    observation = ObservedTrainingSample(
        input_features=torch.tensor([[0.0, 0.0]]), observed_class_labels=torch.tensor([[1.0]])
    )
    return registry, observation


def new_controller():
    from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights import (
        FixedShareWeightController,
    )

    return FixedShareWeightController(
        prediction_combination_settings=client_settings("fedsda").prediction_combination_settings
    )


def build_provider(registry, controller, store=None):
    from federated_learning_experiments.runtime.released_sample_training_responsibilities import (
        build_released_sample_responsibility_provider,
    )

    if store is None:
        return build_released_sample_responsibility_provider(
            registry=registry, weight_controller=controller
        )
    return build_released_sample_responsibility_provider(
        registry=registry, weight_controller=controller, own_assignment_sample_store=store
    )


def own_store(model_ids):
    store = make_store(None, with_responsibility_store=False)
    for model_id in model_ids:
        store.append_model_training_samples(model_id=model_id, training_samples=observed_samples(1))
    return store


def test_restricted_responsibilities_renormalize_full_values_over_own_models(monkeypatch):
    registry, observation = build_three_model_provider_inputs(monkeypatch)
    full_controller, own_controller = new_controller(), new_controller()
    full = build_provider(registry, full_controller)(0, observation)
    restricted = build_provider(registry, own_controller, own_store((0, 1)))(0, observation)
    # (a) 対象外のモデル2は辞書に入らない。
    assert set(restricted) == {0, 1}
    # (b) 全保有モデルで求めた値を、対象の中で正規化し直した値。
    total = full[0] + full[1]
    assert restricted == pytest.approx({0: full[0] / total, 1: full[1] / total}, abs=1e-12)
    assert restricted == pytest.approx({0: 0.8, 1: 0.2}, abs=1e-6)
    # (c) Fixed-Shareの重みは全保有モデルで同期され、制限なしと同じ状態。
    assert own_controller.weights_by_model_id == full_controller.weights_by_model_id
    assert set(own_controller.weights_by_model_id) == {0, 1, 2}


def test_first_assignment_target_is_always_included(monkeypatch):
    registry, observation = build_three_model_provider_inputs(monkeypatch)
    # (d) モデル2は標本を持たないが、いま割り当てる先なので含まれる。
    restricted = build_provider(registry, new_controller(), own_store((0,)))(2, observation)
    assert set(restricted) == {0, 2}
    assert restricted == pytest.approx({0: 0.8 / 1.3, 2: 0.5 / 1.3}, abs=1e-6)
    only_target = build_provider(registry, new_controller(), own_store(()))(1, observation)
    assert only_target == {1: 1.0}


def test_all_held_models_own_gives_identical_responsibilities(monkeypatch):
    registry, observation = build_three_model_provider_inputs(monkeypatch)
    full = build_provider(registry, new_controller())(1, observation)
    # (e) 全保有モデルが対象なら、制限なしと==で同じ（保有していないIDが集合にあっても引かない）。
    restricted = build_provider(registry, new_controller(), own_store((0, 1, 2, 99)))(
        1, observation
    )
    assert restricted == full
    assert list(restricted) == list(full)


@pytest.mark.parametrize("store", [object(), {}, 0])
def test_invalid_store_type_is_rejected_when_building(monkeypatch, store):
    registry, _ = build_three_model_provider_inputs(monkeypatch)
    from federated_learning_experiments.runtime.released_sample_training_responsibilities import (
        build_released_sample_responsibility_provider,
    )

    # (f) 関数を作る時点で拒否する。
    with pytest.raises(TypeError, match="exact ModelTrainingSampleStore"):
        build_released_sample_responsibility_provider(
            registry=registry, weight_controller=new_controller(), own_assignment_sample_store=store
        )


# 構成の登録と掃引（task 3）


def test_configuration_differs_from_soft_training_only_in_the_model_scope():
    from test_fedsda_soft_training_assignment_count_aggregation_weights import build_settings

    from federated_learning_experiments.sweep import experiment_configurations as configurations

    assert configurations.SOFT_TRAINING_OWN_ASSIGNED_MODELS_CONFIGURATION_NAME == NAME
    assert configurations.EXPERIMENT_CONFIGURATION_NAMES[-2:] == (
        "fedsda-soft-training-assignment-count-aggregation-weights",
        NAME,
    )
    for dataset in ("sine2", "mnist4"):
        soft = build_settings("fedsda-soft-training", dataset=dataset)
        participants = soft.run_participant_settings
        assert build_settings(NAME, dataset=dataset) == replace(
            soft,
            run_participant_settings=replace(
                participants,
                run_client_settings=replace(
                    participants.run_client_settings, training_responsibility_model_scope=OWN
                ),
            ),
        )
    mapping = convert_settings_to_plain_mapping(client_settings(NAME))
    assert mapping["training_responsibility_model_scope"] == OWN
    assert mapping["training_assignment_strategy"] == "responsibility_weighted_training"
    assert mapping["record_training_mass_by_concept"] is True
    assert "model_aggregation_weight_source" not in mapping


def test_configuration_is_protected_by_this_test_file():
    from tests.refactoring import behavior_golden_cases as golden

    assert golden.CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN[NAME] == (
        "tests/refactoring/test_fedsda_soft_training_own_assigned_models.py"
    )


def test_plan_adds_runs_and_keeps_existing_identifiers():
    from federated_learning_experiments.sweep.experiment_sweep_plan import (
        ExperimentSweepPlan,
        build_planned_experiment_runs,
    )

    plan = dict(
        dataset_names=("sine2", "mnist4"),
        random_seeds=(0, 1),
        client_count=2,
        per_client_sample_count=100,
        server_aggregation_intervals_per_client_samples=(50, 25),
    )
    existing = (
        "fedsda",
        "fedsda-soft-training",
        "fedsda-training-mass-diagnostics",
        "fedsda-soft-training-assignment-count-aggregation-weights",
    )
    with_new = build_planned_experiment_runs(
        sweep_plan=ExperimentSweepPlan(configuration_names=(*existing, NAME), **plan)
    )
    without = build_planned_experiment_runs(
        sweep_plan=ExperimentSweepPlan(configuration_names=existing, **plan)
    )
    assert len([r for r in with_new if r.configuration_name == NAME]) == 8
    assert [r.run_identifier for r in with_new if r.configuration_name != NAME] == [
        r.run_identifier for r in without
    ]
    assert len({r.run_identifier for r in with_new}) == len(with_new)


@pytest.mark.slow
def test_real_sweep_saves_lists_resumes_and_refuses_conflict(tmp_path):
    import csv

    from run_provenance_examples import TEST_RUN_PROVENANCE

    from federated_learning_experiments.sweep.experiment_sweep_execution import (
        execute_experiment_sweep,
        find_existing_run_state,
    )
    from federated_learning_experiments.sweep.experiment_sweep_plan import (
        ExperimentSweepPlan,
        build_planned_experiment_runs,
    )
    from federated_learning_experiments.sweep.experiment_sweep_records import (
        write_run_summary_table,
    )

    runs = build_planned_experiment_runs(
        sweep_plan=ExperimentSweepPlan(
            configuration_names=(NAME,),
            dataset_names=("sine2",),
            random_seeds=(0, 1),
            client_count=2,
            per_client_sample_count=120,
            server_aggregation_intervals_per_client_samples=(40,),
        )
    )

    def execute():
        return execute_experiment_sweep(
            planned_runs=runs,
            output_directory=tmp_path,
            worker_count=1,
            run_provenance=TEST_RUN_PROVENANCE,
        )

    assert [o.run_status for o in execute()] == ["completed", "completed"]
    assert [o.run_status for o in execute()] == ["skipped_existing"] * 2
    csv_path = write_run_summary_table(
        configuration_directory=tmp_path / NAME, configuration_name=NAME
    ).table_file_path
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and {r["configuration_name"] for r in rows} == {NAME}
    assert all(0.0 <= float(r["prediction_accuracy"]) <= 1.0 for r in rows)
    path = tmp_path / NAME / "runs" / (runs[0].run_identifier + ".json")
    saved = json.loads(path.read_text("utf-8"))
    assert f'"training_responsibility_model_scope": "{OWN}"' in json.dumps(saved, sort_keys=True)
    (tmp_path / NAME / "runs" / (runs[1].run_identifier + ".npz")).unlink()
    assert (
        find_existing_run_state(planned_run=runs[1], output_directory=tmp_path).run_state
        == "conflicting"
    )
    assert execute()[1].run_status == "failed"


# runの挙動（task 4）


def run_with_traced_responsibilities(name, dataset, monkeypatch, count, interval):
    """各clientの責任度の提供の関数を包み、呼出しごとの対象・保有・返した責任度を記録してrunする。"""
    from test_fedsda_soft_training_assignment_count_aggregation_weights import build_settings

    from federated_learning_experiments.learning.training.model_training_sample_store import (
        ModelTrainingSampleStore,
    )
    from federated_learning_experiments.runtime import fedsda_run_client as client_module
    from federated_learning_experiments.runtime.fedsda_experiment_run import (
        execute_fedsda_experiment_run,
    )
    from federated_learning_experiments.runtime.fedsda_run_record_tables import (
        build_fedsda_run_record_tables,
    )

    build = client_module.build_released_sample_responsibility_provider
    append = ModelTrainingSampleStore.append_model_training_samples
    reassign = ModelTrainingSampleStore.reassign_model_training_samples_id
    remap = ModelTrainingSampleStore.remap_model_training_sample_collections
    traces = []
    # 全clientを通した呼出しの順番（clientの処理は同期を挟んで交互に進む）。
    order = [0]
    # storeごとに、固い割り当ての先として標本を足したIDを、付け替え・統合に従って移した集合。
    assigned_ids = {}

    def tracked_append(self, **kwargs):
        append(self, **kwargs)
        if kwargs["training_samples"]:
            assigned_ids.setdefault(id(self), set()).add(kwargs["model_id"])

    def tracked_reassign(self, **kwargs):
        ids = assigned_ids.setdefault(id(self), set())
        if kwargs["original_model_id"] in ids:
            ids.discard(kwargs["original_model_id"])
            ids.add(kwargs["reassigned_model_id"])
        return reassign(self, **kwargs)

    def tracked_remap(self, **kwargs):
        mapping = kwargs["model_id_mapping"]
        ids = assigned_ids.setdefault(id(self), set())
        assigned_ids[id(self)] = {mapping.get(m, m) for m in ids}
        return remap(self, **kwargs)

    def traced_build(**kwargs):
        provider = build(**kwargs)
        registry = kwargs["registry"]
        store = kwargs.get("own_assignment_sample_store")
        calls = []
        traces.append(calls)

        def evaluate(fallback_model_id, sample):
            held = {s.model_id for s in registry.snapshot_ordered_held_model_training_states()}
            own = None if store is None else store.snapshot_model_ids_with_assigned_samples()
            tracked = None if store is None else frozenset(assigned_ids.get(id(store), set()))
            responsibilities = provider(fallback_model_id, sample)
            order[0] += 1
            calls.append(
                dict(
                    order=order[0],
                    fallback=fallback_model_id,
                    held=held,
                    own=own,
                    tracked=tracked,
                    responsibilities=dict(responsibilities),
                )
            )
            return responsibilities

        return evaluate

    with monkeypatch.context() as patch:
        patch.setattr(client_module, "build_released_sample_responsibility_provider", traced_build)
        patch.setattr(ModelTrainingSampleStore, "append_model_training_samples", tracked_append)
        patch.setattr(
            ModelTrainingSampleStore, "reassign_model_training_samples_id", tracked_reassign
        )
        patch.setattr(
            ModelTrainingSampleStore, "remap_model_training_sample_collections", tracked_remap
        )
        run = execute_fedsda_experiment_run(
            run_settings=build_settings(name, dataset=dataset, count=count, interval=interval)
        )
    return build_fedsda_run_record_tables(experiment_run=run), traces


@pytest.mark.slow
@pytest.mark.parametrize("dataset,count", [("sine2", 1500), ("mnist4", 800)])
def test_run_restricts_responsibilities_to_own_models_and_matches_soft_until_restriction(
    dataset, count, monkeypatch
):
    interval = 50
    tables, traces = run_with_traced_responsibilities(NAME, dataset, monkeypatch, count, interval)
    _, soft_traces = run_with_traced_responsibilities(
        "fedsda-soft-training", dataset, monkeypatch, count, interval
    )
    assert len(traces) == len(soft_traces) == 2
    restricted_calls = 0
    matched_prefix = []
    for calls in traces:
        assert calls
        for call in calls:
            eligible = call["own"] | {call["fallback"]}
            # (a) 返した責任度のキーは、対象（標本を持つモデルといま割り当てる先）の中の保有モデル。
            assert set(call["responsibilities"]) == call["held"] & eligible
            # (c2) 標本を持つモデルは、それまでに割り当ての先として標本を足したIDを付け替え・統合で移した集合。
            assert call["own"] == call["tracked"]
            restricted_calls += bool(call["held"] - eligible)
    # (c) どれかのclientで対象外の保有モデルが初めて生じる呼出しより前は、柔らかい学習と同じ責任度。
    # 1つのclientの制限は、集約を通じて他のclientの状態も変えるので、全clientを通した順番で区切る。
    first_order = min(
        (c["order"] for calls in traces for c in calls if c["held"] - (c["own"] | {c["fallback"]})),
        default=None,
    )
    assert first_order is not None and first_order > 1
    for calls, soft_calls in zip(traces, soft_traces, strict=True):
        prefix = [c for c in calls if c["order"] < first_order]
        soft_prefix = [c for c in soft_calls if c["order"] < first_order]
        assert len(prefix) == len(soft_prefix)
        for mine, theirs in zip(prefix, soft_prefix, strict=True):
            assert mine["fallback"] == theirs["fallback"]
            assert mine["held"] == theirs["held"]
            assert mine["responsibilities"] == theirs["responsibilities"]
        matched_prefix.append(len(prefix))
    assert sum(matched_prefix) > 0
    # (b) 制限が実際に効いている。
    assert restricted_calls > 0
    # (d) 学習量のコンセプト別の記録がある。
    assert len(tables["training_mass_by_concept.training_mass"]) > 0
    print(
        json.dumps(
            dict(
                dataset=dataset,
                calls=[len(c) for c in traces],
                restricted_calls=restricted_calls,
                matched_prefix=matched_prefix,
            )
        )
    )
