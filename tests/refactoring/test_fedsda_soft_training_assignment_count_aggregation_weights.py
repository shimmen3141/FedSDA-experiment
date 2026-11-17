"""構成`fedsda-soft-training-assignment-count-aggregation-weights`（柔らかい学習の集約の重みと参加の判定だけを、
固い割り当ての標本の件数に替えた構成）を、設定・重みの関数・run・掃引の順に確かめる。"""

import json
from dataclasses import replace

import pytest

from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)

NAME = "fedsda-soft-training-assignment-count-aggregation-weights"
COUNTS = "single_model_assignment_sample_counts"
AMOUNTS = "training_assignment_amounts"


def build_settings(name, dataset="sine2", count=300, client_count=2, interval=50):
    from federated_learning_experiments.configuration.experiment_run_conditions import (
        ExperimentRunConditions,
    )
    from federated_learning_experiments.sweep.experiment_configurations import (
        build_experiment_run_settings,
    )

    return build_experiment_run_settings(
        configuration_name=name,
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name=dataset,
            random_seed=0,
            client_count=client_count,
            per_client_sample_count=count,
            server_aggregation_interval_per_client_samples=interval,
        ),
    )


def client_settings(name):
    return build_settings(name).run_participant_settings.run_client_settings


# 設定（task 1）


def test_default_source_is_current_amounts_and_is_not_saved():
    for name in ("fedsda", "fedsda-soft-training", "fedsda-training-mass-diagnostics"):
        settings = client_settings(name)
        assert settings.model_aggregation_weight_source == AMOUNTS
        assert "model_aggregation_weight_source" not in convert_settings_to_plain_mapping(settings)


def test_counts_are_accepted_with_soft_training_and_saved():
    soft = client_settings("fedsda-soft-training")
    changed = replace(soft, model_aggregation_weight_source=COUNTS)
    assert changed.model_aggregation_weight_source == COUNTS
    assert convert_settings_to_plain_mapping(changed)["model_aggregation_weight_source"] == COUNTS


@pytest.mark.parametrize("value", ["unknown", True, None, 1])
def test_invalid_source_values_are_rejected(value):
    with pytest.raises(RunSettingsValidationError):
        replace(client_settings("fedsda-soft-training"), model_aggregation_weight_source=value)


@pytest.mark.parametrize("name", ["fedsda", "fedsda-training-mass-diagnostics"])
def test_counts_are_rejected_without_soft_training(name):
    with pytest.raises(RunSettingsValidationError):
        replace(client_settings(name), model_aggregation_weight_source=COUNTS)


def test_client_exposes_the_configured_source():
    from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient

    assert isinstance(FedsdaRunClient.model_aggregation_weight_source, property)
    assert FedsdaRunClient.model_aggregation_weight_source.fset is None
    client = FedsdaRunClient.__new__(FedsdaRunClient)
    client._run_client_settings = replace(
        client_settings("fedsda-soft-training"), model_aggregation_weight_source=COUNTS
    )
    assert client.model_aggregation_weight_source == COUNTS


# 重みを決める関数（task 2）


def make_store(responsibility_provider, *, with_responsibility_store=True):
    from federated_learning_experiments.learning.training.model_training_sample_store import (
        ModelTrainingSampleStore,
    )
    from federated_learning_experiments.learning.training.responsibility_weighted_training import (
        ResponsibilityWeightedTrainingStore,
    )

    store = ModelTrainingSampleStore()
    if with_responsibility_store:
        store.responsibility_training_store = ResponsibilityWeightedTrainingStore()
    store.training_responsibility_provider = responsibility_provider
    return store


def observed_samples(count):
    import torch

    from federated_learning_experiments.learning.training.model_training_sample_records import (
        ObservedTrainingSample,
    )

    return tuple(
        ObservedTrainingSample(
            input_features=torch.tensor([[float(i)]]), observed_class_labels=torch.tensor([[0.0]])
        )
        for i in range(count)
    )


def derive(store, source):
    from federated_learning_experiments.runtime.server_model_registration_and_aggregation import (
        derive_client_model_aggregation_weights,
    )

    return derive_client_model_aggregation_weights(
        training_sample_store=store, model_aggregation_weight_source=source
    )


def test_split_responsibilities_give_amounts_by_default_and_counts_when_selected():
    store = make_store(lambda model_id, sample: {3: 0.75, 5: 0.25})
    store.append_model_training_samples(model_id=3, training_samples=observed_samples(4))
    store.append_model_training_samples(model_id=5, training_samples=observed_samples(2))
    assert derive(store, AMOUNTS) == pytest.approx({3: 4.5, 5: 1.5})
    counts = derive(store, COUNTS)
    assert counts == {3: 4, 5: 2}
    assert all(type(value) is int for value in counts.values())


@pytest.mark.parametrize(
    "weight,probability",
    [(1.0, 0.3), (0.017, 0.9999), (3.7e-5, 1e-200), (0.5, 0.0), (0.0, 0.5)],
)
def test_one_model_responsibilities_make_amounts_exactly_equal_to_counts(weight, probability):
    from federated_learning_experiments.learning.training.responsibility_weighted_training import (
        calculate_training_responsibilities,
    )

    def provider(model_id, sample):
        return calculate_training_responsibilities(
            prediction_weights={model_id: weight},
            true_class_probabilities={model_id: probability},
            fallback_model_id=model_id,
        )

    store = make_store(provider)
    for _ in range(7):
        store.append_model_training_samples(model_id=2, training_samples=observed_samples(13))
    store.reassign_model_training_samples_id(original_model_id=2, reassigned_model_id=4)
    amounts, counts = derive(store, AMOUNTS), derive(store, COUNTS)
    assert counts == {4: 91}
    # 近似の比較を使わない（要求1.3）。
    assert amounts == counts
    assert all(type(value) is float for value in amounts.values())


@pytest.mark.parametrize("with_responsibility_store", [True, False])
def test_records_without_soft_assignment_give_counts_for_both_sources(with_responsibility_store):
    store = make_store(None, with_responsibility_store=with_responsibility_store)
    store.append_model_training_samples(model_id=1, training_samples=observed_samples(3))
    store.append_model_training_samples(model_id=0, training_samples=observed_samples(2))
    assert derive(store, AMOUNTS) == derive(store, COUNTS) == {1: 3, 0: 2}


@pytest.mark.parametrize("value", ["unknown", None, True, 0])
def test_invalid_source_is_rejected_before_reading_the_store(value):
    class UnreadableStore:
        def __getattr__(self, name):
            raise AssertionError(f"read {name}")

    with pytest.raises(ValueError):
        derive(UnreadableStore(), value)


# 構成の登録と掃引（task 3）


def test_configuration_differs_from_soft_training_only_in_the_weight_source():
    from federated_learning_experiments.sweep import experiment_configurations as configurations

    assert (
        configurations.SOFT_TRAINING_ASSIGNMENT_COUNT_AGGREGATION_WEIGHTS_CONFIGURATION_NAME == NAME
    )
    assert configurations.EXPERIMENT_CONFIGURATION_NAMES[-2:] == ("dwm-neural-network", NAME)
    for dataset in ("sine2", "mnist4"):
        soft = build_settings("fedsda-soft-training", dataset=dataset)
        participants = soft.run_participant_settings
        assert build_settings(NAME, dataset=dataset) == replace(
            soft,
            run_participant_settings=replace(
                participants,
                run_client_settings=replace(
                    participants.run_client_settings, model_aggregation_weight_source=COUNTS
                ),
            ),
        )
    mapping = convert_settings_to_plain_mapping(client_settings(NAME))
    assert mapping["model_aggregation_weight_source"] == COUNTS
    assert mapping["training_assignment_strategy"] == "responsibility_weighted_training"
    assert mapping["record_training_mass_by_concept"] is True


def test_configuration_is_protected_by_this_test_file():
    from tests.refactoring import behavior_golden_cases as golden

    assert golden.CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN[NAME] == (
        "tests/refactoring/test_fedsda_soft_training_assignment_count_aggregation_weights.py"
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
    existing = ("fedsda", "fedsda-soft-training", "fedsda-training-mass-diagnostics")
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
    import json

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
    assert '"model_aggregation_weight_source": "single_model_assignment_sample_counts"' in (
        json.dumps(saved, sort_keys=True)
    )
    (tmp_path / NAME / "runs" / (runs[1].run_identifier + ".npz")).unlink()
    assert (
        find_existing_run_state(planned_run=runs[1], output_directory=tmp_path).run_state
        == "conflicting"
    )
    assert execute()[1].run_status == "failed"


# runの挙動（task 4）


def capture_state_at_first_aggregation(run_clients):
    import random

    import numpy as np
    import torch

    return {
        "torch": torch.get_rng_state().clone(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "parameters": [
            {
                state.model_id: {
                    name: value.detach().clone()
                    for name, value in state.classifier.state_dict().items()
                }
                for state in client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
            }
            for client in run_clients
        ],
    }


def run_with_checked_aggregation(name, dataset, monkeypatch, count, interval):
    import torch

    from federated_learning_experiments.evaluation.communication_volume_record_store import (
        CommunicationVolumeRecordStore,
    )
    from federated_learning_experiments.runtime import (
        server_model_registration_and_aggregation as aggregation_parameters,
    )
    from federated_learning_experiments.runtime import server_round_synchronization as aggregation
    from federated_learning_experiments.runtime.fedsda_experiment_run import (
        execute_fedsda_experiment_run,
    )
    from federated_learning_experiments.runtime.fedsda_run_record_tables import (
        build_fedsda_run_record_tables,
    )

    aggregate = aggregation.aggregate_client_models_into_global_models
    add_parameters = aggregation_parameters._add_weighted_parameters
    record_transfers = CommunicationVolumeRecordStore.record_model_transfers
    split = aggregation_parameters.split_shared_and_concept_specific_parameters
    applied_weights, uploads = [], []
    evidence = {"calls": 0, "counts_differ_from_amounts": 0, "kept_without_counts": 0}

    def checked_weights(**kwargs):
        applied_weights.append(kwargs["sample_count"])
        return add_parameters(**kwargs)

    def checked_transfers(self, **kwargs):
        if kwargs["transfer_direction"] == "upload":
            uploads.append(kwargs["model_count"])
        return record_transfers(self, **kwargs)

    def checked_aggregate(**kwargs):
        clients = kwargs["run_clients"]
        repository = kwargs["global_model_repository"]
        if evidence["calls"] == 0:
            evidence["first"] = capture_state_at_first_aggregation(clients)
        held = [
            {
                state.model_id
                for state in client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
            }
            for client in clients
        ]
        target_ids = sorted({m for models in held for m in models if m >= 0})
        expected_weights, expected_uploads = [], 0
        weight_totals = dict.fromkeys(target_ids, 0)
        for client, models in zip(clients, held, strict=True):
            store = client.owners.training_sample_store
            counts = {
                collection.model_id: len(collection.training_samples)
                for collection in store.snapshot_ordered_model_training_samples()
            }
            amounts = store.responsibility_training_store.training_masses_by_model_id
            evidence["counts_differ_from_amounts"] += any(
                counts.get(m, 0) != amounts.get(m, 0.0) for m in models if m >= 0
            )
            participating = [counts[m] for m in target_ids if m in models and counts.get(m, 0) > 0]
            for m in target_ids:
                if m in models:
                    weight_totals[m] += counts.get(m, 0)
            if participating:
                expected_weights.extend([sum(participating), *participating])
                expected_uploads += len(participating)
        without_counts = [
            m for m in target_ids if weight_totals[m] == 0 and m in repository.global_model_ids
        ]
        before = {m: repository.get_global_model_parameters(model_id=m) for m in without_counts}
        applied_weights.clear()
        uploads.clear()
        result = aggregate(**kwargs)
        assert applied_weights == expected_weights
        assert uploads == [expected_uploads]
        for m, parameters in before.items():
            _, concept_before = split(parameter_snapshot=parameters)
            _, concept_after = split(
                parameter_snapshot=repository.get_global_model_parameters(model_id=m)
            )
            assert concept_before.keys() == concept_after.keys()
            for key in concept_before:
                torch.testing.assert_close(concept_before[key], concept_after[key], atol=0, rtol=0)
            evidence["kept_without_counts"] += 1
        evidence["calls"] += 1
        return result

    with monkeypatch.context() as patch:
        patch.setattr(aggregation_parameters, "_add_weighted_parameters", checked_weights)
        patch.setattr(CommunicationVolumeRecordStore, "record_model_transfers", checked_transfers)
        patch.setattr(aggregation, "aggregate_client_models_into_global_models", checked_aggregate)
        run = execute_fedsda_experiment_run(
            run_settings=build_settings(name, dataset=dataset, count=count, interval=interval)
        )
    return build_fedsda_run_record_tables(experiment_run=run), evidence


def first_aggregation_state(name, dataset, monkeypatch, count, interval):
    from federated_learning_experiments.runtime import server_round_synchronization as aggregation
    from federated_learning_experiments.runtime.fedsda_experiment_run import (
        execute_fedsda_experiment_run,
    )
    from federated_learning_experiments.runtime.fedsda_run_record_tables import (
        build_fedsda_run_record_tables,
    )

    aggregate = aggregation.aggregate_client_models_into_global_models
    captured = []

    def capture(**kwargs):
        if not captured:
            captured.append(capture_state_at_first_aggregation(kwargs["run_clients"]))
        return aggregate(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(aggregation, "aggregate_client_models_into_global_models", capture)
        run = execute_fedsda_experiment_run(
            run_settings=build_settings(name, dataset=dataset, count=count, interval=interval)
        )
    return build_fedsda_run_record_tables(experiment_run=run), captured[0]


@pytest.mark.slow
@pytest.mark.parametrize("dataset,count", [("sine2", 1500), ("mnist4", 800)])
def test_run_aggregates_with_assignment_counts_and_matches_soft_before_first_aggregation(
    dataset, count, monkeypatch
):
    import numpy as np
    import torch

    interval = 50
    tables, evidence = run_with_checked_aggregation(NAME, dataset, monkeypatch, count, interval)
    assert evidence["calls"] == count // interval
    # 介入が実際に効いている（件数と学習量が異なる集約がある）。
    assert evidence["counts_differ_from_amounts"] > 0
    # 学習量のコンセプト別の記録がある（要求3.3）。
    assert len(tables["training_mass_by_concept.training_mass"]) > 0
    # 最初の集約の前は`fedsda-soft-training`と一致する（要求1.2）。
    soft_tables, soft_first = first_aggregation_state(
        "fedsda-soft-training", dataset, monkeypatch, count, interval
    )
    first = evidence["first"]
    assert torch.equal(first["torch"], soft_first["torch"])
    assert first["python"] == soft_first["python"]
    assert all(
        np.array_equal(a, b) for a, b in zip(first["numpy"], soft_first["numpy"], strict=True)
    )
    assert [list(p) for p in first["parameters"]] == [list(p) for p in soft_first["parameters"]]
    for mine, theirs in zip(first["parameters"], soft_first["parameters"], strict=True):
        for model_id, values in mine.items():
            for key, value in values.items():
                torch.testing.assert_close(value, theirs[model_id][key], atol=0, rtol=0)
    prediction_columns = [k for k in tables if k.startswith("sample_prediction.")]
    assert prediction_columns
    for column in prediction_columns:
        mine, theirs = np.asarray(tables[column]), np.asarray(soft_tables[column])
        assert mine.ndim == 2 and mine.shape == theirs.shape, column
        np.testing.assert_array_equal(mine[:, :interval], theirs[:, :interval], err_msg=column)
    print(json.dumps({"dataset": dataset, **{k: v for k, v in evidence.items() if k != "first"}}))


@pytest.mark.slow
def test_model_without_assignment_counts_in_any_client_keeps_global_concept_specific_part(
    monkeypatch,
):
    """小さいrunでは件数を持つclientが無い保有モデルが自然には出ないので、重みの関数を包んで
    モデル0の件数を全clientで0にし、要求2.4（既存のグローバルモデルの概念固有部を保つ）を確かめる。"""
    import torch

    from federated_learning_experiments.runtime import (
        server_model_registration_and_aggregation as aggregation_parameters,
    )
    from federated_learning_experiments.runtime.fedsda_experiment_run import (
        execute_fedsda_experiment_run,
    )

    derive_weights = aggregation_parameters.derive_client_model_aggregation_weights
    aggregate = aggregation_parameters.aggregate_client_models_into_global_models
    split = aggregation_parameters.split_shared_and_concept_specific_parameters
    checked = []

    def without_model_zero(**kwargs):
        weights = derive_weights(**kwargs)
        assert kwargs["model_aggregation_weight_source"] == COUNTS
        return {m: (0 if m == 0 else w) for m, w in weights.items()}

    def checked_aggregate(**kwargs):
        repository = kwargs["global_model_repository"]
        before = (
            split(parameter_snapshot=repository.get_global_model_parameters(model_id=0))[1]
            if 0 in repository.global_model_ids
            else None
        )
        result = aggregate(**kwargs)
        if before is not None:
            after = split(parameter_snapshot=repository.get_global_model_parameters(model_id=0))[1]
            assert before.keys() == after.keys()
            for key in before:
                torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
            assert result.aggregated_training_sample_counts_by_model_id.get(0, 0) == 0
            checked.append(True)
        return result

    monkeypatch.setattr(
        aggregation_parameters, "derive_client_model_aggregation_weights", without_model_zero
    )
    monkeypatch.setattr(
        aggregation_parameters, "aggregate_client_models_into_global_models", checked_aggregate
    )
    from federated_learning_experiments.runtime import server_round_synchronization

    monkeypatch.setattr(
        server_round_synchronization,
        "aggregate_client_models_into_global_models",
        checked_aggregate,
    )
    execute_fedsda_experiment_run(run_settings=build_settings(NAME, count=300))
    assert len(checked) >= 5
