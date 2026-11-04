"""真概念の帰属と、ラベルを見る前の予測を全体runの記録で守る。"""

from dataclasses import replace

import numpy as np
import pytest
import torch

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.data.observed_streams import ObservedSample
from federated_learning_experiments.runtime.feddrift_experiment_run import (
    execute_feddrift_experiment_run,
)
from federated_learning_experiments.runtime.feddrift_run_client import assemble_feddrift_run_client
from federated_learning_experiments.runtime.feddrift_run_record_tables import (
    build_feddrift_run_record_tables,
)
from federated_learning_experiments.runtime.feddrift_run_server import FeddriftRunServer
from federated_learning_experiments.sweep import experiment_configurations as configurations
from federated_learning_experiments.sweep.experiment_sweep_plan import make_run_identifier
from tests.refactoring import behavior_golden_cases as golden
from tests.refactoring.test_feddrift_run_client import build_assembly_inputs

CONFIGURATION = "feddrift-oracle-concept-assignment"


def oracle_client(monkeypatch):
    _, inputs = build_assembly_inputs(monkeypatch=monkeypatch)
    inputs["run_client_settings"] = replace(
        inputs["run_client_settings"],
        model_assignment_policy="true_concept_assignment",
        model_comparison_criterion="classification_error_rate",
        new_model_parameter_source="minimum_criterion_held_model",
        detection_reference_after_drift="minimum_held_model_batch_value",
    )
    before = torch.get_rng_state().clone()
    client = assemble_feddrift_run_client(**inputs, initial_model_count=2)
    assert torch.equal(before, torch.get_rng_state())
    return client, inputs


def observe(client, concepts):
    return client.process_observed_block(
        observed_samples=tuple(
            ObservedSample(feature_values=(0.1, 0.2, 0.3), class_label=0) for _ in concepts
        ),
        evaluation_concept_ids=concepts,
    )


def test_initial_models_are_independent_copies_of_pretrained_model(monkeypatch):
    client, inputs = oracle_client(monkeypatch)
    states = client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    assert [state.model_id for state in states] == [0, 1]
    initial = inputs["initial_classifier"]
    for state in states:
        for actual, expected in zip(
            state.classifier.parameters(), initial.parameters(), strict=True
        ):
            assert torch.equal(actual, expected)
            assert actual.data_ptr() != expected.data_ptr()
    for left, right in zip(
        states[0].classifier.parameters(), states[1].classifier.parameters(), strict=True
    ):
        assert left.data_ptr() != right.data_ptr()


@pytest.mark.parametrize("concept", (None, -1, 2, True, "0"))
def test_invalid_concept_rejects_entire_block_before_prediction(monkeypatch, concept):
    client, _ = oracle_client(monkeypatch)
    with pytest.raises((TypeError, ValueError)):
        observe(client, (0, concept))
    assert client.stream_processor.processed_sample_count == 0
    assert client.owners.training_sample_store.snapshot_ordered_model_training_samples() == ()


def test_split_batches_training_eligibility_threshold_rounds_and_weights(monkeypatch):
    client, inputs = oracle_client(monkeypatch)
    server_settings = replace(
        settings(configuration="feddrift").run_participant_settings.server_settings,
        rebuilt_model_optimizer_settings=inputs["run_client_settings"].new_model_optimizer_settings,
        local_training_round_count=2,
    )
    server = FeddriftRunServer(
        run_clients=(client,),
        server_settings=server_settings,
        initial_parameter_snapshot={
            k: v.detach().clone() for k, v in inputs["initial_classifier"].state_dict().items()
        },
        initial_loss_statistics=inputs["initial_loss_statistics"],
        python_random_generator=inputs["python_random_generator"],
    )
    observe(client, (0, 1))
    observe(client, (1,))
    assert [p.prediction_model_id for p in client.stream_processor.sample_predictions] == [0, 0, 1]
    assert server._current_timestep_training_model_ids() == frozenset((0, 1))
    server.synchronize_timestep(timestep_index=0)
    counts = client.owners.model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts()
    assert counts.trained_sample_counts_by_model_id == {}  # 1件と2件は閾値3未満。
    assert server.snapshot_model_weights() == {0: 1, 1: 2}
    observe(client, (0, 0, 1))
    server.synchronize_timestep(timestep_index=1)
    counts = client.owners.model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts()
    assert counts.trained_sample_counts_by_model_id == {0: 36, 1: 36}
    assert server.snapshot_model_weights() == {0: 3, 1: 3}
    observe(client, (0, 0, 0))
    assert server._current_timestep_training_model_ids() == frozenset((0,))
    server.synchronize_timestep(timestep_index=2)
    counts = client.owners.model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts()
    assert counts.trained_sample_counts_by_model_id == {0: 72, 1: 36}
    assert server.snapshot_model_weights() == {0: 6, 1: 3}
    observe(client, (1,))
    server.finalize_observed_stream(timestep_index=3)
    stored = client.owners.training_sample_store.snapshot_ordered_model_training_samples()
    assert {x.model_id: len(x.training_samples) for x in stored} == {0: 6, 1: 4}
    assert server.global_model_repository.global_model_ids == (0, 1)


def settings(dataset="sine2", configuration=CONFIGURATION, *, client_count=10, sample_count=2500):
    return configurations.build_experiment_run_settings(
        configuration_name=configuration,
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name=dataset,
            random_seed=0,
            client_count=client_count,
            per_client_sample_count=sample_count,
            server_aggregation_interval_per_client_samples=50,
        ),
    )


# FedDrift系の小さい全体runの共通の規模。4 client×500標本（間隔50で10時刻）。既定の変化の規則
# （300標本の後、標本ごと0.0015）では短い列でほとんど変化しないので、間隔100・確率0.01に縮める。
# seed 0で変化は計8回、MNIST4の4概念がすべて現れる（変化の規則の形と乱数の順は既定と同じ）。
SMALL_FULL_RUN_CLIENT_COUNT = 4
SMALL_FULL_RUN_SAMPLE_COUNT = 500
SMALL_FULL_RUN_TIMESTEP_COUNT = SMALL_FULL_RUN_SAMPLE_COUNT // 50


def small_full_run_settings(dataset="sine2", configuration=CONFIGURATION):
    base = settings(
        dataset,
        configuration,
        client_count=SMALL_FULL_RUN_CLIENT_COUNT,
        sample_count=SMALL_FULL_RUN_SAMPLE_COUNT,
    )
    execution = base.execution_settings
    return replace(
        base,
        execution_settings=replace(
            execution,
            concept_schedule_settings=replace(
                execution.concept_schedule_settings,
                minimum_sample_index_gap_before_change_trial=100,
                per_eligible_sample_concept_change_probability=0.01,
            ),
        ),
    )


def test_short_run_records_no_detection_batches():
    run_settings = configurations.build_experiment_run_settings(
        configuration_name=CONFIGURATION,
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name="sine2",
            random_seed=0,
            client_count=1,
            per_client_sample_count=6,
            server_aggregation_interval_per_client_samples=3,
        ),
    )
    run = execute_feddrift_experiment_run(run_settings=run_settings)
    tables = build_feddrift_run_record_tables(experiment_run=run)
    assert len(tables["detection_batch.client_id"]) == 0


def test_configuration_changes_only_assignment_policy():
    default = settings(configuration="feddrift")
    actual = settings()
    participant = default.run_participant_settings
    assert actual == replace(
        default,
        run_participant_settings=replace(
            participant,
            run_client_settings=replace(
                participant.run_client_settings, model_assignment_policy="true_concept_assignment"
            ),
        ),
    )
    assert configurations.FEDDRIFT_ORACLE_CONCEPT_ASSIGNMENT_CONFIGURATION_NAME == CONFIGURATION
    assert CONFIGURATION in configurations.EXPERIMENT_CONFIGURATION_NAMES
    assert golden.CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN[CONFIGURATION] == (
        "tests/refactoring/test_feddrift_oracle_concept_assignment_configuration.py"
    )
    assert default.run_participant_settings.run_client_settings.model_assignment_policy == (
        "detection_batch_assignment"
    )
    assert make_run_identifier(configuration_name="feddrift", run_settings=default) == (
        "sine2_interval50_seed0_1fbf192c89f9"
    )
    assert (
        "model_assignment_policy"
        not in convert_settings_to_plain_mapping(default)["run_participant_settings"][
            "run_client_settings"
        ]
    )
    assert (
        convert_settings_to_plain_mapping(actual)["run_participant_settings"][
            "run_client_settings"
        ]["model_assignment_policy"]
        == "true_concept_assignment"
    )


@pytest.mark.slow
@pytest.mark.parametrize("dataset,concept_count", (("sine2", 2), ("mnist4", 4)))
def test_full_run_assigns_each_sample_and_predicts_from_previous_label(dataset, concept_count):
    run = execute_feddrift_experiment_run(run_settings=small_full_run_settings(dataset))
    tables = build_feddrift_run_record_tables(experiment_run=run)
    concepts = tables["sample_prediction.observed_concept_id"]
    changes = int(np.count_nonzero(concepts[:, 1:] != concepts[:, :-1]))
    assert changes > 0
    # 小さい規模でも全概念のモデルへ帰属し学習することを確かめる前提。
    assert set(np.unique(concepts).tolist()) == set(range(concept_count))
    np.testing.assert_array_equal(tables["sample_prediction.training_model_id"], concepts)
    expected_predictions = np.column_stack(
        (np.zeros(SMALL_FULL_RUN_CLIENT_COUNT, dtype=int), concepts[:, :-1])
    )
    np.testing.assert_array_equal(
        tables["sample_prediction.prediction_model_id"], expected_predictions
    )
    participants = run.measured_run.participants
    assert participants.run_server.global_model_repository.global_model_ids == tuple(
        range(concept_count)
    )
    np.testing.assert_array_equal(tables["model_registration.model_id"], np.arange(concept_count))
    assert np.all(tables["model_registration.registered_server_timestep_index"] == -1)
    assert len(tables["model_clustering.model_id"]) == 0
    assert len(tables["detection_batch.client_id"]) == 0
    assert np.all(tables["sample_prediction.held_model_count_at_prediction"] == concept_count)
    assert np.all(
        tables["server_timestep.global_model_count_after_synchronization"] == concept_count
    )
    assert np.all(
        tables["server_timestep.client_held_model_count_after_synchronization"] == concept_count
    )
    trained = {}
    for model_id in range(concept_count):
        mask = tables["client_held_model.model_id"] == model_id
        trained[model_id] = int(tables["client_held_model.trained_sample_count"][mask].sum())
        if model_id in concepts:
            assert trained[model_id] > 0
    for client in participants.run_clients:
        assert client.get_held_model_ids() == set(range(concept_count))
        stored = client.owners.training_sample_store.snapshot_ordered_model_training_samples()
        for collection in stored:
            assert len(collection.training_samples) == np.count_nonzero(
                concepts[client.client_id] == collection.model_id
            )
    print(f"{dataset}: changes={changes}, trained={trained}, models={concept_count}")
