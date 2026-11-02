"""IFCAの初期化、損失による選択、1時刻の学習窓を実体で確かめる。"""

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
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
)
from federated_learning_experiments.runtime import feddrift_detection_batch_processing as processing
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
from federated_learning_experiments.sweep import experiment_configurations as configurations
from federated_learning_experiments.sweep.experiment_sweep_plan import make_run_identifier
from tests.refactoring import behavior_golden_cases as golden
from tests.refactoring.test_feddrift_oracle_concept_assignment_configuration import (
    observe,
    settings,
)
from tests.refactoring.test_feddrift_run_client import build_assembly_inputs

CONFIGURATIONS = ("feddrift-ifca-window", "feddrift-ifca-window-pretrained-cluster0")


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_configuration_registration_defaults_and_identifier(configuration):
    actual = settings(configuration=configuration)
    default = settings(configuration="feddrift")
    client = actual.run_participant_settings.run_client_settings
    assert client.model_assignment_policy == "minimum_loss_cluster_assignment"
    assert client.training_sample_window == "current_timestep"
    assert client.cluster_model_initialization == (
        "all_random" if configuration == CONFIGURATIONS[0] else "pretrained_cluster0"
    )
    assert actual == replace(
        default,
        run_participant_settings=replace(
            default.run_participant_settings,
            run_client_settings=replace(
                default.run_participant_settings.run_client_settings,
                model_assignment_policy="minimum_loss_cluster_assignment",
                training_sample_window="current_timestep",
                cluster_model_initialization=(
                    "all_random" if configuration == CONFIGURATIONS[0] else "pretrained_cluster0"
                ),
            ),
        ),
    )
    assert configurations.EXPERIMENT_CONFIGURATION_NAMES[-5:-3] == CONFIGURATIONS
    assert (
        golden.CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN[configuration]
        == "tests/refactoring/test_feddrift_ifca_window_configuration.py"
    )
    saved = convert_settings_to_plain_mapping(default)["run_participant_settings"][
        "run_client_settings"
    ]
    assert "training_sample_window" not in saved
    assert "cluster_model_initialization" not in saved
    assert (
        make_run_identifier(configuration_name="feddrift", run_settings=default)
        == "sine2_interval50_seed0_1fbf192c89f9"
    )


def test_initialization_repeats_across_clients_and_preserves_all_random_sources(monkeypatch):
    _, inputs = build_assembly_inputs(monkeypatch=monkeypatch)
    states_by_policy = {}
    before = torch.get_rng_state().clone()
    python_before = inputs["python_random_generator"].getstate()
    numpy_before = np.random.get_state()
    for client_id, policy in enumerate(("all_random", "pretrained_cluster0", "all_random")):
        client = assemble_feddrift_run_client(
            **{
                **inputs,
                "client_id": client_id,
                "run_client_settings": replace(
                    inputs["run_client_settings"],
                    model_assignment_policy="minimum_loss_cluster_assignment",
                    training_sample_window="current_timestep",
                    cluster_model_initialization=policy,
                ),
                "initial_model_count": 4,
                "run_random_seed": 0,
            }
        )
        states = client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        values = [
            torch.cat([p.detach().flatten() for p in s.classifier.parameters()]) for s in states
        ]
        assert [s.model_id for s in states] == list(range(4))
        if policy in states_by_policy:
            assert all(
                torch.equal(a, b) for a, b in zip(values, states_by_policy[policy], strict=True)
            )
        states_by_policy[policy] = values
        for state in states:
            if policy == "all_random" or state.model_id:
                assert (
                    state.concept_specific_parameter_optimizer_state.parameter_optimizer.state == {}
                )
    original = torch.cat([p.detach().flatten() for p in inputs["initial_classifier"].parameters()])
    random = states_by_policy["all_random"]
    assert all(not torch.equal(a, b) for i, a in enumerate(random) for b in random[i + 1 :])
    assert all(not torch.equal(a, original) for a in random)
    assert torch.equal(states_by_policy["pretrained_cluster0"][0], original)
    assert all(
        torch.equal(a, b)
        for a, b in zip(random[1:], states_by_policy["pretrained_cluster0"][1:], strict=True)
    )
    assert torch.equal(before, torch.get_rng_state())
    assert python_before == inputs["python_random_generator"].getstate()
    for a, b in zip(numpy_before, np.random.get_state(), strict=True):
        np.testing.assert_equal(a, b)


def test_selection_uses_bounded_loss_skips_initial_comparison_and_breaks_ties_by_id(monkeypatch):
    _, inputs = build_assembly_inputs(monkeypatch=monkeypatch)
    client = assemble_feddrift_run_client(
        **{
            **inputs,
            "initial_model_count": 2,
            "run_client_settings": replace(
                inputs["run_client_settings"],
                model_assignment_policy="minimum_loss_cluster_assignment",
                training_sample_window="current_timestep",
                cluster_model_initialization="all_random",
            ),
        }
    )
    assessor = processing.assess_feddrift_detection_batch
    comparisons = []

    def compare(**kwargs):
        assert "model_comparison_criterion" not in kwargs  # 有界損失の既定を使う。
        result = assessor(**kwargs)
        for model_id, classifier in kwargs["held_models"]:
            expected = evaluate_classifier_per_sample_bounded_losses(
                classifier=classifier,
                input_features=kwargs["input_features"],
                observed_class_labels=kwargs["observed_class_labels"],
            )
            assert dict(result.model_mean_bounded_losses)[model_id] == float(expected.mean().item())
        comparisons.append(result)
        return result

    monkeypatch.setattr(processing, "assess_feddrift_detection_batch", compare)
    observe(client, (0, 0, 0))
    assert comparisons == []
    first_window = client.owners.training_sample_store.snapshot_ordered_model_training_samples()
    observe(client, (1, 1, 1))
    assert len(comparisons) == 1
    assert (
        client.owners.current_training_model_assignment.current_training_model_id
        == comparisons[0].minimum_loss_model_id
    )
    states = client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    states[1].classifier.load_state_dict(states[0].classifier.state_dict())
    # 保有順を逆にしても、同値は小さいIDを選ぶ。
    registry = client.owners.held_model_training_state_registry
    snapshot = registry.snapshot_ordered_held_model_training_states
    monkeypatch.setattr(
        registry, "snapshot_ordered_held_model_training_states", lambda: tuple(reversed(snapshot()))
    )
    observe(client, (1, 0, 1))
    assert client.owners.current_training_model_assignment.current_training_model_id == 0
    current = client.owners.training_sample_store.snapshot_ordered_model_training_samples()
    assert len(first_window[0].training_samples) == 3
    assert sum(len(c.training_samples) for c in current) == 3
    assert current[0].training_samples[0] is not first_window[0].training_samples[0]


@pytest.mark.parametrize(
    "field,value",
    (
        ("training_sample_window", "invalid"),
        ("cluster_model_initialization", "invalid"),
        ("model_assignment_policy", "invalid"),
        ("training_sample_window", "current_timestep"),
        ("cluster_model_initialization", "all_random"),
    ),
)
def test_invalid_or_inactive_options_are_rejected(field, value):
    default = settings(configuration="feddrift").run_participant_settings.run_client_settings
    with pytest.raises(ValueError):
        replace(default, **{field: value})


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_short_run_validates_selection_records_before_saving(configuration):
    run_settings = configurations.build_experiment_run_settings(
        configuration_name=configuration,
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
    assert tables["cluster_selection.model_ids"].shape == (2, 2)
    assert tables["cluster_selection.first_sample_index"].tolist() == [0, 3]
    selections = run.measured_run.participants.run_clients[
        0
    ].detection_batch_processor._cluster_selections
    first, last, selected, losses = selections[1]
    selections[1] = (first, last, 1 - selected, losses)
    with pytest.raises(ValueError, match="平均有界損失"):
        build_feddrift_run_record_tables(experiment_run=run)


@pytest.mark.slow
@pytest.mark.parametrize("configuration", CONFIGURATIONS)
@pytest.mark.parametrize("dataset,concept_count", (("sine2", 2), ("mnist4", 4)))
def test_full_run_selection_window_and_fixed_model_pool(
    monkeypatch, configuration, dataset, concept_count
):
    training_windows = []
    original = FeddriftRunClient.train_local_round

    def inspect_training(client, *, eligible_model_ids=None):
        end = client.stream_processor.processed_sample_count
        selected = client.owners.current_training_model_assignment.current_training_model_id
        assert client.get_held_model_ids() == set(range(concept_count))
        assert selected in eligible_model_ids
        for (
            collection
        ) in client.owners.training_sample_store.snapshot_ordered_model_training_samples():
            assert len(collection.training_samples) == (
                50 if collection.model_id == selected else 0
            )
            for offset, sample in enumerate(collection.training_samples):
                training_windows.append(
                    (
                        client.client_id,
                        end - 50 + offset,
                        sample.input_features.clone(),
                        sample.observed_class_labels.item(),
                    )
                )
        return original(client, eligible_model_ids=eligible_model_ids)

    monkeypatch.setattr(FeddriftRunClient, "train_local_round", inspect_training)
    run = execute_feddrift_experiment_run(run_settings=settings(dataset, configuration))
    tables = build_feddrift_run_record_tables(experiment_run=run)
    concepts = tables["sample_prediction.observed_concept_id"]
    changes = int(np.count_nonzero(concepts[:, 1:] != concepts[:, :-1]))
    assert changes > 0
    selected = tables["cluster_selection.selected_model_id"].reshape(10, 50)
    losses = tables["cluster_selection.model_mean_bounded_losses"].reshape(10, 50, concept_count)
    np.testing.assert_array_equal(selected[:, 0], 0)
    assert np.all(np.isnan(losses[:, 0]))  # 最初の時刻は比較しない。
    np.testing.assert_array_equal(selected[:, 1:], np.argmin(losses[:, 1:], axis=2))
    predicted = np.repeat(np.column_stack((np.zeros(10, dtype=int), selected[:, :-1])), 50, axis=1)
    np.testing.assert_array_equal(tables["sample_prediction.prediction_model_id"], predicted)
    streams = run.measured_run.run_result.observed_client_streams
    assert len(training_windows) == 25000
    for client_id, index, features, label in training_windows:
        sample = streams[client_id].observed_samples[index]
        torch.testing.assert_close(
            features, torch.tensor((sample.feature_values,), dtype=torch.float32), rtol=0, atol=0
        )
        assert label == sample.class_label
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
    counts = np.bincount(selected.flatten(), minlength=concept_count).tolist()
    print(
        f"{configuration}/{dataset}: changes={changes}, selections={counts}, windows={len(training_windows)}, registrations={concept_count}, merges=0"
    )
