"""責任度の式、操作の同期、復元抽出と純度を仕様から検査する。"""

from random import Random

import pytest
import torch

from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.responsibility_weighted_training import (
    ResponsibilityWeightedTrainingStore,
    calculate_training_responsibilities,
    derive_training_mass_purity,
)


def sample(value):
    return ObservedTrainingSample(
        input_features=torch.tensor([[float(value)]]), observed_class_labels=torch.tensor([[0.0]])
    )


@pytest.mark.parametrize("scale", [1.0, 1e-300])
def test_responsibility_formula_and_zero_denominator(scale):
    q = calculate_training_responsibilities(
        prediction_weights={1: 0.25 * scale, 2: 0.75 * scale},
        true_class_probabilities={1: 0.8, 2: 0.2},
        fallback_model_id=2,
    )
    assert q == pytest.approx({1: 4 / 7, 2: 3 / 7})
    assert calculate_training_responsibilities(
        prediction_weights={1: 0.5, 2: 0.5},
        true_class_probabilities={1: 0.0, 2: 0.0},
        fallback_model_id=2,
    ) == {1: 0.0, 2: 1.0}


def test_store_mass_support_merge_rename_and_discard():
    store = ResponsibilityWeightedTrainingStore()
    store.append(
        training_sample=sample(0), responsibilities={1: 0.2, 2: 0.8}, observed_concept_id=0
    )
    store.append(training_sample=sample(1), responsibilities={1: 1.0}, observed_concept_id=None)
    assert store.training_masses_by_model_id == pytest.approx({1: 1.2, 2: 0.8})
    assert store.support_counts_by_model_id == {1: 2, 2: 1}
    store.remap(model_id_mapping={1: 3, 2: 3})
    assert store.training_masses_by_model_id == {3: 2.0}
    assert store.mass_by_model_and_concept == {3: {0: 1.0}}
    store.discard(model_id=3)
    assert store.sample_count == 0
    assert store.mass_by_model_and_concept == {}


def test_discard_does_not_renormalize_remaining_columns():
    store = ResponsibilityWeightedTrainingStore()
    store.append(
        training_sample=sample(0), responsibilities={1: 0.2, 2: 0.8}, observed_concept_id=1
    )
    store.discard(model_id=2)
    assert store.sample_count == 1
    assert store.training_masses_by_model_id == {1: 0.2}
    assert store.mass_by_model_and_concept == {1: {1: 0.2}}


def test_weighted_replacement_frequencies_objective_and_support_eligibility():
    store = ResponsibilityWeightedTrainingStore()
    for i, q in enumerate([0.1, 0.3, 0.6]):
        store.append(
            training_sample=sample(i), responsibilities={1: q, 2: 1 - q}, observed_concept_id=i
        )
    rng = Random(7)
    batches = [
        store.sample_batches(
            held_model_ids=frozenset({1}), batch_sample_count=3, python_random_generator=rng
        )[0]
        for _ in range(6000)
    ]
    values = torch.cat([b.input_features for b in batches]).flatten()
    for i, probability in enumerate([0.1, 0.3, 0.6]):
        assert (values == i).float().mean().item() == pytest.approx(probability, abs=0.012)
    assert values.mean().item() == pytest.approx(1.5, abs=0.025)
    assert (
        store.sample_batches(
            held_model_ids=frozenset({1}), batch_sample_count=4, python_random_generator=rng
        )
        == ()
    )
    assert max(store.duplicate_sample_fractions) > 0


def test_one_hot_samples_only_come_from_assigned_model():
    store = ResponsibilityWeightedTrainingStore()
    for i in range(8):
        store.append(
            training_sample=sample(i), responsibilities={i % 2: 1.0}, observed_concept_id=i % 2
        )
    for batch in store.sample_batches(
        held_model_ids=frozenset({0, 1}), batch_sample_count=4, python_random_generator=Random(0)
    ):
        assert all(int(x) % 2 == batch.model_id for x in batch.input_features.flatten())


def test_purity_and_missing_denominators():
    assert derive_training_mass_purity(
        mass_by_model_and_concept={1: {0: 3.0, 1: 1.0}, 2: {1: 2.0}}
    ) == pytest.approx((5 / 6, 0.5))
    assert derive_training_mass_purity(mass_by_model_and_concept={}) == (None, None)


def build_settings(name, dataset="sine2", count=300):
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
            client_count=2,
            per_client_sample_count=count,
            server_aggregation_interval_per_client_samples=50,
        ),
    )


def test_new_configurations_change_only_assignment_and_diagnostics():
    from dataclasses import replace

    from federated_learning_experiments.core.settings_serialization import (
        convert_settings_to_plain_mapping,
    )

    baseline = build_settings("fedsda")
    client = baseline.run_participant_settings.run_client_settings
    for name, strategy in [
        ("fedsda-soft-training", "responsibility_weighted_training"),
        ("fedsda-training-mass-diagnostics", "single_model_assignment"),
    ]:
        new = build_settings(name)
        assert new == replace(
            baseline,
            run_participant_settings=replace(
                baseline.run_participant_settings,
                run_client_settings=replace(
                    client,
                    training_assignment_strategy=strategy,
                    record_training_mass_by_concept=True,
                ),
            ),
        )
    mapping = convert_settings_to_plain_mapping(client)
    assert "training_assignment_strategy" not in mapping
    assert "record_training_mass_by_concept" not in mapping


def test_all_existing_configuration_settings_and_identifiers_keep_fixed_hashes():
    import hashlib
    import json
    from pathlib import Path

    from federated_learning_experiments.core.settings_serialization import (
        convert_settings_to_plain_mapping,
    )
    from federated_learning_experiments.sweep.experiment_configurations import (
        EXPERIMENT_CONFIGURATION_NAMES,
    )
    from federated_learning_experiments.sweep.experiment_sweep_plan import make_run_identifier

    expected = json.loads(
        Path(__file__).with_name("soft_training_existing_configuration_hashes.json").read_text()
    )
    assert tuple(expected) == EXPERIMENT_CONFIGURATION_NAMES[:-2]
    for name, (digest, identifier) in expected.items():
        settings = build_settings(name)
        assert (
            hashlib.sha256(
                json.dumps(convert_settings_to_plain_mapping(settings), sort_keys=True).encode()
            ).hexdigest()
            == digest
        ), name
        assert make_run_identifier(configuration_name=name, run_settings=settings) == identifier, (
            name
        )


@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["sine2", "mnist4"])
def test_small_run_diagnostics_only_preserves_every_existing_table(dataset):
    import numpy as np

    from federated_learning_experiments.runtime.fedsda_experiment_run import (
        execute_fedsda_experiment_run,
    )
    from federated_learning_experiments.runtime.fedsda_run_record_tables import (
        build_fedsda_run_record_tables,
    )

    baseline = execute_fedsda_experiment_run(
        run_settings=build_settings("fedsda", dataset, count=1500 if dataset == "sine2" else 800)
    )
    diagnostic = execute_fedsda_experiment_run(
        run_settings=build_settings(
            "fedsda-training-mass-diagnostics", dataset, count=1500 if dataset == "sine2" else 800
        )
    )
    left = build_fedsda_run_record_tables(experiment_run=baseline)
    right = build_fedsda_run_record_tables(experiment_run=diagnostic)
    extra = {key for key in right if key.startswith("training_mass_by_concept.")}
    assert extra
    assert right.keys() - extra == left.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)
    assert baseline.run_metrics == diagnostic.run_metrics
    for client in diagnostic.measured_run.participants.client_operations:
        sample_store = client.owners.training_sample_store
        mass = sample_store.responsibility_training_store.mass_by_model_and_concept
        for collection in sample_store.snapshot_ordered_model_training_samples():
            assert sum(mass.get(collection.model_id, {}).values()) == len(
                collection.training_samples
            )
        expected_purity = sum(
            max(concepts.values()) for concepts in mass.values() if concepts
        ) / sum(sum(concepts.values()) for concepts in mass.values())
        assert derive_training_mass_purity(mass_by_model_and_concept=mass)[0] == expected_purity


@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["sine2", "mnist4"])
def test_small_soft_run_uses_all_eligible_models_and_preserves_store_mass(dataset, monkeypatch):
    import json
    import math

    from federated_learning_experiments.learning.training import (
        held_model_joint_training_iterations as iterations,
    )
    from federated_learning_experiments.learning.training.model_training_sample_store import (
        ModelTrainingSampleStore,
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

    append = ModelTrainingSampleStore.append_model_training_samples
    sample_batches = ResponsibilityWeightedTrainingStore.sample_batches
    joint_update = iterations.perform_joint_model_parameter_update
    aggregate = aggregation.aggregate_client_models_into_global_models
    checks = {"append": 0, "batch": 0, "update": 0, "aggregation": 0, "remap": 0, "merge": 0}
    remap = ResponsibilityWeightedTrainingStore.remap
    add_parameters = aggregation_parameters._add_weighted_parameters
    applied_weights = []

    def checked_weights(**kwargs):
        applied_weights.append(kwargs["sample_count"])
        return add_parameters(**kwargs)

    def checked_remap(self, **kwargs):
        mapping = kwargs["model_id_mapping"]
        before = self.training_masses_by_model_id
        expected = {}
        for m, value in before.items():
            target = mapping.get(m, m)
            expected[target] = expected.get(target, 0) + value
        count = self.sample_count
        remap(self, **kwargs)
        assert self.sample_count == count
        assert self.training_masses_by_model_id == pytest.approx(expected)
        assert sum(
            sum(v.values()) for v in self.mass_by_model_and_concept.values()
        ) == pytest.approx(sum(expected.values()))
        checks["remap"] += 1
        checks["merge"] += len(expected) < len(before)

    def checked_append(self, **kwargs):
        append(self, **kwargs)
        if self.training_responsibility_provider is not None:
            store = self.responsibility_training_store
            hard_count = sum(
                len(c.training_samples) for c in self.snapshot_ordered_model_training_samples()
            )
            assert store.sample_count == hard_count
            assert sum(store.training_masses_by_model_id.values()) == pytest.approx(hard_count)
            checks["append"] += 1

    def checked_batches(self, **kwargs):
        result = sample_batches(self, **kwargs)
        expected = [
            m
            for m, count in self.support_counts_by_model_id.items()
            if m in kwargs["held_model_ids"] and count >= kwargs["batch_sample_count"]
        ]
        assert [b.model_id for b in result] == expected
        checks["batch"] += bool(result)
        return result

    def checked_update(**kwargs):
        batches = kwargs["participating_training_batches"]
        before = [
            [p.detach().clone() for p in b.classifier.classification_layer.parameters()]
            for b in batches
        ]
        loss = joint_update(**kwargs)
        for old, batch in zip(before, batches):
            assert any(
                not torch.equal(p, q)
                for p, q in zip(old, batch.classifier.classification_layer.parameters())
            )
        for batch in batches:
            for group in batch.concept_specific_parameter_optimizer.param_groups:
                assert all(p.grad is not None for p in group["params"])
        checks["update"] += 1
        return loss

    def checked_aggregate(**kwargs):
        clients = kwargs["run_clients"]
        expected = {}
        for client in clients:
            for (
                m,
                value,
            ) in client.owners.training_sample_store.responsibility_training_store.training_masses_by_model_id.items():
                if m >= 0:
                    expected[m] = expected.get(m, 0) + value
        expected_weights = []
        for client in clients:
            masses = client.owners.training_sample_store.responsibility_training_store.training_masses_by_model_id
            participating = [masses[m] for m in sorted(expected) if masses.get(m, 0) > 0]
            if participating:
                expected_weights.extend([sum(participating), *participating])
        applied_weights.clear()
        result = aggregate(**kwargs)
        assert applied_weights == pytest.approx(expected_weights)
        for m, value in result.aggregated_training_sample_counts_by_model_id.items():
            assert value == pytest.approx(expected.get(m, 0))
        checks["aggregation"] += 1
        return result

    monkeypatch.setattr(aggregation_parameters, "_add_weighted_parameters", checked_weights)
    monkeypatch.setattr(ResponsibilityWeightedTrainingStore, "remap", checked_remap)
    monkeypatch.setattr(ModelTrainingSampleStore, "append_model_training_samples", checked_append)
    monkeypatch.setattr(ResponsibilityWeightedTrainingStore, "sample_batches", checked_batches)
    monkeypatch.setattr(iterations, "perform_joint_model_parameter_update", checked_update)
    monkeypatch.setattr(
        aggregation, "aggregate_client_models_into_global_models", checked_aggregate
    )
    run = execute_fedsda_experiment_run(
        run_settings=build_settings(
            "fedsda-soft-training", dataset, count=1500 if dataset == "sine2" else 800
        )
    )
    tables = build_fedsda_run_record_tables(experiment_run=run)
    participants = tables["soft_training_update.participating_model_count"]
    effective = tables["training_responsibility.effective_model_count"]
    overlap = tables["soft_training_batch.duplicate_sample_fraction"]
    assert (
        checks["append"] > 0
        and checks["batch"] > 0
        and checks["update"] > 0
        and checks["aggregation"] > 0
    )
    assert participants.max() > 1
    assert effective.max() > 1
    assert all(math.isfinite(x) and x >= 1 - 1e-12 for x in effective)
    assert overlap.max() > 0
    purities = []
    for client in run.measured_run.participants.client_operations:
        store = client.owners.training_sample_store.responsibility_training_store
        purity, baseline = derive_training_mass_purity(
            mass_by_model_and_concept=store.mass_by_model_and_concept
        )
        assert 0 <= baseline <= purity <= 1 + 1e-12
        purities.append((purity, baseline))
    evidence = {
        "dataset": dataset,
        "participating_mean": float(participants.mean()),
        "participating_max": int(participants.max()),
        "effective_mean": float(effective.mean()),
        "effective_max": float(effective.max()),
        "overlap_mean": float(overlap.mean()),
        "overlap_max": float(overlap.max()),
        "purity": purities,
        "checks": checks,
        "alarm_count": len(tables["loss_change_alarm.client_id"]),
    }
    print(json.dumps(evidence))


@pytest.mark.parametrize("optimizer", ["sgd", "standard", "amsgrad"])
def test_one_hot_same_injected_batches_match_hard_joint_update(optimizer, monkeypatch):
    from copy import deepcopy

    from test_joint_model_parameter_update import build_joint_update_oracle_pair

    from federated_learning_experiments.learning.training import (
        held_model_joint_training_iterations as iterations,
    )
    from federated_learning_experiments.learning.training.held_model_training_binding import (
        HeldModelTrainingBinding,
    )
    from federated_learning_experiments.learning.training.local_training_settings import (
        LocalTrainingSettings,
    )
    from federated_learning_experiments.learning.training.model_training_sample_records import (
        SampledModelTrainingBatch,
    )

    batches, shared_optimizer, _ = build_joint_update_oracle_pair(
        class_count=2,
        batch_sample_counts=(4, 4),
        optimizer_variant=optimizer,
        monkeypatch=monkeypatch,
    )
    bindings = tuple(
        HeldModelTrainingBinding(
            model_id=m,
            classifier=b.classifier,
            concept_specific_parameter_optimizer=b.concept_specific_parameter_optimizer,
        )
        for m, b in enumerate(batches)
    )
    copied_bindings, copied_optimizer = deepcopy((bindings, shared_optimizer))
    fixed = tuple(
        SampledModelTrainingBatch(
            model_id=m,
            input_features=b.input_features,
            observed_class_labels=b.observed_class_labels,
        )
        for m, b in enumerate(batches)
    )
    store = ResponsibilityWeightedTrainingStore()
    for m, b in enumerate(batches):
        for x, y in zip(b.input_features.split(1), b.observed_class_labels.split(1)):
            store.append(
                training_sample=ObservedTrainingSample(input_features=x, observed_class_labels=y),
                responsibilities={m: 1.0},
                observed_concept_id=m,
            )
    monkeypatch.setattr(
        iterations, "sample_training_batches_for_held_models", lambda **kwargs: fixed
    )
    monkeypatch.setattr(
        ResponsibilityWeightedTrainingStore, "sample_batches", lambda self, **kwargs: fixed
    )
    common = dict(
        requested_joint_update_iteration_count=3,
        ordered_model_training_samples=(),
        batch_sample_count=4,
        local_training_settings=LocalTrainingSettings(
            local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
            shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
        ),
        update_shared_features=True,
    )
    hard_losses = iterations.perform_held_model_joint_training_iterations(
        **common,
        held_model_training_bindings=bindings,
        python_random_generator=Random(0),
        shared_feature_extractor=bindings[0].classifier.feature_extractor,
        shared_parameter_optimizer=shared_optimizer,
    )
    soft_losses = iterations.perform_held_model_joint_training_iterations(
        **common,
        held_model_training_bindings=copied_bindings,
        python_random_generator=Random(0),
        shared_feature_extractor=copied_bindings[0].classifier.feature_extractor,
        shared_parameter_optimizer=copied_optimizer,
        responsibility_training_store=store,
    )
    from test_joint_model_parameter_update import assert_nested_state_equal

    assert_nested_state_equal(shared_optimizer.state_dict(), copied_optimizer.state_dict())
    for hard, soft in zip(bindings, copied_bindings):
        assert_nested_state_equal(
            hard.concept_specific_parameter_optimizer.state_dict(),
            soft.concept_specific_parameter_optimizer.state_dict(),
        )
    assert hard_losses == pytest.approx(soft_losses, abs=1e-6)
    for hard, soft in zip(bindings, copied_bindings):
        for p, q in zip(hard.classifier.parameters(), soft.classifier.parameters()):
            torch.testing.assert_close(p, q, atol=1e-6, rtol=0)
        for p, q in zip(hard.classifier.parameters(), soft.classifier.parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=0)


def test_run_purity_excludes_missing_clients_and_preserves_equal_weights():
    from federated_learning_experiments.learning.training.responsibility_weighted_training import (
        derive_run_training_mass_purity,
    )

    assert derive_run_training_mass_purity(
        client_masses=({}, {0: {0: 1.0, 1: 1.0}}, {0: {0: 100.0}})
    ) == (0.75, 0.75, 1)
    assert derive_run_training_mass_purity(client_masses=({}, {})) == (None, None, 2)


@pytest.mark.parametrize("dataset", ["sine2", "mnist4"])
def test_prediction_alarm_selection_prefix_matches_before_first_update(dataset, monkeypatch):
    from federated_learning_experiments.runtime import fedsda_run_client as client_module
    from federated_learning_experiments.runtime.fedsda_experiment_run import (
        execute_fedsda_experiment_run,
    )

    process = client_module.process_observed_sample
    traces = {}
    active = []
    finished = set()

    def observe(**kwargs):
        result = process(**kwargs)
        client_id = kwargs["client_id"]
        if client_id not in finished:
            prediction = result.observed_sample_prediction
            active.append(
                (
                    client_id,
                    prediction.prediction_weights_by_model_id,
                    prediction.combined_prediction_probabilities.clone(),
                    result.loss_monitoring_observation.drift_detected,
                    result.alarm_occurrence_handling,
                )
            )
            if result.completed_joint_update_losses:
                finished.add(client_id)
        return result

    monkeypatch.setattr(client_module, "process_observed_sample", observe)
    for name in ("fedsda", "fedsda-soft-training"):
        active.clear()
        finished.clear()
        execute_fedsda_experiment_run(run_settings=build_settings(name, dataset, count=100))
        traces[name] = list(active)
        assert len(finished) == 2
    assert len(traces["fedsda"]) == len(traces["fedsda-soft-training"])
    for hard, soft in zip(traces["fedsda"], traces["fedsda-soft-training"]):
        assert hard[:2] == soft[:2]
        torch.testing.assert_close(hard[2], soft[2], atol=0, rtol=0)
        assert hard[3:] == soft[3:]


@pytest.mark.parametrize(
    "source",
    [
        "learning/training/responsibility_weighted_training.py",
        "runtime/released_sample_training_responsibilities.py",
    ],
)
def test_new_modules_reject_outer_dependencies(source):
    from test_single_run_dependency_boundaries import collect_dependency_boundary_violations

    for module in (
        "federated_drift_experiment.config",
        "sklearn.cluster",
        "tests.refactoring.test_fedsda_soft_training_configuration",
        "federated_learning_experiments.sweep.experiment_configurations",
    ):
        assert collect_dependency_boundary_violations(
            source_module_path=source, source_text=f"import {module}"
        )


def test_release_responsibilities_use_fresh_local_parameters_and_new_pool(monkeypatch):
    from copy import deepcopy
    from math import log
    from types import SimpleNamespace

    from test_joint_model_parameter_update import build_joint_update_oracle_pair

    from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights import (
        FixedShareWeightController,
    )
    from federated_learning_experiments.runtime.released_sample_training_responsibilities import (
        build_released_sample_responsibility_provider,
    )

    batches, _, _ = build_joint_update_oracle_pair(
        class_count=2, batch_sample_counts=(4, 4), optimizer_variant="sgd", monkeypatch=monkeypatch
    )
    states = [SimpleNamespace(model_id=m, classifier=b.classifier) for m, b in enumerate(batches)]
    registry = SimpleNamespace(snapshot_ordered_held_model_training_states=lambda: tuple(states))
    settings = build_settings(
        "fedsda"
    ).run_participant_settings.run_client_settings.prediction_combination_settings
    controller = FixedShareWeightController(prediction_combination_settings=settings)
    provider = build_released_sample_responsibility_provider(
        registry=registry, weight_controller=controller
    )
    for state, probability in zip(states, (0.8, 0.2)):
        with torch.no_grad():
            state.classifier.classification_layer.weight.zero_()
            state.classifier.classification_layer.bias.fill_(log(probability / (1 - probability)))
    observation = ObservedTrainingSample(
        input_features=torch.tensor([[0.0, 0.0]]), observed_class_labels=torch.tensor([[1.0]])
    )
    q = provider(0, observation)
    assert q == pytest.approx({0: 0.8, 1: 0.2}, abs=1e-6)
    store = ResponsibilityWeightedTrainingStore()
    store.append(training_sample=observation, responsibilities=q, observed_concept_id=0)
    with torch.no_grad():
        states[0].classifier.classification_layer.bias.fill_(log(0.4 / 0.6))
    assert provider(0, observation) == pytest.approx({0: 2 / 3, 1: 1 / 3}, abs=1e-6)
    states.append(SimpleNamespace(model_id=2, classifier=deepcopy(states[1].classifier)))
    assert provider(2, observation) == pytest.approx({0: 0.5, 1: 0.25, 2: 0.25}, abs=1e-6)
    assert store.training_masses_by_model_id == pytest.approx({0: 0.8, 1: 0.2}, abs=1e-6)
    assert store.training_masses_by_model_id.get(2, 0) == 0


def test_registration_and_distribution_remap_auxiliary_columns_together():
    from federated_learning_experiments.learning.training.model_training_sample_store import (
        ModelTrainingSampleStore,
    )

    store = ModelTrainingSampleStore()
    store.responsibility_training_store = ResponsibilityWeightedTrainingStore()
    store.append_model_training_samples(
        model_id=-1, training_samples=(sample(0),), observed_concept_ids=(0,)
    )
    store.append_model_training_samples(
        model_id=0, training_samples=(sample(1),), observed_concept_ids=(1,)
    )
    store.reassign_model_training_samples_id(original_model_id=-1, reassigned_model_id=1)
    assert store.responsibility_training_store.training_masses_by_model_id == {1: 1.0, 0: 1.0}
    store.remap_model_training_sample_collections(model_id_mapping={1: 0})
    assert len(store.snapshot_ordered_model_training_samples()[0].training_samples) == 2
    assert store.responsibility_training_store.training_masses_by_model_id == {0: 2.0}
    assert store.responsibility_training_store.mass_by_model_and_concept == {0: {0: 1.0, 1: 1.0}}
    store.clear_model_training_samples()
    assert store.responsibility_training_store.sample_count == 0


@pytest.mark.parametrize("q", [{0: -0.1, 1: 1.1}, {0: float("nan")}, {0: 0.5}, {True: 1.0}])
def test_invalid_responsibilities_are_rejected_before_mutation(q):
    store = ResponsibilityWeightedTrainingStore()
    with pytest.raises((ValueError, TypeError)):
        store.append(training_sample=sample(0), responsibilities=q, observed_concept_id=0)
    assert store.sample_count == 0
    assert store.training_masses_by_model_id == {}


def test_fractional_consolidation_weights_are_preserved():
    from federated_learning_experiments.runtime.model_clustering_and_consolidation import (
        _compute_sample_weighted_mean_parameters,
    )

    parameters, _ = _compute_sample_weighted_mean_parameters(
        model_cluster=(0, 1),
        parameter_snapshots_by_model_id={
            0: {"classification_layer.bias": torch.tensor([2.0])},
            1: {"classification_layer.bias": torch.tensor([6.0])},
        },
        aggregated_training_sample_counts_by_model_id={0: 0.25, 1: 0.75},
    )
    torch.testing.assert_close(parameters["classification_layer.bias"], torch.tensor([5.0]))


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_assignment_strategy", "unknown"),
        ("training_assignment_strategy", True),
        ("record_training_mass_by_concept", 1),
    ],
)
def test_new_setting_values_are_validated(field, value):
    from dataclasses import replace

    from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError

    settings = build_settings("fedsda").run_participant_settings.run_client_settings
    with pytest.raises(RunSettingsValidationError):
        replace(settings, **{field: value})


def test_mass_recording_flag_is_independent_of_soft_assignment():
    from dataclasses import replace

    from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
    from federated_learning_experiments.core.settings_serialization import (
        convert_settings_to_plain_mapping,
    )

    settings = build_settings("fedsda-soft-training").run_participant_settings.run_client_settings
    disabled = replace(settings, record_training_mass_by_concept=False)
    assert "record_training_mass_by_concept" not in convert_settings_to_plain_mapping(disabled)
    assert disabled.training_assignment_strategy == "responsibility_weighted_training"
    with pytest.raises(RunSettingsValidationError):
        replace(
            disabled,
            local_training_settings=replace(
                disabled.local_training_settings,
                local_model_parameter_update_strategy="sequential_per_model_training",
            ),
        )
