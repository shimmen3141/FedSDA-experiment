"""単体予測と独立した学習用Fixed-Shareの契約を検査する。"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_fedsda_soft_training_configuration import build_settings

from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.sweep.experiment_configurations import (
    EXPERIMENT_CONFIGURATION_NAMES,
)
from federated_learning_experiments.sweep.experiment_sweep_plan import make_run_identifier

NAME = "fedsda-soft-training-single-model-prediction"
POLICY = "independent_fixed_share_weight_updates"


def test_existing_configuration_mapping_order_identity_and_protected_files():
    fixture = json.loads(
        Path(__file__)
        .with_name("soft_training_single_model_prediction_existing_configurations.json")
        .read_text(encoding="utf-8")
    )
    assert EXPERIMENT_CONFIGURATION_NAMES == tuple(
        row["name"] for row in fixture["configurations"]
    ) + (NAME,)
    for row in fixture["configurations"]:
        settings = build_settings(row["name"])
        mapping = convert_settings_to_plain_mapping(settings)
        assert mapping == row["settings"], row["name"]
        assert (
            hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
            == row["sha256"]
        )
        assert (
            make_run_identifier(configuration_name=row["name"], run_settings=settings)
            == row["identifier"]
        )
    for path, digest in fixture["protected_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path


def test_new_configuration_has_only_permitted_differences():
    baseline = build_settings("fedsda-single-model-prediction")
    participants = baseline.run_participant_settings
    expected = replace(
        baseline,
        run_participant_settings=replace(
            participants,
            run_client_settings=replace(
                participants.run_client_settings,
                training_assignment_strategy="responsibility_weighted_training",
                record_training_mass_by_concept=True,
                training_responsibility_weight_update_policy=POLICY,
            ),
        ),
    )
    assert build_settings(NAME) == expected
    assert make_run_identifier(
        configuration_name=NAME, run_settings=expected
    ) != make_run_identifier(
        configuration_name="fedsda-single-model-prediction", run_settings=baseline
    )


@pytest.mark.parametrize(
    "change",
    [
        {"training_assignment_strategy": "single_model_assignment"},
        {"training_responsibility_weight_update_policy": "prediction_weight_updates"},
        {"training_responsibility_weight_update_policy": True},
        {"training_responsibility_weight_update_policy": "unknown"},
        {"prediction_combination_strategy": "adahedge_weighted_prediction"},
        {"prediction_combination_strategy": "fixed_share_weighted_prediction"},
        {"prediction_model_parameter_source": "parameters_at_last_synchronization"},
        {
            "prediction_weight_recalibration_after_aggregation_policy": "replay_pending_sample_losses"
        },
        {"local_model_parameter_update_strategy": "sequential_per_model_training"},
    ],
)
def test_invalid_combinations_are_rejected(change):
    client = build_settings(NAME).run_participant_settings.run_client_settings
    with pytest.raises(RunSettingsValidationError):
        if next(iter(change)).startswith("prediction_"):
            replace(
                client,
                prediction_combination_settings=replace(
                    client.prediction_combination_settings, **change
                ),
            )
        elif "local_model_parameter_update_strategy" in change:
            replace(
                client, local_training_settings=replace(client.local_training_settings, **change)
            )
        else:
            replace(client, **change)


def test_auxiliary_observation_and_replay_match_existing_soft_and_preserve_rng():
    import random
    from copy import deepcopy

    import numpy as np
    import torch

    from federated_learning_experiments.evaluation.sample_prediction_record_store import (
        SamplePredictionRecordStore,
    )
    from federated_learning_experiments.evaluation.true_concept_adahedge_diagnostic_evidence_collection import (
        TrueConceptAdaHedgeDiagnosticEvidenceCollection,
    )
    from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
        IndexedObservedTrainingSample,
    )
    from federated_learning_experiments.learning.training.model_training_sample_records import (
        ObservedTrainingSample,
    )
    from federated_learning_experiments.methods.fedsda.prediction_combination.all_sample_adahedge_weight_evidence_holder import (
        AllSampleAdaHedgeWeightEvidenceHolder,
    )
    from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights import (
        FixedShareWeightController,
    )
    from federated_learning_experiments.methods.fedsda.prediction_combination.synchronized_prediction_model_snapshot import (
        SynchronizedPredictionModelSnapshot,
    )
    from federated_learning_experiments.runtime.fedsda_experiment_run import (
        execute_fedsda_experiment_run,
    )
    from federated_learning_experiments.runtime.observed_sample_prediction import (
        predict_observed_sample_and_update_prediction_weights,
    )
    from federated_learning_experiments.runtime.post_aggregation_prediction_recalibration import (
        recalibrate_prediction_state_after_aggregation,
    )
    from federated_learning_experiments.runtime.training_responsibility_weight_updates import (
        recalibrate_training_responsibility_weights_after_aggregation,
        update_training_responsibility_weights_after_label_observation,
    )

    run = execute_fedsda_experiment_run(run_settings=build_settings("fedsda", count=100))
    owners = run.measured_run.participants.client_operations[0].owners
    registry = owners.held_model_training_state_registry
    first = registry.snapshot_ordered_held_model_training_states()[0]
    classifier, optimizer = deepcopy(
        (first.classifier, first.concept_specific_parameter_optimizer_state)
    )
    classifier.feature_extractor = first.classifier.feature_extractor
    registry.register_held_model_training_state(
        model_id=-100, classifier=classifier, concept_specific_parameter_optimizer_state=optimizer
    )
    snapshot = SynchronizedPredictionModelSnapshot()
    snapshot.capture_held_models(held_model_training_state_registry=registry)
    with torch.no_grad():
        classifier.classification_layer.bias.add_(3)
    settings = build_settings(
        "fedsda"
    ).run_participant_settings.run_client_settings.prediction_combination_settings
    left = FixedShareWeightController(prediction_combination_settings=settings)
    right = FixedShareWeightController(prediction_combination_settings=settings)
    records = SamplePredictionRecordStore()
    python_rng = random.Random(19)
    numpy_rng = np.random.RandomState(19)

    def state():
        return (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state().clone(),
            python_rng.getstate(),
            numpy_rng.get_state(),
        )

    def same(before):
        after = state()
        assert before[0] == after[0] and before[3] == after[3]
        torch.testing.assert_close(before[2], after[2], rtol=0, atol=0)
        for index in (1, 4):
            assert before[index][0] == after[index][0]
            np.testing.assert_array_equal(before[index][1], after[index][1])
            assert before[index][2:] == after[index][2:]

    models_before = deepcopy(
        [s.classifier.state_dict() for s in registry.snapshot_ordered_held_model_training_states()]
    )
    from test_joint_model_parameter_update import assert_nested_state_equal

    states = registry.snapshot_ordered_held_model_training_states()
    optimizer_before = deepcopy(
        [
            s.concept_specific_parameter_optimizer_state.parameter_optimizer_if_any.state_dict()
            for s in states
        ]
    )
    gradients_before = [
        [None if p.grad is None else p.grad.clone() for p in s.classifier.parameters()]
        for s in states
    ]
    before = state()
    snapshot.capture_held_models(held_model_training_state_registry=registry)
    same(before)
    _, copies = snapshot.select_prediction_classifiers(held_model_training_state_registry=registry)
    source_storage = {p.data_ptr() for s in states for p in s.classifier.parameters()}
    copy_storage = {
        p.data_ptr(): p.numel() * p.element_size()
        for classifier_copy in copies.values()
        for p in classifier_copy.parameters()
    }
    assert not source_storage.intersection(copy_storage)
    print("学習用同期写しの保持parameter bytes:", sum(copy_storage.values()), "peak RSS: N/A")
    with torch.no_grad():
        classifier.classification_layer.bias.add_(1)
    models_before = deepcopy([s.classifier.state_dict() for s in states])
    from federated_learning_experiments.learning.models.model_computation_measurement import (
        measure_model_computation,
    )
    from federated_learning_experiments.learning.training.responsibility_weighted_training import (
        ResponsibilityWeightedTrainingStore,
    )
    from federated_learning_experiments.runtime.released_sample_training_responsibilities import (
        build_released_sample_responsibility_provider,
    )

    left_store, right_store = (
        ResponsibilityWeightedTrainingStore(),
        ResponsibilityWeightedTrainingStore(),
    )
    for i in range(4):
        if i == 1:
            registry.reassign_held_model_training_state_id(
                original_model_id=-100, reassigned_model_id=7
            )
            for store in (left_store, right_store):
                store.remap(model_id_mapping={-100: 7})
        if i == 2:
            added_classifier, added_optimizer = deepcopy((classifier, optimizer))
            added_classifier.feature_extractor = first.classifier.feature_extractor
            registry.register_held_model_training_state(
                model_id=-200,
                classifier=added_classifier,
                concept_specific_parameter_optimizer_state=added_optimizer,
            )
            assert right_store.training_masses_by_model_id.get(-200, 0) == 0
        if i == 3:
            registry.replace_held_model_training_states(
                held_model_training_states=tuple(
                    s
                    for s in registry.snapshot_ordered_held_model_training_states()
                    if s.model_id != 7
                )
            )
            for store in (left_store, right_store):
                store.remap(model_id_mapping={7: first.model_id})
        observation = IndexedObservedTrainingSample(
            sample_index=i,
            observed_concept_id=None,
            training_sample=ObservedTrainingSample(
                input_features=torch.tensor([[0.1 * i, 0.7]]),
                observed_class_labels=torch.tensor([[float(i % 2)]]),
            ),
        )
        expected = predict_observed_sample_and_update_prediction_weights(
            indexed_observation=observation,
            fixed_share_weight_controller=left,
            all_sample_adahedge_weight_evidence_holder=AllSampleAdaHedgeWeightEvidenceHolder(),
            true_concept_adahedge_diagnostic_evidence_collection=TrueConceptAdaHedgeDiagnosticEvidenceCollection(),
            sample_prediction_record_store=records,
            held_model_training_state_registry=registry,
            current_training_model_assignment=owners.current_training_model_assignment,
            prediction_combination_strategy="fixed_share_weighted_prediction",
            synchronized_prediction_model_snapshot=snapshot,
        )
        before = state()
        with measure_model_computation() as meter:
            actual = update_training_responsibility_weights_after_label_observation(
                indexed_observation=observation,
                held_model_training_state_registry=registry,
                fixed_share_weight_controller=right,
                synchronized_prediction_model_snapshot=snapshot,
            )
        counts = meter.get_model_computation_counts()
        assert counts.shared_part_inference_example_count == 1
        assert counts.concept_specific_part_inference_example_count == len(
            registry.snapshot_ordered_held_model_training_states()
        )
        assert counts.shared_part_inference_forward_multiply_accumulate_count > 0
        assert counts.concept_specific_part_inference_forward_multiply_accumulate_count > 0
        assert (
            counts.shared_part_training_example_count
            == counts.concept_specific_part_training_example_count
            == 0
        )
        assert (
            counts.shared_parameter_optimizer_step_count
            == counts.concept_specific_parameter_optimizer_step_count
            == 0
        )
        assert (
            counts.shared_part_estimated_backward_multiply_accumulate_count
            == counts.concept_specific_part_estimated_backward_multiply_accumulate_count
            == 0
        )
        same(before)
        assert actual == expected.observed_losses_by_model_id
        assert left.weights_by_model_id == right.weights_by_model_id
        assert left.cumulative_observed_loss_variance == right.cumulative_observed_loss_variance
        q_left = build_released_sample_responsibility_provider(
            registry=registry, weight_controller=left
        )(first.model_id, observation.training_sample)
        q_right = build_released_sample_responsibility_provider(
            registry=registry, weight_controller=right
        )(first.model_id, observation.training_sample)
        assert q_left == q_right
        left_store.append(
            training_sample=observation.training_sample,
            responsibilities=q_left,
            observed_concept_id=0,
        )
        right_store.append(
            training_sample=observation.training_sample,
            responsibilities=q_right,
            observed_concept_id=0,
        )
    left_rng, right_rng = random.Random(31), random.Random(31)
    for batch_size in (3, 4, 5):
        left_batches = left_store.sample_batches(
            held_model_ids=frozenset((first.model_id, -100)),
            batch_sample_count=batch_size,
            python_random_generator=left_rng,
        )
        right_batches = right_store.sample_batches(
            held_model_ids=frozenset((first.model_id, -100)),
            batch_sample_count=batch_size,
            python_random_generator=right_rng,
        )
        assert [b.model_id for b in left_batches] == [b.model_id for b in right_batches]
        for left_batch, right_batch in zip(left_batches, right_batches):
            torch.testing.assert_close(
                left_batch.input_features, right_batch.input_features, rtol=0, atol=0
            )
            torch.testing.assert_close(
                left_batch.observed_class_labels, right_batch.observed_class_labels, rtol=0, atol=0
            )
        assert left_rng.getstate() == right_rng.getstate()
    before = state()
    fixed_q_before = deepcopy(right_store._samples)
    with measure_model_computation() as replay_meter:
        actual = recalibrate_training_responsibility_weights_after_aggregation(
            held_model_training_state_registry=registry,
            pending_sample_observation_store=owners.pending_sample_observation_store,
            fixed_share_weight_controller=right,
        )
    replay_counts = replay_meter.get_model_computation_counts()
    assert replay_counts.shared_part_inference_example_count == actual.replayed_sample_count
    assert (
        replay_counts.concept_specific_part_inference_example_count
        == actual.replayed_sample_count * len(actual.replayed_model_ids)
    )
    assert (
        replay_counts.shared_part_training_example_count
        == replay_counts.concept_specific_part_training_example_count
        == 0
    )
    assert (
        replay_counts.shared_parameter_optimizer_step_count
        == replay_counts.concept_specific_parameter_optimizer_step_count
        == 0
    )
    same(before)
    expected = recalibrate_prediction_state_after_aggregation(
        held_model_training_state_registry=registry,
        pending_sample_observation_store=owners.pending_sample_observation_store,
        fixed_share_weight_controller=left,
        all_sample_adahedge_weight_evidence_holder=AllSampleAdaHedgeWeightEvidenceHolder(),
        true_concept_adahedge_diagnostic_evidence_collection=TrueConceptAdaHedgeDiagnosticEvidenceCollection(),
    )
    assert actual.replayed_sample_count == expected.replayed_sample_count
    assert actual.replayed_model_ids == expected.replayed_model_ids
    assert left.weights_by_model_id == right.weights_by_model_id
    assert actual.replayed_sample_count > 0
    for before_row, after_row in zip(fixed_q_before, right_store._samples):
        assert before_row[1:] == after_row[1:]
    for old, s in zip(models_before, states):
        for key, value in old.items():
            torch.testing.assert_close(value, s.classifier.state_dict()[key], rtol=0, atol=0)
    for old_optimizer, old_grads, s in zip(optimizer_before, gradients_before, states):
        assert_nested_state_equal(
            old_optimizer,
            s.concept_specific_parameter_optimizer_state.parameter_optimizer_if_any.state_dict(),
        )
        for old_grad, parameter in zip(old_grads, s.classifier.parameters()):
            if old_grad is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(old_grad, parameter.grad, rtol=0, atol=0)


def test_client_connects_independent_owner_without_changing_prediction_records():
    import numpy as np

    from federated_learning_experiments.runtime.fedsda_experiment_run import (
        execute_fedsda_experiment_run,
    )
    from federated_learning_experiments.runtime.fedsda_run_record_tables import (
        build_fedsda_run_record_tables,
    )

    run = execute_fedsda_experiment_run(run_settings=build_settings(NAME, count=100))
    tables = build_fedsda_run_record_tables(experiment_run=run)
    np.testing.assert_array_equal(tables["sample_prediction.effective_model_count"], 1)
    for client in run.measured_run.participants.client_operations:
        assert (
            client._training_responsibility_weight_controller
            is not client.owners.fixed_share_weight_controller
        )
        assert client._training_responsibility_weight_controller.weights_by_model_id
        assert client.owners.fixed_share_weight_controller.weights_by_model_id == {}


@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["sine2", "mnist4"])
def test_four_configurations_preserve_initialization_generation_order_and_all_client_inputs(
    dataset, monkeypatch
):
    import torch

    from federated_learning_experiments.data.mnist.mnist_sample_generation import (
        MnistSampleGenerator,
    )
    from federated_learning_experiments.data.sine.sine_sample_generation import SineSampleGenerator
    from federated_learning_experiments.runtime import fedsda_run_client as clients
    from federated_learning_experiments.runtime import fedsda_run_participant_factory as factory
    from federated_learning_experiments.runtime.fedsda_experiment_run import (
        execute_fedsda_experiment_run,
    )

    generator_type = SineSampleGenerator if dataset == "sine2" else MnistSampleGenerator
    generate = generator_type.generate_sample
    assemble = factory.assemble_fedsda_run_client
    process = clients.FedsdaRunClient.process_observed_sample
    active_inputs, active_models, active_generated = [], [], []

    def generated(self, **kwargs):
        result = generate(self, **kwargs)
        active_generated.append((kwargs["concept_id"], result))
        return result

    def assembled(**kwargs):
        active_models.append(
            (
                kwargs["client_id"],
                kwargs["initial_model_id"],
                {k: v.clone() for k, v in kwargs["initial_classifier"].state_dict().items()},
            )
        )
        return assemble(**kwargs)

    def processed(self, **kwargs):
        active_inputs.append(
            (
                self.client_id,
                kwargs["sample_index"],
                kwargs.get("evaluation_concept_id"),
                kwargs["observed_sample"],
            )
        )
        return process(self, **kwargs)

    monkeypatch.setattr(generator_type, "generate_sample", generated)
    monkeypatch.setattr(factory, "assemble_fedsda_run_client", assembled)
    monkeypatch.setattr(clients.FedsdaRunClient, "process_observed_sample", processed)
    traces = []
    for name in ("fedsda", "fedsda-soft-training", "fedsda-single-model-prediction", NAME):
        active_inputs.clear()
        active_models.clear()
        active_generated.clear()
        execute_fedsda_experiment_run(run_settings=build_settings(name, dataset, count=100))
        traces.append((list(active_inputs), list(active_models), list(active_generated)))
    for inputs, models, generated in traces[1:]:
        assert inputs == traces[0][0]
        assert generated == traces[0][2]
        assert len(models) == len(traces[0][1]) == 2
        for a, b in zip(models, traces[0][1]):
            assert a[:2] == b[:2]
            for key in a[2]:
                torch.testing.assert_close(a[2][key], b[2][key], rtol=0, atol=0)


def test_same_local_state_keeps_four_monitoring_results_identical():
    from copy import deepcopy

    import torch

    from federated_learning_experiments.data.observed_streams import ObservedSample
    from federated_learning_experiments.runtime.fedsda_experiment_run import (
        execute_fedsda_experiment_run,
    )

    initial = execute_fedsda_experiment_run(
        run_settings=build_settings(NAME, count=30)
    ).measured_run.participants.client_operations[0]
    results = []
    for name in ("fedsda", "fedsda-soft-training", "fedsda-single-model-prediction", NAME):
        client = deepcopy(initial)
        client._run_client_settings = build_settings(
            name
        ).run_participant_settings.run_client_settings
        if name != NAME:
            client._training_responsibility_weight_controller = None
            client._training_responsibility_model_snapshot = None
        last_index = client.owners.sample_prediction_record_store.last_recorded_sample_index
        sample_index = 0 if last_index is None else last_index + 1
        result = client.process_observed_sample(
            observed_sample=ObservedSample(feature_values=(0.2, 0.8), class_label=1),
            sample_index=sample_index,
            evaluation_concept_id=0,
        )
        results.append(result)
    assert all(
        r.loss_monitoring_observation == results[0].loss_monitoring_observation for r in results
    )
    torch.testing.assert_close(
        results[2].observed_sample_prediction.combined_prediction_probabilities,
        results[3].observed_sample_prediction.combined_prediction_probabilities,
        rtol=0,
        atol=0,
    )
    predictions = []
    for label in (0, 1):
        client = deepcopy(initial)
        last_index = client.owners.sample_prediction_record_store.last_recorded_sample_index
        result = client.process_observed_sample(
            observed_sample=ObservedSample(feature_values=(0.2, 0.8), class_label=label),
            sample_index=0 if last_index is None else last_index + 1,
            evaluation_concept_id=0,
        )
        predictions.append(result.observed_sample_prediction.combined_prediction_probabilities)
    torch.testing.assert_close(predictions[0], predictions[1], rtol=0, atol=0)


@pytest.mark.parametrize("empty_pending", [False, True])
def test_replay_empty_pending_and_one_model_are_noops(empty_pending):
    from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_sample_observation_store import (
        PendingSampleObservationStore,
    )
    from federated_learning_experiments.runtime.fedsda_experiment_run import (
        execute_fedsda_experiment_run,
    )
    from federated_learning_experiments.runtime.training_responsibility_weight_updates import (
        recalibrate_training_responsibility_weights_after_aggregation,
    )

    owners = (
        execute_fedsda_experiment_run(run_settings=build_settings("fedsda", count=100))
        .measured_run.participants.client_operations[0]
        .owners
    )
    controller = owners.fixed_share_weight_controller
    before = dict(controller.__dict__)
    result = recalibrate_training_responsibility_weights_after_aggregation(
        held_model_training_state_registry=owners.held_model_training_state_registry,
        pending_sample_observation_store=PendingSampleObservationStore()
        if empty_pending
        else owners.pending_sample_observation_store,
        fixed_share_weight_controller=controller,
    )
    assert result.replayed_sample_count == 0 and result.replayed_model_ids == ()
    assert before == controller.__dict__


def test_prediction_label_update_monitor_release_and_joint_training_order(monkeypatch):
    from federated_learning_experiments.learning.training import (
        held_model_joint_training_iterations as iterations,
    )
    from federated_learning_experiments.runtime import fedsda_run_client as clients
    from federated_learning_experiments.runtime import observed_sample_processing as processing
    from federated_learning_experiments.runtime.fedsda_experiment_run import (
        execute_fedsda_experiment_run,
    )

    predict = processing.predict_observed_sample_and_update_prediction_weights
    update = processing.update_training_responsibility_weights_after_label_observation
    monitor = processing.evaluate_classifier_per_sample_bounded_losses
    build_provider = clients.build_released_sample_responsibility_provider
    train = iterations.perform_joint_model_parameter_update
    process = clients.FedsdaRunClient.process_observed_sample
    events = []
    active = []
    traces = []

    def predicted(**kwargs):
        result = predict(**kwargs)
        if active:
            events.append("predict")
        return result

    def updated(**kwargs):
        if active:
            events.append("aux")
        return update(**kwargs)

    def monitored(**kwargs):
        if active:
            events.append("monitor")
        return monitor(**kwargs)

    def provider(**kwargs):
        callback = build_provider(**kwargs)

        def responsibility(*args):
            if active:
                events.append("q")
            return callback(*args)

        return responsibility

    def trained(**kwargs):
        if active:
            events.append("train")
        return train(**kwargs)

    def processed(self, **kwargs):
        events.clear()
        active.append(True)
        try:
            result = process(self, **kwargs)
            traces.append(tuple(events))
            return result
        finally:
            active.clear()

    monkeypatch.setattr(
        processing, "predict_observed_sample_and_update_prediction_weights", predicted
    )
    monkeypatch.setattr(
        processing, "update_training_responsibility_weights_after_label_observation", updated
    )
    monkeypatch.setattr(processing, "evaluate_classifier_per_sample_bounded_losses", monitored)
    monkeypatch.setattr(clients, "build_released_sample_responsibility_provider", provider)
    monkeypatch.setattr(iterations, "perform_joint_model_parameter_update", trained)
    monkeypatch.setattr(clients.FedsdaRunClient, "process_observed_sample", processed)
    execute_fedsda_experiment_run(run_settings=build_settings(NAME, count=100))
    assert (
        traces
        and any("q" in trace for trace in traces)
        and any("train" in trace for trace in traces)
    )
    for trace in traces:
        assert trace[:3] == ("predict", "aux", "monitor")
        assert trace.count("aux") == 1
        if "q" in trace and "train" in trace:
            assert trace.index("q") < trace.index("train")


def test_auxiliary_cost_is_in_local_and_total_measurement_without_extra_communication(monkeypatch):
    from dataclasses import fields
    from time import perf_counter

    from federated_learning_experiments.learning.models.model_computation_measurement import (
        ModelComputationCounts,
        measure_model_computation,
        subtract_model_computation_counts,
    )
    from federated_learning_experiments.runtime import fedsda_run_client as clients
    from federated_learning_experiments.runtime import fedsda_run_participant_factory as factory
    from federated_learning_experiments.runtime import observed_sample_processing as processing
    from federated_learning_experiments.runtime.fedsda_experiment_run import (
        execute_fedsda_experiment_run,
    )

    update = processing.update_training_responsibility_weights_after_label_observation
    replay = clients.recalibrate_training_responsibility_weights_after_aggregation
    assemble_server = factory.assemble_fedsda_run_server
    communication = []
    observed_counts, replay_counts = [], []
    auxiliary_intervals = []

    def assembled(**kwargs):
        result = assemble_server(**kwargs)
        communication.append(result.owners.communication_volume_record_store)
        return result

    def measured(operation, collection, kwargs):
        before = communication[0].get_state_snapshot()
        earlier_counts = run_meter.get_model_computation_counts()
        started = perf_counter()
        with measure_model_computation() as meter:
            result = operation(**kwargs)
        auxiliary_intervals.append((started, perf_counter()))
        assert communication[0].get_state_snapshot() == before
        counts = meter.get_model_computation_counts()
        # 全体runの実計測器の増分が、補助処理だけを測った全fieldと完全一致する。
        assert (
            subtract_model_computation_counts(
                later_counts=run_meter.get_model_computation_counts(), earlier_counts=earlier_counts
            )
            == counts
        )
        assert (
            counts.shared_parameter_optimizer_step_count
            == counts.concept_specific_parameter_optimizer_step_count
            == 0
        )
        assert (
            counts.shared_part_training_example_count
            == counts.concept_specific_part_training_example_count
            == 0
        )
        collection.append(counts)
        return result

    monkeypatch.setattr(factory, "assemble_fedsda_run_server", assembled)
    monkeypatch.setattr(
        processing,
        "update_training_responsibility_weights_after_label_observation",
        lambda **kwargs: measured(update, observed_counts, kwargs),
    )
    monkeypatch.setattr(
        clients,
        "recalibrate_training_responsibility_weights_after_aggregation",
        lambda **kwargs: measured(replay, replay_counts, kwargs),
    )
    started = perf_counter()
    with measure_model_computation() as run_meter:
        run = execute_fedsda_experiment_run(run_settings=build_settings(NAME, count=1500))
    ended = perf_counter()
    assert len(observed_counts) == run.run_metrics.processed_sample_count
    assert replay_counts
    # 呼び出しの存在だけでなく、複数モデルの保留損失の再計算を実際に通す。
    assert sum(c.shared_part_inference_example_count for c in replay_counts) > 0
    assert sum(c.concept_specific_part_inference_example_count for c in replay_counts) > sum(
        c.shared_part_inference_example_count for c in replay_counts
    )
    assert all(started <= a < b <= ended for a, b in auxiliary_intervals)
    assert 0 < sum(b - a for a, b in auxiliary_intervals) <= ended - started
    for field, collected in (
        ("local_processing_model_computation_counts", observed_counts),
        ("synchronization_model_computation_counts", replay_counts),
    ):
        for name in (
            "shared_part_inference_example_count",
            "concept_specific_part_inference_example_count",
            "shared_part_inference_forward_multiply_accumulate_count",
            "concept_specific_part_inference_forward_multiply_accumulate_count",
        ):
            measured_count = sum(
                getattr(r, field).__getattribute__(name)
                for r in run.measured_run.round_model_computation_counts
            )
            assert measured_count >= sum(getattr(c, name) for c in collected)
    counts = run.run_metrics.model_computation_counts
    # 局所処理・同期処理・終端の内訳の合計と保存されるrun総計を全fieldで照合する。
    for field in fields(ModelComputationCounts):
        assert getattr(counts, field.name) == sum(
            getattr(r.local_processing_model_computation_counts, field.name)
            + getattr(r.synchronization_model_computation_counts, field.name)
            for r in run.measured_run.round_model_computation_counts
        ) + getattr(run.measured_run.finalization_model_computation_counts, field.name)
    assert run.run_metrics.computation_cost_summary.client_forward_multiply_accumulate_count == sum(
        getattr(counts, name)
        for name in (
            "shared_part_inference_forward_multiply_accumulate_count",
            "concept_specific_part_inference_forward_multiply_accumulate_count",
            "shared_part_training_forward_multiply_accumulate_count",
            "concept_specific_part_training_forward_multiply_accumulate_count",
        )
    )
