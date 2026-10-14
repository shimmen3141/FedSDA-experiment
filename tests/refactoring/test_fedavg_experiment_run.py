"""単一モデルの学習と全体runを実旧Obliviousへ照合する。"""

import random
from copy import deepcopy
from dataclasses import fields, replace
from random import Random

import numpy as np
import pytest
import torch
from run_provenance_examples import TEST_RUN_PROVENANCE

from federated_drift_experiment import config, experiment
from federated_drift_experiment.clients.oblivious import ObliviousClient
from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.data.observed_streams import ObservedSample
from federated_learning_experiments.learning.training.initial_model_pretraining_settings import (
    InitialModelPretrainingSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.methods.fedavg.fedavg_client import FedavgClient
from federated_learning_experiments.runtime.fedavg_experiment_run import (
    execute_fedavg_experiment_run,
)
from federated_learning_experiments.runtime.fedavg_run_artifact_files import (
    load_fedavg_run_artifacts,
    save_fedavg_experiment_run,
)
from federated_learning_experiments.runtime.fedavg_run_participant_factory import (
    FedavgRunParticipantFactory,
)
from federated_learning_experiments.runtime.fedavg_run_settings import (
    build_default_fedavg_run_settings,
)
from federated_learning_experiments.runtime.single_run_execution import execute_stream_protocol_run
from tests.refactoring.test_feddrift_client_round_training import assert_values_equal
from tests.refactoring.test_feddrift_run_client import build_assembly_inputs
from tests.refactoring.test_initial_model_pretraining import assert_numpy_random_states_equal


def make_client_pair(monkeypatch, optimizer="adam", interval=3, updates=2, batch=3):
    old, inputs = build_assembly_inputs(monkeypatch=monkeypatch, optimizer_name=optimizer)
    for key, value in dict(
        LOCAL_UPDATE_INTERVAL=interval, UPDATES_PER_SAMPLE=updates, CLIENT_BATCH_SIZE=batch
    ).items():
        monkeypatch.setattr(config, key, value)
    legacy = ObliviousClient(6, old.models, old.model_stats, verbose=False)
    new = FedavgClient(
        client_id=6,
        initial_classifier=inputs["initial_classifier"],
        initial_parameter_optimizer_state=inputs["initial_parameter_optimizer_state"],
        initial_loss_statistics=inputs["initial_loss_statistics"],
        local_training_batch_sample_count=batch,
        updates_per_sample=updates,
        local_update_interval_per_client_samples=interval,
        python_random_generator=Random(17),
    )
    return legacy, new


def assert_client_matches_legacy(old, new):
    assert [float(p.prediction_is_correct) for p in new.sample_predictions] == old.history_accuracy
    assert [p.prediction_model_id for p in new.sample_predictions] == old.history_model_id
    assert [p.observed_concept_id for p in new.sample_predictions] == old.history_concept
    assert old.history_drift_type == [0] * len(new.sample_predictions)
    assert new.pending_update_sample_count == old._pending_updates
    assert new.trained_sample_count == old.model_training_examples.get(0, 0)
    assert new.parameter_update_step_count == old.model_optimizer_steps.get(0, 0)
    for a, b in zip(new.classifier.parameters(), old.models[0].parameters(), strict=True):
        assert torch.equal(a, b)
    assert_values_equal(
        new.parameter_optimizer_state.parameter_optimizer.state_dict(),
        old.models[0].optimizer.state_dict(),
    )
    stats = new.loss_statistics_store.get_model_loss_statistics(model_id=0)
    moments = stats.overall_loss_moments
    assert (
        moments.observed_loss_count,
        moments.mean_loss,
        moments.sum_squared_loss_deviations,
    ) == (old.model_stats[0]["n"], old.model_stats[0]["mean"], old.model_stats[0]["M2"])
    assert {
        k: (m.observed_loss_count, m.mean_loss, m.sum_squared_loss_deviations)
        for k, m in stats.class_loss_moments_by_class_id
    } == {
        k: (m["n"], m["mean"], m["M2"])
        for k, m in old.model_stats[0].get("class_stats", {}).items()
    }
    assert len(new.training_samples) == len(old.train_data_store.get(0, []))
    for (x, y), (ox, oy) in zip(new.training_samples, old.train_data_store.get(0, []), strict=True):
        assert torch.equal(x, ox) and torch.equal(y, oy)


@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
@pytest.mark.parametrize("interval,updates,batch", [(1, 1, 3), (3, 2, 3), (4, 0, 3), (4, 2, 20)])
def test_client_matches_real_oblivious(monkeypatch, optimizer, interval, updates, batch):
    old, new = make_client_pair(monkeypatch, optimizer, interval, updates, batch)
    random.seed(17)
    for index in range(11):
        sample = ObservedSample(feature_values=(index / 20.0, -0.2, 0.7), class_label=index % 2)
        old.process_one_step(
            torch.tensor(sample.feature_values),
            torch.tensor([float(sample.class_label)]),
            index % 3,
        )
        new.process_observed_sample(
            observed_sample=sample, sample_index=index, evaluation_concept_id=index % 3
        )
        if index in (4, 10):
            old.flush_pending_updates()
            new.flush_pending_local_updates(round_index=0)
        assert_client_matches_legacy(old, new)
        assert random.getstate() == new.python_random_generator.getstate()


@pytest.mark.parametrize("bad", ["position", "concept", "width", "class", "finite"])
def test_invalid_observation_leaves_state_and_rng_unchanged(monkeypatch, bad):
    old, new = make_client_pair(monkeypatch)
    sample = ObservedSample(feature_values=(0.2, 0.3, 0.4), class_label=0)
    index = 0
    concept = 0
    if bad == "position":
        index = True
    if bad == "concept":
        concept = True
    if bad == "width":
        sample = ObservedSample(feature_values=(0.2,), class_label=0)
    if bad == "class":
        sample = ObservedSample(feature_values=(0.2, 0.3, 0.4), class_label=2)
    if bad == "finite":
        sample = ObservedSample(feature_values=(float("nan"), 0.3, 0.4), class_label=0)
    rng = new.python_random_generator.getstate()
    state = deepcopy(new.parameter_optimizer_state.parameter_optimizer.state_dict())
    with pytest.raises((ValueError, TypeError)):
        new.process_observed_sample(
            observed_sample=sample, sample_index=index, evaluation_concept_id=concept
        )
    assert new.python_random_generator.getstate() == rng
    assert_values_equal(new.parameter_optimizer_state.parameter_optimizer.state_dict(), state)
    assert_client_matches_legacy(old, new)


def test_true_concept_is_diagnostic_only(monkeypatch):
    _, a = make_client_pair(monkeypatch)
    b = deepcopy(a)
    for index in range(8):
        sample = ObservedSample(feature_values=(index / 20.0, -0.2, 0.7), class_label=index % 2)
        a.process_observed_sample(
            observed_sample=sample, sample_index=index, evaluation_concept_id=0
        )
        b.process_observed_sample(
            observed_sample=sample, sample_index=index, evaluation_concept_id=3
        )
    for pa, pb in zip(a.classifier.parameters(), b.classifier.parameters(), strict=True):
        assert torch.equal(pa, pb)
    assert_values_equal(
        a.parameter_optimizer_state.parameter_optimizer.state_dict(),
        b.parameter_optimizer_state.parameter_optimizer.state_dict(),
    )
    assert a.python_random_generator.getstate() == b.python_random_generator.getstate()
    assert [p.predicted_class_id for p in a.sample_predictions] == [
        p.predicted_class_id for p in b.sample_predictions
    ]


def small_run_settings(
    dataset="sine2", optimizer="adam", interval=4, samples=13, updates=2, local_interval=3, batch=3
):
    conditions = ExperimentRunConditions(
        dataset_name=dataset,
        random_seed=37,
        client_count=2,
        per_client_sample_count=samples,
        server_aggregation_interval_per_client_samples=interval,
    )
    settings = build_default_fedavg_run_settings(experiment_run_conditions=conditions)
    optimization = (
        SgdParameterOptimizerSettings(learning_rate=0.01)
        if optimizer == "sgd"
        else AdamParameterOptimizerSettings(
            learning_rate=0.01, weight_decay=0.0, adam_variant="standard"
        )
    )
    return replace(
        settings,
        execution_settings=replace(
            settings.execution_settings,
            concept_schedule_settings=replace(
                settings.execution_settings.concept_schedule_settings,
                minimum_sample_index_gap_before_change_trial=2,
                per_eligible_sample_concept_change_probability=0.6,
            ),
        ),
        run_participant_settings=replace(
            settings.run_participant_settings,
            hidden_layer_widths=(7, 5),
            parameter_optimizer_settings=optimization,
            initial_model_pretraining_settings=InitialModelPretrainingSettings(
                pretraining_sample_count=7,
                pretraining_epoch_count=2,
                pretraining_batch_sample_count=3,
            ),
            local_training_batch_sample_count=batch,
            updates_per_sample=updates,
            local_update_interval_per_client_samples=local_interval,
        ),
    )


def run_real_oblivious(monkeypatch, settings):
    """旧setupと旧1時刻処理をそのまま呼ぶ。学習/集約の式は複製しない。"""
    conditions = settings.execution_settings.experiment_run_conditions
    participants = settings.run_participant_settings
    optimizer = participants.parameter_optimizer_settings
    original = config.dataset_spec
    monkeypatch.setattr(
        config,
        "dataset_spec",
        lambda dataset=None: replace(
            original(dataset),
            hidden_dims=participants.hidden_layer_widths,
            learning_rate=optimizer.learning_rate,
        ),
    )
    schedule = settings.execution_settings.concept_schedule_settings
    for k, v in dict(
        DATASET=conditions.dataset_name,
        N_CLIENTS=conditions.client_count,
        TOTAL_DATA_POINTS=conditions.per_client_sample_count,
        AGGREGATION_INTERVAL=conditions.server_aggregation_interval_per_client_samples,
        CONCEPT_SCHEDULE="random",
        MIN_STABLE_PERIOD=schedule.minimum_sample_index_gap_before_change_trial,
        DRIFT_PROB=schedule.per_eligible_sample_concept_change_probability,
        PRETRAIN_SAMPLES=participants.initial_model_pretraining_settings.pretraining_sample_count,
        PRETRAIN_EPOCHS=participants.initial_model_pretraining_settings.pretraining_epoch_count,
        PRETRAIN_BATCH_SIZE=participants.initial_model_pretraining_settings.pretraining_batch_sample_count,
        OPTIMIZER="sgd" if type(optimizer) is SgdParameterOptimizerSettings else "adam",
        BASE_LR=optimizer.learning_rate,
        WEIGHT_DECAY=getattr(optimizer, "weight_decay", 0.0),
        AMSGRAD=getattr(optimizer, "adam_variant", None) == "amsgrad",
        CLIENT_BATCH_SIZE=participants.local_training_batch_sample_count,
        UPDATES_PER_SAMPLE=participants.updates_per_sample,
        LOCAL_UPDATE_INTERVAL=participants.local_update_interval_per_client_samples,
    ).items():
        monkeypatch.setattr(config, k, v)
    saved_py = random.getstate()
    saved_np = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(conditions.random_seed)
            np.random.seed(conditions.random_seed)
            torch.manual_seed(conditions.random_seed)
            server, clients = experiment._setup_server_and_clients(
                experiment.MODE_SPECS["Oblivious"], None, False
            )
            traces = experiment.make_concept_schedules(
                conditions.client_count, conditions.per_client_sample_count
            )
            streams = experiment.build_data_streams(traces)
            chunk = conditions.server_aggregation_interval_per_client_samples
            for round_index in range(conditions.per_client_sample_count // chunk):
                start = round_index * chunk
                experiment._run_per_sample_timestep(
                    clients,
                    server,
                    [s[start : start + chunk] for s in streams],
                    [t[start : start + chunk] for t in traces],
                    round_index,
                    True,
                    False,
                )
            server.finalize_protocol(conditions.per_client_sample_count // chunk)
            return (
                server,
                clients,
                traces,
                streams,
                random.getstate(),
                np.random.get_state(),
                torch.get_rng_state().clone(),
            )
    finally:
        random.setstate(saved_py)
        np.random.set_state(saved_np)


@pytest.mark.parametrize("dataset", ["sine2", "sea2", "sea4", "circle2", "mnist2", "mnist4"])
@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
def test_whole_run_matches_real_oblivious(monkeypatch, dataset, optimizer):
    settings = small_run_settings(dataset, optimizer)
    old, clients, traces, streams, py_state, np_state, torch_state = run_real_oblivious(
        monkeypatch, settings
    )
    factory = FedavgRunParticipantFactory(
        run_participant_settings=settings.run_participant_settings
    )
    end_states = []
    original = FedavgClient.apply_global_parameters

    def observe(self, **kwargs):
        original(self, **kwargs)
        end_states.append(torch.get_rng_state().clone())

    monkeypatch.setattr(FedavgClient, "apply_global_parameters", observe)
    caller = torch.get_rng_state().clone()
    result = execute_stream_protocol_run(
        execution_settings=settings.execution_settings, participant_factory=factory
    )
    assert torch.equal(torch.get_rng_state(), caller)
    participants = factory.prepared_run_participants
    new_server = participants.server_operations
    assert result.processed_sample_count_per_client == 12
    assert result.unprocessed_tail_sample_count_per_client == 1
    for old_client, new_client in zip(clients, participants.client_operations, strict=True):
        assert_client_matches_legacy(old_client, new_client)
    for new_value, old_value in zip(
        new_server.global_parameters.values(), old.global_models[0].values(), strict=True
    ):
        assert torch.equal(new_value, old_value)
    assert [t.concept_ids_by_sample_index for t in result.evaluation_concept_traces] == [
        tuple(t) for t in traces
    ]
    for new_stream, old_stream in zip(result.observed_client_streams, streams, strict=True):
        for new_sample, (x, y) in zip(new_stream.observed_samples, old_stream, strict=True):
            assert tuple(float(v) for v in x) == new_sample.feature_values
            assert int(y.item()) == new_sample.class_label
    assert factory._sources.python_random_generator.getstate() == py_state
    assert_numpy_random_states_equal(factory._sources.numpy_random_generator.get_state(), np_state)
    assert torch.equal(end_states[-1], torch_state)
    communication = new_server.communication_volume_record_store.get_state_snapshot()
    assert (
        communication.uploaded_model_count,
        communication.downloaded_model_count,
        communication.uploaded_message_count,
        communication.downloaded_message_count,
        communication.uploaded_parameter_value_count,
        communication.downloaded_parameter_value_count,
        communication.uploaded_byte_count,
        communication.downloaded_byte_count,
    ) == (
        old.comm_models_up,
        old.comm_models_down,
        old.comm_messages_up,
        old.comm_messages_down,
        old.comm_parameter_values_up,
        old.comm_parameter_values_down,
        old.comm_bytes_up,
        old.comm_bytes_down,
    )
    stats = new_server.global_loss_statistics.overall_loss_moments
    assert (stats.observed_loss_count, stats.mean_loss, stats.sum_squared_loss_deviations) == (
        old.global_stats[0]["n"],
        old.global_stats[0]["mean"],
        old.global_stats[0]["M2"],
    )


@pytest.mark.parametrize("interval,samples", [(1, 9), (4, 13), (20, 13)])
@pytest.mark.parametrize("updates,batch", [(0, 3), (2, 3), (2, 30)])
def test_measured_run_has_common_counts_and_preserves_tail(interval, samples, updates, batch):
    run = execute_fedavg_experiment_run(
        run_settings=small_run_settings(
            interval=interval, samples=samples, updates=updates, batch=batch
        )
    )
    measured = run.measured_run
    processed = samples // interval * interval
    clients = measured.participants.client_operations
    metrics = run.run_metrics
    assert metrics.final_global_model_count == 1
    assert metrics.processed_sample_count == 2 * processed == metrics.held_model_sample_count
    assert metrics.training_model_switch_detection_metrics.detection_count == 0
    assert metrics.communication_volume.uploaded_model_count == 2 * (samples // interval)
    assert metrics.communication_volume.downloaded_model_count == 2 * (samples // interval)
    assert (
        metrics.communication_volume.uploaded_message_count
        == metrics.communication_volume.downloaded_message_count
        == 0
    )
    counts = measured.model_computation_counts
    assert (
        counts.concept_specific_part_inference_example_count == 2 * metrics.processed_sample_count
    )
    assert counts.concept_specific_part_training_example_count == sum(
        c.trained_sample_count for c in clients
    )
    assert counts.concept_specific_parameter_optimizer_step_count == sum(
        c.parameter_update_step_count for c in clients
    )
    for field in fields(counts):
        assert getattr(counts, field.name) == sum(
            getattr(r.local_processing_model_computation_counts, field.name)
            + getattr(r.synchronization_model_computation_counts, field.name)
            for r in measured.round_model_computation_counts
        ) + getattr(measured.finalization_model_computation_counts, field.name)
    assert (
        measured.preparation_model_computation_counts.concept_specific_part_training_example_count
        == 14
    )
    assert measured.run_result.unprocessed_tail_sample_count_per_client == samples % interval
    assert all(len(c.training_samples) == processed for c in clients)


@pytest.mark.parametrize("samples,interval", [(13, 4), (3, 20)])
def test_run_artifact_roundtrip(tmp_path, samples, interval):
    run = execute_fedavg_experiment_run(
        run_settings=small_run_settings(samples=samples, interval=interval)
    )
    paths = save_fedavg_experiment_run(
        experiment_run=run,
        output_directory=tmp_path,
        file_stem="baseline",
        run_provenance=TEST_RUN_PROVENANCE,
    )
    summary, tables = load_fedavg_run_artifacts(summary_file_path=paths.summary_file_path)
    assert summary["method_name"] == "fedavg"
    assert summary["run_settings"] == __import__("json").loads(
        __import__("json").dumps(convert_settings_to_plain_mapping(run.run_settings))
    )
    assert summary["run_metrics"] == convert_settings_to_plain_mapping(
        run.run_metrics, non_finite_float_as_none=True
    )
    assert tables["sample_prediction.sample_index"].shape == (2, samples // interval * interval)
    assert tables["generated_concept.concept_id"].shape == (2, samples)
    assert np.all(tables["sample_prediction.prediction_model_id"] == 0)
    assert np.all(tables["sample_prediction.held_model_count_at_prediction"] == 1)
    assert len(tables["synchronization.round_index"]) == samples // interval
    with pytest.raises(FileExistsError):
        save_fedavg_experiment_run(
            experiment_run=run,
            output_directory=tmp_path,
            file_stem="baseline",
            run_provenance=TEST_RUN_PROVENANCE,
        )
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("suffix", [".json", ".npz"])
def test_existing_half_artifact_is_not_overwritten(tmp_path, suffix):
    run = execute_fedavg_experiment_run(run_settings=small_run_settings())
    (tmp_path / ("baseline" + suffix)).write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        save_fedavg_experiment_run(
            experiment_run=run,
            output_directory=tmp_path,
            file_stem="baseline",
            run_provenance=TEST_RUN_PROVENANCE,
        )
    assert list(tmp_path.iterdir()) == [tmp_path / ("baseline" + suffix)]
    assert (tmp_path / ("baseline" + suffix)).read_bytes() == b"keep"


def test_stale_summary_and_inconsistent_predictions_are_rejected_before_write(tmp_path):
    run = execute_fedavg_experiment_run(run_settings=small_run_settings())
    stale = replace(run, run_metrics=replace(run.run_metrics, prediction_accuracy=0.12345))
    with pytest.raises(ValueError):
        save_fedavg_experiment_run(
            experiment_run=stale,
            output_directory=tmp_path,
            file_stem="stale",
            run_provenance=TEST_RUN_PROVENANCE,
        )
    client = run.measured_run.participants.client_operations[0]
    client.sample_predictions[0] = replace(client.sample_predictions[0], sample_index=9)
    with pytest.raises(ValueError):
        save_fedavg_experiment_run(
            experiment_run=run,
            output_directory=tmp_path,
            file_stem="invalid",
            run_provenance=TEST_RUN_PROVENANCE,
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("bad", ["method", "version", "path"])
def test_loader_rejects_other_method_and_unsafe_reference(tmp_path, bad):
    import json

    run = execute_fedavg_experiment_run(run_settings=small_run_settings())
    paths = save_fedavg_experiment_run(
        experiment_run=run,
        output_directory=tmp_path,
        file_stem="baseline",
        run_provenance=TEST_RUN_PROVENANCE,
    )
    summary = json.loads(paths.summary_file_path.read_text("utf-8"))
    if bad == "method":
        summary["method_name"] = "feddrift"
    if bad == "version":
        summary["format_version"] = True
    if bad == "path":
        summary["record_table_file_name"] = "../baseline.npz"
    paths.summary_file_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError):
        load_fedavg_run_artifacts(summary_file_path=paths.summary_file_path)


@pytest.mark.parametrize("weights", [(1, 3), (0, 3), (0, 0)])
def test_server_uses_unequal_retained_counts_and_counts_only_uploading_clients(
    monkeypatch, weights
):
    from federated_learning_experiments.data.observed_sample_generation import (
        create_observed_sample_generator,
    )
    from federated_learning_experiments.execution.run_random_sources import (
        create_run_random_sources,
    )

    settings = small_run_settings(samples=1, interval=20, updates=0)
    old, old_clients, *_ = run_real_oblivious(monkeypatch, settings)
    sources = create_run_random_sources(random_seed=37)
    factory = FedavgRunParticipantFactory(
        run_participant_settings=settings.run_participant_settings
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(37)
        participants = factory.prepare_run(
            experiment_run_conditions=settings.execution_settings.experiment_run_conditions,
            run_random_sources=sources,
            sample_generator=create_observed_sample_generator(
                dataset_name="sine2", numpy_random_generator=sources.numpy_random_generator
            ),
        )
    for client_id, (old_client, new_client) in enumerate(
        zip(old_clients, participants.client_operations, strict=True)
    ):
        with torch.no_grad():
            for op, np_ in zip(
                old_client.models[0].parameters(), new_client.classifier.parameters(), strict=True
            ):
                op.add_(client_id * 0.25)
                np_.add_(client_id * 0.25)
        for i in range(weights[client_id]):
            sample = ObservedSample(feature_values=(0.2, 0.3), class_label=i % 2)
            old_client.process_one_step(
                torch.tensor(sample.feature_values), torch.tensor([float(sample.class_label)]), 0
            )
            new_client.process_observed_sample(
                observed_sample=sample, sample_index=i, evaluation_concept_id=0
            )
    old.run_round(0)
    server = participants.server_operations
    server.synchronize_models(round_index=0, new_model_registration_available=False)
    for ov, nv in zip(
        old.global_models[0].values(), server.global_parameters.values(), strict=True
    ):
        assert torch.equal(ov, nv)
    for oc, nc in zip(old_clients, participants.client_operations, strict=True):
        assert_client_matches_legacy(oc, nc)
    comm = server.communication_volume_record_store.get_state_snapshot()
    assert comm.uploaded_model_count == sum(w > 0 for w in weights) == old.comm_models_up
    assert comm.downloaded_model_count == 2 == old.comm_models_down


@pytest.mark.parametrize("bad", ["nan", "shape", "keys"])
def test_invalid_distribution_preserves_model_optimizer_and_torch_rng(monkeypatch, bad):
    from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
        snapshot_classifier_parameters,
    )

    _, client = make_client_pair(monkeypatch)
    parameters = snapshot_classifier_parameters(classifier=client.classifier)
    if bad == "keys":
        parameters.pop(next(iter(parameters)))
    else:
        key = next(iter(parameters))
        if bad == "nan":
            parameters[key].fill_(float("nan"))
        if bad == "shape":
            parameters[key] = parameters[key].reshape(-1)
    classifier = client.classifier
    optimizer = client.parameter_optimizer_state
    state = torch.get_rng_state().clone()
    with pytest.raises((ValueError, TypeError)):
        client.apply_global_parameters(parameter_snapshot=parameters)
    assert client.classifier is classifier and client.parameter_optimizer_state is optimizer
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize(
    "name,value",
    [
        ("updates_per_sample", True),
        ("local_training_batch_sample_count", 0),
        ("local_update_interval_per_client_samples", 0),
        ("hidden_layer_widths", (True,)),
    ],
)
def test_forged_settings_fail_before_model_generation(monkeypatch, name, value):
    from federated_learning_experiments.execution.run_execution_errors import RunExecutionError

    settings = small_run_settings()
    object.__setattr__(settings.run_participant_settings, name, value)
    called = []
    monkeypatch.setattr(
        "federated_learning_experiments.runtime.fedavg_run_participant_factory.pretrain_initial_model",
        lambda **kwargs: called.append(kwargs),
    )
    state = torch.get_rng_state().clone()
    with pytest.raises((ValueError, TypeError, RunExecutionError)):
        execute_fedavg_experiment_run(run_settings=settings)
    assert not called and torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("dataset", ["sine2", "sea2", "sea4", "circle2", "mnist2", "mnist4"])
def test_default_conditions_match_real_legacy_dataset_and_learning_defaults(dataset):
    conditions = ExperimentRunConditions(
        dataset_name=dataset,
        random_seed=0,
        client_count=2,
        per_client_sample_count=5,
        server_aggregation_interval_per_client_samples=3,
    )
    settings = build_default_fedavg_run_settings(experiment_run_conditions=conditions)
    participant = settings.run_participant_settings
    definition = config.dataset_spec(dataset)
    assert participant.hidden_layer_widths == definition.hidden_dims
    assert participant.parameter_optimizer_settings.learning_rate == (
        definition.learning_rate or config.BASE_LR
    )
    assert participant.parameter_optimizer_settings.weight_decay == config.WEIGHT_DECAY
    assert participant.parameter_optimizer_settings.adam_variant == (
        "amsgrad" if config.AMSGRAD else "standard"
    )
    assert participant.local_training_batch_sample_count == config.CLIENT_BATCH_SIZE
    assert participant.updates_per_sample == config.UPDATES_PER_SAMPLE
    assert participant.local_update_interval_per_client_samples == config.LOCAL_UPDATE_INTERVAL
    pretraining = participant.initial_model_pretraining_settings
    assert (
        pretraining.pretraining_sample_count,
        pretraining.pretraining_epoch_count,
        pretraining.pretraining_batch_sample_count,
    ) == (config.PRETRAIN_SAMPLES, config.PRETRAIN_EPOCHS, config.PRETRAIN_BATCH_SIZE)


def test_runtime_failure_preserves_successful_operations_and_reports_position_and_cause(
    monkeypatch,
):
    from federated_learning_experiments.execution.run_execution_errors import RunExecutionError

    settings = small_run_settings()
    prepared = []
    original_prepare = FedavgRunParticipantFactory.prepare_run

    def prepare(self, **kwargs):
        result = original_prepare(self, **kwargs)
        prepared.append(result)
        return result

    monkeypatch.setattr(FedavgRunParticipantFactory, "prepare_run", prepare)
    failure = RuntimeError("intentional model execution failure")
    original_process = FedavgClient.process_observed_sample

    def process(self, **kwargs):
        if self.client_id == 1 and kwargs["sample_index"] == 2:
            raise failure
        return original_process(self, **kwargs)

    monkeypatch.setattr(FedavgClient, "process_observed_sample", process)
    before = torch.get_rng_state().clone()
    with pytest.raises(RunExecutionError) as caught:
        execute_fedavg_experiment_run(run_settings=settings)
    error = caught.value
    assert error.__cause__ is failure
    assert (error.stage_name, error.client_id, error.sample_index, error.round_index) == (
        "sample_processing",
        1,
        2,
        0,
    )
    clients = prepared[0].client_operations
    assert [len(c.sample_predictions) for c in clients] == [3, 2]
    assert [len(c.training_samples) for c in clients] == [3, 2]
    assert clients[0].parameter_update_step_count == 6
    assert clients[1].pending_update_sample_count == 2
    assert (
        prepared[0]
        .server_operations.communication_volume_record_store.get_state_snapshot()
        .uploaded_model_count
        == 0
    )
    assert torch.equal(before, torch.get_rng_state())
