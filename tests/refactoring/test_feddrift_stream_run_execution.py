"""全体runを実旧入口の実行結果・全状態・局所乱数へ照合する。"""

import random
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from io import StringIO

import numpy as np
import pytest
import torch

from federated_drift_experiment import config, experiment
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_settings import (
    RandomConceptScheduleSettings,
)
from federated_learning_experiments.execution.run_execution_errors import RunExecutionError
from federated_learning_experiments.runtime import feddrift_stream_run_execution as execution
from federated_learning_experiments.runtime.feddrift_run_client import FeddriftRunClient
from federated_learning_experiments.runtime.feddrift_run_participant_factory import (
    FeddriftRunParticipantFactory,
)
from federated_learning_experiments.runtime.feddrift_run_server import FeddriftRunServer
from tests.refactoring.test_feddrift_run_client import assert_rng_equal, rng_snapshot
from tests.refactoring.test_feddrift_run_participant_factory import make_inputs
from tests.refactoring.test_feddrift_server_synchronization import assert_server_equal
from tests.refactoring.test_feddrift_stream_finalization import (
    assert_terminal_values_equal,
    terminal_snapshot,
)
from tests.refactoring.test_initial_model_pretraining import assert_numpy_random_states_equal


def run_inputs(
    monkeypatch,
    dataset="sine2",
    optimizer="sgd",
    count=11,
    threshold=1.0,
    rounds=2,
    seed=37,
    probability=0.5,
):
    settings, inputs = make_inputs(monkeypatch, dataset, optimizer, 1, 2)
    settings = replace(
        settings,
        run_client_settings=replace(
            settings.run_client_settings, loss_increase_threshold=threshold
        ),
        server_settings=replace(
            settings.server_settings,
            maximum_same_cluster_loss_increase=threshold,
            local_training_round_count=rounds,
        ),
    )
    for name, value in dict(
        TOTAL_DATA_POINTS=count,
        MIN_STABLE_PERIOD=1,
        DRIFT_PROB=probability,
        CONCEPT_SCHEDULE="random",
        DELAY_TOLERANCE=3,
        STABLE_WINDOW=3,
        FEDDRIFT_ROUNDS=rounds,
    ).items():
        monkeypatch.setattr(config, name, value)
    conditions = replace(
        inputs["experiment_run_conditions"], per_client_sample_count=count, random_seed=seed
    )
    run_settings = execution.FeddriftStreamProtocolExecutionSettings(
        experiment_run_conditions=conditions,
        concept_schedule_settings=RandomConceptScheduleSettings(
            concept_schedule_strategy="random_changes_after_minimum_index_gap",
            minimum_sample_index_gap_before_change_trial=1,
            per_eligible_sample_concept_change_probability=probability,
        ),
    )
    return settings, run_settings


def run_legacy(monkeypatch, run_settings, threshold):
    """進行や式を複製せず、実旧の入口と生成操作を観測する。"""
    captured = {}
    with monkeypatch.context() as patch:
        for name, key in (
            ("_setup_server_and_clients", "participants"),
            ("make_concept_schedules", "concepts"),
            ("build_data_streams", "data"),
        ):
            operation = getattr(experiment, name)

            def observe(*args, operation=operation, key=key, **kwargs):
                value = operation(*args, **kwargs)
                captured[key] = value
                return value

            patch.setattr(experiment, name, observe)
        before = rng_snapshot()
        try:
            with redirect_stdout(StringIO()):
                captured["metrics"] = experiment.run_random_drift_experiment(
                    mode="FedDrift",
                    distance_threshold=threshold,
                    random_seed=run_settings.experiment_run_conditions.random_seed,
                    verbose=False,
                    show_plot=False,
                )
            captured["rng"] = rng_snapshot()
        finally:
            random.setstate(before[0])
            np.random.set_state(before[1])
            torch.set_rng_state(before[2])
    return captured


@pytest.mark.parametrize(
    "dataset,optimizer,count,threshold,rounds,seed,probability",
    [
        (dataset, optimizer, 11, threshold, 2, 37, 0.5)
        for dataset in ("sine2", "sea4", "mnist2", "mnist4")
        for optimizer in ("sgd", "adam")
        for threshold in (1.0, 0.0)
    ]
    + [
        ("sine2", "sgd", count, 0.03, rounds, seed, probability)
        for count, rounds, seed, probability in (
            (2, 1, 0, 0.0),
            (3, 3, 17, 1.0),
            (12, 1, 99, 0.5),
            (17, 3, 71, 1.0),
        )
    ],
)
def test_full_run_matches_actual_legacy(
    monkeypatch, dataset, optimizer, count, threshold, rounds, seed, probability
):
    settings, run_settings = run_inputs(
        monkeypatch, dataset, optimizer, count, threshold, rounds, seed, probability
    )
    old = run_legacy(monkeypatch, run_settings, threshold)
    factory = FeddriftRunParticipantFactory(run_participant_settings=settings)
    actual = {}
    prepare = factory.prepare_run

    def observe_prepare(**kwargs):
        actual["sources"] = kwargs["run_random_sources"]
        return prepare(**kwargs)

    monkeypatch.setattr(factory, "prepare_run", observe_prepare)
    scope = execution.isolated_cpu_torch_random_state

    @contextmanager
    def observe_scope(**kwargs):
        with scope(**kwargs):
            yield
            actual["torch"] = torch.get_rng_state().clone()

    monkeypatch.setattr(execution, "isolated_cpu_torch_random_state", observe_scope)
    before = rng_snapshot()
    result = execution.execute_feddrift_stream_protocol_run(
        execution_settings=run_settings, participant_factory=factory
    )
    assert_rng_equal(before)

    participants = factory.prepared_run_participants
    assert participants is not None
    assert_server_equal(old["participants"][0], participants.run_server)
    assert actual["sources"].python_random_generator.getstate() == old["rng"][0]
    assert_numpy_random_states_equal(
        actual["sources"].numpy_random_generator.get_state(), old["rng"][1]
    )
    assert torch.equal(actual["torch"], old["rng"][2])
    assert result.processed_sample_count_per_client == count // 3 * 3
    assert result.unprocessed_tail_sample_count_per_client == count % 3
    for stream, trace, concepts, data in zip(
        result.observed_client_streams,
        result.evaluation_concept_traces,
        old["concepts"],
        old["data"],
        strict=True,
    ):
        assert trace.concept_ids_by_sample_index == tuple(concepts)
        for observed, (features, label) in zip(stream.observed_samples, data, strict=True):
            assert observed.feature_values == tuple(float(v) for v in features.flatten())
            assert observed.class_label == int(label.item())
    expected = [
        ("configuration_validation", None, None, None),
        ("initial_preparation", None, None, None),
        ("participant_validation", None, None, None),
        ("concept_trace_generation", None, None, None),
        ("observed_stream_generation", None, None, None),
    ]
    for t in range(count // 3):
        expected.extend(("sample_processing", i, t * 3, t) for i in range(2))
        expected.append(("server_synchronization", None, None, t))
    expected.append(("observed_stream_finalization", None, None, max(0, count // 3 - 1)))
    assert [
        (e.stage_name, e.client_id, e.sample_index, e.round_index) for e in result.execution_events
    ] == expected
    snapshot = deepcopy(terminal_snapshot(participants.run_server))
    repeated = execution.execute_feddrift_stream_protocol_run(
        execution_settings=run_settings, participant_factory=factory
    )
    assert repeated == result
    assert_terminal_values_equal(
        terminal_snapshot(factory.prepared_run_participants.run_server), snapshot
    )
    assert_rng_equal(before)


@pytest.mark.parametrize(
    "invalid",
    (
        "outer",
        "conditions",
        "concept",
        "seed",
        "batch",
        "factory",
        "factory_settings",
        "bool_client_count",
        "probability",
    ),
)
def test_invalid_configuration_rejected_before_preparation(monkeypatch, invalid):
    settings, run_settings = run_inputs(monkeypatch, count=2)
    factory = FeddriftRunParticipantFactory(run_participant_settings=settings)
    execution.execute_feddrift_stream_protocol_run(
        execution_settings=run_settings, participant_factory=factory
    )
    prepared = factory.prepared_run_participants
    before = rng_snapshot()
    if invalid == "outer":
        run_settings = object()
    elif invalid == "conditions":
        object.__setattr__(run_settings, "experiment_run_conditions", object())
    elif invalid == "concept":
        object.__setattr__(run_settings, "concept_schedule_settings", object())
    elif invalid == "seed":
        object.__setattr__(run_settings.experiment_run_conditions, "random_seed", 2**32)
    elif invalid == "bool_client_count":
        object.__setattr__(run_settings.experiment_run_conditions, "client_count", True)
    elif invalid == "probability":
        object.__setattr__(
            run_settings.concept_schedule_settings,
            "per_eligible_sample_concept_change_probability",
            float("nan"),
        )
    elif invalid == "batch":
        object.__setattr__(
            run_settings.experiment_run_conditions,
            "server_aggregation_interval_per_client_samples",
            4,
        )
    elif invalid == "factory_settings":
        object.__setattr__(settings.server_settings, "local_training_round_count", True)
    else:
        factory = object()
    calls = []
    monkeypatch.setattr(
        execution, "create_run_random_sources", lambda **kwargs: calls.append(kwargs)
    )
    with pytest.raises(RunExecutionError) as caught:
        execution.execute_feddrift_stream_protocol_run(
            execution_settings=run_settings, participant_factory=factory
        )
    assert caught.value.stage_name == "configuration_validation"
    assert caught.value.__cause__ is not None
    assert not calls
    if invalid != "factory":
        assert factory.prepared_run_participants is prepared
    assert_rng_equal(before)


@pytest.mark.parametrize(
    "stage",
    (
        "initial_preparation",
        "participant_validation",
        "concept_trace_generation",
        "observed_stream_generation",
        "sample_processing",
        "server_synchronization",
        "observed_stream_finalization",
    ),
)
def test_stage_failures_preserve_cause_positions_rng_and_success_boundary(monkeypatch, stage):
    settings, run_settings = run_inputs(monkeypatch)
    factory = FeddriftRunParticipantFactory(run_participant_settings=settings)
    cause = RuntimeError("injected")
    events = []
    wrapper = execution.invoke_run_operation_and_record_success

    def observe(**kwargs):
        events[:] = kwargs["execution_events"]
        return wrapper(**kwargs)

    def fail(*args, **kwargs):
        raise cause

    monkeypatch.setattr(execution, "invoke_run_operation_and_record_success", observe)
    targets = {
        "initial_preparation": (factory, "prepare_run"),
        "participant_validation": (execution, "_validate_participants"),
        "concept_trace_generation": (execution, "generate_random_client_concept_traces"),
        "observed_stream_generation": (execution, "build_client_observed_streams"),
        "sample_processing": (FeddriftRunClient, "process_observed_block"),
        "server_synchronization": (FeddriftRunServer, "synchronize_timestep"),
        "observed_stream_finalization": (FeddriftRunServer, "finalize_observed_stream"),
    }
    monkeypatch.setattr(*targets[stage], fail)
    before = rng_snapshot()
    with pytest.raises(RunExecutionError) as caught:
        execution.execute_feddrift_stream_protocol_run(
            execution_settings=run_settings, participant_factory=factory
        )
    assert caught.value.stage_name == stage
    assert caught.value.__cause__ is cause
    assert all(event.stage_name != stage for event in events)
    if stage == "sample_processing":
        assert (caught.value.client_id, caught.value.sample_index, caught.value.round_index) == (
            0,
            0,
            0,
        )
    elif stage in ("server_synchronization", "observed_stream_finalization"):
        assert caught.value.round_index == (0 if stage == "server_synchronization" else 2)
    else:
        assert caught.value.round_index is None
    assert_rng_equal(before)


@pytest.mark.parametrize("invalid", ("type", "count", "client_id", "server_reference"))
def test_prepared_bundle_rejected_before_stream_generation(monkeypatch, invalid):
    settings, run_settings = run_inputs(monkeypatch)
    factory = FeddriftRunParticipantFactory(run_participant_settings=settings)
    prepare = factory.prepare_run

    def broken_prepare(**kwargs):
        participants = prepare(**kwargs)
        if invalid == "type":
            return object()
        if invalid == "count":
            run_settings.experiment_run_conditions.__dict__["client_count"] = 3
        elif invalid == "client_id":
            participants.run_clients[0]._client_id = True
        else:
            participants.run_server.run_clients = tuple(reversed(participants.run_clients))
        return participants

    monkeypatch.setattr(factory, "prepare_run", broken_prepare)
    generated = []
    monkeypatch.setattr(
        execution, "generate_random_client_concept_traces", lambda **kw: generated.append(kw)
    )
    before = rng_snapshot()
    with pytest.raises(RunExecutionError) as caught:
        execution.execute_feddrift_stream_protocol_run(
            execution_settings=run_settings, participant_factory=factory
        )
    assert caught.value.stage_name == "participant_validation"
    assert not generated
    assert_rng_equal(before)


@pytest.mark.parametrize(
    "stage,occurrence,position",
    (("sample_processing", 4, (1, 3, 1)), ("server_synchronization", 2, (None, None, 1))),
)
def test_intermediate_failure_positions_and_completed_events(
    monkeypatch, stage, occurrence, position
):
    settings, run_settings = run_inputs(monkeypatch)
    factory = FeddriftRunParticipantFactory(run_participant_settings=settings)
    wrapper = execution.invoke_run_operation_and_record_success
    cause = RuntimeError("middle failure")
    calls = 0
    events = []

    def fail(**kwargs):
        raise cause

    def inject(**kwargs):
        nonlocal calls
        events[:] = kwargs["execution_events"]
        if kwargs["stage_name"] == stage:
            calls += 1
            if calls == occurrence:
                kwargs["run_operation"] = fail
        return wrapper(**kwargs)

    monkeypatch.setattr(execution, "invoke_run_operation_and_record_success", inject)
    before = rng_snapshot()
    with pytest.raises(RunExecutionError) as caught:
        execution.execute_feddrift_stream_protocol_run(
            execution_settings=run_settings, participant_factory=factory
        )
    assert caught.value.__cause__ is cause
    assert (caught.value.client_id, caught.value.sample_index, caught.value.round_index) == position
    assert sum(event.stage_name == stage for event in events) == occurrence - 1
    assert_rng_equal(before)


@pytest.mark.parametrize("dataset", ("sine2", "mnist2"))
def test_sample_generator_initialization_failure_is_preparation_failure(monkeypatch, dataset):
    settings, run_settings = run_inputs(monkeypatch, dataset=dataset, count=2)
    factory = FeddriftRunParticipantFactory(run_participant_settings=settings)
    execution.execute_feddrift_stream_protocol_run(
        execution_settings=run_settings, participant_factory=factory
    )
    previous = factory.prepared_run_participants
    cause = FileNotFoundError("sample data unavailable")

    def fail(**kwargs):
        raise cause

    monkeypatch.setattr(execution, "create_observed_sample_generator", fail)
    before = rng_snapshot()
    with pytest.raises(RunExecutionError) as caught:
        execution.execute_feddrift_stream_protocol_run(
            execution_settings=run_settings, participant_factory=factory
        )
    assert caught.value.stage_name == "initial_preparation"
    assert caught.value.__cause__ is cause
    assert factory.prepared_run_participants is previous
    assert_rng_equal(before)
