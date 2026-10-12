"""計測の無干渉を実旧・通常runへ照合し、独立MLP算式で計数を検証する。"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import fields, replace

import pytest
import torch

from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.evaluation.run_metric_calculations import RunMetricSettings
from federated_learning_experiments.execution.run_execution_errors import RunExecutionError
from federated_learning_experiments.learning.models.model_computation_measurement import (
    ModelComputationCounts,
    measure_model_computation,
    subtract_model_computation_counts,
)
from federated_learning_experiments.runtime import feddrift_stream_run_execution as execution
from federated_learning_experiments.runtime.feddrift_measured_run_execution import (
    execute_feddrift_stream_protocol_run_with_computation_measurement,
)
from federated_learning_experiments.runtime.feddrift_run_metric_derivation import (
    derive_feddrift_run_metrics,
)
from federated_learning_experiments.runtime.feddrift_run_participant_factory import (
    FeddriftRunParticipantFactory,
)
from tests.refactoring.test_feddrift_run_client import assert_rng_equal, rng_snapshot
from tests.refactoring.test_feddrift_server_synchronization import assert_server_equal
from tests.refactoring.test_feddrift_stream_finalization import (
    assert_terminal_values_equal,
    terminal_snapshot,
)
from tests.refactoring.test_feddrift_stream_run_execution import run_inputs, run_legacy
from tests.refactoring.test_initial_model_pretraining import assert_numpy_random_states_equal
from tests.refactoring.test_model_computation_measurement import count_global_hooks


@pytest.mark.parametrize(
    "dataset,optimizer,count,threshold,rounds",
    [
        (dataset, optimizer, 11, 1.0, 2)
        for dataset in ("sine2", "sea4", "mnist2", "mnist4")
        for optimizer in ("sgd", "adam")
    ]
    + [("sine2", "sgd", 17, 0.03, 3), ("sea4", "adam", 12, 0.0, 1)],
)
def test_measured_run_preserves_actual_legacy_and_unmeasured_run(
    monkeypatch, dataset, optimizer, count, threshold, rounds
):
    settings, run_settings = run_inputs(monkeypatch, dataset, optimizer, count, threshold, rounds)
    old = run_legacy(monkeypatch, run_settings, threshold)
    observed = {}
    prepare = FeddriftRunParticipantFactory.prepare_run

    def observe_prepare(self, **kwargs):
        observed["sources"] = kwargs["run_random_sources"]
        return prepare(self, **kwargs)

    monkeypatch.setattr(FeddriftRunParticipantFactory, "prepare_run", observe_prepare)
    scope = execution.isolated_cpu_torch_random_state

    @contextmanager
    def observe_scope(**kwargs):
        with scope(**kwargs):
            yield
            observed["torch"] = torch.get_rng_state().clone()

    monkeypatch.setattr(execution, "isolated_cpu_torch_random_state", observe_scope)
    before, hooks = rng_snapshot(), count_global_hooks()
    measured = execute_feddrift_stream_protocol_run_with_computation_measurement(
        execution_settings=run_settings, run_participant_settings=settings
    )
    assert_rng_equal(before)
    assert count_global_hooks() == hooks
    assert_server_equal(old["participants"][0], measured.participants.run_server)
    assert observed["sources"].python_random_generator.getstate() == old["rng"][0]
    assert_numpy_random_states_equal(
        observed["sources"].numpy_random_generator.get_state(), old["rng"][1]
    )
    assert torch.equal(observed["torch"], old["rng"][2])
    snapshot = deepcopy(terminal_snapshot(measured.participants.run_server))
    factory = FeddriftRunParticipantFactory(run_participant_settings=settings)
    normal = execution.execute_feddrift_stream_protocol_run(
        execution_settings=run_settings, participant_factory=factory
    )
    assert normal == measured.run_result
    assert_terminal_values_equal(
        terminal_snapshot(factory.prepared_run_participants.run_server), snapshot
    )
    metric_settings = RunMetricSettings(
        maximum_detection_delay_sample_count=3, post_change_recovery_window_sample_count=3
    )
    assert derive_feddrift_run_metrics(
        run_result=normal,
        participants=factory.prepared_run_participants,
        run_metric_settings=metric_settings,
    ) == derive_feddrift_run_metrics(
        run_result=measured.run_result,
        participants=measured.participants,
        run_metric_settings=metric_settings,
    )
    repeated = execute_feddrift_stream_protocol_run_with_computation_measurement(
        execution_settings=run_settings, run_participant_settings=settings
    )
    assert repeated.run_result == normal
    assert repeated.model_computation_counts == measured.model_computation_counts
    assert (
        repeated.preparation_model_computation_counts
        == measured.preparation_model_computation_counts
    )
    assert type(measured.model_computation_counts) is ModelComputationCounts
    for counts in (
        measured.model_computation_counts,
        measured.preparation_model_computation_counts,
    ):
        for field in fields(counts):
            value = getattr(counts, field.name)
            assert type(value) is int and value >= 0
            if field.name.startswith("shared_"):
                assert value == 0
        # 全モデルは同じ独立MLPの形。多モデル時も入力標本の延べ数でMACを照合する。
        input_width = len(
            measured.run_result.observed_client_streams[0].observed_samples[0].feature_values
        )
        # dataset名の数字は概念数。SEAはbinary、MNISTは10クラス。
        output_width = 10 if dataset.startswith("mnist") else 1
        forward_macs = input_width * 7 + 7 * 5 + 5 * output_width
        backward_macs = input_width * 7 + 2 * (7 * 5 + 5 * output_width)
        assert (
            counts.concept_specific_part_training_forward_multiply_accumulate_count
            == counts.concept_specific_part_training_example_count * forward_macs
        )
        assert (
            counts.concept_specific_part_inference_forward_multiply_accumulate_count
            == counts.concept_specific_part_inference_example_count * forward_macs
        )
        assert (
            counts.concept_specific_part_estimated_backward_multiply_accumulate_count
            == counts.concept_specific_part_training_example_count * backward_macs
        )
    assert (
        measured.preparation_model_computation_counts.concept_specific_parameter_optimizer_step_count
        > 0
    )
    assert_rng_equal(before)
    assert count_global_hooks() == hooks


def test_single_model_counts_follow_independent_linear_layer_formula(monkeypatch):
    settings, run_settings = run_inputs(monkeypatch, count=3, rounds=2, probability=0.0)
    measured = execute_feddrift_stream_protocol_run_with_computation_measurement(
        execution_settings=run_settings, run_participant_settings=settings
    )
    counts = measured.model_computation_counts
    # 独立MLP: input2→hidden7→hidden5→binary出力1。
    forward = 2 * 7 + 7 * 5 + 5 * 1
    # inputには勾配なし、後の層には重み勾配と入力勾配がある。
    backward = 2 * 7 + 2 * (7 * 5 + 5 * 1)
    # 2 client × R2 × chunk3 × updates_per_sample2、batchは3。
    updates = 2 * 2 * 3 * 2
    training_examples = updates * 3
    # 予測6件、全保有モデル検出評価6件、保持後の損失統計評価6件。
    inference_examples = 18
    assert counts.concept_specific_parameter_optimizer_step_count == updates
    assert counts.concept_specific_part_training_example_count == training_examples
    assert counts.concept_specific_part_inference_example_count == inference_examples
    assert (
        counts.concept_specific_part_training_forward_multiply_accumulate_count
        == training_examples * forward
    )
    assert (
        counts.concept_specific_part_inference_forward_multiply_accumulate_count
        == inference_examples * forward
    )
    assert (
        counts.concept_specific_part_estimated_backward_multiply_accumulate_count
        == training_examples * backward
    )


@pytest.mark.parametrize(
    "count,rounds,threshold", [(2, 2, 1.0), (3, 1, 1.0), (11, 3, 1.0), (17, 3, 0.03)]
)
def test_timestep_parts_partition_all_counts_and_leave_tail_unprocessed(
    monkeypatch, count, rounds, threshold
):
    settings, run_settings = run_inputs(
        monkeypatch, count=count, rounds=rounds, threshold=threshold, seed=71, probability=1.0
    )
    measured = execute_feddrift_stream_protocol_run_with_computation_measurement(
        execution_settings=run_settings, run_participant_settings=settings
    )
    timesteps = measured.timestep_model_computation_counts
    assert len(timesteps) == count // 3
    assert [x.timestep_index for x in timesteps] == list(range(count // 3))
    parts = [measured.finalization_model_computation_counts]
    for timestep in timesteps:
        local = timestep.sample_processing_model_computation_counts
        sync = timestep.synchronization_model_computation_counts
        assert local.concept_specific_parameter_optimizer_step_count == 0
        assert local.concept_specific_part_training_forward_multiply_accumulate_count == 0
        assert local.concept_specific_part_inference_forward_multiply_accumulate_count > 0
        assert sync.concept_specific_parameter_optimizer_step_count > 0
        parts.extend((local, sync))
    for field in fields(ModelComputationCounts):
        assert sum(getattr(part, field.name) for part in parts) == getattr(
            measured.model_computation_counts, field.name
        )
    assert measured.run_result.processed_sample_count_per_client == count // 3 * 3
    assert measured.run_result.unprocessed_tail_sample_count_per_client == count % 3
    # 全体runはtailをflushせず、正常な終端はモデル計算を行わない。
    assert all(
        getattr(measured.finalization_model_computation_counts, f.name) == 0
        for f in fields(ModelComputationCounts)
    )
    if count < 3:
        assert not timesteps
        assert measured.model_computation_counts == measured.finalization_model_computation_counts
    if threshold < 1.0:
        assert measured.participants.run_server.cross_evaluation_record_store.snapshot_client_cross_evaluation_records()


@pytest.mark.parametrize("rounds,updates,batch", [(1, 0, 2), (1, 1, 2), (3, 2, 4)])
def test_sync_training_counts_respect_independent_batch_and_update_settings(
    monkeypatch, rounds, updates, batch
):
    settings, run_settings = run_inputs(monkeypatch, count=3, rounds=rounds, probability=0.0)
    settings = replace(
        settings,
        run_client_settings=replace(
            settings.run_client_settings,
            updates_per_sample=updates,
            local_training_batch_sample_count=batch,
        ),
    )
    measured = execute_feddrift_stream_protocol_run_with_computation_measurement(
        execution_settings=run_settings, run_participant_settings=settings
    )
    # 保持標本がbatch未満なら学習しない。学習batch数とchunkは独立。
    sync = measured.timestep_model_computation_counts[0].synchronization_model_computation_counts
    expected_updates = 2 * rounds * 3 * updates if batch <= 3 else 0
    assert sync.concept_specific_parameter_optimizer_step_count == expected_updates
    assert sync.concept_specific_part_training_example_count == expected_updates * batch


@pytest.mark.parametrize(
    "stage,occurrence",
    [
        ("configuration_validation", 1),
        ("initial_preparation", 1),
        ("participant_validation", 1),
        ("concept_trace_generation", 1),
        ("observed_stream_generation", 1),
        ("sample_processing", 1),
        ("sample_processing", 4),
        ("server_synchronization", 1),
        ("server_synchronization", 2),
        ("observed_stream_finalization", 1),
    ],
)
def test_failure_keeps_stage_cause_rng_and_removes_hooks(monkeypatch, stage, occurrence):
    settings, run_settings = run_inputs(monkeypatch)
    normal = execute_feddrift_stream_protocol_run_with_computation_measurement(
        execution_settings=run_settings, run_participant_settings=settings
    )
    cause = RuntimeError("injected measurement run failure")
    wrapper = execution.invoke_run_operation_and_record_success
    calls = 0

    def fail(**kwargs):
        raise cause

    def inject(**kwargs):
        nonlocal calls
        if kwargs["stage_name"] == stage:
            calls += 1
            if calls == occurrence:
                kwargs["run_operation"] = fail
        return wrapper(**kwargs)

    before, hooks = rng_snapshot(), count_global_hooks()
    with monkeypatch.context() as patch:
        patch.setattr(execution, "invoke_run_operation_and_record_success", inject)
        with pytest.raises(RunExecutionError) as caught:
            execute_feddrift_stream_protocol_run_with_computation_measurement(
                execution_settings=run_settings, run_participant_settings=settings
            )
    assert caught.value.stage_name == stage
    assert caught.value.__cause__ is cause
    assert count_global_hooks() == hooks
    assert_rng_equal(before)
    repeated = execute_feddrift_stream_protocol_run_with_computation_measurement(
        execution_settings=run_settings, run_participant_settings=settings
    )
    assert repeated.model_computation_counts == normal.model_computation_counts
    assert repeated.timestep_model_computation_counts == normal.timestep_model_computation_counts
    assert count_global_hooks() == hooks
    assert_rng_equal(before)


def test_loader_failure_is_preparation_error_and_removes_hooks(monkeypatch):
    settings, run_settings = run_inputs(monkeypatch, dataset="mnist2")
    before, hooks = rng_snapshot(), count_global_hooks()
    cause = FileNotFoundError("missing sample data")

    def fail(**kwargs):
        raise cause

    monkeypatch.setattr(execution, "create_observed_sample_generator", fail)
    with pytest.raises(RunExecutionError) as caught:
        execute_feddrift_stream_protocol_run_with_computation_measurement(
            execution_settings=run_settings, run_participant_settings=settings
        )
    assert caught.value.stage_name == "initial_preparation"
    assert caught.value.__cause__ is cause
    assert count_global_hooks() == hooks
    assert_rng_equal(before)


def test_checkpoint_counts_match_independently_observed_operation_boundaries(monkeypatch):
    settings, run_settings = run_inputs(
        monkeypatch, count=17, rounds=3, threshold=0.03, seed=71, probability=1.0
    )
    expected = {}
    wrapper = execution.invoke_run_operation_and_record_success
    with measure_model_computation() as meter:

        def observe(**kwargs):
            stage = kwargs["stage_name"]
            before = meter.get_model_computation_counts()
            result = wrapper(**kwargs)
            after = meter.get_model_computation_counts()
            expected.setdefault(stage, []).append(
                subtract_model_computation_counts(later_counts=after, earlier_counts=before)
            )
            return result

        with monkeypatch.context() as patch:
            patch.setattr(execution, "invoke_run_operation_and_record_success", observe)
            execution.execute_feddrift_stream_protocol_run(
                execution_settings=run_settings,
                participant_factory=FeddriftRunParticipantFactory(
                    run_participant_settings=settings
                ),
            )
    measured = execute_feddrift_stream_protocol_run_with_computation_measurement(
        execution_settings=run_settings, run_participant_settings=settings
    )
    assert measured.preparation_model_computation_counts == expected["initial_preparation"][0]
    assert (
        measured.finalization_model_computation_counts
        == expected["observed_stream_finalization"][0]
    )
    for index, timestep in enumerate(measured.timestep_model_computation_counts):
        assert (
            timestep.synchronization_model_computation_counts
            == expected["server_synchronization"][index]
        )
        for field in fields(ModelComputationCounts):
            assert getattr(timestep.sample_processing_model_computation_counts, field.name) == sum(
                getattr(x, field.name)
                for x in expected["sample_processing"][2 * index : 2 * index + 2]
            )


@pytest.mark.parametrize(
    "invalid", ["execution_type", "execution_seed", "participant_type", "participant_field"]
)
def test_actual_invalid_settings_leave_rng_and_existing_hooks_unchanged(monkeypatch, invalid):
    settings, run_settings = run_inputs(monkeypatch)
    if invalid == "execution_type":
        run_settings = object()
    elif invalid == "execution_seed":
        object.__setattr__(run_settings.experiment_run_conditions, "random_seed", True)
    elif invalid == "participant_type":
        settings = object()
    else:
        object.__setattr__(settings.run_client_settings, "updates_per_sample", True)
    before, hooks = rng_snapshot(), count_global_hooks()
    # 呼出し元の別計測のhookも、新計測の終了で外してはならない。
    with measure_model_computation() as outer_meter:
        nested_hooks = count_global_hooks()
        with pytest.raises(RunExecutionError) as caught:
            execute_feddrift_stream_protocol_run_with_computation_measurement(
                execution_settings=run_settings, run_participant_settings=settings
            )
        assert caught.value.stage_name == "configuration_validation"
        assert (
            type(caught.value.__cause__)
            is {
                "execution_type": TypeError,
                "execution_seed": RunSettingsValidationError,
                "participant_type": TypeError,
                "participant_field": ValueError,
            }[invalid]
        )
        assert count_global_hooks() == nested_hooks
        assert all(
            getattr(outer_meter.get_model_computation_counts(), f.name) == 0
            for f in fields(ModelComputationCounts)
        )
    assert count_global_hooks() == hooks
    assert_rng_equal(before)
