"""単一runの実行条件・乱数・呼出順を検証する。"""

import random
from contextlib import nullcontext
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch

from test_run_settings_validation import valid_run_settings_mapping
from federated_learning_experiments.configuration.experiment_run_conditions import ExperimentRunConditions
from federated_learning_experiments.configuration.run_settings import ValidatedExperimentRunSettingsSubset
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_settings import RandomConceptScheduleSettings
from federated_learning_experiments.execution.stream_protocol_execution_settings import (
    StreamProtocolExecutionSettings,
    validate_stream_protocol_execution_settings,
)

from federated_learning_experiments.execution.run_random_sources import create_run_random_sources
from federated_learning_experiments.learning.models.torch_random_state_scope import (
    isolated_cpu_torch_random_state,
)


def test_run_random_sources_are_independent_and_reproduce_legacy_sequences():
    global_python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    run_random_sources = create_run_random_sources(random_seed=17)
    repeated_run_random_sources = create_run_random_sources(random_seed=17)
    expected_python_random_generator = random.Random(17)
    expected_numpy_random_generator = np.random.RandomState(17)
    assert run_random_sources.python_random_generator is not repeated_run_random_sources.python_random_generator
    assert run_random_sources.numpy_random_generator is not repeated_run_random_sources.numpy_random_generator
    assert run_random_sources.python_random_generator.random() == expected_python_random_generator.random()
    np.testing.assert_array_equal(
        run_random_sources.numpy_random_generator.uniform(size=10),
        expected_numpy_random_generator.uniform(size=10),
    )
    assert repeated_run_random_sources.python_random_generator.random() == random.Random(17).random()
    np.testing.assert_array_equal(
        repeated_run_random_sources.numpy_random_generator.uniform(size=10),
        np.random.RandomState(17).uniform(size=10),
    )
    assert random.getstate() == global_python_random_state
    np.testing.assert_equal(np.random.get_state(), global_numpy_random_state)


@pytest.mark.parametrize("fail_inside_scope", [False, True])
def test_cpu_torch_random_scope_restores_state_on_success_and_failure(fail_inside_scope):
    original_cpu_torch_random_state = torch.get_rng_state().clone()
    expected_cpu_random_generator = torch.Generator(device="cpu").manual_seed(17)
    with pytest.raises(ValueError, match="テスト内の失敗") if fail_inside_scope else nullcontext():
        with isolated_cpu_torch_random_state(random_seed=17):
            observed_cpu_random_values = torch.rand(10)
            assert torch.equal(
                observed_cpu_random_values,
                torch.rand(10, generator=expected_cpu_random_generator),
            )
            if fail_inside_scope:
                raise ValueError("テスト内の失敗")
    assert torch.equal(torch.get_rng_state(), original_cpu_torch_random_state)


def test_cpu_torch_random_scope_does_not_seed_other_devices_or_change_runtime_defaults(monkeypatch):
    non_cpu_seed_calls = []
    original_torch_thread_count = torch.get_num_threads()
    original_torch_default_dtype = torch.get_default_dtype()
    monkeypatch.setattr(torch, "manual_seed", non_cpu_seed_calls.append)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", non_cpu_seed_calls.append)
    monkeypatch.setattr(torch.xpu, "manual_seed_all", non_cpu_seed_calls.append)
    with isolated_cpu_torch_random_state(random_seed=0):
        torch.rand(3)
    assert non_cpu_seed_calls == []
    assert torch.get_num_threads() == original_torch_thread_count
    assert torch.get_default_dtype() == original_torch_default_dtype


@pytest.fixture
def valid_stream_protocol_execution_settings():
    return StreamProtocolExecutionSettings(
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name="sine2", random_seed=0, client_count=3,
            per_client_sample_count=1500, server_aggregation_interval_per_client_samples=50,
        ),
        concept_schedule_settings=RandomConceptScheduleSettings(
            concept_schedule_strategy="random_changes_after_minimum_index_gap",
            minimum_sample_index_gap_before_change_trial=100,
            per_eligible_sample_concept_change_probability=0.015,
        ),
        execution_strategy="sample_index_then_client_order_with_interval_synchronization",
    )


@pytest.mark.parametrize("random_seed", [0, 2**32 - 1])
@pytest.mark.parametrize("per_client_sample_count,server_aggregation_interval_per_client_samples", [(16, 7), (3, 7)])
def test_stream_protocol_execution_settings_accept_seed_and_interval_boundaries(
    valid_stream_protocol_execution_settings, random_seed, per_client_sample_count,
    server_aggregation_interval_per_client_samples,
):
    execution_settings = replace(
        valid_stream_protocol_execution_settings,
        experiment_run_conditions=replace(
            valid_stream_protocol_execution_settings.experiment_run_conditions,
            random_seed=random_seed, per_client_sample_count=per_client_sample_count,
            server_aggregation_interval_per_client_samples=server_aggregation_interval_per_client_samples,
        ),
    )
    assert validate_stream_protocol_execution_settings(execution_settings=execution_settings) is None


@pytest.mark.parametrize("configuration_parameter_name,specified_parameter_value", [
    ("experiment_run_conditions", None), ("concept_schedule_settings", {}),
    ("execution_strategy", "unknown"), ("execution_strategy", None),
    ("dataset_name", "sea2"), ("dataset_name", "mnist2"), ("random_seed", 2**32),
])
def test_stream_protocol_execution_settings_reject_invalid_components_and_choices(
    valid_stream_protocol_execution_settings, configuration_parameter_name, specified_parameter_value,
):
    with pytest.raises(RunSettingsValidationError) as exception_info:
        if configuration_parameter_name in ("dataset_name", "random_seed"):
            replace(
                valid_stream_protocol_execution_settings,
                experiment_run_conditions=replace(
                    valid_stream_protocol_execution_settings.experiment_run_conditions,
                    **{configuration_parameter_name: specified_parameter_value},
                ),
            )
        else:
            replace(valid_stream_protocol_execution_settings, **{configuration_parameter_name: specified_parameter_value})
    assert exception_info.value.configuration_parameter_name == configuration_parameter_name
    assert exception_info.value.specified_parameter_value == specified_parameter_value
    assert exception_info.value.validation_failure_reason


def test_stream_protocol_execution_validation_rejects_partial_settings(valid_run_settings_mapping):
    partial_run_settings = ValidatedExperimentRunSettingsSubset(**valid_run_settings_mapping)
    with pytest.raises(RunSettingsValidationError) as exception_info:
        validate_stream_protocol_execution_settings(execution_settings=partial_run_settings)
    assert exception_info.value.configuration_parameter_name == "execution_settings"
    assert exception_info.value.specified_parameter_value is partial_run_settings


@pytest.mark.parametrize("missing_parameter_name", [
    "experiment_run_conditions", "concept_schedule_settings", "execution_strategy",
])
def test_stream_protocol_execution_settings_are_required_frozen_and_keyword_only(
    valid_stream_protocol_execution_settings, missing_parameter_name,
):
    with pytest.raises(TypeError):
        StreamProtocolExecutionSettings(**{
            configuration_parameter_name: specified_parameter_value
            for configuration_parameter_name, specified_parameter_value
            in vars(valid_stream_protocol_execution_settings).items()
            if configuration_parameter_name != missing_parameter_name
        })
    with pytest.raises(TypeError):
        StreamProtocolExecutionSettings(*vars(valid_stream_protocol_execution_settings).values())
    with pytest.raises(FrozenInstanceError):
        setattr(valid_stream_protocol_execution_settings, missing_parameter_name, None)
    with pytest.raises(FrozenInstanceError):
        delattr(valid_stream_protocol_execution_settings, missing_parameter_name)
