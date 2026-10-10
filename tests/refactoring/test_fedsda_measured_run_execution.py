"""計測つきの全体run: 計測なしの全体runと同じ結果、準備の間の計算の分離、計測の後始末。"""

import pytest
from test_fedsda_run_client import snapshot_run_client_state
from test_fedsda_stream_protocol_run import (
    PRETRAINING_VALUES,
    execute_stream_protocol_run_with_factory,
    make_execution_settings,
    make_run_participant_settings,
)
from test_model_computation_measurement import count_global_hooks
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

from federated_learning_experiments.evaluation.run_metric_calculations import RunMetricSettings
from federated_learning_experiments.learning.models.model_computation_measurement import (
    ModelComputationCounts,
    measure_model_computation,
    subtract_model_computation_counts,
)
from federated_learning_experiments.runtime.fedsda_measured_run_execution import (
    FedsdaMeasuredRun,
    execute_fedsda_stream_protocol_run_with_computation_measurement,
)
from federated_learning_experiments.runtime.fedsda_run_metric_derivation import (
    derive_fedsda_run_metrics,
)


def make_small_run_settings(valid_run_settings_mapping):
    execution_settings = make_execution_settings(
        random_seed=7,
        client_count=3,
        per_client_sample_count=300,
        aggregation_interval=10,
        minimum_change_gap=30,
        concept_change_probability=0.05,
    )
    run_participant_settings = make_run_participant_settings(
        valid_run_settings_mapping, update_interval=2
    )
    return execution_settings, run_participant_settings


def test_measured_run_matches_unmeasured_run_and_separates_preparation_computation(
    valid_run_settings_mapping,
):
    """計測つきの全体runは、計測なしと同じ結果になり、準備の間の計算（事前学習）を、別に返す。"""
    execution_settings, run_participant_settings = make_small_run_settings(
        valid_run_settings_mapping
    )
    run_result, participants, run_random_sources = execute_stream_protocol_run_with_factory(
        execution_settings=execution_settings, run_participant_settings=run_participant_settings
    )
    hook_counts_before = count_global_hooks()
    # 外側の計測は、準備の間と、その後の、両方の計算を数える。
    with measure_model_computation() as enclosing_meter:
        measured_run = execute_fedsda_stream_protocol_run_with_computation_measurement(
            execution_settings=execution_settings,
            run_participant_settings=run_participant_settings,
        )
    assert count_global_hooks() == hook_counts_before
    assert type(measured_run) is FedsdaMeasuredRun
    # 結果と、全clientの全状態が、計測なしの全体runと同じ。
    assert measured_run.run_result == run_result
    for run_client, measured_run_client in zip(
        participants.client_operations, measured_run.participants.client_operations, strict=True
    ):
        assert run_client is not measured_run_client
        assert repr(
            snapshot_run_client_state(
                run_client=run_client,
                python_random_generator=run_random_sources.python_random_generator,
            )
        ) == repr(
            snapshot_run_client_state(
                run_client=measured_run_client,
                python_random_generator=measured_run_client._python_random_generator,
            )
        )
    run_metric_settings = RunMetricSettings(
        maximum_detection_delay_sample_count=20, post_change_recovery_window_sample_count=5
    )
    assert derive_fedsda_run_metrics(
        run_result=measured_run.run_result,
        participants=measured_run.participants,
        run_metric_settings=run_metric_settings,
    ) == derive_fedsda_run_metrics(
        run_result=run_result, participants=participants, run_metric_settings=run_metric_settings
    )
    # 準備の間の計算: 事前学習の学習（標本数×epoch数）と、その後の、初期の損失統計のための評価。
    preparation_counts = measured_run.preparation_model_computation_counts
    assert type(preparation_counts) is ModelComputationCounts
    pretraining_sample_count = PRETRAINING_VALUES["pretraining_sample_count"]
    assert preparation_counts.concept_specific_part_training_example_count == (
        pretraining_sample_count * PRETRAINING_VALUES["pretraining_epoch_count"]
    )
    assert preparation_counts.shared_part_training_example_count == (
        preparation_counts.concept_specific_part_training_example_count
    )
    assert preparation_counts.concept_specific_part_inference_example_count == (
        pretraining_sample_count
    )
    assert preparation_counts.concept_specific_parameter_optimizer_step_count > 0
    assert preparation_counts.shared_parameter_optimizer_step_count == (
        preparation_counts.concept_specific_parameter_optimizer_step_count
    )
    # 全体runの計数は、準備の後から終わりまで（外側の計測の合計−準備の間）。
    assert measured_run.model_computation_counts == subtract_model_computation_counts(
        later_counts=enclosing_meter.get_model_computation_counts(),
        earlier_counts=preparation_counts,
    )
    assert measured_run.model_computation_counts.concept_specific_part_training_example_count > 0
    assert measured_run.model_computation_counts.concept_specific_part_inference_example_count > (
        3 * measured_run.run_result.processed_sample_count_per_client
    )


def test_measured_run_removes_measurement_and_propagates_failure(valid_run_settings_mapping):
    """実行の枠の例外は、そのまま伝わり、計測の登録は残らない。"""
    execution_settings, run_participant_settings = make_small_run_settings(
        valid_run_settings_mapping
    )
    hook_counts_before = count_global_hooks()
    for invalid_arguments in (
        dict(execution_settings=None, run_participant_settings=run_participant_settings),
        dict(execution_settings=execution_settings, run_participant_settings=None),
    ):
        with pytest.raises(Exception) as raised:
            execute_fedsda_stream_protocol_run_with_computation_measurement(**invalid_arguments)
        assert type(raised.value) is not AssertionError
        assert count_global_hooks() == hook_counts_before
    with pytest.raises(TypeError):
        execute_fedsda_stream_protocol_run_with_computation_measurement(
            execution_settings, run_participant_settings
        )
    assert count_global_hooks() == hook_counts_before
