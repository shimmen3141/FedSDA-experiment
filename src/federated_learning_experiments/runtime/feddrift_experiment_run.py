"""完全条件からFedDriftの実験runを実行する。"""

from dataclasses import dataclass

from federated_learning_experiments.runtime.feddrift_measured_run_execution import (
    FeddriftMeasuredRun,
    execute_feddrift_stream_protocol_run_with_computation_measurement,
)
from federated_learning_experiments.runtime.feddrift_run_metric_derivation import (
    FeddriftRunMetrics,
    derive_feddrift_run_metrics,
)
from federated_learning_experiments.runtime.feddrift_run_settings import FeddriftRunSettings


@dataclass(frozen=True, kw_only=True, eq=False)
class FeddriftExperimentRun:
    """条件・計測結果・全指標。参加者状態を含むため値として比べない。"""

    run_settings: FeddriftRunSettings
    measured_run: FeddriftMeasuredRun
    run_metrics: FeddriftRunMetrics


def execute_feddrift_experiment_run(*, run_settings: FeddriftRunSettings) -> FeddriftExperimentRun:
    """計測つき全体runと指標導出を接続する。"""
    if type(run_settings) is not FeddriftRunSettings:
        raise TypeError("exact FeddriftRunSettingsが必要です。")
    run_settings.__post_init__()
    measured = execute_feddrift_stream_protocol_run_with_computation_measurement(
        execution_settings=run_settings.execution_settings,
        run_participant_settings=run_settings.run_participant_settings,
    )
    metrics = derive_feddrift_run_metrics(
        run_result=measured.run_result,
        participants=measured.participants,
        run_metric_settings=run_settings.run_metric_settings,
        model_computation_counts=measured.model_computation_counts,
    )
    return FeddriftExperimentRun(
        run_settings=run_settings, measured_run=measured, run_metrics=metrics
    )
