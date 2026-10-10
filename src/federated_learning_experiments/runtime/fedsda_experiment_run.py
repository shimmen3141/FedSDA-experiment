"""完全なrun設定から、最終構成のFedSDAの1 run（計測つきの全体runと、指標の導出）を実行する。"""

from dataclasses import dataclass

from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.runtime.fedsda_measured_run_execution import (
    FedsdaMeasuredRun,
    execute_fedsda_stream_protocol_run_with_computation_measurement,
)
from federated_learning_experiments.runtime.fedsda_run_metric_derivation import (
    FedsdaRunMetrics,
    derive_fedsda_run_metrics,
)
from federated_learning_experiments.runtime.fedsda_run_settings import FedsdaRunSettings


@dataclass(frozen=True, kw_only=True, eq=False)
class FedsdaExperimentRun:
    """1 runの、条件と結果。

    `measured_run`は、計測つきの全体runの結果（実行の枠の結果、参加者、モデルの計算の計数）。
    `run_metrics`は、そこから導出した指標。参加者（状態を持つ）を含むので、値としては比べない。
    """

    run_settings: FedsdaRunSettings
    measured_run: FedsdaMeasuredRun
    run_metrics: FedsdaRunMetrics


def execute_fedsda_experiment_run(*, run_settings: FedsdaRunSettings) -> FedsdaExperimentRun:
    """完全なrun設定で、計測つきの全体runを実行し、指標を導出して返す。呼出し側の乱数は進めない。"""
    if type(run_settings) is not FedsdaRunSettings:
        raise RunSettingsValidationError(
            configuration_parameter_name="run_settings",
            specified_parameter_value=run_settings,
            validation_failure_reason="FedsdaRunSettingsを指定してください。派生型は受理しません。",
        )
    # frozenを回避して組み立てた値も、実行の前に拒否する。
    run_settings.__post_init__()
    measured_run = execute_fedsda_stream_protocol_run_with_computation_measurement(
        execution_settings=run_settings.execution_settings,
        run_participant_settings=run_settings.run_participant_settings,
    )
    run_metrics = derive_fedsda_run_metrics(
        run_result=measured_run.run_result,
        participants=measured_run.participants,
        run_metric_settings=run_settings.run_metric_settings,
        model_computation_counts=measured_run.model_computation_counts,
    )
    return FedsdaExperimentRun(
        run_settings=run_settings, measured_run=measured_run, run_metrics=run_metrics
    )
