"""FedDriftの完全なrun条件。"""

from dataclasses import dataclass

from federated_learning_experiments.evaluation.run_metric_calculations import RunMetricSettings
from federated_learning_experiments.runtime.feddrift_run_participant_factory import (
    FeddriftRunParticipantSettings,
)
from federated_learning_experiments.runtime.feddrift_stream_run_execution import (
    FeddriftStreamProtocolExecutionSettings,
)


@dataclass(frozen=True, kw_only=True)
class FeddriftRunSettings:
    """実行・参加者・指標条件の束。"""

    execution_settings: FeddriftStreamProtocolExecutionSettings
    run_participant_settings: FeddriftRunParticipantSettings
    run_metric_settings: RunMetricSettings

    def __post_init__(self) -> None:
        for value, expected in (
            (self.execution_settings, FeddriftStreamProtocolExecutionSettings),
            (self.run_participant_settings, FeddriftRunParticipantSettings),
            (self.run_metric_settings, RunMetricSettings),
        ):
            if type(value) is not expected:
                raise TypeError(f"exact {expected.__name__}が必要です。")
            value.__post_init__()
        if (
            self.execution_settings.experiment_run_conditions.server_aggregation_interval_per_client_samples
            != self.run_participant_settings.run_client_settings.detection_batch_sample_count
        ):
            raise ValueError("FedDriftの集約間隔と検出バッチ件数を一致させてください。")
