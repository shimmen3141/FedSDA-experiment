"""`fedavg`と同じ学習器を専門家にしたDWM（構成`dwm-neural-network`）の凍結設定。"""

from dataclasses import dataclass

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.evaluation.run_metric_calculations import RunMetricSettings
from federated_learning_experiments.execution.stream_protocol_execution_settings import (
    StreamProtocolExecutionSettings,
    validate_stream_protocol_execution_settings,
)
from federated_learning_experiments.learning.training.initial_model_pretraining_settings import (
    InitialModelPretrainingSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.runtime.fedavg_run_settings import (
    build_default_fedavg_run_settings,
)

# Kolter & Maloof (2007) 4.2節の、大きく雑音のある問題（SEA、5万標本）の値。公式の実装の既定はp=1。
DEFAULT_WEIGHT_DECREASE_FACTOR = 0.5
DEFAULT_REMOVAL_WEIGHT_THRESHOLD = 0.01
DEFAULT_PERIOD_SAMPLE_COUNT = 50


@dataclass(frozen=True, kw_only=True)
class DynamicWeightedMajorityRunParticipantSettings:
    """事前学習・専門家の学習器・DWMの固定値。"""

    initial_model_pretraining_settings: InitialModelPretrainingSettings
    hidden_layer_widths: tuple[int, ...]
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings
    local_training_batch_sample_count: int
    updates_per_sample: int
    weight_decrease_factor: float
    removal_weight_threshold: float
    period_sample_count: int

    def __post_init__(self) -> None:
        if type(self.initial_model_pretraining_settings) is not InitialModelPretrainingSettings:
            raise TypeError("exact事前学習設定が必要です。")
        self.initial_model_pretraining_settings.__post_init__()
        if type(self.parameter_optimizer_settings) not in (
            AdamParameterOptimizerSettings,
            SgdParameterOptimizerSettings,
        ):
            raise TypeError("exact Adam/SGD設定が必要です。")
        self.parameter_optimizer_settings.__post_init__()
        if (
            type(self.hidden_layer_widths) is not tuple
            or not self.hidden_layer_widths
            or any(type(w) is not int or w < 1 for w in self.hidden_layer_widths)
        ):
            raise ValueError("隠れ層は正のbuiltin intの非空tupleが必要です。")
        for value, minimum in (
            (self.local_training_batch_sample_count, 1),
            (self.updates_per_sample, 0),
            (self.period_sample_count, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError("件数は範囲内のbuiltin intが必要です。")
        for value in (self.weight_decrease_factor, self.removal_weight_threshold):
            if type(value) is not float:
                raise TypeError("βとθにはbuiltin floatが必要です。")
            if not 0.0 < value < 1.0:
                raise ValueError("βとθは開区間(0, 1)が必要です。")


@dataclass(frozen=True, kw_only=True)
class DynamicWeightedMajorityRunSettings:
    """実行・参加者・評価の完全条件。"""

    execution_settings: StreamProtocolExecutionSettings
    run_participant_settings: DynamicWeightedMajorityRunParticipantSettings
    run_metric_settings: RunMetricSettings

    def __post_init__(self) -> None:
        validate_stream_protocol_execution_settings(execution_settings=self.execution_settings)
        for settings, expected in (
            (self.run_participant_settings, DynamicWeightedMajorityRunParticipantSettings),
            (self.run_metric_settings, RunMetricSettings),
        ):
            if type(settings) is not expected:
                raise TypeError(f"exact {expected.__name__}が必要です。")
            settings.__post_init__()


def build_default_dynamic_weighted_majority_run_settings(
    *, experiment_run_conditions: ExperimentRunConditions
) -> DynamicWeightedMajorityRunSettings:
    """`fedavg`の既定から実行・事前学習・構造・optimizer・batch・更新の回数・評価の設定を取り出し、β・θ・pを足す。

    `fedavg`の局所更新をまとめる間隔は取り出さない（DWMは毎標本で学習する）。
    """
    fedavg = build_default_fedavg_run_settings(experiment_run_conditions=experiment_run_conditions)
    participant = fedavg.run_participant_settings
    return DynamicWeightedMajorityRunSettings(
        execution_settings=fedavg.execution_settings,
        run_participant_settings=DynamicWeightedMajorityRunParticipantSettings(
            initial_model_pretraining_settings=participant.initial_model_pretraining_settings,
            hidden_layer_widths=participant.hidden_layer_widths,
            parameter_optimizer_settings=participant.parameter_optimizer_settings,
            local_training_batch_sample_count=participant.local_training_batch_sample_count,
            updates_per_sample=participant.updates_per_sample,
            weight_decrease_factor=DEFAULT_WEIGHT_DECREASE_FACTOR,
            removal_weight_threshold=DEFAULT_REMOVAL_WEIGHT_THRESHOLD,
            period_sample_count=DEFAULT_PERIOD_SAMPLE_COUNT,
        ),
        run_metric_settings=fedavg.run_metric_settings,
    )
