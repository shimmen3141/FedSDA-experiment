"""全データ保持オンラインFedAvgの凍結設定。"""

from dataclasses import dataclass

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_settings import (
    RandomConceptScheduleSettings,
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


@dataclass(frozen=True, kw_only=True)
class FedavgRunParticipantSettings:
    """事前学習とclientのローカル学習の条件。"""

    initial_model_pretraining_settings: InitialModelPretrainingSettings
    hidden_layer_widths: tuple[int, ...]
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings
    local_training_batch_sample_count: int
    updates_per_sample: int
    local_update_interval_per_client_samples: int

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
            (self.local_update_interval_per_client_samples, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError("学習件数は範囲内のbuiltin intが必要です。")


@dataclass(frozen=True, kw_only=True)
class FedavgRunSettings:
    """実行・参加者・評価の完全条件。"""

    execution_settings: StreamProtocolExecutionSettings
    run_participant_settings: FedavgRunParticipantSettings
    run_metric_settings: RunMetricSettings

    def __post_init__(self) -> None:
        validate_stream_protocol_execution_settings(execution_settings=self.execution_settings)
        for settings, expected in (
            (self.run_participant_settings, FedavgRunParticipantSettings),
            (self.run_metric_settings, RunMetricSettings),
        ):
            if type(settings) is not expected:
                raise TypeError(f"exact {expected.__name__}が必要です。")
            settings.__post_init__()


def build_default_fedavg_run_settings(
    *, experiment_run_conditions: ExperimentRunConditions
) -> FedavgRunSettings:
    """旧単一モデル比較の標準条件を設定へ写す。"""
    if type(experiment_run_conditions) is not ExperimentRunConditions:
        raise TypeError("exact実行条件が必要です。")
    experiment_run_conditions.__post_init__()
    mnist = experiment_run_conditions.dataset_name in (
        "mnist2",
        "mnist4",
        "fmnist2",
        "fmnist4",
        "rotmnist2",
        "rotmnist4",
    )
    return FedavgRunSettings(
        execution_settings=StreamProtocolExecutionSettings(
            experiment_run_conditions=experiment_run_conditions,
            concept_schedule_settings=RandomConceptScheduleSettings(
                concept_schedule_strategy="random_changes_after_minimum_index_gap",
                minimum_sample_index_gap_before_change_trial=300,
                per_eligible_sample_concept_change_probability=0.0015,
            ),
            execution_strategy="sample_index_then_client_order_with_interval_synchronization",
        ),
        run_participant_settings=FedavgRunParticipantSettings(
            initial_model_pretraining_settings=InitialModelPretrainingSettings(
                pretraining_sample_count=500,
                pretraining_epoch_count=10,
                pretraining_batch_sample_count=32,
            ),
            hidden_layer_widths=(1568,) if mnist else (32, 32),
            parameter_optimizer_settings=AdamParameterOptimizerSettings(
                learning_rate=0.001 if mnist else 0.01, weight_decay=0.001, adam_variant="amsgrad"
            ),
            local_training_batch_sample_count=32,
            updates_per_sample=1,
            local_update_interval_per_client_samples=1,
        ),
        run_metric_settings=RunMetricSettings(
            maximum_detection_delay_sample_count=100, post_change_recovery_window_sample_count=200
        ),
    )
