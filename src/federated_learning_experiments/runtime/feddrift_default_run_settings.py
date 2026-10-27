"""FedDriftの既定完全条件を作る。"""

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_settings import (
    RandomConceptScheduleSettings,
)
from federated_learning_experiments.evaluation.run_metric_calculations import RunMetricSettings
from federated_learning_experiments.learning.training.initial_model_pretraining_settings import (
    InitialModelPretrainingSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
)
from federated_learning_experiments.runtime.feddrift_run_client_settings import (
    FeddriftRunClientSettings,
)
from federated_learning_experiments.runtime.feddrift_run_participant_factory import (
    FeddriftRunParticipantSettings,
)
from federated_learning_experiments.runtime.feddrift_run_settings import FeddriftRunSettings
from federated_learning_experiments.runtime.feddrift_server_settings import FeddriftServerSettings
from federated_learning_experiments.runtime.feddrift_stream_run_execution import (
    FeddriftStreamProtocolExecutionSettings,
)

_LOSS_INCREASE_THRESHOLDS_BY_DATASET = {
    "sea2": 0.04,
    "sea4": 0.04,
    "sine2": 0.20,
    "circle2": 0.10,
    "mnist2": 0.10,
    "mnist4": 0.10,
}

_MODEL_DEFAULTS_BY_DATASET = {
    "sine2": ((32, 32), 0.01),
    "sea2": ((32, 32), 0.01),
    "sea4": ((32, 32), 0.01),
    "circle2": ((32, 32), 0.01),
    "mnist2": ((1568,), 0.001),
    "mnist4": ((1568,), 0.001),
}


def build_default_feddrift_run_settings(
    *,
    experiment_run_conditions: ExperimentRunConditions,
    loss_increase_threshold: float | None = None,
) -> FeddriftRunSettings:
    """公式にそろえた既定条件を作る。生成・実行や乱数消費は行わない。"""
    if type(experiment_run_conditions) is not ExperimentRunConditions:
        raise TypeError("experiment_run_conditions must be exact ExperimentRunConditions")
    experiment_run_conditions.__post_init__()
    if loss_increase_threshold is None:
        loss_increase_threshold = _LOSS_INCREASE_THRESHOLDS_BY_DATASET[
            experiment_run_conditions.dataset_name
        ]
    execution_settings = FeddriftStreamProtocolExecutionSettings(
        experiment_run_conditions=experiment_run_conditions,
        concept_schedule_settings=RandomConceptScheduleSettings(
            concept_schedule_strategy="random_changes_after_minimum_index_gap",
            minimum_sample_index_gap_before_change_trial=300,
            per_eligible_sample_concept_change_probability=0.0015,
        ),
    )
    hidden_layer_widths, learning_rate = _MODEL_DEFAULTS_BY_DATASET[
        experiment_run_conditions.dataset_name
    ]
    optimizer = AdamParameterOptimizerSettings(
        learning_rate=learning_rate, weight_decay=0.001, adam_variant="amsgrad"
    )
    return FeddriftRunSettings(
        execution_settings=execution_settings,
        run_participant_settings=FeddriftRunParticipantSettings(
            run_client_settings=FeddriftRunClientSettings(
                model_comparison_criterion="classification_error_rate",
                new_model_parameter_source="minimum_criterion_held_model",
                detection_reference_after_drift="minimum_held_model_batch_value",
                loss_increase_threshold=loss_increase_threshold,
                detection_batch_sample_count=experiment_run_conditions.server_aggregation_interval_per_client_samples,
                local_training_batch_sample_count=32,
                updates_per_sample=1,
                maximum_stored_evaluation_sample_count_per_model=50,
                added_evaluation_batch_sample_count=20,
                new_model_optimizer_settings=optimizer,
            ),
            server_settings=FeddriftServerSettings(
                model_comparison_criterion="classification_error_rate",
                training_model_eligibility="models_assigned_in_current_timestep",
                consolidation_weight_timing="weights_including_current_timestep_assignments",
                maximum_same_cluster_loss_increase=loss_increase_threshold,
                cluster_linkage="complete",
                minimum_cross_evaluation_sample_count=5,
                maximum_cross_evaluation_client_count=experiment_run_conditions.client_count,
                maximum_client_evaluation_sample_count=50,
                new_model_isolation_timestep_count=1,
                local_training_round_count=1,
                initial_model_aggregation_weight=500,
                rebuilt_model_optimizer_settings=optimizer,
            ),
            initial_model_pretraining_settings=InitialModelPretrainingSettings(
                pretraining_sample_count=500,
                pretraining_epoch_count=10,
                pretraining_batch_sample_count=32,
            ),
            hidden_layer_widths=hidden_layer_widths,
        ),
        run_metric_settings=RunMetricSettings(
            maximum_detection_delay_sample_count=100, post_change_recovery_window_sample_count=200
        ),
    )
