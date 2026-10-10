"""最終構成のFedSDAの既定値を持つ、完全なrun設定を作る。

最終構成は、docs/overview/proposed-method.mdの「最終提案構成」。規模（client数、標本数、集約間隔）・seed・
datasetは、実行条件として、呼出し側が渡す。ここが持つのは、それ以外の全部の既定値である。
既定値を変えるときは、返された値の一部を`dataclasses.replace`で置き換える（検証は、もう一度行われる）。
"""

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_settings import (
    RandomConceptScheduleSettings,
)
from federated_learning_experiments.evaluation.run_metric_calculations import RunMetricSettings
from federated_learning_experiments.execution.stream_protocol_execution_settings import (
    StreamProtocolExecutionSettings,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.training.candidate_epoch_training_settings import (
    CandidateEpochTrainingSettings,
)
from federated_learning_experiments.learning.training.initial_model_pretraining_settings import (
    InitialModelPretrainingSettings,
)
from federated_learning_experiments.learning.training.local_training_schedule_settings import (
    LocalTrainingScheduleSettings,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import (
    CandidateModelTrainingAndAcceptanceSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import (
    CandidateParameterInitializationSettings,
)
from federated_learning_experiments.methods.fedsda.consolidation.model_clustering_calculations import (
    ModelClusteringCriteria,
)
from federated_learning_experiments.methods.fedsda.consolidation.model_consolidation_settings import (
    ModelConsolidationSettings,
)
from federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings import (
    LossChangeDetectionSettings,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.prediction_combination_settings import (
    PredictionCombinationSettings,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings import (
    TrainingDataAssignmentSettings,
)
from federated_learning_experiments.runtime.fedsda_run_client_settings import (
    FedsdaRunClientScalarSettings,
    FedsdaRunClientSettings,
)
from federated_learning_experiments.runtime.fedsda_run_participant_factory import (
    FedsdaRunParticipantSettings,
)
from federated_learning_experiments.runtime.fedsda_run_settings import FedsdaRunSettings

# ---- datasetごとの、モデルの既定（隠れ層の幅と学習率） ----
_SYNTHETIC_HIDDEN_LAYER_WIDTHS = (32, 32)
_SYNTHETIC_LEARNING_RATE = 0.01
_MNIST_HIDDEN_LAYER_WIDTHS = (1568,)
_MNIST_LEARNING_RATE = 0.001
_HIDDEN_LAYER_WIDTHS_AND_LEARNING_RATE_BY_DATASET_NAME: dict[str, tuple[tuple[int, ...], float]] = {
    "sine2": (_SYNTHETIC_HIDDEN_LAYER_WIDTHS, _SYNTHETIC_LEARNING_RATE),
    "sea2": (_SYNTHETIC_HIDDEN_LAYER_WIDTHS, _SYNTHETIC_LEARNING_RATE),
    "sea4": (_SYNTHETIC_HIDDEN_LAYER_WIDTHS, _SYNTHETIC_LEARNING_RATE),
    "circle2": (_SYNTHETIC_HIDDEN_LAYER_WIDTHS, _SYNTHETIC_LEARNING_RATE),
    "mnist2": (_MNIST_HIDDEN_LAYER_WIDTHS, _MNIST_LEARNING_RATE),
    "mnist4": (_MNIST_HIDDEN_LAYER_WIDTHS, _MNIST_LEARNING_RATE),
}

# ---- 概念の変更（評価用の概念列） ----
_MINIMUM_SAMPLE_INDEX_GAP_BEFORE_CHANGE_TRIAL = 300
_PER_ELIGIBLE_SAMPLE_CONCEPT_CHANGE_PROBABILITY = 0.0015

# ---- モデルと学習 ----
_RESIDUAL_ADAPTER_REQUESTED_RANK = 8
_WEIGHT_DECAY = 0.001
_LOCAL_TRAINING_BATCH_SAMPLE_COUNT = 32
_TRAINING_REQUESTS_PER_UPDATE_INTERVAL = 1
_JOINT_UPDATE_ITERATIONS_PER_TRAINING_REQUEST = 1
_PRETRAINING_SAMPLE_COUNT = 500
_PRETRAINING_EPOCH_COUNT = 10
_PRETRAINING_BATCH_SAMPLE_COUNT = 32

# ---- 損失の変化の検出 ----
_E_SR_FALSE_ALARM_CONTROL_ALPHA = 0.001
_LOSS_MONITOR_BETTING_FRACTIONS = (0.05, 0.1, 0.2, 0.4, 0.8)
_LOSS_MONITOR_MAXIMUM_RETAINED_CANDIDATE_COUNT = 1000
_DETECTOR_NAME = "overall + class-conditional e-SR mixture"
_MINIMUM_CHANGE_INTERVAL_SAMPLE_COUNT = 5

# ---- 帰属の保留（FIFO）と、予測の結合（Fixed-Shareの時間尺度は、FIFOの容量と同じ値） ----
_PENDING_ASSIGNMENT_BUFFER_CAPACITY_SAMPLES = 30

# ---- モデルの再利用と、候補の作成・検証 ----
# 履歴の平均損失からの増加の許容量（γ）。クラスタリングの、同じクラスタとみなす判定の値の上限にも使う。
_MAXIMUM_TOLERATED_MEAN_LOSS_INCREASE = 0.1
_CANDIDATE_POST_ALARM_VALIDATION_SAMPLE_COUNT = 10
_CANDIDATE_MAXIMUM_EPOCH_COUNT = 30
_CANDIDATE_VALIDATION_SAMPLE_FRACTION = 0.2
_CANDIDATE_CONSECUTIVE_NON_IMPROVING_EPOCH_LIMIT = 3
# 候補の平均損失の最小改善量（候補の学習の、早期終了の最小改善量と同じ値）。
_MINIMUM_CANDIDATE_MEAN_LOSS_IMPROVEMENT = 0.0001
_NEW_MODEL_UPLOAD_DELAY_ROUND_COUNT = 1

# ---- 評価標本と、サーバのクロス評価・クラスタリング ----
_MAXIMUM_STORED_EVALUATION_SAMPLE_COUNT_PER_MODEL = 50
_ADDED_EVALUATION_BATCH_SAMPLE_COUNT = 20
_MAXIMUM_CROSS_EVALUATION_SAMPLE_COUNT = 50
_MAXIMUM_EVALUATING_CLIENT_COUNT_PER_MODEL = 3
_MINIMUM_PAIR_EVALUATION_SAMPLE_COUNT = 5
_CLUSTERING_CONFIDENCE_LEVEL = 0.95

# ---- 指標 ----
_MAXIMUM_DETECTION_DELAY_SAMPLE_COUNT = 100
_POST_CHANGE_RECOVERY_WINDOW_SAMPLE_COUNT = 200


def build_final_configuration_fedsda_run_settings(
    *, experiment_run_conditions: ExperimentRunConditions
) -> FedsdaRunSettings:
    """実行条件（dataset、seed、規模）から、最終構成の既定値を持つ、完全なrun設定を返す。"""
    if type(experiment_run_conditions) is not ExperimentRunConditions:
        raise RunSettingsValidationError(
            configuration_parameter_name="experiment_run_conditions",
            specified_parameter_value=experiment_run_conditions,
            validation_failure_reason="ExperimentRunConditionsを指定してください。",
        )
    dataset_name = experiment_run_conditions.dataset_name
    if dataset_name not in _HIDDEN_LAYER_WIDTHS_AND_LEARNING_RATE_BY_DATASET_NAME:
        raise RunSettingsValidationError(
            configuration_parameter_name="dataset_name",
            specified_parameter_value=dataset_name,
            validation_failure_reason=(
                "最終構成の、モデルの既定（隠れ層の幅と学習率）があるdatasetを指定してください: "
                + "・".join(_HIDDEN_LAYER_WIDTHS_AND_LEARNING_RATE_BY_DATASET_NAME)
                + "。"
            ),
        )
    hidden_layer_widths, learning_rate = _HIDDEN_LAYER_WIDTHS_AND_LEARNING_RATE_BY_DATASET_NAME[
        dataset_name
    ]
    # 2つのoptimizerの設定（候補とつなぎ直し、事前学習と配布での作り直し）は、既定では、同じ学習率。
    parameter_optimizer_settings = AdamParameterOptimizerSettings(
        learning_rate=learning_rate, weight_decay=_WEIGHT_DECAY, adam_variant="amsgrad"
    )
    run_client_settings = FedsdaRunClientSettings(
        loss_change_detection_settings=LossChangeDetectionSettings(
            drift_detector_name="e_sr",
            loss_monitoring_scope="overall_and_true_class_losses",
            e_sr_false_alarm_control_alpha=_E_SR_FALSE_ALARM_CONTROL_ALPHA,
        ),
        prediction_combination_settings=PredictionCombinationSettings(
            prediction_combination_strategy="fixed_share_weighted_prediction",
            prediction_mixture_activation_policy="always",
            prediction_weight_recalibration_after_aggregation_policy=(
                "recompute_buffer_losses_and_replay_weight_updates"
            ),
            prediction_state_reset_on_training_assignment_change_policy=(
                "restart_adahedge_preserve_fixed_share_prediction_state"
            ),
            fixed_share_weight_redistribution_time_scale_samples=(
                _PENDING_ASSIGNMENT_BUFFER_CAPACITY_SAMPLES
            ),
        ),
        local_training_settings=LocalTrainingSettings(
            local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
            shared_backbone_gradient_combination_strategy=(
                "sample_weighted_mean_per_concept_gradients"
            ),
        ),
        local_training_schedule_settings=LocalTrainingScheduleSettings(
            training_requests_per_update_interval=_TRAINING_REQUESTS_PER_UPDATE_INTERVAL,
            joint_update_iterations_per_training_request=(
                _JOINT_UPDATE_ITERATIONS_PER_TRAINING_REQUEST
            ),
        ),
        training_data_assignment_settings=TrainingDataAssignmentSettings(
            pending_assignment_buffer_capacity_samples=_PENDING_ASSIGNMENT_BUFFER_CAPACITY_SAMPLES
        ),
        candidate_model_training_and_acceptance_settings=(
            CandidateModelTrainingAndAcceptanceSettings(
                candidate_model_acceptance_policy=(
                    "current_model_first_reuse_then_two_segment_candidate_validation"
                ),
                candidate_post_alarm_validation_sample_count=(
                    _CANDIDATE_POST_ALARM_VALIDATION_SAMPLE_COUNT
                ),
            )
        ),
        candidate_parameter_initialization_settings=CandidateParameterInitializationSettings(
            candidate_parameter_initialization_source="lowest_evaluated_mean_loss_model"
        ),
        candidate_epoch_training_settings=CandidateEpochTrainingSettings(
            candidate_training_strategy="validation_loss_early_stopping",
            maximum_epoch_count=_CANDIDATE_MAXIMUM_EPOCH_COUNT,
            maximum_batch_sample_count=_LOCAL_TRAINING_BATCH_SAMPLE_COUNT,
            validation_sample_fraction=_CANDIDATE_VALIDATION_SAMPLE_FRACTION,
            consecutive_non_improving_epoch_limit=_CANDIDATE_CONSECUTIVE_NON_IMPROVING_EPOCH_LIMIT,
            minimum_validation_loss_decrease=_MINIMUM_CANDIDATE_MEAN_LOSS_IMPROVEMENT,
        ),
        parameter_optimizer_settings=parameter_optimizer_settings,
        rebuilt_model_parameter_optimizer_settings=parameter_optimizer_settings,
        scalar_settings=FedsdaRunClientScalarSettings(
            maximum_tolerated_mean_loss_increase=_MAXIMUM_TOLERATED_MEAN_LOSS_INCREASE,
            minimum_candidate_mean_loss_improvement=_MINIMUM_CANDIDATE_MEAN_LOSS_IMPROVEMENT,
            new_model_upload_delay_round_count=_NEW_MODEL_UPLOAD_DELAY_ROUND_COUNT,
            minimum_change_interval_sample_count=_MINIMUM_CHANGE_INTERVAL_SAMPLE_COUNT,
            local_training_batch_sample_count=_LOCAL_TRAINING_BATCH_SAMPLE_COUNT,
            maximum_stored_evaluation_sample_count_per_model=(
                _MAXIMUM_STORED_EVALUATION_SAMPLE_COUNT_PER_MODEL
            ),
            added_evaluation_batch_sample_count=_ADDED_EVALUATION_BATCH_SAMPLE_COUNT,
            maximum_cross_evaluation_sample_count=_MAXIMUM_CROSS_EVALUATION_SAMPLE_COUNT,
            loss_monitor_maximum_retained_candidate_count=(
                _LOSS_MONITOR_MAXIMUM_RETAINED_CANDIDATE_COUNT
            ),
            detector_name=_DETECTOR_NAME,
        ),
        loss_monitor_betting_fractions=_LOSS_MONITOR_BETTING_FRACTIONS,
    )
    return FedsdaRunSettings(
        execution_settings=StreamProtocolExecutionSettings(
            experiment_run_conditions=experiment_run_conditions,
            concept_schedule_settings=RandomConceptScheduleSettings(
                concept_schedule_strategy="random_changes_after_minimum_index_gap",
                minimum_sample_index_gap_before_change_trial=(
                    _MINIMUM_SAMPLE_INDEX_GAP_BEFORE_CHANGE_TRIAL
                ),
                per_eligible_sample_concept_change_probability=(
                    _PER_ELIGIBLE_SAMPLE_CONCEPT_CHANGE_PROBABILITY
                ),
            ),
            execution_strategy="sample_index_then_client_order_with_interval_synchronization",
        ),
        run_participant_settings=FedsdaRunParticipantSettings(
            run_client_settings=run_client_settings,
            initial_model_pretraining_settings=InitialModelPretrainingSettings(
                pretraining_sample_count=_PRETRAINING_SAMPLE_COUNT,
                pretraining_epoch_count=_PRETRAINING_EPOCH_COUNT,
                pretraining_batch_sample_count=_PRETRAINING_BATCH_SAMPLE_COUNT,
            ),
            model_architecture_settings=ModelArchitectureSettings(
                model_architecture_name="shared_backbone_residual_adapter",
                residual_adapter_requested_rank=_RESIDUAL_ADAPTER_REQUESTED_RANK,
            ),
            hidden_layer_widths=hidden_layer_widths,
            model_clustering_criteria=ModelClusteringCriteria(
                maximum_same_cluster_decision_score=_MAXIMUM_TOLERATED_MEAN_LOSS_INCREASE,
                minimum_pair_evaluation_sample_count=_MINIMUM_PAIR_EVALUATION_SAMPLE_COUNT,
                clustering_confidence_level=_CLUSTERING_CONFIDENCE_LEVEL,
            ),
            maximum_evaluating_client_count_per_model=_MAXIMUM_EVALUATING_CLIENT_COUNT_PER_MODEL,
        ),
        model_consolidation_settings=ModelConsolidationSettings(
            model_clustering_trigger_policy="on_new_model_registration",
            model_pair_comparison_strategy="classwise_unique_correctness_lower_confidence_bound",
            model_clustering_linkage="average_linkage",
            model_consolidation_policy="weighted_parameter_average_and_merge_ids",
        ),
        run_metric_settings=RunMetricSettings(
            maximum_detection_delay_sample_count=_MAXIMUM_DETECTION_DELAY_SAMPLE_COUNT,
            post_change_recovery_window_sample_count=_POST_CHANGE_RECOVERY_WINDOW_SAMPLE_COUNT,
        ),
    )
