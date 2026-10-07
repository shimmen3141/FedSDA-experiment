"""警報の変化区間へ区間評価の結果を適用し、再利用・維持・候補検証開始のいずれかを行う。"""

from dataclasses import dataclass

from torch import Tensor, cat, float32, strided

from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training.candidate_epoch_training_settings import (
    CandidateEpochTrainingSettings,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
    TrainingModelAssignmentChange,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.model_training_and_assignment_counts import (
    ModelTrainingAndAssignmentCountsStore,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment import (
    AlarmIntervalModelReuseAssessment,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import (
    CandidateModelTrainingAndAcceptanceSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization import (
    select_candidate_initial_parameter_snapshot,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import (
    CandidateParameterInitializationSettings,
)
from federated_learning_experiments.runtime.alarm_interval_model_reuse_assessment import (
    evaluate_held_models_for_alarm_interval_reuse,
)
from federated_learning_experiments.runtime.assigned_training_sample_absorption import (
    absorb_assigned_training_samples_into_held_model,
)
from federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start import (
    PostAlarmCandidateValidationSession,
    start_post_alarm_candidate_validation_session,
)

ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES: tuple[str, ...] = (
    "alarm_interval_held_model_reused",
    "alarm_interval_current_model_maintained",
    "alarm_interval_candidate_validation_started",
)


@dataclass(frozen=True, kw_only=True)
class AlarmChangeIntervalResolution:
    """適用した結果種別、区間評価の情報、標本の吸収先、学習帰属の変更記録、開始したsession。"""

    resolution_outcome: str
    alarm_interval_reuse_assessment: AlarmIntervalModelReuseAssessment
    assigned_model_id: int | None
    training_model_assignment_change: TrainingModelAssignmentChange | None
    started_validation_session: PostAlarmCandidateValidationSession | None

    def __post_init__(self) -> None:
        if self.resolution_outcome not in ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES:
            raise ValueError("resolution_outcomeは正式な結果種別のいずれかが必要です。")


def _validate_alarm_change_interval_resolution_inputs(
    *,
    change_interval_training_samples: tuple[ObservedTrainingSample, ...],
    change_interval_sample_concept_ids: tuple[int | None, ...],
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    training_sample_store: ModelTrainingSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    current_training_model_assignment: CurrentTrainingModelAssignment,
) -> None:
    """分岐に依らない入力を、分類器の評価・状態変更・乱数消費より前に検査する。"""
    if type(held_model_training_state_registry) is not HeldModelTrainingStateRegistry:
        raise TypeError(
            "held_model_training_state_registryはexact HeldModelTrainingStateRegistryが必要です。"
        )
    if type(loss_statistics_store) is not ModelAndClassLossStatisticsStore:
        raise TypeError("loss_statistics_storeはexact ModelAndClassLossStatisticsStoreが必要です。")
    if type(training_sample_store) is not ModelTrainingSampleStore:
        raise TypeError("training_sample_storeはexact ModelTrainingSampleStoreが必要です。")
    if (
        type(model_training_and_assignment_counts_store)
        is not ModelTrainingAndAssignmentCountsStore
    ):
        raise TypeError(
            "model_training_and_assignment_counts_storeは"
            "exact ModelTrainingAndAssignmentCountsStoreが必要です。"
        )
    if type(current_training_model_assignment) is not CurrentTrainingModelAssignment:
        raise TypeError(
            "current_training_model_assignmentはexact CurrentTrainingModelAssignmentが必要です。"
        )
    if type(change_interval_training_samples) is not tuple:
        raise TypeError("change_interval_training_samplesはexact tupleが必要です。")
    if not change_interval_training_samples:
        raise ValueError("change_interval_training_samplesは1件以上が必要です。")
    for training_sample in change_interval_training_samples:
        if type(training_sample) is not ObservedTrainingSample:
            raise TypeError(
                "change_interval_training_samplesの各要素はexact ObservedTrainingSampleが必要です。"
            )
        if (
            type(training_sample.input_features) is not Tensor
            or type(training_sample.observed_class_labels) is not Tensor
        ):
            raise TypeError("変化区間の標本の特徴とラベルはexact Tensorが必要です。")
    first_training_sample = change_interval_training_samples[0]
    if first_training_sample.input_features.ndim != 2:
        raise ValueError("変化区間の標本の特徴はshape[1, F]が必要です。")
    input_feature_count = first_training_sample.input_features.shape[1]
    for training_sample in change_interval_training_samples:
        for parameter_name, specified_value in (
            ("input_features", training_sample.input_features),
            ("observed_class_labels", training_sample.observed_class_labels),
        ):
            if (
                specified_value.device.type != "cpu"
                or specified_value.dtype != float32
                or specified_value.layout != strided
                or specified_value.is_nested
            ):
                raise ValueError(
                    f"変化区間の標本の{parameter_name}はCPU float32 strided Tensorが必要です。"
                )
        if training_sample.input_features.shape != (1, input_feature_count):
            raise ValueError("変化区間の標本の特徴は、特徴数が揃ったshape[1, F]が必要です。")
        if training_sample.observed_class_labels.shape != (1, 1):
            raise ValueError("変化区間の標本のラベルはshape[1, 1]が必要です。")
    if type(change_interval_sample_concept_ids) is not tuple:
        raise TypeError("change_interval_sample_concept_idsはexact tupleが必要です。")
    if len(change_interval_sample_concept_ids) != len(change_interval_training_samples):
        raise ValueError("change_interval_sample_concept_idsは標本列と同じ長さが必要です。")
    for observed_concept_id in change_interval_sample_concept_ids:
        if observed_concept_id is not None and type(observed_concept_id) is not int:
            raise TypeError(
                "change_interval_sample_concept_idsの各要素は"
                "Noneまたはbool以外のbuiltin intが必要です。"
            )
    held_model_training_states = (
        held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    )
    if not held_model_training_states:
        raise LookupError("保有モデルがないため変化区間を解決できません。")
    if current_training_model_assignment.current_training_model_id not in tuple(
        held_model_training_state.model_id
        for held_model_training_state in held_model_training_states
    ):
        raise LookupError("現在の学習帰属IDは保有モデルに含まれる必要があります。")


def resolve_alarm_change_interval(
    *,
    change_interval_training_samples: tuple[ObservedTrainingSample, ...],
    change_interval_sample_concept_ids: tuple[int | None, ...],
    maximum_alarm_interval_mean_loss_increase: float,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    training_sample_store: ModelTrainingSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    current_training_model_assignment: CurrentTrainingModelAssignment,
    candidate_parameter_initialization_settings: CandidateParameterInitializationSettings,
    architecture_reference_classifier: ResidualAdapterClassifier,
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
    candidate_epoch_training_settings: CandidateEpochTrainingSettings,
    candidate_model_training_and_acceptance_settings: CandidateModelTrainingAndAcceptanceSettings,
    proposal_sample_index: int,
    estimated_change_point_sample_index: int | None,
    detection_episode_id: int | None,
    detector_name: str,
) -> AlarmChangeIntervalResolution:
    """区間を評価し、選択があれば吸収と帰属切替え、なければ候補検証sessionを開始する。"""
    _validate_alarm_change_interval_resolution_inputs(
        change_interval_training_samples=change_interval_training_samples,
        change_interval_sample_concept_ids=change_interval_sample_concept_ids,
        held_model_training_state_registry=held_model_training_state_registry,
        loss_statistics_store=loss_statistics_store,
        training_sample_store=training_sample_store,
        model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
        current_training_model_assignment=current_training_model_assignment,
    )
    # 標本順に連結した区間全体を、評価と候補の学習の両方に使う。
    change_interval_input_features = cat(
        [training_sample.input_features for training_sample in change_interval_training_samples]
    )
    change_interval_observed_class_labels = cat(
        [
            training_sample.observed_class_labels
            for training_sample in change_interval_training_samples
        ]
    )
    alarm_interval_reuse_assessment = evaluate_held_models_for_alarm_interval_reuse(
        input_features=change_interval_input_features,
        observed_class_labels=change_interval_observed_class_labels,
        held_model_training_state_registry=held_model_training_state_registry,
        loss_statistics_store=loss_statistics_store,
        maximum_alarm_interval_mean_loss_increase=maximum_alarm_interval_mean_loss_increase,
    )
    selected_reuse_model_id = alarm_interval_reuse_assessment.selected_reuse_model_id
    if selected_reuse_model_id is not None:
        # 吸収は自身の検査と全標本の損失評価の後に更新する。帰属の切替えはその後に行う。
        absorb_assigned_training_samples_into_held_model(
            model_id=selected_reuse_model_id,
            assigned_training_samples=change_interval_training_samples,
            assigned_sample_concept_ids=change_interval_sample_concept_ids,
            held_model_training_state_registry=held_model_training_state_registry,
            training_sample_store=training_sample_store,
            model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
            loss_statistics_store=loss_statistics_store,
        )
        training_model_assignment_change = (
            current_training_model_assignment.assign_model_for_training(
                model_id=selected_reuse_model_id
            )
        )
        return AlarmChangeIntervalResolution(
            resolution_outcome=(
                "alarm_interval_current_model_maintained"
                if training_model_assignment_change is None
                else "alarm_interval_held_model_reused"
            ),
            alarm_interval_reuse_assessment=alarm_interval_reuse_assessment,
            assigned_model_id=selected_reuse_model_id,
            training_model_assignment_change=training_model_assignment_change,
            started_validation_session=None,
        )
    # 適合なし。区間は吸収せず、保留標本として候補検証sessionへ渡す。
    held_model_training_states = (
        held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    )
    available_parameter_snapshots_by_model_id = {
        held_model_training_state.model_id: snapshot_classifier_parameters(
            classifier=held_model_training_state.classifier
        )
        for held_model_training_state in held_model_training_states
    }
    initial_candidate_parameter_snapshot = select_candidate_initial_parameter_snapshot(
        settings=candidate_parameter_initialization_settings,
        available_parameter_snapshots_by_model_id=available_parameter_snapshots_by_model_id,
        current_training_model_id=current_training_model_assignment.current_training_model_id,
        evaluated_mean_losses_by_model_id=alarm_interval_reuse_assessment.baseline_supported_interval_mean_losses_by_model_id,
    )
    if initial_candidate_parameter_snapshot is None:
        # 保有モデルが1件以上あるので既存の初期値選択はNoneを返さない。型上の分岐を明示して拒否する。
        raise LookupError("候補の初期parameterを選べません。")
    started_validation_session = start_post_alarm_candidate_validation_session(
        architecture_reference_classifier=architecture_reference_classifier,
        initial_candidate_parameter_snapshot=initial_candidate_parameter_snapshot,
        parameter_optimizer_settings=parameter_optimizer_settings,
        candidate_epoch_training_settings=candidate_epoch_training_settings,
        candidate_model_training_and_acceptance_settings=candidate_model_training_and_acceptance_settings,
        held_model_training_state_registry=held_model_training_state_registry,
        loss_statistics_store=loss_statistics_store,
        current_training_model_assignment=current_training_model_assignment,
        input_features=change_interval_input_features,
        observed_class_labels=change_interval_observed_class_labels,
        pending_assignment_training_samples=change_interval_training_samples,
        proposal_sample_index=proposal_sample_index,
        estimated_change_point_sample_index=estimated_change_point_sample_index,
        detection_episode_id=detection_episode_id,
        detector_name=detector_name,
    )
    return AlarmChangeIntervalResolution(
        resolution_outcome="alarm_interval_candidate_validation_started",
        alarm_interval_reuse_assessment=alarm_interval_reuse_assessment,
        assigned_model_id=None,
        training_model_assignment_change=None,
        started_validation_session=started_validation_session,
    )
