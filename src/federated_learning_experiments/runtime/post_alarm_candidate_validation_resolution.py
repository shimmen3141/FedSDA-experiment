"""警報後の候補検証を確定し、評価結果に応じた採用・再利用・維持・棄却の状態更新を適用する。"""

from dataclasses import dataclass

from torch import Tensor

from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
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
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.learning.training.temporary_model_id_allocation import (
    TemporaryModelIdAllocator,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import (
    PostAlarmCandidateLossEvaluation,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import (
    PendingModelUploadState,
)
from federated_learning_experiments.runtime.adopted_candidate_local_adoption import (
    adopt_candidate_as_current_training_model,
)
from federated_learning_experiments.runtime.assigned_training_sample_absorption import (
    absorb_assigned_training_samples_into_held_model,
)

POST_ALARM_CANDIDATE_VALIDATION_RESOLUTION_OUTCOMES: tuple[str, ...] = (
    "candidate_adopted_as_new_model",
    "held_reference_model_reused",
    "current_model_maintained",
    "candidate_rejected",
)


@dataclass(frozen=True, kw_only=True)
class PostAlarmCandidateValidationResolution:
    """適用した結果種別、保留標本の帰属先、学習帰属が変わった場合の変更記録。"""

    resolution_outcome: str
    assigned_model_id: int
    training_model_assignment_change: TrainingModelAssignmentChange | None

    def __post_init__(self) -> None:
        if self.resolution_outcome not in POST_ALARM_CANDIDATE_VALIDATION_RESOLUTION_OUTCOMES:
            raise ValueError("resolution_outcomeは正式な結果種別のいずれかが必要です。")


def _validate_resolution_inputs(
    *,
    post_alarm_candidate_loss_evaluation: PostAlarmCandidateLossEvaluation,
    pending_assignment_training_samples: tuple[ObservedTrainingSample, ...],
    pending_assignment_sample_concept_ids: tuple[int | None, ...],
    current_training_model_assignment: CurrentTrainingModelAssignment,
) -> None:
    if type(post_alarm_candidate_loss_evaluation) is not PostAlarmCandidateLossEvaluation:
        raise TypeError(
            "post_alarm_candidate_loss_evaluationはexact PostAlarmCandidateLossEvaluationが必要です。"
        )
    if type(post_alarm_candidate_loss_evaluation.candidate_accepted) is not bool:
        raise TypeError("評価結果のcandidate_acceptedはbuiltin boolが必要です。")
    reusable_reference_model_id = post_alarm_candidate_loss_evaluation.reusable_reference_model_id
    if reusable_reference_model_id is not None and type(reusable_reference_model_id) is not int:
        raise TypeError("評価結果のreusable_reference_model_idはNoneまたはbuiltin intが必要です。")
    if (
        post_alarm_candidate_loss_evaluation.candidate_accepted
        and reusable_reference_model_id is not None
    ):
        raise ValueError("評価結果が候補の採用と再利用可能な参照の両方を示しています。")
    if type(current_training_model_assignment) is not CurrentTrainingModelAssignment:
        raise TypeError(
            "current_training_model_assignmentはexact CurrentTrainingModelAssignmentが必要です。"
        )
    if type(pending_assignment_training_samples) is not tuple:
        raise TypeError("pending_assignment_training_samplesはexact tupleが必要です。")
    if type(pending_assignment_sample_concept_ids) is not tuple:
        raise TypeError("pending_assignment_sample_concept_idsはexact tupleが必要です。")
    if len(pending_assignment_sample_concept_ids) != len(pending_assignment_training_samples):
        raise ValueError("pending_assignment_sample_concept_idsは保留標本列と同じ長さが必要です。")
    for observed_concept_id in pending_assignment_sample_concept_ids:
        if observed_concept_id is not None and type(observed_concept_id) is not int:
            raise TypeError(
                "pending_assignment_sample_concept_idsの各要素はNoneまたはbuiltin intが必要です。"
            )


def apply_post_alarm_candidate_validation_resolution(
    *,
    post_alarm_candidate_loss_evaluation: PostAlarmCandidateLossEvaluation,
    pending_assignment_training_samples: tuple[ObservedTrainingSample, ...],
    pending_assignment_sample_concept_ids: tuple[int | None, ...],
    temporary_model_id_allocator: TemporaryModelIdAllocator,
    adopted_candidate_classifier: ResidualAdapterClassifier,
    candidate_concept_specific_parameter_optimizer_state: ParameterOptimizerState,
    initial_statistics_input_features: Tensor,
    initial_statistics_observed_class_labels: Tensor,
    upload_delay_round_count: int,
    candidate_trained_sample_count: int,
    candidate_parameter_update_step_count: int,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    training_sample_store: ModelTrainingSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    current_training_model_assignment: CurrentTrainingModelAssignment,
    pending_model_upload_state: PendingModelUploadState,
) -> PostAlarmCandidateValidationResolution:
    """採用は採用の組立へ、それ以外は吸収の後に必要な学習帰属の切替えへ振り分ける。"""
    _validate_resolution_inputs(
        post_alarm_candidate_loss_evaluation=post_alarm_candidate_loss_evaluation,
        pending_assignment_training_samples=pending_assignment_training_samples,
        pending_assignment_sample_concept_ids=pending_assignment_sample_concept_ids,
        current_training_model_assignment=current_training_model_assignment,
    )
    previous_model_id = current_training_model_assignment.current_training_model_id
    if post_alarm_candidate_loss_evaluation.candidate_accepted:
        # 採用では保留標本を追加するだけで、概念計数と損失統計は更新しない。
        assignment_change = adopt_candidate_as_current_training_model(
            temporary_model_id_allocator=temporary_model_id_allocator,
            adopted_candidate_classifier=adopted_candidate_classifier,
            candidate_concept_specific_parameter_optimizer_state=candidate_concept_specific_parameter_optimizer_state,
            initial_statistics_input_features=initial_statistics_input_features,
            initial_statistics_observed_class_labels=initial_statistics_observed_class_labels,
            upload_delay_round_count=upload_delay_round_count,
            candidate_trained_sample_count=candidate_trained_sample_count,
            candidate_parameter_update_step_count=candidate_parameter_update_step_count,
            pending_assignment_training_samples=pending_assignment_training_samples,
            held_model_training_state_registry=held_model_training_state_registry,
            loss_statistics_store=loss_statistics_store,
            training_sample_store=training_sample_store,
            model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
            current_training_model_assignment=current_training_model_assignment,
            pending_model_upload_state=pending_model_upload_state,
        )
        return PostAlarmCandidateValidationResolution(
            resolution_outcome="candidate_adopted_as_new_model",
            assigned_model_id=assignment_change.current_model_id,
            training_model_assignment_change=assignment_change,
        )
    reusable_reference_model_id = post_alarm_candidate_loss_evaluation.reusable_reference_model_id
    if reusable_reference_model_id is None:
        absorb_assigned_training_samples_into_held_model(
            model_id=previous_model_id,
            assigned_training_samples=pending_assignment_training_samples,
            assigned_sample_concept_ids=pending_assignment_sample_concept_ids,
            held_model_training_state_registry=held_model_training_state_registry,
            training_sample_store=training_sample_store,
            model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
            loss_statistics_store=loss_statistics_store,
        )
        return PostAlarmCandidateValidationResolution(
            resolution_outcome="candidate_rejected",
            assigned_model_id=previous_model_id,
            training_model_assignment_change=None,
        )
    # 吸収が全検証を終えてから学習帰属を切り替え、拒否時に現在IDを変えない。
    absorb_assigned_training_samples_into_held_model(
        model_id=reusable_reference_model_id,
        assigned_training_samples=pending_assignment_training_samples,
        assigned_sample_concept_ids=pending_assignment_sample_concept_ids,
        held_model_training_state_registry=held_model_training_state_registry,
        training_sample_store=training_sample_store,
        model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
        loss_statistics_store=loss_statistics_store,
    )
    assignment_change = current_training_model_assignment.assign_model_for_training(
        model_id=reusable_reference_model_id
    )
    return PostAlarmCandidateValidationResolution(
        resolution_outcome="current_model_maintained"
        if assignment_change is None
        else "held_reference_model_reused",
        assigned_model_id=reusable_reference_model_id,
        training_model_assignment_change=assignment_change,
    )
