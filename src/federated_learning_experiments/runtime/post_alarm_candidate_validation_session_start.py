"""警報後の候補生成・学習・参照固定・損失収集開始を順に組み立てる。"""

from dataclasses import dataclass

from torch import Tensor

from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.model_architecture_classifier import (
    ModelArchitectureClassifier,
)
from federated_learning_experiments.learning.training.candidate_epoch_training import (
    CandidateEpochTrainingResult,
)
from federated_learning_experiments.learning.training.candidate_epoch_training_settings import (
    CandidateEpochTrainingSettings,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import (
    CandidateModelTrainingAndAcceptanceSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import (
    PostAlarmCandidateLossCollection,
)

from .alarm_interval_candidate_training import train_candidate_on_alarm_change_interval
from .candidate_classifier_construction import (
    IndependentCandidateTrainingState,
)
from .post_alarm_reference_model_fixation import (
    FixedPostAlarmReferenceModels,
    fix_reference_models_at_alarm,
)


@dataclass(frozen=True, kw_only=True)
class PostAlarmCandidateValidationSession:
    """開始時のbindingを固定し、候補学習・損失収集の可変部品を保持する。"""

    proposal_sample_index: int
    estimated_change_point_sample_index: int | None
    detection_episode_id: int | None
    detector_name: str
    initial_training_model_id: int
    candidate_training_state: IndependentCandidateTrainingState
    candidate_epoch_training_result: CandidateEpochTrainingResult
    training_input_features: Tensor
    training_observed_class_labels: Tensor
    pending_assignment_training_samples: tuple[ObservedTrainingSample, ...]
    fixed_reference_models: FixedPostAlarmReferenceModels
    post_alarm_candidate_loss_collection: PostAlarmCandidateLossCollection


def start_post_alarm_candidate_validation_session(
    *,
    architecture_reference_classifier: ModelArchitectureClassifier,
    initial_candidate_parameter_snapshot: dict[str, Tensor],
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
    candidate_epoch_training_settings: CandidateEpochTrainingSettings,
    candidate_model_training_and_acceptance_settings: CandidateModelTrainingAndAcceptanceSettings,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    current_training_model_assignment: CurrentTrainingModelAssignment,
    input_features: Tensor,
    observed_class_labels: Tensor,
    pending_assignment_training_samples: tuple[ObservedTrainingSample, ...],
    proposal_sample_index: int,
    estimated_change_point_sample_index: int | None,
    detection_episode_id: int | None,
    detector_name: str,
) -> PostAlarmCandidateValidationSession:
    """借用ownerは更新せず、独立候補と固定参照を含む新sessionを返す。"""
    trained_candidate = train_candidate_on_alarm_change_interval(
        architecture_reference_classifier=architecture_reference_classifier,
        initial_candidate_parameter_snapshot=initial_candidate_parameter_snapshot,
        parameter_optimizer_settings=parameter_optimizer_settings,
        candidate_epoch_training_settings=candidate_epoch_training_settings,
        candidate_model_training_and_acceptance_settings=candidate_model_training_and_acceptance_settings,
        held_model_training_state_registry=held_model_training_state_registry,
        loss_statistics_store=loss_statistics_store,
        current_training_model_assignment=current_training_model_assignment,
        input_features=input_features,
        observed_class_labels=observed_class_labels,
        pending_assignment_training_samples=pending_assignment_training_samples,
        proposal_sample_index=proposal_sample_index,
        estimated_change_point_sample_index=estimated_change_point_sample_index,
        detection_episode_id=detection_episode_id,
        detector_name=detector_name,
    )
    initial_training_model_id = current_training_model_assignment.current_training_model_id
    candidate_training_state = trained_candidate.candidate_training_state
    candidate_epoch_training_result = trained_candidate.candidate_epoch_training_result
    fixed_reference_models = fix_reference_models_at_alarm(
        held_model_training_state_registry=held_model_training_state_registry,
        loss_statistics_store=loss_statistics_store,
    )
    post_alarm_candidate_loss_collection = PostAlarmCandidateLossCollection(
        candidate_model_training_and_acceptance_settings=candidate_model_training_and_acceptance_settings,
        proposal_sample_index=proposal_sample_index,
        reference_model_ids=tuple(fixed_reference_models.reference_classifiers_by_model_id),
    )
    return PostAlarmCandidateValidationSession(
        proposal_sample_index=proposal_sample_index,
        estimated_change_point_sample_index=estimated_change_point_sample_index,
        detection_episode_id=detection_episode_id,
        detector_name=detector_name,
        initial_training_model_id=initial_training_model_id,
        candidate_training_state=candidate_training_state,
        candidate_epoch_training_result=candidate_epoch_training_result,
        training_input_features=input_features,
        training_observed_class_labels=observed_class_labels,
        pending_assignment_training_samples=pending_assignment_training_samples,
        fixed_reference_models=fixed_reference_models,
        post_alarm_candidate_loss_collection=post_alarm_candidate_loss_collection,
    )
