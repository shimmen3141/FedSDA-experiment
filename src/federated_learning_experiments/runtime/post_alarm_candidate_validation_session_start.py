"""警報後の候補生成・学習・参照固定・損失収集開始を順に組み立てる。"""

from dataclasses import dataclass

from torch import Tensor, float32, is_grad_enabled, isfinite, strided

from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training.candidate_epoch_training import (
    CandidateEpochTrainingResult,
    train_candidate_classifier_epochs,
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

from .candidate_classifier_construction import (
    IndependentCandidateTrainingState,
    create_independent_candidate_training_state,
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


def _validate_session_training_tensor_pair(
    *,
    input_features: Tensor,
    observed_class_labels: Tensor,
    input_feature_count: int,
    class_count: int,
    required_sample_count: int | None,
) -> None:
    for training_tensor in (input_features, observed_class_labels):
        if type(training_tensor) is not Tensor:
            raise TypeError("学習区間と保留標本はexact Tensorが必要です。")
        if (
            training_tensor.device.type != "cpu"
            or training_tensor.dtype != float32
            or training_tensor.layout != strided
            or training_tensor.is_nested
        ):
            raise ValueError("学習区間と保留標本はCPU float32 stridedが必要です。")
        if training_tensor.ndim != 2 or len(training_tensor) < 1:
            raise ValueError("学習区間と保留標本は非空の2次元Tensorが必要です。")
        if not isfinite(training_tensor).all().item():
            raise ValueError("学習区間と保留標本は有限値が必要です。")
    sample_count = len(input_features)
    if (
        input_features.shape[1] != input_feature_count
        or observed_class_labels.shape != (sample_count, 1)
        or (required_sample_count is not None and sample_count != required_sample_count)
    ):
        raise ValueError("特徴とラベルのshapeまたは要求標本件数が一致しません。")
    if (
        not (
            (observed_class_labels == observed_class_labels.round())
            & (observed_class_labels >= 0)
            & (observed_class_labels < class_count)
        )
        .all()
        .item()
    ):
        raise ValueError("観測ラベルは分類器の整数クラス範囲が必要です。")


def _validate_post_alarm_candidate_validation_start_inputs(
    *,
    architecture_reference_classifier: ResidualAdapterClassifier,
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
) -> None:
    if type(proposal_sample_index) is not int:
        raise TypeError("proposal_sample_indexはbool以外のbuiltin intが必要です。")
    if proposal_sample_index < 0:
        raise ValueError("proposal_sample_indexは0以上が必要です。")
    if estimated_change_point_sample_index is not None:
        if type(estimated_change_point_sample_index) is not int:
            raise TypeError(
                "estimated_change_point_sample_indexはbuiltin intまたはNoneが必要です。"
            )
        if not 0 <= estimated_change_point_sample_index <= proposal_sample_index:
            raise ValueError("推定変化点は0以上かつ提案位置以下が必要です。")
    if detection_episode_id is not None:
        if type(detection_episode_id) is not int:
            raise TypeError("detection_episode_idはbuiltin intまたはNoneが必要です。")
        if detection_episode_id < 0:
            raise ValueError("detection_episode_idは0以上が必要です。")
    if type(detector_name) is not str:
        raise TypeError("detector_nameはexact strが必要です。")
    if not detector_name.strip():
        raise ValueError("detector_nameは空白以外の文字が必要です。")
    if type(candidate_epoch_training_settings) is not CandidateEpochTrainingSettings:
        raise TypeError("candidate_epoch_training_settingsはexact設定型が必要です。")
    CandidateEpochTrainingSettings(
        candidate_training_strategy=candidate_epoch_training_settings.candidate_training_strategy,
        maximum_epoch_count=candidate_epoch_training_settings.maximum_epoch_count,
        maximum_batch_sample_count=candidate_epoch_training_settings.maximum_batch_sample_count,
        validation_sample_fraction=candidate_epoch_training_settings.validation_sample_fraction,
        consecutive_non_improving_epoch_limit=candidate_epoch_training_settings.consecutive_non_improving_epoch_limit,
        minimum_validation_loss_decrease=candidate_epoch_training_settings.minimum_validation_loss_decrease,
    )
    if (
        type(candidate_model_training_and_acceptance_settings)
        is not CandidateModelTrainingAndAcceptanceSettings
    ):
        raise TypeError("candidate_model_training_and_acceptance_settingsはexact設定型が必要です。")
    if (
        type(
            candidate_model_training_and_acceptance_settings.candidate_post_alarm_validation_sample_count
        )
        is not int
    ):
        raise TypeError("候補の検証標本件数はbool以外のbuiltin intが必要です。")
    CandidateModelTrainingAndAcceptanceSettings(
        candidate_model_acceptance_policy=candidate_model_training_and_acceptance_settings.candidate_model_acceptance_policy,
        candidate_post_alarm_validation_sample_count=candidate_model_training_and_acceptance_settings.candidate_post_alarm_validation_sample_count,
    )
    if type(held_model_training_state_registry) is not HeldModelTrainingStateRegistry:
        raise TypeError("held_model_training_state_registryはexact登録owner型が必要です。")
    if type(loss_statistics_store) is not ModelAndClassLossStatisticsStore:
        raise TypeError("loss_statistics_storeはexact統計owner型が必要です。")
    if type(current_training_model_assignment) is not CurrentTrainingModelAssignment:
        raise TypeError("current_training_model_assignmentはexact帰属owner型が必要です。")
    held_model_training_states = (
        held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    )
    if not held_model_training_states:
        raise LookupError("候補検証の開始には保有モデルが必要です。")
    if current_training_model_assignment.current_training_model_id not in tuple(
        held_model_training_state.model_id
        for held_model_training_state in held_model_training_states
    ):
        raise LookupError("現在の学習帰属IDは保有モデルに含まれる必要があります。")
    snapshot_classifier_parameters(classifier=architecture_reference_classifier)
    input_feature_count = architecture_reference_classifier.feature_extractor.input_feature_count
    class_count = architecture_reference_classifier.class_count
    for held_model_training_state in held_model_training_states:
        snapshot_classifier_parameters(classifier=held_model_training_state.classifier)
        if (
            held_model_training_state.classifier.feature_extractor.input_feature_count
            != input_feature_count
            or held_model_training_state.classifier.class_count != class_count
        ):
            raise ValueError("保有分類器と構造参照の入力幅・クラス数が一致しません。")
        # public取得は全fieldをconstructorへ渡して独立copyし、統計を再検査する。
        loss_statistics_store.get_model_loss_statistics(model_id=held_model_training_state.model_id)
    _validate_session_training_tensor_pair(
        input_features=input_features,
        observed_class_labels=observed_class_labels,
        input_feature_count=input_feature_count,
        class_count=class_count,
        required_sample_count=None,
    )
    if type(pending_assignment_training_samples) is not tuple:
        raise TypeError("pending_assignment_training_samplesはexact tupleが必要です。")
    for training_sample in pending_assignment_training_samples:
        if type(training_sample) is not ObservedTrainingSample:
            raise TypeError("保留標本はexact ObservedTrainingSampleが必要です。")
        _validate_session_training_tensor_pair(
            input_features=training_sample.input_features,
            observed_class_labels=training_sample.observed_class_labels,
            input_feature_count=input_feature_count,
            class_count=class_count,
            required_sample_count=1,
        )
    if (
        candidate_epoch_training_settings.candidate_training_strategy != "skip_training"
        and candidate_epoch_training_settings.maximum_epoch_count > 0
        and not is_grad_enabled()
    ):
        raise ValueError("候補の学習更新にはgrad有効が必要です。")


def start_post_alarm_candidate_validation_session(
    *,
    architecture_reference_classifier: ResidualAdapterClassifier,
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
    _validate_post_alarm_candidate_validation_start_inputs(
        architecture_reference_classifier=architecture_reference_classifier,
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
    # 初期値・optimizer設定の検査は、この既存public生成API内で生成前に行う。
    candidate_training_state = create_independent_candidate_training_state(
        architecture_reference_classifier=architecture_reference_classifier,
        initial_candidate_parameter_snapshot=initial_candidate_parameter_snapshot,
        parameter_optimizer_settings=parameter_optimizer_settings,
    )
    candidate_epoch_training_result = train_candidate_classifier_epochs(
        candidate_classifier=candidate_training_state.candidate_classifier,
        candidate_shared_parameter_optimizer_state=candidate_training_state.candidate_shared_parameter_optimizer_state,
        candidate_concept_specific_parameter_optimizer_state=candidate_training_state.candidate_concept_specific_parameter_optimizer_state,
        input_features=input_features,
        observed_class_labels=observed_class_labels,
        candidate_epoch_training_settings=candidate_epoch_training_settings,
    )
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
