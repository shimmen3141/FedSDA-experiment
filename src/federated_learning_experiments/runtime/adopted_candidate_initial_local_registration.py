"""採用された候補を、一時IDの保有モデルとして独立した状態ownerへ登録する。"""

from torch import Tensor

from federated_learning_experiments.learning.loss_statistics.batch_loss_statistics_initialization import (
    initialize_model_and_class_loss_statistics_from_batch,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.models.shared_feature_extractor import (
    SharedFeatureExtractor,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
)
from federated_learning_experiments.learning.training.adopted_candidate_shared_feature_integration import (
    integrate_adopted_candidate_shared_features,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingState,
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import (
    PendingModelUploadState,
)


def _validate_registration_inputs(
    *,
    temporary_model_id: int,
    upload_delay_round_count: int,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    current_training_model_assignment: CurrentTrainingModelAssignment,
    pending_model_upload_state: PendingModelUploadState,
) -> None:
    if type(temporary_model_id) is not int:
        raise TypeError("temporary_model_idはbool・派生型以外のbuiltin intが必要です。")
    if temporary_model_id >= 0:
        raise ValueError("temporary_model_idは負が必要です。")
    if type(upload_delay_round_count) is not int:
        raise TypeError("upload_delay_round_countはbool・派生型以外のbuiltin intが必要です。")
    if upload_delay_round_count < 1:
        raise ValueError("upload_delay_round_countは1以上が必要です。")
    for state_owner, expected_owner_type, owner_name in (
        (
            held_model_training_state_registry,
            HeldModelTrainingStateRegistry,
            "held_model_training_state_registry",
        ),
        (loss_statistics_store, ModelAndClassLossStatisticsStore, "loss_statistics_store"),
        (
            current_training_model_assignment,
            CurrentTrainingModelAssignment,
            "current_training_model_assignment",
        ),
        (pending_model_upload_state, PendingModelUploadState, "pending_model_upload_state"),
    ):
        if type(state_owner) is not expected_owner_type:
            raise TypeError(f"{owner_name}はexact {expected_owner_type.__name__}が必要です。")


def _reject_temporary_model_id_already_in_use(
    *,
    temporary_model_id: int,
    held_model_training_states: tuple[HeldModelTrainingState, ...],
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    pending_model_upload_state: PendingModelUploadState,
) -> None:
    if any(
        held_model_training_state.model_id == temporary_model_id
        for held_model_training_state in held_model_training_states
    ):
        raise ValueError("temporary_model_idは学習状態一覧に保有済みです。")
    if loss_statistics_store.get_model_loss_statistics(model_id=temporary_model_id) is not None:
        raise ValueError("temporary_model_idは損失統計に登録済みです。")
    pending_model_upload = pending_model_upload_state.get_pending_model_upload()
    if pending_model_upload is not None and pending_model_upload.model_id == temporary_model_id:
        raise ValueError("temporary_model_idは既存の送信保留の対応IDです。")


def _select_active_shared_feature_extractor(
    *,
    held_model_training_states: tuple[HeldModelTrainingState, ...],
    current_training_model_id: int,
) -> SharedFeatureExtractor:
    """現在の学習帰属IDのモデルを優先し、保有されていなければ一覧先頭を使う。"""
    if not held_model_training_states:
        raise LookupError("保有モデルがないため共有特徴抽出部の反映先を選べません。")
    for held_model_training_state in held_model_training_states:
        if held_model_training_state.model_id == current_training_model_id:
            return held_model_training_state.classifier.feature_extractor
    return held_model_training_states[0].classifier.feature_extractor


def register_adopted_candidate_as_temporary_held_model(
    *,
    temporary_model_id: int,
    adopted_candidate_classifier: ResidualAdapterClassifier,
    candidate_concept_specific_parameter_optimizer_state: ParameterOptimizerState,
    initial_statistics_input_features: Tensor,
    initial_statistics_observed_class_labels: Tensor,
    upload_delay_round_count: int,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    current_training_model_assignment: CurrentTrainingModelAssignment,
    pending_model_upload_state: PendingModelUploadState,
) -> None:
    """全検証と数値生成の後に、共有反映→一覧→統計→送信保留を順次更新する。"""
    _validate_registration_inputs(
        temporary_model_id=temporary_model_id,
        upload_delay_round_count=upload_delay_round_count,
        held_model_training_state_registry=held_model_training_state_registry,
        loss_statistics_store=loss_statistics_store,
        current_training_model_assignment=current_training_model_assignment,
        pending_model_upload_state=pending_model_upload_state,
    )
    held_model_training_states = (
        held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    )
    _reject_temporary_model_id_already_in_use(
        temporary_model_id=temporary_model_id,
        held_model_training_states=held_model_training_states,
        loss_statistics_store=loss_statistics_store,
        pending_model_upload_state=pending_model_upload_state,
    )
    active_shared_feature_extractor = _select_active_shared_feature_extractor(
        held_model_training_states=held_model_training_states,
        current_training_model_id=current_training_model_assignment.current_training_model_id,
    )
    # 共有反映は候補共有部の値をそのまま複写するため、反映前の候補で同じ値を得る。
    per_sample_bounded_losses = evaluate_classifier_per_sample_bounded_losses(
        classifier=adopted_candidate_classifier,
        input_features=initial_statistics_input_features,
        observed_class_labels=initial_statistics_observed_class_labels,
    )
    initial_loss_statistics = initialize_model_and_class_loss_statistics_from_batch(
        per_sample_bounded_losses=per_sample_bounded_losses,
        observed_class_labels=initial_statistics_observed_class_labels,
        class_count=adopted_candidate_classifier.class_count,
    )
    parameter_snapshot = snapshot_classifier_parameters(classifier=adopted_candidate_classifier)
    integrate_adopted_candidate_shared_features(
        adopted_candidate_classifier=adopted_candidate_classifier,
        candidate_concept_specific_parameter_optimizer_state=candidate_concept_specific_parameter_optimizer_state,
        active_shared_feature_extractor=active_shared_feature_extractor,
    )
    held_model_training_state_registry.register_held_model_training_state(
        model_id=temporary_model_id,
        classifier=adopted_candidate_classifier,
        concept_specific_parameter_optimizer_state=candidate_concept_specific_parameter_optimizer_state,
    )
    loss_statistics_store.set_model_loss_statistics(
        model_id=temporary_model_id, loss_statistics=initial_loss_statistics
    )
    pending_model_upload_state.queue_model_upload(
        model_id=temporary_model_id,
        parameter_snapshot=parameter_snapshot,
        upload_delay_round_count=upload_delay_round_count,
    )
