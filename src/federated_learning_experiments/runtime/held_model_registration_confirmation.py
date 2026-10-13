"""保有済みモデルの正式ID通知を、独立した状態ownerへ適用する。"""

from federated_learning_experiments.evaluation.model_evaluation_sample_store import (
    ModelEvaluationSampleStore,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
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
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_registration_settings import (
    KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import (
    PendingModelUploadState,
)


def _validate_confirmation_inputs(
    *,
    registered_global_model_id: int,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    training_sample_store: ModelTrainingSampleStore,
    evaluation_sample_store: ModelEvaluationSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    current_training_model_assignment: CurrentTrainingModelAssignment,
    pending_model_upload_state: PendingModelUploadState,
) -> None:
    if type(registered_global_model_id) is not int:
        raise TypeError("registered_global_model_idはbool・派生型以外のbuiltin intが必要です。")
    if registered_global_model_id < 0:
        raise ValueError("registered_global_model_idは非負が必要です。")
    for state_owner, expected_owner_type, owner_name in (
        (
            held_model_training_state_registry,
            HeldModelTrainingStateRegistry,
            "held_model_training_state_registry",
        ),
        (loss_statistics_store, ModelAndClassLossStatisticsStore, "loss_statistics_store"),
        (training_sample_store, ModelTrainingSampleStore, "training_sample_store"),
        (evaluation_sample_store, ModelEvaluationSampleStore, "evaluation_sample_store"),
        (
            model_training_and_assignment_counts_store,
            ModelTrainingAndAssignmentCountsStore,
            "model_training_and_assignment_counts_store",
        ),
        (
            current_training_model_assignment,
            CurrentTrainingModelAssignment,
            "current_training_model_assignment",
        ),
        (pending_model_upload_state, PendingModelUploadState, "pending_model_upload_state"),
    ):
        if type(state_owner) is not expected_owner_type:
            raise TypeError(f"{owner_name}はexact {expected_owner_type.__name__}が必要です。")


def confirm_held_model_registration(
    *,
    registered_global_model_id: int,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    training_sample_store: ModelTrainingSampleStore,
    evaluation_sample_store: ModelEvaluationSampleStore,
    model_training_and_assignment_counts_store: ModelTrainingAndAssignmentCountsStore,
    current_training_model_assignment: CurrentTrainingModelAssignment,
    pending_model_upload_state: PendingModelUploadState,
) -> TrainingModelAssignmentChange | None:
    """入力/保有を先に確認し、付替え→加算→現在ID→保留解除を順次行う。

    付け替えるモデルは、送信保留のownerの方針で決まる。現行を付け替える方針では、現在の学習帰属の
    モデル（非負なら、送信保留を外すだけ）。送信保留を全部保持する2つの方針では、ownerが、現行のIDから
    選ぶ、次に登録する送信保留のモデル（それ自身のIDの方針では、待ちの済んだ最も古いもの。現行モデルとして
    待った後に登録する方針——FedSDAの既定——では、待ちの済んだ現行の送信保留）で、現行がそのモデルのとき
    だけ、現在の学習帰属も正式IDへ変える。
    """
    _validate_confirmation_inputs(
        registered_global_model_id=registered_global_model_id,
        held_model_training_state_registry=held_model_training_state_registry,
        loss_statistics_store=loss_statistics_store,
        training_sample_store=training_sample_store,
        evaluation_sample_store=evaluation_sample_store,
        model_training_and_assignment_counts_store=model_training_and_assignment_counts_store,
        current_training_model_assignment=current_training_model_assignment,
        pending_model_upload_state=pending_model_upload_state,
    )
    keeps_every_pending_model = (
        pending_model_upload_state.pending_model_registration_policy
        != KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME
    )
    if keeps_every_pending_model:
        # 現行が非負でも、送信保留を捨てない（現行を付け替える方針の、下の短絡を通らない）。
        ready_pending_model_upload = pending_model_upload_state.get_next_pending_model_upload_to_register(
            current_training_model_id=current_training_model_assignment.current_training_model_id
        )
        if ready_pending_model_upload is None:
            raise ValueError("この同期で登録する送信保留がありません。")
        original_model_id = ready_pending_model_upload.model_id
        if original_model_id >= 0:
            raise ValueError("送信保留のモデルIDは、一時ID（負）が必要です。")
    else:
        original_model_id = current_training_model_assignment.current_training_model_id
        if original_model_id >= 0:
            pending_model_upload_state.clear_pending_model_upload()
            return None
    held_model_training_state_registry.get_held_model_training_state(model_id=original_model_id)
    held_model_training_state_registry.reassign_held_model_training_state_id(
        original_model_id=original_model_id, reassigned_model_id=registered_global_model_id
    )
    loss_statistics_store.reassign_model_loss_statistics_id(
        original_model_id=original_model_id, reassigned_model_id=registered_global_model_id
    )
    training_sample_store.reassign_model_training_samples_id(
        original_model_id=original_model_id, reassigned_model_id=registered_global_model_id
    )
    evaluation_sample_store.reassign_model_evaluation_samples_id(
        original_model_id=original_model_id, reassigned_model_id=registered_global_model_id
    )
    model_training_and_assignment_counts_store.transfer_model_training_and_assignment_counts(
        original_model_id=original_model_id, receiving_model_id=registered_global_model_id
    )
    if keeps_every_pending_model:
        assignment_change = None
        if current_training_model_assignment.current_training_model_id == original_model_id:
            assignment_change = current_training_model_assignment.assign_model_for_training(
                model_id=registered_global_model_id
            )
        pending_model_upload_state.remove_pending_model_upload(model_id=original_model_id)
        return assignment_change
    assignment_change = current_training_model_assignment.assign_model_for_training(
        model_id=registered_global_model_id
    )
    pending_model_upload_state.clear_pending_model_upload()
    return assignment_change
