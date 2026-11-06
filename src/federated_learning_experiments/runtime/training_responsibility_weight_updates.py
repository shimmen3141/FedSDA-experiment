"""実際の単体予測とは独立に、学習責任度の事前重みだけを更新する。"""

from torch import no_grad

from federated_learning_experiments.learning.prediction.class_probability_calculations import (
    compute_model_mean_bounded_losses_after_label_observation,
    convert_model_outputs_to_prediction_probabilities,
    normalize_model_prediction_weights,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
    IndexedObservedTrainingSample,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights import (
    FixedShareWeightController,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.synchronized_prediction_model_snapshot import (
    SynchronizedPredictionModelSnapshot,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_sample_observation_store import (
    PendingSampleObservationStore,
)
from federated_learning_experiments.runtime.post_aggregation_prediction_recalibration import (
    PostAggregationPredictionRecalibration,
    _compute_pending_sample_loss_sequence,
)


def _validate_owners(**owners: object) -> None:
    expected = {
        "held_model_training_state_registry": HeldModelTrainingStateRegistry,
        "fixed_share_weight_controller": FixedShareWeightController,
        "synchronized_prediction_model_snapshot": SynchronizedPredictionModelSnapshot,
        "pending_sample_observation_store": PendingSampleObservationStore,
        "indexed_observation": IndexedObservedTrainingSample,
    }
    for name, owner in owners.items():
        if type(owner) is not expected[name]:
            raise TypeError(f"{name}には{expected[name].__name__}の実体を指定してください。")


def update_training_responsibility_weights_after_label_observation(
    *,
    indexed_observation: IndexedObservedTrainingSample,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    fixed_share_weight_controller: FixedShareWeightController,
    synchronized_prediction_model_snapshot: SynchronizedPredictionModelSnapshot,
) -> dict[int, float]:
    """予測確定・ラベル開示後に同期損失で更新し、補助損失列を返す。予測記録は作らない。"""
    _validate_owners(
        indexed_observation=indexed_observation,
        held_model_training_state_registry=held_model_training_state_registry,
        fixed_share_weight_controller=fixed_share_weight_controller,
        synchronized_prediction_model_snapshot=synchronized_prediction_model_snapshot,
    )
    shared_classifier, classifiers = (
        synchronized_prediction_model_snapshot.select_prediction_classifiers(
            held_model_training_state_registry=held_model_training_state_registry
        )
    )
    class_count = shared_classifier.class_count
    # run全体のhook計測内で実行する。共有部1回・概念固有部全モデル分は
    # local_processingの推論件数/MACへ入り、学習のforward/backwardには含めない。
    with no_grad():
        features = shared_classifier.extract_shared_features(
            indexed_observation.training_sample.input_features
        )
        probabilities = convert_model_outputs_to_prediction_probabilities(
            model_outputs_by_model_id={
                model_id: classifiers[model_id].forward_from_shared_features(features)
                for model_id in sorted(classifiers)
            },
            class_count=class_count,
        )
    losses = compute_model_mean_bounded_losses_after_label_observation(
        prediction_probabilities_by_model_id=probabilities,
        observed_class_labels=indexed_observation.training_sample.observed_class_labels,
        class_count=class_count,
        prediction_probabilities_are_validated=True,
    )
    weights = normalize_model_prediction_weights(
        prediction_weights_by_model_id=fixed_share_weight_controller.get_weights_before_label_observation(
            model_ids=sorted(classifiers)
        )
    )
    fixed_share_weight_controller.update_weights_after_loss_observation(
        observed_losses_by_model_id=losses, prediction_weights_by_model_id=weights
    )
    return losses


def recalibrate_training_responsibility_weights_after_aggregation(
    *,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    pending_sample_observation_store: PendingSampleObservationStore,
    fixed_share_weight_controller: FixedShareWeightController,
) -> PostAggregationPredictionRecalibration:
    """配布後の保留標本の局所損失を再生する。AdaHedgeや固定済み責任度は変更しない。"""
    _validate_owners(
        held_model_training_state_registry=held_model_training_state_registry,
        pending_sample_observation_store=pending_sample_observation_store,
        fixed_share_weight_controller=fixed_share_weight_controller,
    )
    losses = _compute_pending_sample_loss_sequence(
        held_model_training_state_registry=held_model_training_state_registry,
        pending_sample_observation_store=pending_sample_observation_store,
    )
    result = PostAggregationPredictionRecalibration(
        replayed_sample_count=len(losses),
        replayed_model_ids=tuple(losses[0]) if losses else (),
        restarted_true_concept_ids=(),
    )
    fixed_share_weight_controller.replay_observed_losses_after_aggregation(
        observed_loss_sequence=losses
    )
    return result
