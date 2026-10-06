"""警報後の検証標本1件について、候補と参照モデルの損失を評価して損失収集へ追加する。"""

from torch import Tensor

from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import (
    PostAlarmCandidateLossCollection,
)


def _evaluate_single_sample_bounded_loss(
    *,
    classifier: ResidualAdapterClassifier,
    input_features: Tensor,
    observed_class_labels: Tensor,
) -> float:
    per_sample_bounded_losses = evaluate_classifier_per_sample_bounded_losses(
        classifier=classifier,
        input_features=input_features,
        observed_class_labels=observed_class_labels,
    )
    return per_sample_bounded_losses[0].item()


def observe_post_alarm_candidate_validation_sample(
    *,
    sample_index: int,
    input_features: Tensor,
    observed_class_labels: Tensor,
    candidate_classifier: ResidualAdapterClassifier,
    reference_classifiers_by_model_id: dict[int, ResidualAdapterClassifier],
    post_alarm_candidate_loss_collection: PostAlarmCandidateLossCollection,
) -> bool:
    """全分類器の損失評価を終えてから収集へ1回だけ追加し、規定件数への到達を返す。"""
    if type(post_alarm_candidate_loss_collection) is not PostAlarmCandidateLossCollection:
        raise TypeError(
            "post_alarm_candidate_loss_collectionはexact PostAlarmCandidateLossCollectionが必要です。"
        )
    if type(reference_classifiers_by_model_id) is not dict:
        raise TypeError("reference_classifiers_by_model_idはexact dictが必要です。")
    # 1標本契約は、どの分類器のforwardよりも前に確認する。
    if isinstance(input_features, Tensor) and (
        input_features.ndim != 2 or input_features.shape[0] != 1
    ):
        raise ValueError("input_featuresは1標本ぶんのshape[1,F]が必要です。")
    candidate_loss = _evaluate_single_sample_bounded_loss(
        classifier=candidate_classifier,
        input_features=input_features,
        observed_class_labels=observed_class_labels,
    )
    reference_losses_by_model_id = {
        model_id: _evaluate_single_sample_bounded_loss(
            classifier=reference_classifier,
            input_features=input_features,
            observed_class_labels=observed_class_labels,
        )
        for model_id, reference_classifier in reference_classifiers_by_model_id.items()
    }
    post_alarm_candidate_loss_collection.observe_losses_after_label_observation(
        sample_index=sample_index,
        candidate_loss=candidate_loss,
        reference_losses_by_model_id=reference_losses_by_model_id,
    )
    return post_alarm_candidate_loss_collection.ready_for_acceptance_evaluation
