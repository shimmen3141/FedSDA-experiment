"""保有モデルを警報区間で評価し、履歴基準と再利用適合の判定を組み立てる。"""

from math import isfinite

from torch import Tensor, mean

from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.alarm_interval_model_reuse_assessment import (
    AlarmIntervalModelReuseAssessment,
    assess_alarm_interval_model_reuse,
)
from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import (
    select_alarm_interval_reuse_baseline_mean_loss,
)


def _validate_alarm_interval_reuse_evaluation_inputs(
    *,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    maximum_alarm_interval_mean_loss_increase: float,
) -> None:
    """分類器の評価より前に、状態所有者・許容増加量・保有モデルの有無を検査する。"""
    if type(held_model_training_state_registry) is not HeldModelTrainingStateRegistry:
        raise TypeError(
            "held_model_training_state_registryはexact HeldModelTrainingStateRegistryが必要です。"
        )
    if type(loss_statistics_store) is not ModelAndClassLossStatisticsStore:
        raise TypeError("loss_statistics_storeはexact ModelAndClassLossStatisticsStoreが必要です。")
    if type(maximum_alarm_interval_mean_loss_increase) not in (int, float):
        raise TypeError(
            "maximum_alarm_interval_mean_loss_increaseは"
            "bool以外のbuiltin intまたはfloatが必要です。"
        )
    try:
        specified_value = float(maximum_alarm_interval_mean_loss_increase)
    except OverflowError as overflow_error:
        raise ValueError(
            "maximum_alarm_interval_mean_loss_increaseは有限のfloatへ表現できる値が必要です。"
        ) from overflow_error
    if not isfinite(specified_value) or specified_value < 0:
        raise ValueError("maximum_alarm_interval_mean_loss_increaseは有限の非負値が必要です。")
    if not held_model_training_state_registry.snapshot_ordered_held_model_training_states():
        raise ValueError("保有モデルがないため警報区間の再利用を評価できません。")


def evaluate_held_models_for_alarm_interval_reuse(
    *,
    input_features: Tensor,
    observed_class_labels: Tensor,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    loss_statistics_store: ModelAndClassLossStatisticsStore,
    maximum_alarm_interval_mean_loss_increase: float,
) -> AlarmIntervalModelReuseAssessment:
    """保有順に全モデルを区間評価し、履歴基準を使えるモデルだけを適合判定へ渡す。"""
    _validate_alarm_interval_reuse_evaluation_inputs(
        held_model_training_state_registry=held_model_training_state_registry,
        loss_statistics_store=loss_statistics_store,
        maximum_alarm_interval_mean_loss_increase=maximum_alarm_interval_mean_loss_increase,
    )
    baseline_supported_interval_mean_losses_by_model_id: list[tuple[int, float]] = []
    reuse_baseline_mean_losses_by_model_id: dict[int, float] = {}
    held_model_training_states = (
        held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    )
    for held_model_training_state in held_model_training_states:
        # 履歴基準を使えないモデルも区間全体を評価する。特徴/ラベルの検査は損失評価が行う。
        per_sample_bounded_losses = evaluate_classifier_per_sample_bounded_losses(
            classifier=held_model_training_state.classifier,
            input_features=input_features,
            observed_class_labels=observed_class_labels,
        )
        # float32の標本損失をtorchの平均で求める。Pythonの和では再計算しない。
        interval_mean_loss = mean(per_sample_bounded_losses).item()
        model_loss_statistics = loss_statistics_store.get_model_loss_statistics(
            model_id=held_model_training_state.model_id
        )
        reuse_baseline_mean_loss = select_alarm_interval_reuse_baseline_mean_loss(
            loss_moments=None
            if model_loss_statistics is None
            else model_loss_statistics.overall_loss_moments
        )
        if reuse_baseline_mean_loss is None:
            continue
        baseline_supported_interval_mean_losses_by_model_id.append(
            (held_model_training_state.model_id, interval_mean_loss)
        )
        reuse_baseline_mean_losses_by_model_id[held_model_training_state.model_id] = (
            reuse_baseline_mean_loss
        )
    return assess_alarm_interval_model_reuse(
        baseline_supported_interval_mean_losses_by_model_id=tuple(
            baseline_supported_interval_mean_losses_by_model_id
        ),
        reuse_baseline_mean_losses_by_model_id=reuse_baseline_mean_losses_by_model_id,
        maximum_alarm_interval_mean_loss_increase=maximum_alarm_interval_mean_loss_increase,
    )
