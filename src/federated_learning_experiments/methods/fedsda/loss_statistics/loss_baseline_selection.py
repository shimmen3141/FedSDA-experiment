"""保存された一系列の損失集計から用途別の基準平均を選択する。"""

from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)


def _validate_optional_loss_moments(
    *,
    loss_moments: BoundedLossMoments | None,
) -> BoundedLossMoments | None:
    if loss_moments is None:
        return None
    if type(loss_moments) is not BoundedLossMoments:
        raise TypeError("loss_momentsはBoundedLossMomentsのexact型またはNoneが必要です。")
    return BoundedLossMoments(
        observed_loss_count=loss_moments.observed_loss_count,
        mean_loss=loss_moments.mean_loss,
        sum_squared_loss_deviations=loss_moments.sum_squared_loss_deviations,
    )


def select_loss_monitoring_baseline_mean_loss(
    *,
    loss_moments: BoundedLossMoments | None,
) -> float:
    """不足時は0.01、1件以上は保存平均を監視用の範囲に制限する。"""
    validated_loss_moments = _validate_optional_loss_moments(loss_moments=loss_moments)
    if validated_loss_moments is None or validated_loss_moments.observed_loss_count < 1:
        return 0.01
    return min(1.0 - 1e-6, max(0.01, validated_loss_moments.mean_loss))


def select_alarm_interval_reuse_baseline_mean_loss(
    *,
    loss_moments: BoundedLossMoments | None,
) -> float | None:
    """警報区間の再利用比較では2件以上の非零平均だけを使用する。"""
    validated_loss_moments = _validate_optional_loss_moments(loss_moments=loss_moments)
    if (
        validated_loss_moments is None
        or validated_loss_moments.observed_loss_count < 2
        or validated_loss_moments.mean_loss == 0.0
    ):
        return None
    return validated_loss_moments.mean_loss


def select_post_alarm_reference_historical_mean_loss(
    *,
    loss_moments: BoundedLossMoments | None,
) -> float | None:
    """警報後の参照比較には2件以上の保存平均を零も含めて使用する。"""
    validated_loss_moments = _validate_optional_loss_moments(loss_moments=loss_moments)
    if validated_loss_moments is None or validated_loss_moments.observed_loss_count < 2:
        return None
    return validated_loss_moments.mean_loss
