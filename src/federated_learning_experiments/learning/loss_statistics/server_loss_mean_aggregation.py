"""参加済みクライアントの全体損失平均を旧順で集約する。"""

from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)


def _copy_validated_participating_client_loss_moments(
    *, participating_client_loss_moments: tuple[BoundedLossMoments, ...],
) -> tuple[BoundedLossMoments, ...]:
    if type(participating_client_loss_moments) is not tuple:
        raise TypeError("participating_client_loss_momentsはexact tupleが必要です。")
    copied_participating_client_loss_moments = []
    for loss_moments in participating_client_loss_moments:
        if type(loss_moments) is not BoundedLossMoments:
            raise TypeError(
                "participating_client_loss_momentsの要素はexact BoundedLossMomentsが必要です。")
        try:
            copied_loss_moments = BoundedLossMoments(
                observed_loss_count=loss_moments.observed_loss_count,
                mean_loss=loss_moments.mean_loss,
                sum_squared_loss_deviations=loss_moments.sum_squared_loss_deviations,
            )
        except (TypeError, ValueError) as validation_error:
            raise type(validation_error)(
                f"participating_client_loss_momentsの要素: {validation_error}") from validation_error
        copied_participating_client_loss_moments.append(copied_loss_moments)
    return tuple(copied_participating_client_loss_moments)


def aggregate_participating_client_loss_means(
    *, participating_client_loss_moments: tuple[BoundedLossMoments, ...],
) -> BoundedLossMoments | None:
    """件数で加重し、更新不要ならNone、更新時はM2を0にして返す。"""
    validated_participating_client_loss_moments = (
        _copy_validated_participating_client_loss_moments(
            participating_client_loss_moments=participating_client_loss_moments))
    total_observed_loss_count = 0
    weighted_mean_sum = 0.0
    try:
        for loss_moments in validated_participating_client_loss_moments:
            weighted_mean_sum += loss_moments.mean_loss * loss_moments.observed_loss_count
            total_observed_loss_count += loss_moments.observed_loss_count
        if total_observed_loss_count == 0:
            return None
        aggregated_mean_loss = weighted_mean_sum / total_observed_loss_count
    except OverflowError as aggregation_error:
        raise ValueError(
            "participating_client_loss_momentsの集約結果は有限floatで表現できません。") from aggregation_error
    try:
        return BoundedLossMoments(
            observed_loss_count=total_observed_loss_count,
            mean_loss=aggregated_mean_loss,
            sum_squared_loss_deviations=0.0,
        )
    except (TypeError, ValueError) as aggregation_error:
        raise type(aggregation_error)(
            f"participating_client_loss_momentsの集約結果: {aggregation_error}") from aggregation_error
