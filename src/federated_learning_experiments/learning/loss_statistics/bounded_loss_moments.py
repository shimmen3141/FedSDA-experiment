"""一系列の有界損失を旧Welfordの演算順で不変集計する。"""

import math
from dataclasses import dataclass


def _validate_finite_nonnegative_number(
    *,
    specified_value: int | float,
    parameter_name: str,
    maximum_value: float | None = None,
) -> float:
    if type(specified_value) not in (int, float):
        raise TypeError(f"{parameter_name}はbool以外のbuiltin int/floatが必要です。")
    try:
        specified_value = float(specified_value)
    except OverflowError:
        raise ValueError(f"{parameter_name}は有限floatで表現できる値が必要です。") from None
    if not math.isfinite(specified_value) or specified_value < 0:
        raise ValueError(f"{parameter_name}は有限の非負値が必要です。")
    if maximum_value is not None and specified_value > maximum_value:
        raise ValueError(f"{parameter_name}は{maximum_value}以下が必要です。")
    return specified_value


def _validate_loss_moment_fields(
    *,
    observed_loss_count: int,
    mean_loss: float,
    sum_squared_loss_deviations: float,
) -> tuple[float, float]:
    if type(observed_loss_count) is not int:
        raise TypeError("observed_loss_countはbool以外のbuiltin intが必要です。")
    _validate_finite_nonnegative_number(
        specified_value=observed_loss_count, parameter_name="observed_loss_count"
    )
    mean_loss = _validate_finite_nonnegative_number(
        specified_value=mean_loss, parameter_name="mean_loss", maximum_value=1.0
    )
    sum_squared_loss_deviations = _validate_finite_nonnegative_number(
        specified_value=sum_squared_loss_deviations, parameter_name="sum_squared_loss_deviations"
    )
    if observed_loss_count == 0:
        if mean_loss != 0.0:
            raise ValueError("mean_lossは0件では0が必要です。")
        if sum_squared_loss_deviations != 0.0:
            raise ValueError("sum_squared_loss_deviationsは0件では0が必要です。")
    return mean_loss, sum_squared_loss_deviations


@dataclass(frozen=True, kw_only=True)
class BoundedLossMoments:
    """明示seedを補正せず保持する件数・平均・偏差平方和。"""

    observed_loss_count: int
    mean_loss: float
    sum_squared_loss_deviations: float

    def __post_init__(self) -> None:
        mean_loss, sum_squared_loss_deviations = _validate_loss_moment_fields(
            observed_loss_count=self.observed_loss_count,
            mean_loss=self.mean_loss,
            sum_squared_loss_deviations=self.sum_squared_loss_deviations,
        )
        object.__setattr__(self, "mean_loss", mean_loss)
        object.__setattr__(self, "sum_squared_loss_deviations", sum_squared_loss_deviations)


@dataclass(frozen=True, kw_only=True)
class LossMeanAndSampleVariance:
    """2件以上の保存平均と不偏標本分散。"""

    observed_loss_count: int
    mean_loss: float
    sample_variance: float


def _validate_loss_moments(*, loss_moments: BoundedLossMoments) -> None:
    if type(loss_moments) is not BoundedLossMoments:
        raise TypeError("loss_momentsはBoundedLossMomentsのexact型が必要です。")
    _validate_loss_moment_fields(
        observed_loss_count=loss_moments.observed_loss_count,
        mean_loss=loss_moments.mean_loss,
        sum_squared_loss_deviations=loss_moments.sum_squared_loss_deviations,
    )


def accumulate_bounded_loss_observation(
    *,
    loss_moments: BoundedLossMoments,
    observed_loss: float,
) -> BoundedLossMoments:
    """入力を変更せず、件数→差→平均→差→偏差平方和の順に一件追加する。"""
    _validate_loss_moments(loss_moments=loss_moments)
    observed_loss = _validate_finite_nonnegative_number(
        specified_value=observed_loss, parameter_name="observed_loss", maximum_value=1.0
    )
    observed_loss_count = loss_moments.observed_loss_count + 1
    _validate_finite_nonnegative_number(
        specified_value=observed_loss_count, parameter_name="observed_loss_count"
    )
    first_mean_difference = observed_loss - loss_moments.mean_loss
    updated_mean_loss = loss_moments.mean_loss + first_mean_difference / observed_loss_count
    second_mean_difference = observed_loss - updated_mean_loss
    updated_sum_squared_loss_deviations = (
        loss_moments.sum_squared_loss_deviations + first_mean_difference * second_mean_difference
    )
    return BoundedLossMoments(
        observed_loss_count=observed_loss_count,
        mean_loss=updated_mean_loss,
        sum_squared_loss_deviations=updated_sum_squared_loss_deviations,
    )


def estimate_loss_mean_and_sample_variance(
    *,
    loss_moments: BoundedLossMoments,
) -> LossMeanAndSampleVariance | None:
    """不足件数はNone、2件以上は保存平均とM2/(n-1)を返す。"""
    _validate_loss_moments(loss_moments=loss_moments)
    if loss_moments.observed_loss_count < 2:
        return None
    return LossMeanAndSampleVariance(
        observed_loss_count=loss_moments.observed_loss_count,
        mean_loss=float(loss_moments.mean_loss),
        sample_variance=(
            loss_moments.sum_squared_loss_deviations / (loss_moments.observed_loss_count - 1)
        ),
    )
