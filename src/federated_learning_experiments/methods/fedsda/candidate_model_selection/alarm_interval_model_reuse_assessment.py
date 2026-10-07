"""警報区間の平均損失と履歴基準から、保有モデルの再利用適合を判定する。"""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, kw_only=True)
class AlarmIntervalModelReuseAssessment:
    """履歴基準を使えるモデルの区間平均列と、増加量条件を満たした同順の部分列。"""

    baseline_supported_interval_mean_losses_by_model_id: tuple[tuple[int, float], ...]
    reusable_mean_losses_by_model_id: tuple[tuple[int, float], ...]

    @property
    def selected_reuse_model_id(self) -> int | None:
        """適合列の区間平均が最小のID。同率は列の先着、適合なしはNone。"""
        if not self.reusable_mean_losses_by_model_id:
            return None
        return min(
            self.reusable_mean_losses_by_model_id,
            key=lambda evaluated_model_loss: evaluated_model_loss[1],
        )[0]


def _validate_bounded_mean_loss(*, parameter_name: str, specified_value: float) -> None:
    if type(specified_value) not in (int, float):
        raise TypeError(f"{parameter_name}はbool以外のbuiltin intまたはfloatが必要です。")
    if not 0 <= specified_value <= 1:
        # NaNと無限大、floatへ表現できない巨大な整数もこの範囲検査で拒否される。
        raise ValueError(f"{parameter_name}は0以上1以下の有限値が必要です。")


def _validate_alarm_interval_reuse_assessment_inputs(
    *,
    baseline_supported_interval_mean_losses_by_model_id: tuple[tuple[int, float], ...],
    reuse_baseline_mean_losses_by_model_id: dict[int, float],
    maximum_alarm_interval_mean_loss_increase: float,
) -> None:
    if type(baseline_supported_interval_mean_losses_by_model_id) is not tuple:
        raise TypeError(
            "baseline_supported_interval_mean_losses_by_model_idはexact tupleが必要です。"
        )
    seen_model_ids: set[int] = set()
    for evaluated_model_loss in baseline_supported_interval_mean_losses_by_model_id:
        if type(evaluated_model_loss) is not tuple:
            raise TypeError(
                "baseline_supported_interval_mean_losses_by_model_idの要素はexact tupleが必要です。"
            )
        if len(evaluated_model_loss) != 2:
            raise ValueError(
                "baseline_supported_interval_mean_losses_by_model_idの要素は"
                "(model_id, mean_loss)の2要素が必要です。"
            )
        model_id, mean_loss = evaluated_model_loss
        if type(model_id) is not int:
            raise TypeError("model_idはbool以外のbuiltin intが必要です。")
        _validate_bounded_mean_loss(parameter_name="mean_loss", specified_value=mean_loss)
        if model_id in seen_model_ids:
            raise ValueError(
                "baseline_supported_interval_mean_losses_by_model_idのmodel_idは一意が必要です。"
            )
        seen_model_ids.add(model_id)
    if type(reuse_baseline_mean_losses_by_model_id) is not dict:
        raise TypeError("reuse_baseline_mean_losses_by_model_idはexact dictが必要です。")
    for model_id, reuse_baseline_mean_loss in reuse_baseline_mean_losses_by_model_id.items():
        if type(model_id) is not int:
            raise TypeError(
                "reuse_baseline_mean_losses_by_model_idのkeyはbool以外のbuiltin intが必要です。"
            )
        _validate_bounded_mean_loss(
            parameter_name="reuse_baseline_mean_loss", specified_value=reuse_baseline_mean_loss
        )
        if reuse_baseline_mean_loss == 0:
            raise ValueError("reuse_baseline_mean_lossは非零が必要です。")
    expected_model_ids = set(reuse_baseline_mean_losses_by_model_id)
    if seen_model_ids != expected_model_ids:
        raise ValueError(
            "reuse_baseline_mean_losses_by_model_idのkeyは評価済みのmodel_idと同じ集合が必要です。"
        )
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


def assess_alarm_interval_model_reuse(
    *,
    baseline_supported_interval_mean_losses_by_model_id: tuple[tuple[int, float], ...],
    reuse_baseline_mean_losses_by_model_id: dict[int, float],
    maximum_alarm_interval_mean_loss_increase: float,
) -> AlarmIntervalModelReuseAssessment:
    """全入力検査後、区間平均−履歴平均が許容増加量以下のモデルを同じ順序で残す。"""
    _validate_alarm_interval_reuse_assessment_inputs(
        baseline_supported_interval_mean_losses_by_model_id=baseline_supported_interval_mean_losses_by_model_id,
        reuse_baseline_mean_losses_by_model_id=reuse_baseline_mean_losses_by_model_id,
        maximum_alarm_interval_mean_loss_increase=maximum_alarm_interval_mean_loss_increase,
    )
    return AlarmIntervalModelReuseAssessment(
        baseline_supported_interval_mean_losses_by_model_id=baseline_supported_interval_mean_losses_by_model_id,
        reusable_mean_losses_by_model_id=tuple(
            (model_id, mean_loss)
            for model_id, mean_loss in baseline_supported_interval_mean_losses_by_model_id
            if mean_loss - reuse_baseline_mean_losses_by_model_id[model_id]
            <= maximum_alarm_interval_mean_loss_increase
        ),
    )
