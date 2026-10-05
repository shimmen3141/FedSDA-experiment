"""単一の有界損失系列をe-SRで監視する数値状態。"""

import math
from dataclasses import dataclass

import numpy as np


def _validate_bounded_number(*, specified_value: float, parameter_name: str) -> float:
    """bool・強制変換・非有限値を許さず、0～1の数値を得る。"""
    if type(specified_value) not in (int, float):
        raise TypeError(f"{parameter_name}はbool以外のbuiltin int/floatにしてください。")
    if not 0 <= specified_value <= 1 or not math.isfinite(specified_value):
        raise ValueError(f"{parameter_name}は有限な0～1にしてください。")
    return float(specified_value)


def _log_sum_exp(*, log_values: np.ndarray) -> float:
    """候補合計を旧NumPy float64の加算順で計算する。"""
    log_values = np.asarray(log_values, dtype=np.float64)
    maximum_log_value = float(np.max(log_values))
    if not math.isfinite(maximum_log_value):
        return maximum_log_value
    return maximum_log_value + math.log(float(np.exp(log_values - maximum_log_value).sum()))


@dataclass(frozen=True, kw_only=True)
class BoundedLossESRObservation:
    """一更新後の結果。内部1始まり番号をglobal位置と区別する。"""

    observed_loss_count: int
    log_e_value: float
    drift_detected: bool
    candidate_start_observation_number: int | None
    oldest_retained_candidate_observation_number: int | None
    estimated_change_span_sample_count: int
    evaluated_candidate_bet_count: int


@dataclass(frozen=True, kw_only=True)
class BoundedLossESRState:
    """明示取得時だけコピーする候補capitalと最新結果。"""

    baseline_loss_mean: float
    last_observation: BoundedLossESRObservation
    candidate_start_observation_numbers: tuple[int, ...]
    candidate_log_capitals: tuple[tuple[float, ...], ...]


class BoundedLossESRDetector:
    """候補×賭け率をfloat64で更新し、各候補の寄与を合計する。"""

    def __init__(
        self,
        *,
        baseline_loss_mean: float,
        false_alarm_control_alpha: float,
        maximum_retained_candidate_count: int,
        betting_fractions: tuple[float, ...],
    ) -> None:
        false_alarm_control_alpha = _validate_bounded_number(
            specified_value=false_alarm_control_alpha,
            parameter_name="false_alarm_control_alpha",
        )
        if not 0 < false_alarm_control_alpha < 1:
            raise ValueError("false_alarm_control_alphaは厳密に0と1の間にしてください。")
        if type(maximum_retained_candidate_count) is not int:
            raise TypeError("maximum_retained_candidate_countはbuiltin intにしてください。")
        if maximum_retained_candidate_count < 1:
            raise ValueError("maximum_retained_candidate_countは1以上にしてください。")
        if type(betting_fractions) is not tuple or not betting_fractions:
            raise ValueError("betting_fractionsは非空tupleにしてください。")
        for betting_fraction in betting_fractions:
            betting_fraction = _validate_bounded_number(
                specified_value=betting_fraction, parameter_name="betting_fractions"
            )
            if not 0 < betting_fraction < 1:
                raise ValueError("betting_fractionsの各値は厳密に0と1の間にしてください。")
        self._maximum_retained_candidate_count = maximum_retained_candidate_count
        self._betting_fractions = np.asarray(betting_fractions, dtype=np.float64)
        self._log_betting_weights = np.full(
            len(betting_fractions), -math.log(len(betting_fractions))
        )
        self._log_alarm_threshold = math.log(1.0 / false_alarm_control_alpha)
        self.reset(baseline_loss_mean=baseline_loss_mean)

    @property
    def last_observation(self) -> BoundedLossESRObservation:
        """内部配列を含まないimmutable最新結果を返す。"""
        return self._last_observation

    def reset(self, *, baseline_loss_mean: float) -> None:
        """検証後にbaselineを固定して新しい監視区間を開始する。"""
        baseline_loss_mean = _validate_bounded_number(
            specified_value=baseline_loss_mean, parameter_name="baseline_loss_mean"
        )
        self._baseline_loss_mean = min(1.0 - 1e-6, max(1e-6, baseline_loss_mean))
        self._candidate_log_capitals = np.empty((0, len(self._betting_fractions)), dtype=np.float64)
        self._candidate_start_observation_numbers = np.empty(0, dtype=np.int64)
        self._observed_loss_count = 0
        self._last_observation = BoundedLossESRObservation(
            observed_loss_count=0,
            log_e_value=-math.inf,
            drift_detected=False,
            candidate_start_observation_number=None,
            oldest_retained_candidate_observation_number=None,
            estimated_change_span_sample_count=0,
            evaluated_candidate_bet_count=0,
        )

    def observe_loss(self, *, observed_loss: float) -> BoundedLossESRObservation:
        """一つのラベル観測後損失を旧演算順で取り込む。"""
        observed_loss = _validate_bounded_number(
            specified_value=observed_loss, parameter_name="observed_loss"
        )
        self._observed_loss_count += 1
        loss_increments = 1.0 + self._betting_fractions * (
            observed_loss / self._baseline_loss_mean - 1.0
        )
        log_loss_increments = np.log(np.maximum(loss_increments, np.finfo(np.float64).tiny))
        new_candidate_log_capitals = np.zeros((1, len(self._betting_fractions)), dtype=np.float64)
        self._candidate_log_capitals = np.vstack(
            (self._candidate_log_capitals, new_candidate_log_capitals)
        )
        self._candidate_start_observation_numbers = np.append(
            self._candidate_start_observation_numbers, self._observed_loss_count
        )
        self._candidate_log_capitals += log_loss_increments
        if len(self._candidate_start_observation_numbers) > self._maximum_retained_candidate_count:
            excess_candidate_count = (
                len(self._candidate_start_observation_numbers)
                - self._maximum_retained_candidate_count
            )
            self._candidate_log_capitals = self._candidate_log_capitals[excess_candidate_count:]
            self._candidate_start_observation_numbers = self._candidate_start_observation_numbers[
                excess_candidate_count:
            ]
        weighted_candidate_logs = self._candidate_log_capitals + self._log_betting_weights
        row_maximum_logs = np.max(weighted_candidate_logs, axis=1)
        candidate_logs = row_maximum_logs + np.log(
            np.exp(weighted_candidate_logs - row_maximum_logs[:, None]).sum(axis=1)
        )
        log_e_value = _log_sum_exp(log_values=candidate_logs)
        best_candidate_index = int(np.argmax(candidate_logs))
        candidate_start_observation_number = int(
            self._candidate_start_observation_numbers[best_candidate_index]
        )
        self._last_observation = BoundedLossESRObservation(
            observed_loss_count=self._observed_loss_count,
            log_e_value=log_e_value,
            drift_detected=log_e_value >= self._log_alarm_threshold,
            candidate_start_observation_number=candidate_start_observation_number,
            oldest_retained_candidate_observation_number=int(
                self._candidate_start_observation_numbers[0]
            ),
            estimated_change_span_sample_count=self._observed_loss_count
            - candidate_start_observation_number
            + 1,
            evaluated_candidate_bet_count=len(self._candidate_start_observation_numbers)
            * len(self._betting_fractions),
        )
        return self._last_observation

    def get_state_snapshot(self) -> BoundedLossESRState:
        """候補配列を変更不能な独立tupleへコピーする。"""
        return BoundedLossESRState(
            baseline_loss_mean=self._baseline_loss_mean,
            last_observation=self._last_observation,
            candidate_start_observation_numbers=tuple(
                self._candidate_start_observation_numbers.tolist()
            ),
            candidate_log_capitals=tuple(
                tuple(log_values) for log_values in self._candidate_log_capitals.tolist()
            ),
        )
