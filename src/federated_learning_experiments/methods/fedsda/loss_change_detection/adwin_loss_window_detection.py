"""単一の損失系列を、ADWIN（全部の分割点を調べる方式）で監視する数値状態。"""

import math
from collections import deque
from dataclasses import dataclass
from itertools import accumulate


@dataclass(frozen=True, kw_only=True)
class AdwinLossWindowObservation:
    """1回の更新の後の結果。"""

    # 更新（と、検知による古い側の削除）の後の、窓の標本数。
    window_sample_count: int
    drift_detected: bool
    # この更新で調べた分割点の数（窓が、検定を始める標本数に満たなければ0）。
    evaluated_split_count: int


@dataclass(frozen=True, kw_only=True)
class AdwinLossWindowState:
    """窓の損失と、その和・二乗和（取得後の更新で変わらない写し）。"""

    window_losses: tuple[float, ...]
    loss_sum: float
    squared_loss_sum: float
    last_observation: AdwinLossWindowObservation


class AdwinLossWindowDetector:
    """窓を2つに分けたときの平均の差が、Bernstein型の閾値を超える分割があれば、検知して、古い側を捨てる。

    演算の順は、旧の`FullScanADWIN`と同じ（累積和は、窓の先頭からの逐次の和）。
    """

    def __init__(
        self,
        *,
        confidence_delta: float,
        maximum_window_sample_count: int,
        minimum_test_window_sample_count: int,
    ) -> None:
        if type(confidence_delta) is not float:
            raise TypeError("confidence_deltaはbuiltin floatにしてください。")
        if not 0.0 < confidence_delta < 1.0:
            raise ValueError("confidence_deltaは0より大きく1未満にしてください。")
        for count_name, count in (
            ("maximum_window_sample_count", maximum_window_sample_count),
            ("minimum_test_window_sample_count", minimum_test_window_sample_count),
        ):
            if type(count) is not int:
                raise TypeError(f"{count_name}はbool以外のbuiltin intにしてください。")
            if count < 1:
                raise ValueError(f"{count_name}は1以上にしてください。")
        self._confidence_delta = confidence_delta
        self._maximum_window_sample_count = maximum_window_sample_count
        self._minimum_test_window_sample_count = minimum_test_window_sample_count
        self._window_losses: deque[float] = deque()
        self._loss_sum = 0.0
        self._squared_loss_sum = 0.0
        self._last_observation = AdwinLossWindowObservation(
            window_sample_count=0, drift_detected=False, evaluated_split_count=0
        )

    @property
    def last_observation(self) -> AdwinLossWindowObservation:
        return self._last_observation

    def _remove_oldest_loss(self) -> None:
        removed_loss = self._window_losses.popleft()
        self._loss_sum -= removed_loss
        self._squared_loss_sum -= removed_loss**2

    def _find_best_split_old_side_sample_count(self) -> int | None:
        """閾値を最も大きく超える分割の、古い側の標本数。超える分割がなければNone。"""
        window_sample_count = len(self._window_losses)
        cumulative_loss_sums = tuple(accumulate(self._window_losses))
        adjusted_delta = self._confidence_delta / max(1, window_sample_count)
        log_term = math.log(max(1e-12, 2.0 / adjusted_delta))
        window_mean_loss = self._loss_sum / window_sample_count
        window_loss_variance = max(
            0.0, (self._squared_loss_sum / window_sample_count) - (window_mean_loss**2)
        )
        best_old_side_sample_count = None
        maximum_excess_over_threshold = -1.0
        for old_side_sample_count in range(1, window_sample_count):
            new_side_sample_count = window_sample_count - old_side_sample_count
            old_side_loss_sum = cumulative_loss_sums[old_side_sample_count - 1]
            new_side_loss_sum = self._loss_sum - old_side_loss_sum
            mean_loss_difference = abs(
                old_side_loss_sum / old_side_sample_count
                - new_side_loss_sum / new_side_sample_count
            )
            harmonic_sample_count = 1.0 / (
                1.0 / old_side_sample_count + 1.0 / new_side_sample_count
            )
            threshold = (
                math.sqrt(
                    (2.0 / harmonic_sample_count) * max(1e-12, window_loss_variance) * log_term
                )
                + (2.0 / (3.0 * harmonic_sample_count)) * log_term
            )
            if mean_loss_difference > threshold:
                excess_over_threshold = mean_loss_difference - threshold
                if excess_over_threshold > maximum_excess_over_threshold:
                    maximum_excess_over_threshold = excess_over_threshold
                    best_old_side_sample_count = old_side_sample_count
        return best_old_side_sample_count

    def observe_loss(self, *, observed_loss: float) -> AdwinLossWindowObservation:
        """損失を1つ窓へ足し、上限を超えた分を捨ててから、分割を調べる。"""
        if type(observed_loss) not in (int, float):
            raise TypeError("observed_lossはbool以外のbuiltin int/floatにしてください。")
        if not 0 <= observed_loss <= 1 or not math.isfinite(observed_loss):
            raise ValueError("observed_lossは有限な0～1にしてください。")
        observed_loss = float(observed_loss)
        self._window_losses.append(observed_loss)
        self._loss_sum += observed_loss
        self._squared_loss_sum += observed_loss**2
        if len(self._window_losses) > self._maximum_window_sample_count:
            self._remove_oldest_loss()
        tested_window_sample_count = len(self._window_losses)
        drift_detected = False
        evaluated_split_count = 0
        if tested_window_sample_count >= self._minimum_test_window_sample_count:
            evaluated_split_count = tested_window_sample_count - 1
            best_old_side_sample_count = self._find_best_split_old_side_sample_count()
            if best_old_side_sample_count is not None:
                drift_detected = True
                for _ in range(best_old_side_sample_count):
                    self._remove_oldest_loss()
        self._last_observation = AdwinLossWindowObservation(
            window_sample_count=len(self._window_losses),
            drift_detected=drift_detected,
            evaluated_split_count=evaluated_split_count,
        )
        return self._last_observation

    def reset(self) -> None:
        """窓を空にする。"""
        self._window_losses.clear()
        self._loss_sum = 0.0
        self._squared_loss_sum = 0.0
        self._last_observation = AdwinLossWindowObservation(
            window_sample_count=0, drift_detected=False, evaluated_split_count=0
        )

    def get_state_snapshot(self) -> AdwinLossWindowState:
        return AdwinLossWindowState(
            window_losses=tuple(self._window_losses),
            loss_sum=self._loss_sum,
            squared_loss_sum=self._squared_loss_sum,
            last_observation=self._last_observation,
        )
