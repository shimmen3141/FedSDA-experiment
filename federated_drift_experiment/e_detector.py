"""bounded mean向け混合Shiryaev--Roberts型e-detector。

ADWINとは独立に、bounded mean向けShiryaev--Roberts型e-detectorを提供する。
入力は[0, 1]の損失で、定常時の条件付き平均がbaseline以下という仮定を置く。
"""
import math

import numpy as np


def _logsumexp(values):
    """外部依存を増やさず、対数e値を安定に合成する。"""
    values = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(values))
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log(float(np.exp(values - maximum).sum()))


class BoundedMeanEDetector:
    """損失平均の上昇を検知する混合e-SR検知器。

    候補変化点ごとにe-processを開始し、複数の賭け率lambdaを等重みで混合する。
    全候補の混合e値の和が1/alpha以上になったとき検知し、最大寄与候補を
    推定分割点として返す。保持候補数の上限で古い候補を捨てる操作は、e値を
    小さくする方向なので保守的である。
    """

    DEFAULT_LAMBDAS = (0.05, 0.1, 0.2, 0.4, 0.8)

    def __init__(self, baseline, alpha=0.001, max_candidates=1000, lambdas=None):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alphaは0と1の間である必要があります")
        if max_candidates < 1:
            raise ValueError("max_candidatesは1以上である必要があります")
        lambdas = tuple(lambdas or self.DEFAULT_LAMBDAS)
        if not lambdas or any(not 0.0 < value < 1.0 for value in lambdas):
            raise ValueError("lambdaは0と1の間の値を1つ以上指定してください")

        self.alpha = float(alpha)
        self.max_candidates = int(max_candidates)
        self.lambdas = np.asarray(lambdas, dtype=np.float64)
        self.log_weights = np.full(len(lambdas), -math.log(len(lambdas)))
        self.log_threshold = math.log(1.0 / self.alpha)
        self.reset(baseline)

    def reset(self, baseline=None):
        """候補e-processを破棄し、新しい定常区間を開始する。"""
        if baseline is not None:
            # m=0または1では増分式が退化するため、数値的に安全な範囲へ制限する。
            self.baseline = min(1.0 - 1e-6, max(1e-6, float(baseline)))
        elif not hasattr(self, "baseline"):
            raise ValueError("初回resetではbaselineが必要です")
        self._log_capitals = np.empty((0, len(self.lambdas)), dtype=np.float64)
        self._candidate_starts = np.empty(0, dtype=np.int64)
        self._time = 0
        self.width = 0
        self.drift_detected_flag = False
        self.log_e_value = -math.inf
        self.split_start = None

    def update(self, value):
        """1損失を投入し、全候補変化点のe-processを更新する。"""
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("e-detectorへの入力損失は[0, 1]である必要があります")

        self.drift_detected_flag = False
        self.split_start = None
        self._time += 1

        # L=1+lambda(X/m-1)。帰無仮説E[X|past]<=mの下でE[L|past]<=1。
        increments = 1.0 + self.lambdas * (value / self.baseline - 1.0)
        log_increments = np.log(np.maximum(increments, np.finfo(np.float64).tiny))

        new_candidate = np.zeros((1, len(self.lambdas)), dtype=np.float64)
        self._log_capitals = np.vstack((self._log_capitals, new_candidate))
        self._candidate_starts = np.append(self._candidate_starts, self._time)
        self._log_capitals += log_increments

        if len(self._candidate_starts) > self.max_candidates:
            excess = len(self._candidate_starts) - self.max_candidates
            self._log_capitals = self._log_capitals[excess:]
            self._candidate_starts = self._candidate_starts[excess:]

        weighted_logs = self._log_capitals + self.log_weights
        row_maxima = np.max(weighted_logs, axis=1)
        candidate_logs = row_maxima + np.log(
            np.exp(weighted_logs - row_maxima[:, None]).sum(axis=1)
        )
        self.log_e_value = _logsumexp(candidate_logs)
        best_index = int(np.argmax(candidate_logs))
        self.split_start = int(self._candidate_starts[best_index])
        self.width = self._time - self.split_start + 1

        if self.log_e_value >= self.log_threshold:
            self.drift_detected_flag = True

    @property
    def drift_detected(self):
        return self.drift_detected_flag

    @property
    def e_value(self):
        """表示・記録用のe値。浮動小数点上限を超える場合はinfを返す。"""
        if self.log_e_value >= math.log(np.finfo(np.float64).max):
            return math.inf
        return math.exp(self.log_e_value)

    @property
    def active_hypothesis_count(self):
        """今回の更新で評価した候補変化点×賭け率の数。"""
        return len(self._candidate_starts) * len(self.lambdas)
