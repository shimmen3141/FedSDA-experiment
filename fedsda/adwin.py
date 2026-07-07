"""ADWIN (ADaptive WINdowing) によるドリフト検出器。

全分割点を走査する素朴な実装(FullScan)。Bernstein型の統計的閾値
epsilon を用い、|mu0 - mu1| > epsilon となる分割が存在したらドリフトと
判定してウィンドウの古い側を削除する。
"""
import math
from collections import deque

import numpy as np

from . import config


class FullScanADWIN:
    def __init__(self, delta=None, max_window_size=None):
        self.delta = delta if delta is not None else config.ADWIN_DELTA
        max_window_size = max_window_size if max_window_size is not None else config.ADWIN_MAX_WINDOW
        self.window = deque()
        self.total = 0.0
        self.total_sq = 0.0
        self.width = 0
        self.max_window_size = max_window_size
        self.drift_detected_flag = False

    def update(self, value):
        self.window.append(value)
        self.total += value
        self.total_sq += value ** 2
        self.width += 1
        self.drift_detected_flag = False

        if self.width > self.max_window_size:
            removed = self.window.popleft()
            self.total -= removed
            self.total_sq -= removed ** 2
            self.width -= 1

        self._check_drift()

    def _check_drift(self):
        if self.width < config.ADWIN_MIN_WIDTH:
            return

        window_arr = np.array(self.window)
        cumsum = np.cumsum(window_arr)

        total_sum = self.total
        total_width = self.width

        delta_prime = self.delta / max(1, total_width)
        ln_term = math.log(max(1e-12, 2.0 / delta_prime))

        best_cut_n0 = -1
        max_diff_vs_epsilon = -1.0
        drift_found = False

        mean_W = self.total / self.width
        variance_W = max(0.0, (self.total_sq / self.width) - (mean_W ** 2))

        for n0 in range(1, total_width):
            n1 = total_width - n0
            sum0 = cumsum[n0 - 1]
            sum1 = total_sum - sum0
            mu0 = sum0 / n0
            mu1 = sum1 / n1
            diff = abs(mu0 - mu1)

            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            epsilon = math.sqrt((2.0 / m) * max(1e-12, variance_W) * ln_term) + (2.0 / (3.0 * m)) * ln_term

            if diff > epsilon:
                metric = diff - epsilon
                if metric > max_diff_vs_epsilon:
                    max_diff_vs_epsilon = metric
                    best_cut_n0 = n0
                    drift_found = True

        if drift_found:
            self.drift_detected_flag = True
            for _ in range(best_cut_n0):
                rm = self.window.popleft()
                self.total -= rm
                self.total_sq -= rm ** 2
                self.width -= 1
            return

    @property
    def drift_detected(self):
        return self.drift_detected_flag

    def reset(self):
        self.window.clear()
        self.total = 0.0
        self.total_sq = 0.0
        self.width = 0
        self.drift_detected_flag = False
