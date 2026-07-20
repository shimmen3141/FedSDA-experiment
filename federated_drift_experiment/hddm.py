"""有界損失系列向けHDDM-A / HDDM-Wドリフト検出器。"""

import math
from dataclasses import dataclass


def _validate_probability(name, value):
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name}は0と1の間である必要があります")
    return value


class HDDMA:
    """Hoeffding境界と累積平均を用いて平均損失の上昇を検出する。"""

    def __init__(self, drift_confidence=0.001, warning_confidence=0.005):
        self.drift_confidence = _validate_probability(
            "drift_confidence", drift_confidence
        )
        self.warning_confidence = _validate_probability(
            "warning_confidence", warning_confidence
        )
        self.reset()

    def reset(self):
        self.total_n = 0
        self.total_sum = 0.0
        self.min_n = 0
        self.min_sum = 0.0
        self.width = 0
        self.drift_detected_flag = False
        self.warning_detected = False
        self._last_hypothesis_count = 0

    def update(self, value):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("HDDMへの入力損失は[0, 1]である必要があります")

        self.drift_detected_flag = False
        self.warning_detected = False
        self._last_hypothesis_count = 0
        self.total_n += 1
        self.total_sum += value
        self.width = self.total_n

        if self.min_n == 0:
            self.min_n = self.total_n
            self.min_sum = self.total_sum

        min_bound = math.sqrt(
            math.log(1.0 / self.drift_confidence) / (2.0 * self.min_n)
        )
        total_bound = math.sqrt(
            math.log(1.0 / self.drift_confidence) / (2.0 * self.total_n)
        )
        if (self.min_sum / self.min_n + min_bound
                >= self.total_sum / self.total_n + total_bound):
            self.min_n = self.total_n
            self.min_sum = self.total_sum

        can_compare = self.min_n != self.total_n
        self._last_hypothesis_count = int(can_compare)
        if self._mean_increased(self.drift_confidence):
            detected_width = self.total_n - self.min_n
            self._clear_statistics()
            self.width = max(1, detected_width)
            self.drift_detected_flag = True
        elif self._mean_increased(self.warning_confidence):
            self._last_hypothesis_count += int(can_compare)
            self.warning_detected = True
        else:
            self._last_hypothesis_count += int(can_compare)

    def _mean_increased(self, confidence):
        if self.min_n == self.total_n:
            return False
        scale = (self.total_n - self.min_n) / (self.min_n * self.total_n)
        bound = math.sqrt(scale * math.log(2.0 / confidence) / 2.0)
        return self.total_sum / self.total_n - self.min_sum / self.min_n >= bound

    def _clear_statistics(self):
        self.total_n = 0
        self.total_sum = 0.0
        self.min_n = 0
        self.min_sum = 0.0

    @property
    def drift_detected(self):
        return self.drift_detected_flag

    @property
    def active_hypothesis_count(self):
        return self._last_hypothesis_count


@dataclass
class _WeightedSample:
    mean: float = -1.0
    squared_weight_sum: float = 0.0


class HDDMW:
    """EWMAとMcDiarmid境界を用いて平均損失の上昇を検出する。"""

    def __init__(self, drift_confidence=0.001, warning_confidence=0.005,
                 lambda_option=0.05):
        self.drift_confidence = _validate_probability(
            "drift_confidence", drift_confidence
        )
        self.warning_confidence = _validate_probability(
            "warning_confidence", warning_confidence
        )
        self.lambda_option = _validate_probability("lambda_option", lambda_option)
        self.reset()

    def reset(self):
        self.total = _WeightedSample()
        self.before_increment = _WeightedSample()
        self.after_increment = _WeightedSample()
        self.increment_cutpoint = math.inf
        self.width = 0
        self._delay = 0
        self.drift_detected_flag = False
        self.warning_detected = False
        self._last_hypothesis_count = 0

    def update(self, value):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("HDDMへの入力損失は[0, 1]である必要があります")

        self.drift_detected_flag = False
        self.warning_detected = False
        self._last_hypothesis_count = 0
        self.width += 1
        self._update_weighted(self.total, value)
        self._update_increment_statistics(value)

        can_compare = (
            self.before_increment.mean >= 0.0
            and self.after_increment.mean >= 0.0
        )
        self._last_hypothesis_count = int(can_compare)
        if self._mean_increased(self.drift_confidence):
            detected_width = max(1, self._delay)
            hypothesis_count = self._last_hypothesis_count
            self.reset()
            self.width = detected_width
            self.drift_detected_flag = True
            self._last_hypothesis_count = hypothesis_count
        elif self._mean_increased(self.warning_confidence):
            self._last_hypothesis_count += int(can_compare)
            self.warning_detected = True
        else:
            self._last_hypothesis_count += int(can_compare)

    def _update_weighted(self, sample, value):
        decay = 1.0 - self.lambda_option
        if sample.mean < 0.0:
            sample.mean = value
            sample.squared_weight_sum = 1.0
            return
        sample.mean = self.lambda_option * value + decay * sample.mean
        sample.squared_weight_sum = (
            self.lambda_option ** 2 + decay ** 2 * sample.squared_weight_sum
        )

    def _update_increment_statistics(self, value):
        bound = math.sqrt(
            self.total.squared_weight_sum
            * math.log(1.0 / self.drift_confidence) / 2.0
        )
        if self.total.mean + bound < self.increment_cutpoint:
            self.increment_cutpoint = self.total.mean + bound
            self.before_increment = _WeightedSample(
                self.total.mean, self.total.squared_weight_sum
            )
            self.after_increment = _WeightedSample()
            self._delay = 0
        else:
            self._delay += 1
            self._update_weighted(self.after_increment, value)

    def _mean_increased(self, confidence):
        if self.before_increment.mean < 0.0 or self.after_increment.mean < 0.0:
            return False
        bound = math.sqrt(
            (self.before_increment.squared_weight_sum
             + self.after_increment.squared_weight_sum)
            * math.log(1.0 / confidence) / 2.0
        )
        return self.after_increment.mean - self.before_increment.mean > bound

    @property
    def drift_detected(self):
        return self.drift_detected_flag

    @property
    def active_hypothesis_count(self):
        return self._last_hypothesis_count
