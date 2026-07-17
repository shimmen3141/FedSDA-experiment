"""e-detectorの帰無仮説で使う基準平均の推定戦略。"""
import math
from abc import ABC, abstractmethod


class EDetectorBaselineEstimator(ABC):
    """モデル損失統計から条件付き平均上限の候補を作る戦略。"""

    MIN_BASELINE = 0.01
    MAX_BASELINE = 1.0 - 1e-6

    def _clip(self, value):
        return min(self.MAX_BASELINE, max(self.MIN_BASELINE, float(value)))

    @abstractmethod
    def estimate(self, stats):
        """`n`, `mean`, `M2`を持つ統計辞書から基準平均を返す。"""

    @property
    @abstractmethod
    def name(self):
        """実験結果へ保存する安定した戦略名。"""


class HistoricalMeanBaseline(EDetectorBaselineEstimator):
    """履歴損失の標本平均をそのまま使う従来戦略。"""

    name = "historical_mean"

    def estimate(self, stats):
        if not stats or stats.get("n", 0) < 1:
            return self.MIN_BASELINE
        return self._clip(stats["mean"])


class EmpiricalBernsteinUCB(EDetectorBaselineEstimator):
    """[0,1]有界損失のempirical Bernstein型上側信頼限界。

    固定された較正標本に対する有限標本境界を用いる。オンライン学習中の将来の
    条件付き平均まで自動的に保証するものではないため、検出側では前提を明示する。
    """

    name = "empirical_bernstein_ucb"

    def __init__(self, beta=0.05):
        if not 0.0 < beta < 1.0:
            raise ValueError("betaは0と1の間である必要があります")
        self.beta = float(beta)

    def estimate(self, stats):
        if not stats or stats.get("n", 0) < 2:
            # 較正不足時に小さな基準を置くと誤警報保証と感度を過大評価するため保守化する。
            return self.MAX_BASELINE
        n = int(stats["n"])
        variance = max(0.0, float(stats["M2"]) / (n - 1))
        log_term = math.log(2.0 / self.beta)
        radius = math.sqrt(2.0 * variance * log_term / n)
        radius += 7.0 * log_term / (3.0 * (n - 1))
        return self._clip(float(stats["mean"]) + radius)


def make_baseline_estimator(strategy, beta=0.05):
    """設定名または既存インスタンスから基準平均戦略を構築する。"""
    if isinstance(strategy, EDetectorBaselineEstimator):
        return strategy
    if strategy == HistoricalMeanBaseline.name:
        return HistoricalMeanBaseline()
    if strategy == EmpiricalBernsteinUCB.name:
        return EmpiricalBernsteinUCB(beta=beta)
    raise ValueError(f"未知のe-detector基準平均戦略です: {strategy!r}")
