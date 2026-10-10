"""損失の監視（検出器）の計算の計数を持つ。"""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class LossMonitoringComputationCounts:
    """検出器の部品の更新回数と、その更新で評価した「候補の変化点×賭け率」の数の合計。"""

    detector_component_update_count: int
    evaluated_candidate_bet_count: int


def _validate_nonnegative_count(*, count: int, count_name: str) -> None:
    if type(count) is not int:
        raise TypeError(f"{count_name} must be builtin int")
    if count < 0:
        raise ValueError(f"{count_name} must be nonnegative")


class LossMonitoringComputationCountStore:
    """標本ごとの監視の計算の計数を足して、合計を所有する。"""

    def __init__(self) -> None:
        self._detector_component_update_count = 0
        self._evaluated_candidate_bet_count = 0

    def record_loss_monitoring_computation(
        self, *, detector_component_update_count: int, evaluated_candidate_bet_count: int
    ) -> None:
        """標本1件ぶんの計数を足す。2つとも確かめてから、足す。"""
        _validate_nonnegative_count(
            count=detector_component_update_count, count_name="detector_component_update_count"
        )
        _validate_nonnegative_count(
            count=evaluated_candidate_bet_count, count_name="evaluated_candidate_bet_count"
        )
        self._detector_component_update_count += detector_component_update_count
        self._evaluated_candidate_bet_count += evaluated_candidate_bet_count

    def get_loss_monitoring_computation_counts(self) -> LossMonitoringComputationCounts:
        return LossMonitoringComputationCounts(
            detector_component_update_count=self._detector_component_update_count,
            evaluated_candidate_bet_count=self._evaluated_candidate_bet_count,
        )
