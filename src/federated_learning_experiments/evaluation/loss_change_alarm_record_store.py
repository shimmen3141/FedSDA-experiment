"""損失監視の標本ごとの値と、警報ごとの位置（警報、変化区間の先頭、検出器の候補開始）を記録する。"""

from dataclasses import dataclass
from math import inf, isfinite, isnan


@dataclass(frozen=True, kw_only=True)
class LossChangeAlarmRecordSnapshot:
    """独立したtuple。警報と基準の引下げは、それぞれ同じ添字が同じ事象に対応する。"""

    monitored_log_e_values: tuple[float, ...]
    alarm_sample_indices: tuple[int, ...]
    estimated_change_point_sample_indices: tuple[int, ...]
    detector_candidate_start_sample_indices: tuple[int, ...]
    baseline_lowering_sample_indices: tuple[int, ...] = ()
    previous_baseline_loss_means: tuple[float, ...] = ()
    lowered_baseline_loss_means: tuple[float, ...] = ()


class LossChangeAlarmRecordStore:
    """記録だけを所有し、検出や応答の判断をしない。"""

    def __init__(self) -> None:
        self._monitored_log_e_values: list[float] = []
        self._alarm_sample_indices: list[int] = []
        self._estimated_change_point_sample_indices: list[int] = []
        self._detector_candidate_start_sample_indices: list[int] = []
        self._baseline_lowering_sample_indices: list[int] = []
        self._previous_baseline_loss_means: list[float] = []
        self._lowered_baseline_loss_means: list[float] = []

    def append_baseline_lowering_record(
        self,
        *,
        sample_index: int,
        previous_baseline_loss_mean: float,
        lowered_baseline_loss_mean: float,
    ) -> None:
        """警報ではない基準の引下げを、全項目の検査後に足す。"""
        if type(sample_index) is not int:
            raise TypeError("sample_index must be builtin int")
        if sample_index < 0 or (
            self._baseline_lowering_sample_indices
            and sample_index <= self._baseline_lowering_sample_indices[-1]
        ):
            raise ValueError("sample_index must be nonnegative and follow the last record")
        for value in (previous_baseline_loss_mean, lowered_baseline_loss_mean):
            if type(value) is not float:
                raise TypeError("baseline loss means must be builtin float")
            if not isfinite(value) or not 1e-6 <= value <= 1.0 - 1e-6:
                raise ValueError("baseline loss means must be bounded and finite")
        if lowered_baseline_loss_mean >= previous_baseline_loss_mean:
            raise ValueError("lowered baseline must be smaller than previous baseline")
        self._baseline_lowering_sample_indices.append(sample_index)
        self._previous_baseline_loss_means.append(previous_baseline_loss_mean)
        self._lowered_baseline_loss_means.append(lowered_baseline_loss_mean)

    def append_monitored_log_e_value(self, *, log_e_value: float) -> None:
        """標本1件の監視の値（混合したe値の対数）を足す。負の無限大は値として受け入れる。"""
        if type(log_e_value) is not float:
            raise TypeError("log_e_value must be builtin float")
        if isnan(log_e_value) or log_e_value == inf:
            raise ValueError("log_e_value must not be NaN or positive infinity")
        self._monitored_log_e_values.append(log_e_value)

    def append_alarm_record(
        self,
        *,
        alarm_sample_index: int,
        estimated_change_point_sample_index: int,
        detector_candidate_start_sample_index: int,
    ) -> None:
        """警報1回の位置を足す。候補開始≦変化区間の先頭≦警報、であること。"""
        for sample_index in (
            alarm_sample_index,
            estimated_change_point_sample_index,
            detector_candidate_start_sample_index,
        ):
            if type(sample_index) is not int:
                raise TypeError("sample indices must be builtin int")
        if not (
            0
            <= detector_candidate_start_sample_index
            <= estimated_change_point_sample_index
            <= alarm_sample_index
        ):
            raise ValueError(
                "sample indices must satisfy 0 <= candidate start <= change point <= alarm"
            )
        if self._alarm_sample_indices and alarm_sample_index <= self._alarm_sample_indices[-1]:
            raise ValueError("alarm_sample_index must follow the last recorded alarm")
        self._alarm_sample_indices.append(alarm_sample_index)
        self._estimated_change_point_sample_indices.append(estimated_change_point_sample_index)
        self._detector_candidate_start_sample_indices.append(detector_candidate_start_sample_index)

    def get_state_snapshot(self) -> LossChangeAlarmRecordSnapshot:
        return LossChangeAlarmRecordSnapshot(
            baseline_lowering_sample_indices=tuple(self._baseline_lowering_sample_indices),
            previous_baseline_loss_means=tuple(self._previous_baseline_loss_means),
            lowered_baseline_loss_means=tuple(self._lowered_baseline_loss_means),
            monitored_log_e_values=tuple(self._monitored_log_e_values),
            alarm_sample_indices=tuple(self._alarm_sample_indices),
            estimated_change_point_sample_indices=tuple(
                self._estimated_change_point_sample_indices
            ),
            detector_candidate_start_sample_indices=tuple(
                self._detector_candidate_start_sample_indices
            ),
        )
