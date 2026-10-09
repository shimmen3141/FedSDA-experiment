"""損失監視の標本ごとの値と、警報ごとの位置（警報、変化区間の先頭、検出器の候補開始）を記録する。"""

from dataclasses import dataclass
from math import inf, isnan


@dataclass(frozen=True, kw_only=True)
class LossChangeAlarmRecordSnapshot:
    """4つの列は独立したtuple。警報の3列は同じ長さで、同じ添字が同じ警報に対応する。"""

    monitored_log_e_values: tuple[float, ...]
    alarm_sample_indices: tuple[int, ...]
    estimated_change_point_sample_indices: tuple[int, ...]
    detector_candidate_start_sample_indices: tuple[int, ...]


class LossChangeAlarmRecordStore:
    """記録だけを所有し、検出や応答の判断をしない。"""

    def __init__(self) -> None:
        self._monitored_log_e_values: list[float] = []
        self._alarm_sample_indices: list[int] = []
        self._estimated_change_point_sample_indices: list[int] = []
        self._detector_candidate_start_sample_indices: list[int] = []

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
            monitored_log_e_values=tuple(self._monitored_log_e_values),
            alarm_sample_indices=tuple(self._alarm_sample_indices),
            estimated_change_point_sample_indices=tuple(
                self._estimated_change_point_sample_indices
            ),
            detector_candidate_start_sample_indices=tuple(
                self._detector_candidate_start_sample_indices
            ),
        )
