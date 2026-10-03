"""モデル帰属を確定する前の観測標本位置を明示操作で保持する。"""

from collections import deque
from dataclasses import dataclass

from federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings import TrainingDataAssignmentSettings


@dataclass(frozen=True, kw_only=True)
class PendingTrainingAssignmentState:
    """内部dequeを公開しない診断用の状態copy。"""

    pending_sample_indices: tuple[int, ...]
    last_observed_sample_index: int | None


def _validate_nonnegative_sample_index(*, sample_index: int) -> None:
    """位置を暗黙変換せず、builtin intの非負値だけ受理する。"""
    if type(sample_index) is not int:
        raise TypeError("sample_index must be a builtin int, excluding bool")
    if sample_index < 0:
        raise ValueError("sample_index must be nonnegative")


class PendingTrainingAssignmentBuffer:
    """観測順を所有し、容量調整や消費は呼出側が明示する。"""

    def __init__(self, *, training_data_assignment_settings: TrainingDataAssignmentSettings) -> None:
        if type(training_data_assignment_settings) is not TrainingDataAssignmentSettings:
            raise TypeError("training_data_assignment_settings must be TrainingDataAssignmentSettings")
        training_data_assignment_settings.__post_init__()
        self._training_data_assignment_settings = training_data_assignment_settings
        self._pending_sample_indices: deque[int] = deque()
        self._last_observed_sample_index: int | None = None

    def append_observed_sample_index(self, *, sample_index: int) -> None:
        """追加時に容量で切り捨てず、警報時の容量＋1件を保持する。"""
        _validate_nonnegative_sample_index(sample_index=sample_index)
        if self._last_observed_sample_index is not None and sample_index != self._last_observed_sample_index + 1:
            raise ValueError("sample_index must follow last_observed_sample_index by one")
        self._pending_sample_indices.append(sample_index)
        self._last_observed_sample_index = sample_index

    def release_sample_indices_exceeding_capacity(self) -> tuple[int, ...]:
        """容量を超える最古の位置だけを順に解放する。"""
        released_sample_indices: list[int] = []
        while len(self._pending_sample_indices) > self._training_data_assignment_settings.pending_assignment_buffer_capacity_samples:
            released_sample_indices.append(self._pending_sample_indices.popleft())
        return tuple(released_sample_indices)

    def get_state_snapshot(self) -> PendingTrainingAssignmentState:
        """最後の観測位置と保留順を変更不能なcopyで返す。"""
        return PendingTrainingAssignmentState(
            pending_sample_indices=tuple(self._pending_sample_indices),
            last_observed_sample_index=self._last_observed_sample_index,
        )

