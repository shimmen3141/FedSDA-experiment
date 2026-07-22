"""近接した複数の検出を一つの適応エピソードとして扱う状態管理。"""
from dataclasses import dataclass


@dataclass
class _ActiveEpisode:
    episode_id: int
    start_position: int
    operation_performed: bool = False


class DetectionEpisodeController:
    """一定区間内の検出証拠を統合し、モデル操作を最大一回に制限する。"""

    def __init__(self, enabled, length):
        if length < 1:
            raise ValueError("Detection episode length must be at least 1")
        self.enabled = bool(enabled)
        self.length = int(length)
        self._next_episode_id = 0
        self._active = None

    def observe_detection(self, position):
        """(モデル操作を許可するか, episode_id) を返す。"""
        if not self.enabled:
            return True, None
        if (
            self._active is None
            or position >= self._active.start_position + self.length
        ):
            self._active = _ActiveEpisode(
                episode_id=self._next_episode_id,
                start_position=position,
            )
            self._next_episode_id += 1
        return not self._active.operation_performed, self._active.episode_id

    def mark_operation(self):
        """現在のエピソードでモデル操作が行われたことを記録する。"""
        if self.enabled and self._active is not None:
            self._active.operation_performed = True
