"""ドリフト検出後のモデル操作を表す構造化イベント。"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AdaptationEvent:
    """検出を契機とした判断と、その結果のモデル操作を記録する。"""

    position: int
    detector: str
    action: str
    old_model_id: int
    new_model_id: int
    estimated_change_point: Optional[int] = None
    episode_id: Optional[int] = None
