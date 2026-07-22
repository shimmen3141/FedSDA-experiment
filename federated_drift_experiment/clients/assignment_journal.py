"""遅延検知後の学習データ再割当を管理する実験用journal。"""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class AssignmentEntry:
    sample_idx: int
    data: tuple
    loss: float
    class_id: int


class RecentAssignmentJournal:
    """確定済みの直近割当を保持し、指定位置以降を再割当可能にする。"""

    def __init__(self, capacity):
        if capacity < 0:
            raise ValueError("journal capacity must be non-negative")
        self._entries = deque(maxlen=capacity)

    def record(self, sample_idx, data, loss, class_id):
        self._entries.append(AssignmentEntry(
            sample_idx=int(sample_idx),
            data=data,
            loss=float(loss),
            class_id=int(class_id),
        ))

    def preview_since(self, estimated_start):
        return [entry.data for entry in self.entries_since(estimated_start)]

    def entries_since(self, estimated_start):
        return [
            entry for entry in self._entries
            if entry.sample_idx >= estimated_start
        ]

    def count_since(self, estimated_start):
        return sum(
            entry.sample_idx >= estimated_start for entry in self._entries
        )

    def reassign_since(self, estimated_start, train_data_store, model_stats):
        """対象データを現在のストア・統計から除き、時系列順に返す。"""
        selected = [
            entry for entry in self._entries
            if entry.sample_idx >= estimated_start
        ]
        for entry in selected:
            self._remove_entry(entry, train_data_store, model_stats)
        return [entry.data for entry in selected]

    def clear(self):
        self._entries.clear()

    @classmethod
    def _remove_entry(cls, entry, train_data_store, model_stats):
        for model_id, data_store in train_data_store.items():
            position = next(
                (index for index, item in enumerate(data_store)
                 if item is entry.data),
                None,
            )
            if position is None:
                continue
            del data_store[position]
            cls._remove_model_stat(
                model_stats, model_id, entry.loss, entry.class_id
            )
            return

    @classmethod
    def _remove_model_stat(cls, model_stats, model_id, value, class_id):
        stats = model_stats.get(model_id)
        if not stats or stats.get("n", 0) == 0:
            return
        cls._remove_running_stat(stats, value)
        class_stats = stats.get("class_stats", {})
        per_class = class_stats.get(class_id)
        if per_class is not None:
            cls._remove_running_stat(per_class, value)
            if per_class["n"] == 0:
                del class_stats[class_id]

    @staticmethod
    def _remove_running_stat(stats, value):
        """Welford統計から、過去に追加した1標本を取り除く。"""
        count = stats.get("n", 0)
        if count <= 1:
            stats.update({"n": 0, "mean": 0.0, "M2": 0.0})
            return
        old_mean = stats["mean"]
        new_count = count - 1
        new_mean = (count * old_mean - value) / new_count
        new_m2 = stats["M2"] - (value - old_mean) * (value - new_mean)
        stats.update({
            "n": new_count,
            "mean": new_mean,
            "M2": max(0.0, new_m2),
        })
