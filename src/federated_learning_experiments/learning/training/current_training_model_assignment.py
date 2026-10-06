"""現在の単一学習帰属IDを保持し、実変更だけを呼出し側へ返す。"""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class TrainingModelAssignmentChange:
    """変更理由や位置を持たない、変更前後のID記録。"""

    previous_model_id: int
    current_model_id: int


def _validate_model_id(*, model_id: int) -> None:
    if type(model_id) is not int:
        raise TypeError("モデルIDはbool・派生型以外のbuiltin intが必要です。")


class CurrentTrainingModelAssignment:
    """モデル選択や通知を行わず、上位が選択した帰属IDを所有する。"""

    def __init__(self, *, initial_model_id: int) -> None:
        _validate_model_id(model_id=initial_model_id)
        self._current_training_model_id = initial_model_id

    @property
    def current_training_model_id(self) -> int:
        return self._current_training_model_id

    def assign_model_for_training(self, *, model_id: int) -> TrainingModelAssignmentChange | None:
        _validate_model_id(model_id=model_id)
        if model_id == self._current_training_model_id:
            return None
        assignment_change = TrainingModelAssignmentChange(
            previous_model_id=self._current_training_model_id, current_model_id=model_id
        )
        self._current_training_model_id = model_id
        return assignment_change

    def remap_current_training_model_id(
        self, *, model_id_mapping: dict[int, int]
    ) -> TrainingModelAssignmentChange | None:
        """全項目検証後、一段だけ適用する。入力対応表は保持しない。"""
        if type(model_id_mapping) is not dict:
            raise TypeError("model_id_mappingはbuiltin dictが必要です。")
        for original_model_id, receiving_model_id in model_id_mapping.items():
            _validate_model_id(model_id=original_model_id)
            _validate_model_id(model_id=receiving_model_id)
        return self.assign_model_for_training(
            model_id=model_id_mapping.get(
                self._current_training_model_id, self._current_training_model_id
            )
        )
