"""clientの、標本ごとの保有モデル数の列を持つ。"""


class HeldModelCountRecordStore:
    """標本を処理するたびの、その時点の保有モデル数を、標本の順に所有する。

    合計は、「保有モデル×標本」の延べ数になる（1モデルあたりの計算量の分母）。
    """

    def __init__(self) -> None:
        self._held_model_counts: list[int] = []

    def append_held_model_count(self, *, held_model_count: int) -> None:
        """標本1件ぶんの保有モデル数を、末尾へ足す。clientは、いつも1つ以上のモデルを保有する。"""
        if type(held_model_count) is not int:
            raise TypeError("held_model_count must be builtin int")
        if held_model_count < 1:
            raise ValueError("held_model_count must be positive")
        self._held_model_counts.append(held_model_count)

    def snapshot_held_model_counts(self) -> tuple[int, ...]:
        """標本の順の列（後からの追加で変わらない読取り）。"""
        return tuple(self._held_model_counts)
