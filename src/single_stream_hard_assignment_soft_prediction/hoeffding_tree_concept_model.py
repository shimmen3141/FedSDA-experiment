"""概念ごとのモデル1つ（RiverのHoeffding tree、既定のハイパーパラメータ）。

確率は、木が予測を持たなければ一様分布、木が返さないクラスは0とし、その上で一様分布と重みwで混ぜて平滑化する
（最大のクラスは変わらない。MDLの対数損失を有限にする）。損失は、監視用が1−（正解クラスの確率）、MDL用が
−ln（正解クラスの確率）。
"""

from copy import deepcopy
from math import isfinite, log

from river.tree import HoeffdingTreeClassifier


def convert_features_to_dictionary(*, row) -> dict[int, float]:
    """特徴の1行を、列の番号をキーとしfloatを値とする辞書にする（比較手法の研究ツールと同じ変換）。"""
    return {index: float(value) for index, value in enumerate(row)}


def _validate_class_count(class_count: object) -> None:
    if type(class_count) is not int:
        raise TypeError("class_countはbool以外のbuiltin intが必要です。")
    if class_count < 2:
        raise ValueError("class_countは2以上が必要です。")


def _validate_smoothing_weight(weight: object) -> None:
    if type(weight) is not float:
        raise TypeError("probability_smoothing_weightはbuiltin floatが必要です。")
    if not isfinite(weight) or not 0.0 < weight < 1.0:
        raise ValueError("probability_smoothing_weightは0より大きく1より小さい値が必要です。")


class HoeffdingTreeConceptModel:
    """Hoeffding treeを1つ持ち、平滑化した確率・損失・1件の学習・写し・空の新しいモデルを提供する。"""

    def __init__(
        self,
        *,
        tree: HoeffdingTreeClassifier,
        class_count: int,
        probability_smoothing_weight: float,
    ) -> None:
        if type(tree) is not HoeffdingTreeClassifier:
            raise TypeError("treeはHoeffdingTreeClassifierが必要です。")
        _validate_class_count(class_count)
        _validate_smoothing_weight(probability_smoothing_weight)
        self.tree = tree
        self.class_count = class_count
        self.probability_smoothing_weight = probability_smoothing_weight

    @classmethod
    def create_empty(
        cls, *, class_count: int, probability_smoothing_weight: float
    ) -> "HoeffdingTreeConceptModel":
        """空の木（既定のハイパーパラメータ）から始める概念モデル。"""
        return cls(
            tree=HoeffdingTreeClassifier(),
            class_count=class_count,
            probability_smoothing_weight=probability_smoothing_weight,
        )

    def create_empty_like(self) -> "HoeffdingTreeConceptModel":
        """同じクラス数・平滑化の、空の新しい概念モデル。"""
        return HoeffdingTreeConceptModel.create_empty(
            class_count=self.class_count,
            probability_smoothing_weight=self.probability_smoothing_weight,
        )

    def copy(self) -> "HoeffdingTreeConceptModel":
        """元と独立に学習できる写し。"""
        return HoeffdingTreeConceptModel(
            tree=deepcopy(self.tree),
            class_count=self.class_count,
            probability_smoothing_weight=self.probability_smoothing_weight,
        )

    @property
    def node_count(self) -> int:
        """木の節点の数（空の木は0）。"""
        node_count = self.tree.n_nodes
        return 0 if node_count is None else int(node_count)

    def predict_class_probabilities(self, *, features: dict[int, float]) -> list[float]:
        """K個のクラスの平滑化した確率（和は1）。"""
        raw = self.tree.predict_proba_one(features)
        class_count = self.class_count
        uniform = 1.0 / class_count
        if not raw:
            base = [uniform] * class_count
        else:
            base = [float(raw.get(class_id, 0.0)) for class_id in range(class_count)]
            total = sum(base)
            base = [uniform] * class_count if total <= 0.0 else [p / total for p in base]
        weight = self.probability_smoothing_weight
        return [(1.0 - weight) * p + weight * uniform for p in base]

    def _validate_class_id(self, observed_class_id: object) -> int:
        if type(observed_class_id) is not int:
            raise TypeError("observed_class_idはbool以外のbuiltin intが必要です。")
        if not 0 <= observed_class_id < self.class_count:
            raise ValueError("observed_class_idは0以上class_count未満が必要です。")
        return observed_class_id

    def monitoring_loss(self, *, features: dict[int, float], observed_class_id: int) -> float:
        """監視用の損失 1−（正解クラスの平滑化した確率）。"""
        class_id = self._validate_class_id(observed_class_id)
        return 1.0 - self.predict_class_probabilities(features=features)[class_id]

    def description_length_loss(
        self, *, features: dict[int, float], observed_class_id: int
    ) -> float:
        """MDL用の損失 −ln（正解クラスの平滑化した確率）（nats）。"""
        class_id = self._validate_class_id(observed_class_id)
        return -log(self.predict_class_probabilities(features=features)[class_id])

    def learn_one(self, *, features: dict[int, float], observed_class_id: int) -> None:
        """1件で1回学習する。"""
        class_id = self._validate_class_id(observed_class_id)
        self.tree.learn_one(features, class_id)
