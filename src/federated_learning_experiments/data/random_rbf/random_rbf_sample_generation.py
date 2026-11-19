"""借りたNumPy乱数で、Random RBF（randomrbf4）の観測標本を供給する。

概念の定義はRiver 0.27.0の`river.datasets.synth.RandomRBF`（`n_classes=2`、`n_features=10`、`n_centroids=50`）と同じ。
概念cの中心は`seed_model=c`のRiverと同じ値（Pythonの`random.Random(c)`で、Riverの`_generate_centroids`と同じ順序）。
runのseedは中心に使わない。標本はRiverの`_generate_sample`と同じ順序・式で、乱数だけをrunのNumPyの乱数にする
（`random()`→`random_sample()`、`gauss(0, 1)`→`standard_normal()`）。特徴の尺度は変えない。
"""

import math
import random
from dataclasses import dataclass

import numpy as np

from federated_learning_experiments.data.observed_streams import ObservedSample

_MAXIMUM_CONCEPT_COUNT = 4
_CLASS_COUNT = 2
_FEATURE_COUNT = 10
_CENTROID_COUNT = 50


@dataclass(frozen=True, kw_only=True)
class RandomRbfCentroid:
    """1つの中心。Riverの`Centroid`の位置・ラベル・標準偏差と、`centroid_weights`の重みを持つ。"""

    centre: tuple[float, ...]
    class_label: int
    standard_deviation: float
    weight: float


def generate_random_rbf_centroids(*, concept_id: int) -> tuple[RandomRbfCentroid, ...]:
    """概念の番号をseedとして、Riverの`RandomRBF(seed_model=concept_id)`と同じ50個の中心を返す。"""
    if type(concept_id) is not int:
        raise TypeError("concept_idには整数を指定してください。boolは受理しません。")
    if not 0 <= concept_id < _MAXIMUM_CONCEPT_COUNT:
        raise ValueError(f"concept_idには0以上{_MAXIMUM_CONCEPT_COUNT}未満を指定してください。")
    model_random_generator = random.Random(concept_id)
    centroids = []
    for _ in range(_CENTROID_COUNT):
        centre = tuple(model_random_generator.random() for _ in range(_FEATURE_COUNT))
        class_label = model_random_generator.randint(0, _CLASS_COUNT - 1)
        standard_deviation = model_random_generator.random()
        weight = model_random_generator.random()
        centroids.append(
            RandomRbfCentroid(
                centre=centre,
                class_label=class_label,
                standard_deviation=standard_deviation,
                weight=weight,
            )
        )
    return tuple(centroids)


class RandomRbfSampleGenerator:
    """runのRandomStateを保持し、概念ごとの固定の中心から標本を作る。"""

    def __init__(
        self, *, numpy_random_generator: np.random.RandomState, concept_count: int
    ) -> None:
        if type(concept_count) is not int:
            raise TypeError("concept_countには整数を指定してください。boolは受理しません。")
        if not 1 <= concept_count <= _MAXIMUM_CONCEPT_COUNT:
            raise ValueError(
                f"concept_countには1以上{_MAXIMUM_CONCEPT_COUNT}以下を指定してください。"
            )
        self.numpy_random_generator = numpy_random_generator
        self.concept_count = concept_count
        # 中心はrunの乱数を使わずに作る（構築では乱数を進めない）。
        self.centroids_by_concept = tuple(
            generate_random_rbf_centroids(concept_id=concept_id)
            for concept_id in range(concept_count)
        )
        self._weight_sums = tuple(
            sum(centroid.weight for centroid in centroids)
            for centroids in self.centroids_by_concept
        )

    def _choose_centroid(self, *, concept_id: int) -> RandomRbfCentroid:
        """Riverの`random_index_based_on_weights`と同じ: 逐次の累積が値を超えた最初の中心（超えなければ最後）。"""
        centroids = self.centroids_by_concept[concept_id]
        value = float(self.numpy_random_generator.random_sample()) * self._weight_sums[concept_id]
        index = 0
        cumulative_weight = 0.0
        while (cumulative_weight <= value) & (index < len(centroids)):
            cumulative_weight += centroids[index].weight
            index += 1
        return centroids[index - 1]

    def generate_sample(self, *, concept_id: int) -> ObservedSample:
        """指定した概念から、観測標本を1つ生成する。概念を検査してから乱数を使う。"""
        if type(concept_id) is not int:
            raise TypeError("concept_idには整数を指定してください。boolは受理しません。")
        if not 0 <= concept_id < self.concept_count:
            raise ValueError(f"concept_idには0以上{self.concept_count}未満を指定してください。")
        centroid = self._choose_centroid(concept_id=concept_id)
        attribute_values = []
        magnitude = 0.0
        for _ in range(_FEATURE_COUNT):
            attribute_value = (float(self.numpy_random_generator.random_sample()) * 2.0) - 1.0
            attribute_values.append(attribute_value)
            magnitude += attribute_value * attribute_value
        magnitude = math.sqrt(magnitude)
        desired_magnitude = (
            float(self.numpy_random_generator.standard_normal()) * centroid.standard_deviation
        )
        scale = desired_magnitude / magnitude
        float64_feature_values = np.array(
            [
                centroid.centre[feature_index] + attribute_values[feature_index] * scale
                for feature_index in range(_FEATURE_COUNT)
            ],
            dtype=np.float64,
        )
        float32_feature_values = float64_feature_values.astype(np.float32)
        return ObservedSample(
            feature_values=tuple(float(feature_value) for feature_value in float32_feature_values),
            class_label=centroid.class_label,
        )
