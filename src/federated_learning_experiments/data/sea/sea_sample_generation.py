"""借りたNumPy乱数で、SEA（sea2・sea4）の観測標本を供給する。"""

import numpy as np

from federated_learning_experiments.data.observed_streams import ObservedSample

# 概念ごとの閾値（特徴0＋特徴1が、閾値以下ならラベル1）。sea2は、先頭の2つを使う。
_CONCEPT_DECISION_THRESHOLDS = (9.0, 8.0, 7.0, 9.5)
# 各概念に内在する、ラベルを反転する確率。
_LABEL_NOISE_PROBABILITY = 0.10


class SeaSampleGenerator:
    """runのRandomStateを保持し、ラベル判定と雑音の後に、特徴をfloat32へ丸める。"""

    def __init__(
        self, *, numpy_random_generator: np.random.RandomState, concept_count: int
    ) -> None:
        if type(concept_count) is not int:
            raise TypeError("concept_countには整数を指定してください。boolは受理しません。")
        if not 1 <= concept_count <= len(_CONCEPT_DECISION_THRESHOLDS):
            raise ValueError(
                f"concept_countには1以上{len(_CONCEPT_DECISION_THRESHOLDS)}以下を指定してください。"
            )
        self.numpy_random_generator = numpy_random_generator
        self.concept_count = concept_count

    def generate_sample(self, *, concept_id: int) -> ObservedSample:
        """指定した概念から、観測標本を1つ生成する。雑音の乱数は、毎回引く。"""
        if type(concept_id) is not int:
            raise TypeError("concept_idには整数を指定してください。boolは受理しません。")
        if not 0 <= concept_id < self.concept_count:
            raise ValueError(f"concept_idには0以上{self.concept_count}未満を指定してください。")
        float64_feature_values = self.numpy_random_generator.uniform(0.0, 10.0, size=3)
        class_label = int(
            float64_feature_values[0] + float64_feature_values[1]
            <= _CONCEPT_DECISION_THRESHOLDS[concept_id]
        )
        if self.numpy_random_generator.rand() < _LABEL_NOISE_PROBABILITY:
            class_label = 1 - class_label
        float32_feature_values = float64_feature_values.astype(np.float32)
        return ObservedSample(
            feature_values=tuple(float(feature_value) for feature_value in float32_feature_values),
            class_label=class_label,
        )
