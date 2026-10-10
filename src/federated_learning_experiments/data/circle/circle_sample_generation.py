"""借りたNumPy乱数で、CIRCLE-2の観測標本を供給する。"""

import numpy as np

from federated_learning_experiments.data.observed_streams import ObservedSample

# 概念ごとの円（中心のx、中心のy、半径）。円の外ならラベル1。
_CONCEPT_CIRCLES = ((0.2, 0.5, 0.15), (0.6, 0.5, 0.25))


class CircleSampleGenerator:
    """runのRandomStateを保持し、ラベル判定の後に、特徴をfloat32へ丸める。"""

    def __init__(self, *, numpy_random_generator: np.random.RandomState) -> None:
        self.numpy_random_generator = numpy_random_generator

    def generate_sample(self, *, concept_id: int) -> ObservedSample:
        """2つの概念のどちらかから、観測標本を1つ生成する。"""
        if type(concept_id) is not int:
            raise TypeError("concept_idには整数を指定してください。boolは受理しません。")
        if not 0 <= concept_id < len(_CONCEPT_CIRCLES):
            raise ValueError("concept_idには0または1を指定してください。")
        center_x, center_y, radius = _CONCEPT_CIRCLES[concept_id]
        float64_feature_values = self.numpy_random_generator.uniform(0.0, 1.0, size=2)
        signed_squared_distance = (
            (float64_feature_values[0] - center_x) ** 2
            + (float64_feature_values[1] - center_y) ** 2
            - radius**2
        )
        float32_feature_values = float64_feature_values.astype(np.float32)
        return ObservedSample(
            feature_values=tuple(float(feature_value) for feature_value in float32_feature_values),
            class_label=int(signed_squared_distance > 0),
        )
