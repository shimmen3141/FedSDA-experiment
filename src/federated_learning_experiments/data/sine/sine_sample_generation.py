"""借りたNumPy乱数でSINEの観測標本を供給する。"""

import numpy as np

from federated_learning_experiments.data.observed_streams import ObservedSample


class SineSampleGenerator:
    """runのRandomStateを保持し、ラベル判定後に特徴をfloat32へ丸める。"""

    def __init__(self, *, numpy_random_generator: np.random.RandomState) -> None:
        self.numpy_random_generator = numpy_random_generator

    def generate_sample(self, *, concept_id: int) -> ObservedSample:
        """二つの概念のどちらかから観測標本を一つ生成する。"""
        if type(concept_id) is not int:
            raise TypeError("concept_idには整数の0/1を指定してください。boolは受理しません。")
        if concept_id not in (0, 1):
            raise ValueError("concept_idには0または1を指定してください。")
        float64_feature_values = self.numpy_random_generator.uniform(0.0, 1.0, size=2)
        below_sine_boundary = float64_feature_values[1] <= np.sin(float64_feature_values[0])
        class_label = int(below_sine_boundary) if concept_id == 0 else int(not below_sine_boundary)
        float32_feature_values = float64_feature_values.astype(np.float32)
        return ObservedSample(
            feature_values=(float(float32_feature_values[0]), float(float32_feature_values[1])),
            class_label=class_label,
        )
