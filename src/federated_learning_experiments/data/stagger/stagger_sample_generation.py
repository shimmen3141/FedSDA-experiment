"""借りたNumPy乱数で、STAGGER（stagger3）の観測標本を供給する。

概念の定義はRiver 0.27.0の`river.datasets.synth.STAGGER`（`balance_classes=False`）と同じ。
size・color・shapeを、この順に0・1・2から等確率に選び、概念の分類の関数でラベルを決める。
"""

import numpy as np

from federated_learning_experiments.data.observed_streams import ObservedSample

_MAXIMUM_CONCEPT_COUNT = 3
_CATEGORY_VALUES = (0, 1, 2)


def compute_stagger_class_label(*, size: int, color: int, shape: int, concept_id: int) -> int:
    """尺度を写す前のsize・color・shape（0〜2）と概念（0〜2）から、Riverの分類の関数と同じラベルを返す。"""
    for value in (size, color, shape, concept_id):
        if type(value) is not int:
            raise TypeError(
                "size・color・shape・concept_idには整数を指定してください。boolは受理しません。"
            )
    if (
        size not in _CATEGORY_VALUES
        or color not in _CATEGORY_VALUES
        or shape not in _CATEGORY_VALUES
    ):
        raise ValueError("size・color・shapeには0・1・2を指定してください。")
    if concept_id == 0:
        # sizeがsmall（0）かつcolorがred（0）なら1。
        return 1 if (size == 0 and color == 0) else 0
    if concept_id == 1:
        # colorがgreen（2）またはshapeがcircle（0）なら1。
        return 1 if (color == 2 or shape == 0) else 0
    if concept_id == 2:
        # sizeがmedium（1）またはlarge（2）なら1。
        return 1 if (size == 1 or size == 2) else 0
    raise ValueError(f"concept_idには0以上{_MAXIMUM_CONCEPT_COUNT}未満を指定してください。")


class StaggerSampleGenerator:
    """runのRandomStateを保持し、特徴を値/2（0・0.5・1）にして返す。"""

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

    def generate_sample(self, *, concept_id: int) -> ObservedSample:
        """指定した概念から、観測標本を1つ生成する。概念を検査してから乱数を使う。"""
        if type(concept_id) is not int:
            raise TypeError("concept_idには整数を指定してください。boolは受理しません。")
        if not 0 <= concept_id < self.concept_count:
            raise ValueError(f"concept_idには0以上{self.concept_count}未満を指定してください。")
        # Riverの`randint(0, 2)`（両端を含む）を、size・color・shapeの順に1回ずつ。
        size = int(self.numpy_random_generator.randint(0, 3))
        color = int(self.numpy_random_generator.randint(0, 3))
        shape = int(self.numpy_random_generator.randint(0, 3))
        class_label = compute_stagger_class_label(
            size=size, color=color, shape=shape, concept_id=concept_id
        )
        float32_feature_values = (np.array((size, color, shape), dtype=np.float64) / 2.0).astype(
            np.float32
        )
        return ObservedSample(
            feature_values=tuple(float(feature_value) for feature_value in float32_feature_values),
            class_label=class_label,
        )
