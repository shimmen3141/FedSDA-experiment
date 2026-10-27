"""公式FedDriftの回転の定義で、MNISTの画素を並べ替える。"""

from numpy import arange, rot90
from numpy.random import RandomState

from federated_learning_experiments.data.mnist.mnist_sample_generation import MnistSampleGenerator
from federated_learning_experiments.data.mnist.mnist_training_data import MnistTrainingData
from federated_learning_experiments.data.observed_streams import ObservedSample

# reshape→rot90→flattenの並べ替えを、標本間で使い回す（公式18e7fbe・62ad4e2）。
_PIXEL_POSITIONS_BY_ROTATION_K: tuple[tuple[int, ...], ...] = tuple(
    tuple(rot90(arange(784).reshape(28, 28), k=rotation_k).flatten().tolist())
    for rotation_k in range(4)
)


class RotatedMnistSampleGenerator(MnistSampleGenerator):
    """画像の選択はMNISTと共通。2概念は0°・180°、4概念は0°・90°・180°・270°。"""

    def __init__(
        self,
        *,
        numpy_random_generator: RandomState,
        concept_count: int,
        mnist_training_data: MnistTrainingData,
    ) -> None:
        super().__init__(
            numpy_random_generator=numpy_random_generator,
            concept_count=concept_count,
            mnist_training_data=mnist_training_data,
        )
        if concept_count not in (2, 4):
            raise ValueError("concept_countには2または4を指定してください。")

    def generate_sample(self, *, concept_id: int) -> ObservedSample:
        """概念の検査の後、元のラベルの標本を選び、画素だけを回す。乱数の消費は1回。"""
        if type(concept_id) is not int:
            raise TypeError("concept_idには整数を指定してください。boolは受理しません。")
        if not 0 <= concept_id < self.concept_count:
            raise ValueError(f"concept_idには0以上{self.concept_count}未満を指定してください。")
        sample = super().generate_sample(concept_id=0)
        if concept_id == 0:
            return sample
        rotation_k = 2 * concept_id if self.concept_count == 2 else concept_id
        return ObservedSample(
            feature_values=tuple(
                map(sample.feature_values.__getitem__, _PIXEL_POSITIONS_BY_ROTATION_K[rotation_k])
            ),
            class_label=sample.class_label,
        )
