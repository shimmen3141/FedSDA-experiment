"""借りたNumPy乱数で、MNIST（mnist2・mnist4）の観測標本を供給する。"""

from numpy import arange, float32, uint8
from numpy.random import RandomState

from federated_learning_experiments.data.mnist.mnist_training_data import MnistTrainingData
from federated_learning_experiments.data.observed_streams import ObservedSample

# 概念の数の上限（概念0は元のラベル、概念kは、数字2k−1と2kを入れ替える）。
_MAXIMUM_CONCEPT_COUNT = 4
# 画素の値（0〜255）ごとの特徴の値。float32で255.0で割った値を、Pythonのfloatにしたもの。
# 標本のtupleは、この256個のfloatを使い回す（標本ごとにfloatを作らない）。
_FEATURE_VALUE_BY_PIXEL_VALUE: tuple[float, ...] = tuple(
    (arange(256, dtype=uint8).astype(float32) / 255.0).tolist()
)


class MnistSampleGenerator:
    """runのRandomStateと、読込み済みの学習用データを保持し、画像を1つ選んで、概念でラベルを入れ替える。"""

    def __init__(
        self,
        *,
        numpy_random_generator: RandomState,
        concept_count: int,
        mnist_training_data: MnistTrainingData,
    ) -> None:
        if type(concept_count) is not int:
            raise TypeError("concept_countには整数を指定してください。boolは受理しません。")
        if not 1 <= concept_count <= _MAXIMUM_CONCEPT_COUNT:
            raise ValueError(
                f"concept_countには1以上{_MAXIMUM_CONCEPT_COUNT}以下を指定してください。"
            )
        if type(mnist_training_data) is not MnistTrainingData:
            raise TypeError("mnist_training_dataにはMnistTrainingDataを指定してください。")
        self.numpy_random_generator = numpy_random_generator
        self.concept_count = concept_count
        self.mnist_training_data = mnist_training_data

    def generate_sample(self, *, concept_id: int) -> ObservedSample:
        """指定した概念から、観測標本を1つ生成する。乱数は、画像の位置の選択に、1回だけ使う。"""
        if type(concept_id) is not int:
            raise TypeError("concept_idには整数を指定してください。boolは受理しません。")
        if not 0 <= concept_id < self.concept_count:
            raise ValueError(f"concept_idには0以上{self.concept_count}未満を指定してください。")
        mnist_training_data = self.mnist_training_data
        image_position = int(
            self.numpy_random_generator.randint(0, len(mnist_training_data.pixel_values), size=1)[0]
        )
        class_label = int(mnist_training_data.digit_labels[image_position])
        if concept_id > 0:
            first_swapped_digit = 2 * concept_id - 1
            second_swapped_digit = first_swapped_digit + 1
            if class_label == first_swapped_digit:
                class_label = second_swapped_digit
            elif class_label == second_swapped_digit:
                class_label = first_swapped_digit
        return ObservedSample(
            feature_values=tuple(
                map(
                    _FEATURE_VALUE_BY_PIXEL_VALUE.__getitem__,
                    mnist_training_data.pixel_values[image_position].tolist(),
                )
            ),
            class_label=class_label,
        )
