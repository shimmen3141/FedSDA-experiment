"""Fashion-MNISTのラベルの入れ替えを、MNISTと同じ規則で生成する。"""

from federated_learning_experiments.data.mnist.mnist_sample_generation import MnistSampleGenerator


class FashionMnistSampleGenerator(MnistSampleGenerator):
    """Fashion-MNISTの学習用データを使う生成器。元の画像が異なるので、MNISTと型で区別する。"""
