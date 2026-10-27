"""datasetごとの観測標本の生成器を作り、概念列からclient別の観測列を供給する。"""

from typing import Protocol

from numpy.random import RandomState

from federated_learning_experiments.data.circle.circle_sample_generation import (
    CircleSampleGenerator,
)
from federated_learning_experiments.data.dataset_definitions import get_dataset_definition
from federated_learning_experiments.data.mnist.fashion_mnist_sample_generation import (
    FashionMnistSampleGenerator,
)
from federated_learning_experiments.data.mnist.mnist_sample_generation import (
    MnistSampleGenerator,
)
from federated_learning_experiments.data.mnist.mnist_training_data import (
    load_fashion_mnist_training_data,
    load_mnist_training_data,
    resolve_fashion_mnist_data_directory,
    resolve_mnist_data_directory,
)
from federated_learning_experiments.data.mnist.rotated_mnist_sample_generation import (
    RotatedMnistSampleGenerator,
)
from federated_learning_experiments.data.observed_streams import (
    ClientConceptTrace,
    ClientObservedStream,
    ObservedSample,
)
from federated_learning_experiments.data.sea.sea_sample_generation import SeaSampleGenerator
from federated_learning_experiments.data.sine.sine_sample_generation import SineSampleGenerator


class ObservedSampleGenerator(Protocol):
    """概念IDから、そのdatasetの観測標本を1つ生成する。借りた乱数を、標本ごとに進める。"""

    def generate_sample(self, *, concept_id: int) -> ObservedSample: ...


# dataset名から、生成器の型（生成と、datasetとの対応の判定が、同じ表を使う）。
_GENERATOR_TYPE_BY_DATASET_NAME: dict[str, type] = {
    "sine2": SineSampleGenerator,
    "sea2": SeaSampleGenerator,
    "sea4": SeaSampleGenerator,
    "circle2": CircleSampleGenerator,
    "mnist2": MnistSampleGenerator,
    "mnist4": MnistSampleGenerator,
    "fmnist2": FashionMnistSampleGenerator,
    "fmnist4": FashionMnistSampleGenerator,
    "rotmnist2": RotatedMnistSampleGenerator,
    "rotmnist4": RotatedMnistSampleGenerator,
}

# 下の関数が返す、生成器の型（受け取る側の、exact型の検査に使う）。
OBSERVED_SAMPLE_GENERATOR_TYPES: tuple[type, ...] = tuple(
    dict.fromkeys(_GENERATOR_TYPE_BY_DATASET_NAME.values())
)


def _get_generator_type(*, dataset_name: str) -> type:
    if dataset_name not in _GENERATOR_TYPE_BY_DATASET_NAME:
        raise ValueError(f"no observed sample generator for dataset {dataset_name!r}")
    return _GENERATOR_TYPE_BY_DATASET_NAME[dataset_name]


def create_observed_sample_generator(
    *, dataset_name: str, numpy_random_generator: RandomState
) -> ObservedSampleGenerator:
    """dataset名と、借りたNumPyの乱数から、そのdatasetの生成器を作る。乱数は進めない。"""
    dataset_definition = get_dataset_definition(dataset_name=dataset_name)
    if type(numpy_random_generator) is not RandomState:
        raise TypeError("numpy_random_generator must be exact numpy.random.RandomState")
    generator_type = _get_generator_type(dataset_name=dataset_name)
    if generator_type is SeaSampleGenerator:
        # SEAは、受け付ける概念IDの範囲を、datasetの概念数で決める。
        return SeaSampleGenerator(
            numpy_random_generator=numpy_random_generator,
            concept_count=dataset_definition.concept_count,
        )
    if generator_type in (
        MnistSampleGenerator,
        FashionMnistSampleGenerator,
        RotatedMnistSampleGenerator,
    ):
        # MNIST形式の画像は、置き場所ごとに学習用データを読み、processの中で使い回す。
        training_data = (
            load_fashion_mnist_training_data(data_directory=resolve_fashion_mnist_data_directory())
            if generator_type is FashionMnistSampleGenerator
            else load_mnist_training_data(data_directory=resolve_mnist_data_directory())
        )
        return generator_type(
            numpy_random_generator=numpy_random_generator,
            concept_count=dataset_definition.concept_count,
            mnist_training_data=training_data,
        )
    if generator_type is CircleSampleGenerator:
        return CircleSampleGenerator(numpy_random_generator=numpy_random_generator)
    if generator_type is SineSampleGenerator:
        return SineSampleGenerator(numpy_random_generator=numpy_random_generator)
    raise ValueError(f"no observed sample generator for dataset {dataset_name!r}")


def is_observed_sample_generator_of_dataset(*, sample_generator: object, dataset_name: str) -> bool:
    """生成器の型と、SEA・MNIST形式の画像なら概念数も、そのdatasetと一致するかを返す。"""
    dataset_definition = get_dataset_definition(dataset_name=dataset_name)
    if type(sample_generator) is not _get_generator_type(dataset_name=dataset_name):
        return False
    if isinstance(sample_generator, (SeaSampleGenerator, MnistSampleGenerator)):
        return sample_generator.concept_count == dataset_definition.concept_count
    return True


def build_client_observed_streams(
    *,
    evaluation_concept_traces: tuple[ClientConceptTrace, ...],
    sample_generator: ObservedSampleGenerator,
) -> tuple[ClientObservedStream, ...]:
    """全概念列を受け取り、client順・標本位置順に観測値を生成する。"""
    observed_client_streams: list[ClientObservedStream] = []
    for client_concept_trace in evaluation_concept_traces:
        observed_samples = tuple(
            sample_generator.generate_sample(concept_id=concept_id)
            for concept_id in client_concept_trace.concept_ids_by_sample_index
        )
        observed_client_streams.append(
            ClientObservedStream(
                client_id=client_concept_trace.client_id,
                observed_samples=observed_samples,
            )
        )
    return tuple(observed_client_streams)
