"""datasetごとの観測標本の生成器を作り、概念列からclient別の観測列を供給する。"""

from typing import Protocol

from numpy.random import RandomState

from federated_learning_experiments.data.circle.circle_sample_generation import (
    CircleSampleGenerator,
)
from federated_learning_experiments.data.dataset_definitions import get_dataset_definition
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


# 下の関数が返す、生成器の型（受け取る側の、exact型の検査に使う）。
OBSERVED_SAMPLE_GENERATOR_TYPES: tuple[type, ...] = (
    SineSampleGenerator,
    SeaSampleGenerator,
    CircleSampleGenerator,
)


def create_observed_sample_generator(
    *, dataset_name: str, numpy_random_generator: RandomState
) -> ObservedSampleGenerator:
    """dataset名と、借りたNumPyの乱数から、そのdatasetの生成器を作る。乱数は進めない。"""
    dataset_definition = get_dataset_definition(dataset_name=dataset_name)
    if type(numpy_random_generator) is not RandomState:
        raise TypeError("numpy_random_generator must be exact numpy.random.RandomState")
    if dataset_name == "sine2":
        return SineSampleGenerator(numpy_random_generator=numpy_random_generator)
    if dataset_name in ("sea2", "sea4"):
        return SeaSampleGenerator(
            numpy_random_generator=numpy_random_generator,
            concept_count=dataset_definition.concept_count,
        )
    if dataset_name == "circle2":
        return CircleSampleGenerator(numpy_random_generator=numpy_random_generator)
    raise ValueError(f"no observed sample generator for dataset {dataset_name!r}")


def is_observed_sample_generator_of_dataset(*, sample_generator: object, dataset_name: str) -> bool:
    """生成器が、そのdatasetのもの（型が一致し、SEAなら概念数も一致する）かを返す。"""
    dataset_definition = get_dataset_definition(dataset_name=dataset_name)
    if dataset_name == "sine2":
        return type(sample_generator) is SineSampleGenerator
    if dataset_name in ("sea2", "sea4"):
        return (
            isinstance(sample_generator, SeaSampleGenerator)
            and type(sample_generator) is SeaSampleGenerator
            and sample_generator.concept_count == dataset_definition.concept_count
        )
    if dataset_name == "circle2":
        return type(sample_generator) is CircleSampleGenerator
    raise ValueError(f"no observed sample generator for dataset {dataset_name!r}")


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
