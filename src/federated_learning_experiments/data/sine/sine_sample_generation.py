"""借りたNumPy乱数でSINEの観測標本とclient別観測列を供給する。"""

import numpy as np

from federated_learning_experiments.data.observed_streams import (
    ClientConceptTrace,
    ClientObservedStream,
    ObservedSample,
)


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


def build_sine_client_observed_streams(
    *,
    evaluation_concept_traces: tuple[ClientConceptTrace, ...],
    sample_generator: SineSampleGenerator,
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
