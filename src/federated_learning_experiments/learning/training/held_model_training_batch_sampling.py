"""保有モデルの検証済みpopulationから一回の学習batchを抽出する。"""

from random import Random

from torch import Tensor, cat, float32, isfinite, strided

from .model_training_sample_records import (
    ModelTrainingSampleCollection,
    ObservedTrainingSample,
    SampledModelTrainingBatch,
)


def _validate_model_id(*, model_id: int, tensor_name: str) -> None:
    if type(model_id) is not int:
        raise ValueError(f"{tensor_name}はbool以外のbuiltin intが必要です。")


def _validate_observed_training_sample(
    *,
    training_sample: ObservedTrainingSample,
    expected_input_feature_count: int | None,
    tensor_name: str,
) -> int:
    if type(training_sample) is not ObservedTrainingSample:
        raise ValueError(f"{tensor_name}はexact ObservedTrainingSampleが必要です。")
    for training_tensor, tensor_field_name in (
        (training_sample.input_features, "input_features"),
        (training_sample.observed_class_labels, "observed_class_labels"),
    ):
        if not isinstance(training_tensor, Tensor):
            raise ValueError(f"{tensor_name}.{tensor_field_name}はTensorが必要です。")
        if (
            training_tensor.device.type != "cpu"
            or training_tensor.dtype != float32
            or training_tensor.layout != strided
            or training_tensor.is_nested
        ):
            raise ValueError(
                f"{tensor_name}.{tensor_field_name}はCPU float32 strided/notnestedが必要です。"
            )
        if training_tensor.dim() != 2 or training_tensor.shape[0] != 1:
            raise ValueError(f"{tensor_name}.{tensor_field_name}は一標本のrank2/一行が必要です。")
        if tensor_field_name == "input_features":
            if training_tensor.shape[1] < 1:
                raise ValueError(f"{tensor_name}.input_featuresの特徴数は正値が必要です。")
            if (
                expected_input_feature_count is not None
                and training_tensor.shape[1] != expected_input_feature_count
            ):
                raise ValueError(f"{tensor_name}.input_featuresの列内特徴数が一致しません。")
        elif training_tensor.shape[1] != 1:
            raise ValueError(f"{tensor_name}.observed_class_labelsは[1,1]が必要です。")
        if not isfinite(training_tensor).all().item():
            raise ValueError(f"{tensor_name}.{tensor_field_name}は有限値が必要です。")
    return training_sample.input_features.shape[1]


def _validate_sampling_request(
    *,
    held_model_ids: frozenset[int],
    ordered_model_training_samples: tuple[ModelTrainingSampleCollection, ...],
    batch_sample_count: int,
    python_random_generator: Random,
) -> tuple[ModelTrainingSampleCollection, ...]:
    if type(held_model_ids) is not frozenset:
        raise ValueError("held_model_idsはexact frozensetが必要です。")
    if type(ordered_model_training_samples) is not tuple:
        raise ValueError("ordered_model_training_samplesはexact tupleが必要です。")
    if type(batch_sample_count) is not int or batch_sample_count < 1:
        raise ValueError("batch_sample_countはbool以外の正のbuiltin intが必要です。")
    if type(python_random_generator) is not Random:
        raise ValueError("python_random_generatorはexact random.Randomが必要です。")
    for model_id in held_model_ids:
        _validate_model_id(model_id=model_id, tensor_name="held_model_idsの要素")
    seen_model_ids = set()
    eligible_model_training_samples = []
    for collection_index, model_training_samples in enumerate(ordered_model_training_samples):
        tensor_name = f"ordered_model_training_samples[{collection_index}]"
        if type(model_training_samples) is not ModelTrainingSampleCollection:
            raise ValueError(f"{tensor_name}はexact ModelTrainingSampleCollectionが必要です。")
        _validate_model_id(
            model_id=model_training_samples.model_id, tensor_name=f"{tensor_name}.model_id"
        )
        if model_training_samples.model_id in seen_model_ids:
            raise ValueError(f"{tensor_name}.model_idが重複しています。")
        seen_model_ids.add(model_training_samples.model_id)
        if model_training_samples.model_id not in held_model_ids:
            continue
        if type(model_training_samples.training_samples) is not tuple:
            raise ValueError(f"{tensor_name}.training_samplesはexact tupleが必要です。")
        if len(model_training_samples.training_samples) < batch_sample_count:
            continue
        expected_input_feature_count = None
        for training_sample_index, training_sample in enumerate(
            model_training_samples.training_samples
        ):
            expected_input_feature_count = _validate_observed_training_sample(
                training_sample=training_sample,
                expected_input_feature_count=expected_input_feature_count,
                tensor_name=f"{tensor_name}.training_samples[{training_sample_index}]",
            )
        eligible_model_training_samples.append(model_training_samples)
    return tuple(eligible_model_training_samples)


def sample_training_batches_for_held_models(
    *,
    held_model_ids: frozenset[int],
    ordered_model_training_samples: tuple[ModelTrainingSampleCollection, ...],
    batch_sample_count: int,
    python_random_generator: Random,
) -> tuple[SampledModelTrainingBatch, ...]:
    """全候補の検証後、借用Randomを参加順に一回ずつ進める。"""
    eligible_model_training_samples = _validate_sampling_request(
        held_model_ids=held_model_ids,
        ordered_model_training_samples=ordered_model_training_samples,
        batch_sample_count=batch_sample_count,
        python_random_generator=python_random_generator,
    )
    sampled_model_training_batches = []
    for model_training_samples in eligible_model_training_samples:
        sampled_training_samples = python_random_generator.sample(
            model_training_samples.training_samples, batch_sample_count
        )
        sampled_model_training_batches.append(
            SampledModelTrainingBatch(
                model_id=model_training_samples.model_id,
                input_features=cat(
                    [training_sample.input_features for training_sample in sampled_training_samples]
                ),
                observed_class_labels=cat(
                    [
                        training_sample.observed_class_labels
                        for training_sample in sampled_training_samples
                    ]
                ),
            )
        )
    return tuple(sampled_model_training_batches)
