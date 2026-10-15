"""保有モデルの学習標本のpopulationから、一回の学習batchを抽出する。

入れ物（ID、標本の列の型、件数、乱数の生成器）は、乱数を進める前に検査する。標本の中身（tensorの種類・形・有限性）は、
抽出した標本だけを検査する（1回の費用を、保持する標本の数に依らないようにするため）。抽出した標本が不正なら、
乱数の生成器の状態を、呼出しの前へ戻して、拒否する。

標本の中身は、観測標本の処理の入口（標本1件の処理と、その最初の予測・損失の評価）が、保持へ入る前に検査している。
ここでの検査は、その後に標本が書き換えられた場合の、使う時点での防御である。
"""

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
    """一標本の、型と、tensorの種類・形を検査する（有限性は、連結したbatchで、まとめて確かめる）。"""
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
    return training_sample.input_features.shape[1]


def _validate_sampling_request(
    *,
    held_model_ids: frozenset[int],
    ordered_model_training_samples: tuple[ModelTrainingSampleCollection, ...],
    batch_sample_count: int,
    python_random_generator: Random,
) -> tuple[tuple[int, ModelTrainingSampleCollection], ...]:
    """入れ物を検査し、抽出の対象（列の位置と、保有モデルの、件数の足りる列）を返す。標本の中身は、見ない。"""
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
        eligible_model_training_samples.append((collection_index, model_training_samples))
    return tuple(eligible_model_training_samples)


def sample_training_batches_for_held_models(
    *,
    held_model_ids: frozenset[int],
    ordered_model_training_samples: tuple[ModelTrainingSampleCollection, ...],
    batch_sample_count: int,
    python_random_generator: Random,
) -> tuple[SampledModelTrainingBatch, ...]:
    """入れ物の検証後、借用Randomを参加順に一回ずつ進め、抽出した標本の中身を検証する。

    抽出した標本が不正なら、借用Randomの状態を、呼出しの前へ戻して、拒否する。
    """
    eligible_model_training_samples = _validate_sampling_request(
        held_model_ids=held_model_ids,
        ordered_model_training_samples=ordered_model_training_samples,
        batch_sample_count=batch_sample_count,
        python_random_generator=python_random_generator,
    )
    random_state_before_sampling = python_random_generator.getstate()
    sampled_model_training_batches = []
    try:
        for collection_index, model_training_samples in eligible_model_training_samples:
            sampled_training_samples = python_random_generator.sample(
                model_training_samples.training_samples, batch_sample_count
            )
            expected_input_feature_count = None
            for training_sample in sampled_training_samples:
                expected_input_feature_count = _validate_observed_training_sample(
                    training_sample=training_sample,
                    expected_input_feature_count=expected_input_feature_count,
                    tensor_name=(
                        f"ordered_model_training_samples[{collection_index}]"
                        ".training_samplesから抽出した標本"
                    ),
                )
            sampled_training_batch = SampledModelTrainingBatch(
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
            # 有限性は、抽出した標本を連結したbatchで、1回ずつ確かめる。
            for batch_tensor, tensor_field_name in (
                (sampled_training_batch.input_features, "input_features"),
                (sampled_training_batch.observed_class_labels, "observed_class_labels"),
            ):
                if not isfinite(batch_tensor).all().item():
                    raise ValueError(
                        f"ordered_model_training_samples[{collection_index}]"
                        f".training_samplesから抽出した標本の{tensor_field_name}は有限値が必要です。"
                    )
            sampled_model_training_batches.append(sampled_training_batch)
    except BaseException:
        # 例外の種類に依らず、乱数の状態を、呼出しの前へ戻す。
        python_random_generator.setstate(random_state_before_sampling)
        raise
    return tuple(sampled_model_training_batches)
