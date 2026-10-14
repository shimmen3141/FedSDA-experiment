"""保有モデルの学習標本のpopulationから、一回の学習batchを抽出する。

入れ物（ID、標本の列の型、件数、乱数の生成器）は、乱数を進める前に検査する。標本の中身（tensorの種類・形・有限性）は、
抽出した標本だけを検査する（1回の費用を、保持する標本の数に依らないようにするため）。抽出した標本が不正なら、
乱数の生成器の状態を、呼出しの前へ戻して、拒否する。

標本の中身は、観測標本の処理の入口（標本1件の処理と、その最初の予測・損失の評価）が、保持へ入る前に検査している。
ここでの検査は、その後に標本が書き換えられた場合の、使う時点での防御である。
"""

from bisect import insort
from math import floor, log
from random import Random

from torch import Tensor, cat, float32, isfinite, strided

from .local_training_settings import LocalTrainingSettings
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


def _validate_training_sample_selection(
    *, local_training_settings: LocalTrainingSettings | None, batch_sample_count: int
) -> None:
    """抜き出し方の設定を、乱数を進める前に、対象のモデルの有無に依らず確かめる（凍結を迂回した値も拒否する）。"""
    if local_training_settings is None:
        return
    if type(local_training_settings) is not LocalTrainingSettings:
        raise ValueError("local_training_settingsはexact LocalTrainingSettingsかNoneが必要です。")
    policy = local_training_settings.training_sample_selection_policy
    most_recent_count = local_training_settings.most_recent_stored_sample_count
    decay_factor = local_training_settings.stored_sample_recency_decay_factor
    if policy == "uniform_over_most_recent_stored_samples":
        if type(most_recent_count) is not int or most_recent_count < batch_sample_count:
            raise ValueError("直近K件のKは、bool以外のbuiltin intで、batchの件数以上が必要です。")
    elif policy == "recency_decayed_stored_samples":
        if type(decay_factor) is not float or not 0.0 < decay_factor < 1.0:
            raise ValueError("指数減衰のαは、0より大きく1より小さいbuiltin floatが必要です。")
    elif policy != "uniform_over_all_stored_samples":
        raise ValueError(f"未知の抜き出し方です: {policy!r}")


def _sample_recency_decayed_positions(
    *, population_size: int, sample_count: int, decay_factor: float, python_random_generator: Random
) -> list[int]:
    """末尾からa件前（a=0が最新）を、未選択の位置の中でαのa乗に比例する条件付き確率で、重複なしに選ぶ。

    選んだ位置で区切られた区間ごとの重みの和を閉じた式で求め、1件あたり`random()`を1回だけ使う。手数は、
    選んだ件数に比例し、列の長さに依らない。重みは、まだ選んでいない最新の位置を1とする相対値で扱うので、
    αが極端に小さくても、全体の和は0にならない。戻り値は、列の位置（0が最古）を、選んだ順に並べたもの。
    """
    selected_ages: list[int] = []
    selected_positions: list[int] = []
    for _ in range(sample_count):
        # 未選択の範囲を、選んだ位置で区切られた区間[start, end)の列にする（新しい順。長さ0の区間は作らない）。
        intervals: list[tuple[int, int]] = []
        interval_start = 0
        for selected_age in selected_ages:
            if interval_start < selected_age:
                intervals.append((interval_start, selected_age))
            interval_start = selected_age + 1
        if interval_start < population_size:
            intervals.append((interval_start, population_size))
        newest_unselected_age = intervals[0][0]
        interval_weights = []
        for start, end in intervals:
            start_weight = decay_factor ** (start - newest_unselected_age)
            interval_weights.append(
                start_weight * (1.0 - decay_factor ** (end - start)) / (1.0 - decay_factor)
            )
        remaining_weight = python_random_generator.random() * sum(interval_weights)
        chosen_age = None
        for (start, end), interval_weight in zip(intervals, interval_weights, strict=True):
            if interval_weight <= 0.0:
                continue
            if remaining_weight < interval_weight:
                start_weight = decay_factor ** (start - newest_unselected_age)
                inverse_argument = 1.0 - remaining_weight * (1.0 - decay_factor) / start_weight
                distance = (
                    floor(log(inverse_argument) / log(decay_factor))
                    if inverse_argument > 0.0
                    else end - start - 1
                )
                chosen_age = start + min(max(distance, 0), end - start - 1)
                break
            remaining_weight -= interval_weight
        if chosen_age is None:
            # 丸めで、どの区間でも止まらなかった: 和が正の最後の区間の、最後の位置。
            start, end = next(
                interval
                for interval, interval_weight in reversed(
                    tuple(zip(intervals, interval_weights, strict=True))
                )
                if interval_weight > 0.0
            )
            chosen_age = end - 1
        insort(selected_ages, chosen_age)
        selected_positions.append(population_size - 1 - chosen_age)
    return selected_positions


def sample_training_batches_for_held_models(
    *,
    held_model_ids: frozenset[int],
    ordered_model_training_samples: tuple[ModelTrainingSampleCollection, ...],
    batch_sample_count: int,
    python_random_generator: Random,
    local_training_settings: LocalTrainingSettings | None = None,
) -> tuple[SampledModelTrainingBatch, ...]:
    """入れ物の検証後、借用Randomを参加順に一回ずつ進め、抽出した標本の中身を検証する。

    抽出した標本が不正なら、借用Randomの状態を、呼出しの前へ戻して、拒否する。
    `local_training_settings`の抜き出し方（省略か既定なら、全部の標本から一様。今と同じ乱数の使い方）で抜き出す。
    """
    eligible_model_training_samples = _validate_sampling_request(
        held_model_ids=held_model_ids,
        ordered_model_training_samples=ordered_model_training_samples,
        batch_sample_count=batch_sample_count,
        python_random_generator=python_random_generator,
    )
    # 抜き出し方は、乱数を進める前に、対象のモデルの有無に依らず確かめる。
    _validate_training_sample_selection(
        local_training_settings=local_training_settings,
        batch_sample_count=batch_sample_count,
    )
    # 設定の検査で、直近K件ならKだけ、指数減衰ならαだけが値を持つ（全履歴ならどちらもNone）。
    most_recent_count = (
        None
        if local_training_settings is None
        else local_training_settings.most_recent_stored_sample_count
    )
    decay_factor = (
        None
        if local_training_settings is None
        else local_training_settings.stored_sample_recency_decay_factor
    )
    random_state_before_sampling = python_random_generator.getstate()
    sampled_model_training_batches = []
    try:
        for collection_index, model_training_samples in eligible_model_training_samples:
            training_samples = model_training_samples.training_samples
            if most_recent_count is not None:
                # 直近K件（列がK件未満なら、全部）から一様に。
                sampled_training_samples = python_random_generator.sample(
                    training_samples[-most_recent_count:], batch_sample_count
                )
            elif decay_factor is not None:
                sampled_training_samples = [
                    training_samples[position]
                    for position in _sample_recency_decayed_positions(
                        population_size=len(training_samples),
                        sample_count=batch_sample_count,
                        decay_factor=decay_factor,
                        python_random_generator=python_random_generator,
                    )
                ]
            else:
                sampled_training_samples = python_random_generator.sample(
                    training_samples, batch_sample_count
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
