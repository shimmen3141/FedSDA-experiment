"""DWM（Kolter & Maloof, JMLR 8, 2007の図1）の投票と、周期の標本での重みの更新の規則。

学習と乱数を持たない純粋な関数。公式の実装（`guml.classifiers.DWM`）と同じく、投票は各専門家が自分の予測クラスへ
重みを足し、最大のクラスを選ぶ（同点はクラスの番号が最小のもの。`Utils.maxIndex`と同じ）。
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class DynamicWeightedMajorityPeriodUpdate:
    """周期の標本での更新の結果。

    `updated_weights`と`kept_expert_positions`は、残す専門家だけを、元の専門家の順で並べる。
    `reweighted_global_prediction_class_id`は、減衰の直後（正規化と削除の前）の重みで投票し直した全体の予測で、
    追加の判断に使う（評価の予測ではない）。
    """

    updated_weights: tuple[float, ...]
    kept_expert_positions: tuple[int, ...]
    reweighted_global_prediction_class_id: int
    new_expert_is_added: bool


def _validate_builtin_int(value: object, *, name: str, minimum: int, maximum: int | None) -> int:
    if type(value) is not int:
        raise TypeError(f"{name}にはbuiltin int（boolを除く）が必要です。")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name}が範囲外です: {value}")
    return value


def _validate_votes(
    *, expert_predicted_class_ids: object, expert_weights: object, class_count: object
) -> int:
    _validate_builtin_int(class_count, name="class_count", minimum=2, maximum=None)
    assert type(class_count) is int
    if type(expert_predicted_class_ids) is not tuple or type(expert_weights) is not tuple:
        raise TypeError("専門家の予測と重みにはtupleが必要です。")
    if not expert_predicted_class_ids or len(expert_predicted_class_ids) != len(expert_weights):
        raise ValueError("専門家の予測と重みは、1つ以上の同じ数が必要です。")
    for class_id in expert_predicted_class_ids:
        _validate_builtin_int(class_id, name="予測クラス", minimum=0, maximum=class_count - 1)
    for weight in expert_weights:
        if type(weight) is not float:
            raise TypeError("重みにはbuiltin floatが必要です。")
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("重みには0以上の有限の値が必要です。")
    return class_count


def _vote(
    expert_predicted_class_ids: tuple[int, ...], expert_weights: tuple[float, ...], class_count: int
) -> int:
    totals = [0.0] * class_count
    for class_id, weight in zip(expert_predicted_class_ids, expert_weights, strict=True):
        totals[class_id] += weight
    # 最初の最大（クラスの番号が最小のもの）。
    return max(range(class_count), key=lambda class_id: (totals[class_id], -class_id))


def vote_weighted_majority(
    *,
    expert_predicted_class_ids: tuple[int, ...],
    expert_weights: tuple[float, ...],
    class_count: int,
) -> int:
    """各専門家の予測クラスへ重みを足し、最大のクラスを返す（同点はクラスの番号が最小）。"""
    class_count = _validate_votes(
        expert_predicted_class_ids=expert_predicted_class_ids,
        expert_weights=expert_weights,
        class_count=class_count,
    )
    return _vote(expert_predicted_class_ids, expert_weights, class_count)


def update_weights_at_period(
    *,
    expert_predicted_class_ids: tuple[int, ...],
    expert_weights: tuple[float, ...],
    observed_class_id: int,
    class_count: int,
    weight_decrease_factor: float,
    removal_weight_threshold: float,
) -> DynamicWeightedMajorityPeriodUpdate:
    """図1の8〜9行目と12〜19行目: 誤った専門家の減衰→投票し直し→最大の重みでの正規化→θ未満の削除→追加の判断。

    最大の重みは正規化で1になり、θは1未満なので、少なくとも1つの専門家が残る。
    """
    class_count = _validate_votes(
        expert_predicted_class_ids=expert_predicted_class_ids,
        expert_weights=expert_weights,
        class_count=class_count,
    )
    _validate_builtin_int(
        observed_class_id, name="observed_class_id", minimum=0, maximum=class_count - 1
    )
    for value, name in (
        (weight_decrease_factor, "weight_decrease_factor"),
        (removal_weight_threshold, "removal_weight_threshold"),
    ):
        if type(value) is not float:
            raise TypeError(f"{name}にはbuiltin floatが必要です。")
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name}は開区間(0, 1)が必要です: {value}")
    if max(expert_weights) <= 0.0:
        raise ValueError("正の重みの専門家が1つ以上必要です。")
    decreased = tuple(
        weight * weight_decrease_factor if class_id != observed_class_id else weight
        for class_id, weight in zip(expert_predicted_class_ids, expert_weights, strict=True)
    )
    reweighted = _vote(expert_predicted_class_ids, decreased, class_count)
    maximum = max(decreased)
    normalized = tuple(weight / maximum for weight in decreased)
    kept = tuple(
        position for position, weight in enumerate(normalized) if weight >= removal_weight_threshold
    )
    return DynamicWeightedMajorityPeriodUpdate(
        updated_weights=tuple(normalized[position] for position in kept),
        kept_expert_positions=kept,
        reweighted_global_prediction_class_id=reweighted,
        new_expert_is_added=reweighted != observed_class_id,
    )
