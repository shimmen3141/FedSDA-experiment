"""prequentialの指標と概念の同定（設計の「指標」）。riverとtorchをimportしない。"""

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

_CHANGE_WINDOWS = (30, 100, 500)


def _kappa(p0: float, reference: float) -> float:
    return float("nan") if reference >= 1.0 else (p0 - reference) / (1.0 - reference)


def compute_prequential_metrics(
    *,
    predicted_class_ids: np.ndarray,
    observed_class_ids: np.ndarray,
    concept_ids: np.ndarray,
    class_count: int,
) -> dict[str, float]:
    """精度、Kappa、Kappa-M、Kappa-Temporal、真の変化点の後の最初の30・100・500件の精度。"""
    predicted = np.asarray(predicted_class_ids, dtype=np.int64)
    observed = np.asarray(observed_class_ids, dtype=np.int64)
    concepts = np.asarray(concept_ids, dtype=np.int64)
    if (
        not predicted.shape == observed.shape == concepts.shape
        or predicted.ndim != 1
        or not predicted.size
    ):
        raise ValueError("予測・観測・コンセプトは同じ長さの空でない1次元が必要です。")
    count = predicted.size
    correct = predicted == observed
    p0 = float(correct.mean())
    predicted_share = np.bincount(predicted, minlength=class_count) / count
    observed_share = np.bincount(observed, minlength=class_count) / count
    pc = float(np.dot(predicted_share, observed_share))
    # それまでに最も多く観測したクラス（観測が無いとき・同点は小さい番号）を予測する分類器。
    seen = np.zeros(class_count, dtype=np.int64)
    majority_correct = 0
    for label in observed:
        majority_correct += int(int(np.argmax(seen)) == label)
        seen[label] += 1
    # 直前の標本の観測クラス（最初はクラス0）を予測する分類器。
    previous = np.concatenate([[0], observed[:-1]])
    metrics = dict(
        prequential_accuracy=p0,
        kappa=_kappa(p0, pc),
        kappa_m=_kappa(p0, majority_correct / count),
        kappa_temporal=_kappa(p0, float((previous == observed).mean())),
    )
    change_points = [int(i) for i in np.flatnonzero(np.diff(concepts)) + 1]
    boundaries = change_points + [count]
    for window in _CHANGE_WINDOWS:
        hit = total = 0
        for start, end in zip(change_points, boundaries[1:], strict=True):
            stop = min(end, start + window)
            hit += int(correct[start:stop].sum())
            total += stop - start
        metrics[f"accuracy_first_{window}_after_change"] = (
            float("nan") if total == 0 else hit / total
        )
    metrics["true_change_count"] = len(change_points)
    return metrics


def _majority(values: Sequence[int]) -> int | None:
    if not values:
        return None
    counts = Counter(int(v) for v in values)
    best = max(counts.values())
    return min(value for value, c in counts.items() if c == best)


def classify_concept_identification(
    *,
    alarms: Sequence[Mapping[str, object]],
    assigned_model_ids: np.ndarray,
    concept_ids: np.ndarray,
    model_created_sample_index_by_model_id: Mapping[int, int],
) -> dict[str, int]:
    """警報としたものを、正しい再利用・誤った再利用・再来なのに新規作成・初めての概念での新規作成・そのまま、に分類した数。

    各モデルの概念は、最後の時点でそのモデルに割り当てた全標本の最多の真のコンセプト（同点は小さい番号、無ければなし）。
    各警報の新しい概念は、列の末尾k件の最多の真のコンセプト。`new`の比較の対象は、その警報の時点で保有していた
    モデル（その警報より前に作ったモデル。初期のモデルは-1）。
    """
    assigned = np.asarray(assigned_model_ids, dtype=np.int64)
    concepts = np.asarray(concept_ids, dtype=np.int64)
    model_concepts = {
        model_id: _majority(concepts[assigned == model_id].tolist())
        for model_id in model_created_sample_index_by_model_id
    }
    counts = dict(
        correct_reuse=0,
        incorrect_reuse=0,
        new_model_for_recurring_concept=0,
        new_model_for_first_seen_concept=0,
        stay=0,
    )
    for alarm in alarms:
        outcome = alarm["adaptation_outcome"]
        new_concept = _majority(list(alarm["tail_concept_ids"]))
        if outcome == "stay":
            counts["stay"] += 1
        elif outcome == "reuse":
            destination_concept = model_concepts.get(int(alarm["destination_model_id"]))
            key = (
                "correct_reuse"
                if destination_concept is not None and destination_concept == new_concept
                else "incorrect_reuse"
            )
            counts[key] += 1
        elif outcome == "new":
            alarm_index = int(alarm["sample_index"])
            earlier = [
                model_id
                for model_id, created in model_created_sample_index_by_model_id.items()
                if created < alarm_index
            ]
            recurring = any(model_concepts.get(model_id) == new_concept for model_id in earlier)
            counts[
                "new_model_for_recurring_concept"
                if recurring
                else "new_model_for_first_seen_concept"
            ] += 1
    return counts
