"""手法に依存しない、モデル別の診断の計算（値の列のまとめ、同じ概念のAUC、クラスタの割当と真の概念の一致の比較）。"""

from dataclasses import dataclass
from math import isfinite

from numpy import asarray, float64


@dataclass(frozen=True, kw_only=True)
class ValueDistributionSummary:
    """値の列の、合計・平均・最小・変動係数。列が空なら、すべて0。"""

    total_value: float
    mean_value: float
    minimum_value: float
    # 母標準偏差÷平均。平均が0以下なら0。
    coefficient_of_variation: float


@dataclass(frozen=True, kw_only=True)
class ClusterAssignmentTrueConceptAgreement:
    """モデル対を同じクラスタにしたか、と、真の概念が一致するか、の比較。

    適合率は、同じクラスタにした対のうち、真の概念が一致する対の割合。再現率は、真の概念が一致する対のうち、
    同じクラスタにした対の割合。分母が0なら0。
    """

    same_cluster_same_concept_pair_count: int
    same_cluster_different_concept_pair_count: int
    different_cluster_same_concept_pair_count: int
    different_cluster_different_concept_pair_count: int
    same_cluster_precision: float
    same_cluster_recall: float
    same_cluster_f1: float


def summarize_value_distribution(*, values: tuple[int | float, ...]) -> ValueDistributionSummary:
    """0以上の有限の値（builtin intまたはfloat）の列を、まとめる。"""
    if type(values) is not tuple:
        raise TypeError("values must be exact tuple")
    for value in values:
        if type(value) not in (int, float):
            raise TypeError("values must hold builtin int or float")
        if not isfinite(value):
            raise ValueError("values must be finite")
        if value < 0:
            raise ValueError("values must be nonnegative")
    if not values:
        return ValueDistributionSummary(
            total_value=0.0, mean_value=0.0, minimum_value=0.0, coefficient_of_variation=0.0
        )
    value_array = asarray(values, dtype=float64)
    mean_value = float(value_array.mean())
    return ValueDistributionSummary(
        total_value=float(value_array.sum()),
        mean_value=mean_value,
        minimum_value=float(value_array.min()),
        coefficient_of_variation=float(value_array.std() / mean_value) if mean_value > 0.0 else 0.0,
    )


def _select_finite_values(*, values: object, parameter_name: str) -> list[float]:
    if type(values) is not tuple:
        raise TypeError(f"{parameter_name} must be exact tuple")
    for value in values:
        if type(value) is not float:
            raise TypeError(f"{parameter_name} must hold builtin float")
    return [value for value in values if isfinite(value)]


def calculate_lower_value_same_concept_auc(
    *, same_concept_values: tuple[float, ...], different_concept_values: tuple[float, ...]
) -> float | None:
    """小さい値を、同じ概念の証拠とみなしたROC-AUCを返す。

    同じ概念の対の値と、違う概念の対の値の、すべての組のうち、同じ概念の値のほうが小さい組の割合
    （等しい組は、半分と数える）。有限でない値（無限大、NaN）は、数えない。どちらかに、有限の値がなければ、定義できないので、None。
    """
    finite_same_concept_values = _select_finite_values(
        values=same_concept_values, parameter_name="same_concept_values"
    )
    finite_different_concept_values = _select_finite_values(
        values=different_concept_values, parameter_name="different_concept_values"
    )
    if not finite_same_concept_values or not finite_different_concept_values:
        return None
    favorable_pair_count = 0.0
    for same_concept_value in finite_same_concept_values:
        for different_concept_value in finite_different_concept_values:
            favorable_pair_count += float(same_concept_value < different_concept_value)
            favorable_pair_count += 0.5 * float(same_concept_value == different_concept_value)
    return favorable_pair_count / (
        len(finite_same_concept_values) * len(finite_different_concept_values)
    )


def _validate_flags(*, flags: object, parameter_name: str) -> None:
    if type(flags) is not tuple:
        raise TypeError(f"{parameter_name} must be exact tuple")
    for flag in flags:
        if type(flag) is not bool:
            raise TypeError(f"{parameter_name} must hold builtin bool")


def calculate_cluster_assignment_true_concept_agreement(
    *,
    assigned_to_same_cluster_flags: tuple[bool, ...],
    true_concepts_match_flags: tuple[bool, ...],
) -> ClusterAssignmentTrueConceptAgreement:
    """対ごとの、同じクラスタにしたか、と、真の概念が一致するか（同じ長さの列）を、比べる。"""
    _validate_flags(
        flags=assigned_to_same_cluster_flags, parameter_name="assigned_to_same_cluster_flags"
    )
    _validate_flags(flags=true_concepts_match_flags, parameter_name="true_concepts_match_flags")
    if len(assigned_to_same_cluster_flags) != len(true_concepts_match_flags):
        raise ValueError(
            "assigned_to_same_cluster_flags and true_concepts_match_flags must have the same length"
        )
    pair_counts = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for assigned_to_same_cluster, true_concepts_match in zip(
        assigned_to_same_cluster_flags, true_concepts_match_flags, strict=True
    ):
        pair_counts[(assigned_to_same_cluster, true_concepts_match)] += 1
    same_cluster_same_concept_pair_count = pair_counts[(True, True)]
    same_cluster_pair_count = same_cluster_same_concept_pair_count + pair_counts[(True, False)]
    same_concept_pair_count = same_cluster_same_concept_pair_count + pair_counts[(False, True)]
    same_cluster_precision = (
        same_cluster_same_concept_pair_count / same_cluster_pair_count
        if same_cluster_pair_count
        else 0.0
    )
    same_cluster_recall = (
        same_cluster_same_concept_pair_count / same_concept_pair_count
        if same_concept_pair_count
        else 0.0
    )
    return ClusterAssignmentTrueConceptAgreement(
        same_cluster_same_concept_pair_count=same_cluster_same_concept_pair_count,
        same_cluster_different_concept_pair_count=pair_counts[(True, False)],
        different_cluster_same_concept_pair_count=pair_counts[(False, True)],
        different_cluster_different_concept_pair_count=pair_counts[(False, False)],
        same_cluster_precision=same_cluster_precision,
        same_cluster_recall=same_cluster_recall,
        same_cluster_f1=(
            2.0
            * same_cluster_precision
            * same_cluster_recall
            / (same_cluster_precision + same_cluster_recall)
            if same_cluster_precision + same_cluster_recall
            else 0.0
        ),
    )
