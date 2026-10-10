"""モデルのクラスタリングの判定の計算（乱数もownerも使わない）と、判定の基準。"""

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import NormalDist


def _validate_count(*, count: int, count_name: str, minimum_count: int = 0) -> None:
    if type(count) is not int:
        raise TypeError(f"{count_name} must be builtin int")
    if count < minimum_count:
        raise ValueError(f"{count_name} must be at least {minimum_count}")


def _validate_confidence_level(*, confidence_level: float) -> None:
    if type(confidence_level) is not float:
        raise TypeError("confidence_level must be builtin float")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be greater than 0.5 and less than 1")


def _validate_finite_score(*, score: float, score_name: str) -> None:
    if type(score) is not float:
        raise TypeError(f"{score_name} must be builtin float")
    if not isfinite(score):
        raise ValueError(f"{score_name} must be finite")


@dataclass(frozen=True, kw_only=True)
class ModelClusteringCriteria:
    """クラスタリングの判定の基準。生成時に範囲を確かめる。"""

    # 判定の値（の、クラスタ間の平均）が、これ以下のモデルを、同じクラスタにする。
    maximum_same_cluster_decision_score: float
    # モデル対を判定するのに要る、4つの損失の統計それぞれの評価の件数。クラス別の集計を使うのに要る件数でもある。
    minimum_pair_evaluation_sample_count: int
    # 判定の値（信頼下限）の信頼水準。
    clustering_confidence_level: float

    def __post_init__(self) -> None:
        _validate_finite_score(
            score=self.maximum_same_cluster_decision_score,
            score_name="maximum_same_cluster_decision_score",
        )
        _validate_count(
            count=self.minimum_pair_evaluation_sample_count,
            count_name="minimum_pair_evaluation_sample_count",
            minimum_count=1,
        )
        _validate_confidence_level(confidence_level=self.clustering_confidence_level)


def _compute_wilson_lower_bound(
    *, success_count: int, sample_count: int, confidence_level: float
) -> float:
    proportion = success_count / sample_count
    z_value = NormalDist().inv_cdf(confidence_level)
    denominator = 1.0 + z_value * z_value / sample_count
    center = proportion + z_value * z_value / (2.0 * sample_count)
    radius = z_value * sqrt(
        proportion * (1.0 - proportion) / sample_count
        + z_value * z_value / (4.0 * sample_count * sample_count)
    )
    return max((center - radius) / denominator, 0.0)


def _validate_success_and_sample_counts(
    *, success_count: int, sample_count: int, counts_name: str
) -> None:
    _validate_count(count=sample_count, count_name=f"{counts_name} sample count", minimum_count=1)
    _validate_count(count=success_count, count_name=f"{counts_name} success count")
    if success_count > sample_count:
        raise ValueError(f"{counts_name} success count must not exceed the sample count")


def compute_binomial_proportion_lower_confidence_bound(
    *, success_count: int, sample_count: int, confidence_level: float
) -> float:
    """二項比率の、Wilsonの片側信頼下限（0未満は0）。"""
    _validate_success_and_sample_counts(
        success_count=success_count, sample_count=sample_count, counts_name="binomial"
    )
    _validate_confidence_level(confidence_level=confidence_level)
    return _compute_wilson_lower_bound(
        success_count=success_count, sample_count=sample_count, confidence_level=confidence_level
    )


def _validate_unique_correctness_counts(
    *, unique_correctness_counts: tuple[int, ...], counts_name: str
) -> None:
    """（評価した標本数、小さいIDのモデルだけ正解、大きいIDのモデルだけ正解）。標本数は0でもよい。"""
    if type(unique_correctness_counts) is not tuple:
        raise TypeError(f"{counts_name} must be builtin tuple")
    if len(unique_correctness_counts) != 3:
        raise ValueError(f"{counts_name} must hold three counts")
    for count in unique_correctness_counts:
        _validate_count(count=count, count_name=counts_name)
    if max(unique_correctness_counts[1:]) > unique_correctness_counts[0]:
        raise ValueError(f"{counts_name} unique correct counts must not exceed the sample count")


def compute_classwise_unique_correctness_decision_score(
    *,
    overall_unique_correctness_counts: tuple[int, int, int],
    class_unique_correctness_counts: tuple[tuple[int, int, int, int], ...],
    minimum_class_sample_count: int,
    confidence_level: float,
) -> float:
    """モデル対の判定の値: クラス別の「片方だけが正解した割合」の、同時信頼下限の最大。

    数の組は（評価した標本数、小さいIDのモデルだけ正解、大きいIDのモデルだけ正解）。クラス別は、先頭にクラスを足す。
    標本数が下限以上のクラスだけを使い、そのようなクラスがなければ、全体を使う。信頼水準は、
    左右の2方向と、使うクラスの数で補正する（Bonferroni）。値が小さいほど、2つのモデルの正誤が似ている。
    """
    _validate_unique_correctness_counts(
        unique_correctness_counts=overall_unique_correctness_counts,
        counts_name="overall_unique_correctness_counts",
    )
    if type(class_unique_correctness_counts) is not tuple:
        raise TypeError("class_unique_correctness_counts must be builtin tuple")
    observed_class_ids: set[int] = set()
    for class_counts in class_unique_correctness_counts:
        if type(class_counts) is not tuple:
            raise TypeError("class_unique_correctness_counts elements must be builtin tuple")
        if len(class_counts) != 4:
            raise ValueError("class_unique_correctness_counts elements must hold four values")
        if type(class_counts[0]) is not int:
            raise TypeError("class ID must be builtin int")
        if class_counts[0] in observed_class_ids:
            raise ValueError("class_unique_correctness_counts must not hold duplicate classes")
        observed_class_ids.add(class_counts[0])
        _validate_unique_correctness_counts(
            unique_correctness_counts=class_counts[1:],
            counts_name="class_unique_correctness_counts",
        )
    _validate_count(
        count=minimum_class_sample_count, count_name="minimum_class_sample_count", minimum_count=1
    )
    _validate_confidence_level(confidence_level=confidence_level)
    supported_counts = [
        class_counts[1:]
        for class_counts in class_unique_correctness_counts
        if class_counts[1] >= minimum_class_sample_count
    ]
    if not supported_counts:
        if overall_unique_correctness_counts[0] < 1:
            raise ValueError(
                "overall_unique_correctness_counts must hold at least one sample when no class is used"
            )
        supported_counts = [overall_unique_correctness_counts]
    simultaneous_confidence_level = 1.0 - ((1.0 - confidence_level) / (2 * len(supported_counts)))
    return max(
        _compute_wilson_lower_bound(
            success_count=unique_correct_count,
            sample_count=sample_count,
            confidence_level=simultaneous_confidence_level,
        )
        for sample_count, lower_only_correct_count, higher_only_correct_count in supported_counts
        for unique_correct_count in (lower_only_correct_count, higher_only_correct_count)
    )


def cluster_model_ids_by_average_linkage(
    *,
    model_ids: tuple[int, ...],
    decision_scores_by_model_pair: dict[tuple[int, int], float],
    maximum_same_cluster_decision_score: float,
) -> tuple[tuple[int, ...], ...]:
    """判定の値の、クラスタ間の平均（average linkage）で、決定的に逐次統合したクラスタを返す。

    対のキーは（小さいID、大きいID）。値のないモデル対を含むクラスタどうしは、統合の候補にしない。
    平均が最小のクラスタの対を、平均が閾値以下の間、統合する（同じ平均なら、クラスタの並びの小さいほう）。
    戻り値は、クラスタ（モデルIDの昇順）を、並べ直した順（先頭のIDの昇順）に持つ。
    """
    if type(model_ids) is not tuple:
        raise TypeError("model_ids must be builtin tuple")
    if not model_ids:
        raise ValueError("model_ids must not be empty")
    for model_id in model_ids:
        if type(model_id) is not int:
            raise TypeError("model_ids must hold builtin int")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("model_ids must not hold duplicates")
    if type(decision_scores_by_model_pair) is not dict:
        raise TypeError("decision_scores_by_model_pair must be builtin dict")
    for model_pair, decision_score in decision_scores_by_model_pair.items():
        if type(model_pair) is not tuple:
            raise TypeError("decision_scores_by_model_pair keys must be builtin tuple")
        if len(model_pair) != 2 or any(type(model_id) is not int for model_id in model_pair):
            raise ValueError("decision_scores_by_model_pair keys must be pairs of builtin int")
        if not model_pair[0] < model_pair[1]:
            raise ValueError("decision_scores_by_model_pair keys must be (lower ID, higher ID)")
        if model_pair[0] not in model_ids or model_pair[1] not in model_ids:
            raise ValueError("decision_scores_by_model_pair keys must be pairs of model_ids")
        _validate_finite_score(score=decision_score, score_name="decision score")
    _validate_finite_score(
        score=maximum_same_cluster_decision_score,
        score_name="maximum_same_cluster_decision_score",
    )

    model_clusters: list[tuple[int, ...]] = [(model_id,) for model_id in sorted(model_ids)]
    while True:
        best_candidate = None
        for left_position, left_cluster in enumerate(model_clusters):
            for right_position in range(left_position + 1, len(model_clusters)):
                right_cluster = model_clusters[right_position]
                cluster_model_pairs = [
                    (min(left_model_id, right_model_id), max(left_model_id, right_model_id))
                    for left_model_id in left_cluster
                    for right_model_id in right_cluster
                ]
                if any(
                    model_pair not in decision_scores_by_model_pair
                    for model_pair in cluster_model_pairs
                ):
                    continue
                pair_decision_scores = [
                    decision_scores_by_model_pair[model_pair] for model_pair in cluster_model_pairs
                ]
                candidate = (
                    sum(pair_decision_scores) / len(pair_decision_scores),
                    left_cluster,
                    right_cluster,
                    left_position,
                    right_position,
                )
                if best_candidate is None or candidate < best_candidate:
                    best_candidate = candidate
        if best_candidate is None or best_candidate[0] > maximum_same_cluster_decision_score:
            break
        _, left_cluster, right_cluster, left_position, right_position = best_candidate
        model_clusters = [
            model_cluster
            for cluster_position, model_cluster in enumerate(model_clusters)
            if cluster_position not in (left_position, right_position)
        ]
        model_clusters.append(tuple(sorted(left_cluster + right_cluster)))
        model_clusters.sort()
    return tuple(model_clusters)
