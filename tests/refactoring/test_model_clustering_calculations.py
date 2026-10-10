"""クラスタリングの判定の計算を、実旧の対応する関数（Wilsonの下限、クラス別の同時信頼下限、average linkage）と照合する。"""

import itertools
import math
import random

import pytest
from test_held_candidate_validation_progress import make_subclass_copy

from federated_drift_experiment.clustering import (
    FunctionalPairStats,
    binomial_proportion_lower_bound,
    cluster_models,
)
from federated_learning_experiments.methods.fedsda.consolidation.model_clustering_calculations import (
    ModelClusteringCriteria,
    cluster_model_ids_by_average_linkage,
    compute_binomial_proportion_lower_confidence_bound,
    compute_classwise_unique_correctness_decision_score,
)

CONFIDENCE_LEVELS = (0.95, 0.6, 0.999, 0.5000001, 0.9875)


@pytest.mark.parametrize("confidence_level", CONFIDENCE_LEVELS)
def test_binomial_lower_confidence_bound_matches_real_legacy(confidence_level):
    for sample_count in (1, 2, 3, 5, 12, 50, 1000):
        for success_count in sorted({0, 1, sample_count // 3, sample_count - 1, sample_count}):
            lower_bound = compute_binomial_proportion_lower_confidence_bound(
                success_count=success_count,
                sample_count=sample_count,
                confidence_level=confidence_level,
            )
            assert type(lower_bound) is float
            assert lower_bound == binomial_proportion_lower_bound(
                success_count, sample_count, confidence_level
            ), (success_count, sample_count)
            # 下限は、0以上で、標本の比率を（丸めの分を除いて）超えない。
            assert 0.0 <= lower_bound <= success_count / sample_count + 1e-12


@pytest.mark.parametrize(
    "invalid_arguments,expected_exception",
    [
        (dict(success_count=True), TypeError),
        (dict(success_count=1.0), TypeError),
        (dict(success_count=-1), ValueError),
        (dict(success_count=6), ValueError),
        (dict(sample_count=5.0), TypeError),
        (dict(sample_count=0, success_count=0), ValueError),
        (dict(confidence_level=1), TypeError),
        (dict(confidence_level="0.95"), TypeError),
        (dict(confidence_level=0.5), ValueError),
        (dict(confidence_level=1.0), ValueError),
        (dict(confidence_level=math.nan), ValueError),
    ],
)
def test_binomial_lower_confidence_bound_rejects_invalid_arguments(
    invalid_arguments, expected_exception
):
    with pytest.raises(expected_exception):
        compute_binomial_proportion_lower_confidence_bound(
            **dict(success_count=2, sample_count=5, confidence_level=0.95) | invalid_arguments
        )


def make_legacy_pair_statistics(overall_counts, class_counts):
    legacy_statistics = FunctionalPairStats()
    legacy_statistics.add(*overall_counts)
    for class_id, *counts in class_counts:
        legacy_statistics.add_class(class_id, *counts)
    return legacy_statistics


# (全体の数の組, クラス別の数の組)。数の組は（評価した標本数、小さいIDだけ正解、大きいIDだけ正解）。
UNIQUE_CORRECTNESS_CASES = [
    ((20, 0, 0), ((0, 12, 0, 0), (1, 8, 0, 0))),
    ((20, 3, 1), ((0, 12, 3, 0), (1, 8, 0, 1))),
    # クラスの一部が下限未満（下限以上のクラスだけを使う）。
    ((20, 3, 4), ((0, 16, 1, 4), (1, 4, 2, 0))),
    # 全クラスが下限未満（全体を使う）。
    ((8, 2, 1), ((0, 4, 2, 0), (1, 4, 0, 1))),
    # クラス別がない（全体を使う）。
    ((30, 30, 0), ()),
    ((5, 0, 5), ()),
    # 多クラス。初めて現れた順は、昇順とは限らない。
    ((40, 6, 9), ((3, 10, 1, 4), (0, 10, 5, 0), (2, 15, 0, 5), (1, 5, 0, 0))),
    ((7, 7, 0), ((2, 7, 7, 0),)),
]


@pytest.mark.parametrize("overall_counts,class_counts", UNIQUE_CORRECTNESS_CASES)
@pytest.mark.parametrize("confidence_level", CONFIDENCE_LEVELS)
@pytest.mark.parametrize("minimum_class_sample_count", [1, 5, 11])
def test_classwise_decision_score_matches_real_legacy(
    overall_counts, class_counts, confidence_level, minimum_class_sample_count
):
    decision_score = compute_classwise_unique_correctness_decision_score(
        overall_unique_correctness_counts=overall_counts,
        class_unique_correctness_counts=class_counts,
        minimum_class_sample_count=minimum_class_sample_count,
        confidence_level=confidence_level,
    )
    assert type(decision_score) is float
    assert decision_score == make_legacy_pair_statistics(
        overall_counts, class_counts
    ).class_conditional_confidence_distance(minimum_class_sample_count, confidence_level)


@pytest.mark.parametrize(
    "invalid_arguments,expected_exception",
    [
        (dict(overall_unique_correctness_counts=[20, 3, 1]), TypeError),
        (dict(overall_unique_correctness_counts=(20, 3)), ValueError),
        (dict(overall_unique_correctness_counts=(20, True, 1)), TypeError),
        (dict(overall_unique_correctness_counts=(20, -1, 1)), ValueError),
        (dict(overall_unique_correctness_counts=(20, 21, 0)), ValueError),
        # 全体を使うことになるのに、全体の標本数が0。
        (
            dict(overall_unique_correctness_counts=(0, 0, 0), class_unique_correctness_counts=()),
            ValueError,
        ),
        (dict(class_unique_correctness_counts=[(0, 12, 3, 0)]), TypeError),
        (dict(class_unique_correctness_counts=([0, 12, 3, 0],)), TypeError),
        (dict(class_unique_correctness_counts=((0, 12, 3),)), ValueError),
        (dict(class_unique_correctness_counts=((0.0, 12, 3, 0),)), TypeError),
        (dict(class_unique_correctness_counts=((0, 12, 13, 0),)), ValueError),
        (dict(class_unique_correctness_counts=((0, 12, 3, 0), (0, 8, 0, 1))), ValueError),
        (dict(minimum_class_sample_count=5.0), TypeError),
        (dict(minimum_class_sample_count=0), ValueError),
        (dict(confidence_level=0.5), ValueError),
        (dict(confidence_level=None), TypeError),
    ],
)
def test_classwise_decision_score_rejects_invalid_arguments(invalid_arguments, expected_exception):
    with pytest.raises(expected_exception):
        compute_classwise_unique_correctness_decision_score(
            **dict(
                overall_unique_correctness_counts=(20, 3, 1),
                class_unique_correctness_counts=((0, 12, 3, 0), (1, 8, 0, 1)),
                minimum_class_sample_count=5,
                confidence_level=0.95,
            )
            | invalid_arguments
        )


def make_pair_scores(model_ids, score_values, missing_pairs=()):
    """モデルIDの昇順の、全部の対（小さいID、大きいID）へ、値を順に割り当てる。欠けさせる対は除く。"""
    pairs = list(itertools.combinations(sorted(model_ids), 2))
    return {
        pair: float(score_value)
        for pair, score_value in zip(pairs, score_values, strict=True)
        if pair not in missing_pairs
    }


# (モデルIDの列, 対ごとの判定の値, 閾値)。
AVERAGE_LINKAGE_CASES = [
    ((0,), {}, 0.1),
    ((0, 1), {}, 0.1),
    ((0, 1), {(0, 1): 0.1}, 0.1),
    ((0, 1), {(0, 1): 0.1000001}, 0.1),
    ((3, 1, 2), make_pair_scores((1, 2, 3), (0.874, 0.0, 0.874)), 0.1),
    # 3つが1つへ集まる（最初の統合の後、平均が閾値以下）。
    ((1, 2, 3), make_pair_scores((1, 2, 3), (0.0, 0.005, 0.006)), 0.1),
    # 最初の統合の後、平均が閾値を超える。
    ((1, 2, 3), make_pair_scores((1, 2, 3), (0.0, 0.3, 0.05)), 0.1),
    # 同じ値の対が複数ある（クラスタの並びで決まる）。
    ((0, 1, 2, 3), make_pair_scores((0, 1, 2, 3), (0.05,) * 6), 0.1),
    ((0, 1, 2, 3), make_pair_scores((0, 1, 2, 3), (0.05, 0.9, 0.9, 0.9, 0.9, 0.05)), 0.1),
    # 値のない対があるクラスタどうしは、統合の候補にならない。
    ((0, 1, 2), make_pair_scores((0, 1, 2), (0.0, 0.0, 0.0), missing_pairs={(0, 2)}), 0.1),
    ((0, 1, 2, 3), make_pair_scores((0, 1, 2, 3), (0.0,) * 6, missing_pairs={(0, 1), (0, 2)}), 0.1),
    (
        (5, 0, 9, 2, 7),
        make_pair_scores((0, 2, 5, 7, 9), (0.02, 0.5, 0.08, 0.9, 0.5, 0.01, 0.7, 0.4, 0.03, 0.6)),
        0.1,
    ),
    ((0, 1, 2), make_pair_scores((0, 1, 2), (-0.5, 0.0, 0.2)), 0.0),
]


@pytest.mark.parametrize("model_ids,decision_scores,threshold", AVERAGE_LINKAGE_CASES)
def test_average_linkage_clusters_match_real_legacy(model_ids, decision_scores, threshold):
    model_clusters = cluster_model_ids_by_average_linkage(
        model_ids=model_ids,
        decision_scores_by_model_pair=decision_scores,
        maximum_same_cluster_decision_score=threshold,
    )
    assert type(model_clusters) is tuple
    assert all(type(model_cluster) is tuple for model_cluster in model_clusters)
    assert [list(model_cluster) for model_cluster in model_clusters] == cluster_models(
        list(model_ids), dict(decision_scores), threshold, "average"
    )
    # 全モデルが、ちょうど1つのクラスタに入る。
    assert sorted(itertools.chain.from_iterable(model_clusters)) == sorted(model_ids)


def test_average_linkage_clusters_match_real_legacy_for_generated_scores():
    """決まった擬似乱数で作った値（一部の対を欠けさせる）でも、実旧と同じクラスタになる。"""
    score_generator = random.Random(53)
    merged_cluster_observed = False
    for _ in range(300):
        model_ids = tuple(score_generator.sample(range(12), score_generator.randint(2, 7)))
        decision_scores = {
            pair: round(score_generator.choice((0.0, 0.05, 0.1, 0.3)) * score_generator.random(), 3)
            for pair in itertools.combinations(sorted(model_ids), 2)
            if score_generator.random() < 0.85
        }
        threshold = score_generator.choice((0.0, 0.02, 0.1))
        model_clusters = cluster_model_ids_by_average_linkage(
            model_ids=model_ids,
            decision_scores_by_model_pair=decision_scores,
            maximum_same_cluster_decision_score=threshold,
        )
        assert [list(model_cluster) for model_cluster in model_clusters] == cluster_models(
            list(model_ids), dict(decision_scores), threshold, "average"
        )
        merged_cluster_observed |= any(len(model_cluster) >= 3 for model_cluster in model_clusters)
    assert merged_cluster_observed


@pytest.mark.parametrize(
    "invalid_arguments,expected_exception",
    [
        (dict(model_ids=[0, 1, 2]), TypeError),
        (dict(model_ids=(0, True, 2)), TypeError),
        (dict(model_ids=(0, 1, 1)), ValueError),
        (dict(model_ids=()), ValueError),
        (dict(decision_scores_by_model_pair=[((0, 1), 0.0)]), TypeError),
        (dict(decision_scores_by_model_pair={(1, 0): 0.0}), ValueError),
        (dict(decision_scores_by_model_pair={(0, 0): 0.0}), ValueError),
        (dict(decision_scores_by_model_pair={(0, 9): 0.0}), ValueError),
        (dict(decision_scores_by_model_pair={(0, 1, 2): 0.0}), ValueError),
        (dict(decision_scores_by_model_pair={2: 0.0}), TypeError),
        (dict(decision_scores_by_model_pair={(0, 1): 0}), TypeError),
        (dict(decision_scores_by_model_pair={(0, 1): math.nan}), ValueError),
        (dict(decision_scores_by_model_pair={(0, 1): math.inf}), ValueError),
        (dict(maximum_same_cluster_decision_score=1), TypeError),
        (dict(maximum_same_cluster_decision_score=math.nan), ValueError),
    ],
)
def test_average_linkage_rejects_invalid_arguments(invalid_arguments, expected_exception):
    with pytest.raises(expected_exception):
        cluster_model_ids_by_average_linkage(
            **dict(
                model_ids=(0, 1, 2),
                decision_scores_by_model_pair={(0, 1): 0.0, (1, 2): 0.5},
                maximum_same_cluster_decision_score=0.1,
            )
            | invalid_arguments
        )


def test_clustering_criteria_validate_values():
    criteria = ModelClusteringCriteria(
        maximum_same_cluster_decision_score=0.1,
        minimum_pair_evaluation_sample_count=5,
        clustering_confidence_level=0.95,
    )
    assert criteria.maximum_same_cluster_decision_score == 0.1
    for invalid_fields, expected_exception in (
        (dict(maximum_same_cluster_decision_score=1), TypeError),
        (dict(maximum_same_cluster_decision_score=math.inf), ValueError),
        (dict(maximum_same_cluster_decision_score=math.nan), ValueError),
        (dict(minimum_pair_evaluation_sample_count=5.0), TypeError),
        (dict(minimum_pair_evaluation_sample_count=True), TypeError),
        (dict(minimum_pair_evaluation_sample_count=0), ValueError),
        (dict(clustering_confidence_level=1), TypeError),
        (dict(clustering_confidence_level=0.5), ValueError),
        (dict(clustering_confidence_level=1.0), ValueError),
    ):
        with pytest.raises(expected_exception):
            ModelClusteringCriteria(
                **dict(
                    maximum_same_cluster_decision_score=0.1,
                    minimum_pair_evaluation_sample_count=5,
                    clustering_confidence_level=0.95,
                )
                | invalid_fields
            )
    # 派生型の値と区別できる（利用側は、exact型を確かめる）。
    assert type(make_subclass_copy(criteria)) is not ModelClusteringCriteria
