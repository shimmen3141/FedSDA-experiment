"""値のまとめ、同じ概念のAUC、クラスタの割当と真の概念の一致の比較を、旧と同じ式・実旧の関数と照合する。"""

import random
from dataclasses import FrozenInstanceError
from math import inf, isnan, nan
from types import SimpleNamespace

import numpy as np
import pytest

from federated_drift_experiment.servers.clustering import CrossEvaluationClusteringServer
from federated_learning_experiments.evaluation.model_diagnostic_metric_calculations import (
    ClusterAssignmentTrueConceptAgreement,
    ValueDistributionSummary,
    calculate_cluster_assignment_true_concept_agreement,
    calculate_lower_value_same_concept_auc,
    summarize_value_distribution,
)


def summarize_like_legacy(values):
    """旧の`_add_model_diagnostic_results`の`summarize`と同じ式。"""
    array = np.asarray(list(values), dtype=float)
    mean = float(array.mean()) if array.size else 0.0
    return dict(
        total_value=float(array.sum()) if array.size else 0.0,
        mean_value=mean,
        minimum_value=float(array.min()) if array.size else 0.0,
        coefficient_of_variation=float(array.std() / mean) if mean > 0.0 else 0.0,
    )


@pytest.mark.parametrize("random_seed", range(30))
def test_value_distribution_summary_matches_legacy_formula(random_seed):
    """乱数で作った列で、合計・平均・最小・変動係数が、旧と同じ式の値と、`==`で一致する。"""
    random_generator = random.Random(random_seed)
    value_count = random_generator.randrange(0, 9)
    if random_seed % 3 == 0:
        values = tuple(random_generator.randrange(0, 5000) for _ in range(value_count))
    elif random_seed % 3 == 1:
        values = tuple(random_generator.uniform(0.0, 50.0) for _ in range(value_count))
    else:
        # すべて0（平均が0で、変動係数は0）。
        values = (0,) * value_count
    summary = summarize_value_distribution(values=values)
    assert type(summary) is ValueDistributionSummary
    expected = summarize_like_legacy(values)
    for field_name, expected_value in expected.items():
        field_value = getattr(summary, field_name)
        assert type(field_value) is float, field_name
        assert field_value == expected_value, field_name


def test_value_distribution_summary_of_known_values():
    assert summarize_value_distribution(values=()) == ValueDistributionSummary(
        total_value=0.0, mean_value=0.0, minimum_value=0.0, coefficient_of_variation=0.0
    )
    assert summarize_value_distribution(values=(2, 4)) == ValueDistributionSummary(
        total_value=6.0, mean_value=3.0, minimum_value=2.0, coefficient_of_variation=1.0 / 3.0
    )
    summary = summarize_value_distribution(values=(5,))
    assert summary.coefficient_of_variation == 0.0
    with pytest.raises(FrozenInstanceError):
        summary.total_value = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("values", "error_type"),
    [
        ([1, 2], TypeError),
        ((1, True), TypeError),
        ((1, "2"), TypeError),
        ((1, np.int64(2)), TypeError),
        ((1.0, nan), ValueError),
        ((1.0, inf), ValueError),
        ((1, -1), ValueError),
    ],
)
def test_value_distribution_summary_rejects_invalid_values(values, error_type):
    with pytest.raises(error_type):
        summarize_value_distribution(values=values)


def make_legacy_pair_observations(same_concept_values, different_concept_values):
    return [
        SimpleNamespace(oracle_same_concept=True, value=value) for value in same_concept_values
    ] + [
        SimpleNamespace(oracle_same_concept=False, value=value)
        for value in different_concept_values
    ]


@pytest.mark.parametrize("random_seed", range(40))
def test_lower_value_same_concept_auc_matches_legacy_function(random_seed):
    """乱数で作った列（同じ値、有限でない値、空を含む）で、AUCが、実旧の`_lower_score_auc`と一致する。"""
    random_generator = random.Random(random_seed)

    def make_values():
        return tuple(
            random_generator.choice(
                # 等しい値の組と、有限でない値を含める。
                (0.0, 0.5, 1.0, 1.5, 2.0, inf, random_generator.uniform(0.0, 2.0))
            )
            for _ in range(random_generator.randrange(0, 7))
        )

    same_concept_values, different_concept_values = make_values(), make_values()
    legacy_auc = CrossEvaluationClusteringServer._lower_score_auc(
        make_legacy_pair_observations(same_concept_values, different_concept_values),
        lambda observation: observation.value,
    )
    auc = calculate_lower_value_same_concept_auc(
        same_concept_values=same_concept_values,
        different_concept_values=different_concept_values,
    )
    # 旧は、定義できないときNaN。新は、None。
    if isnan(legacy_auc):
        assert auc is None
    else:
        assert type(auc) is float
        assert auc == legacy_auc


def test_lower_value_same_concept_auc_of_known_values():
    # 同じ概念の値が、すべて小さい。
    assert (
        calculate_lower_value_same_concept_auc(
            same_concept_values=(0.1, 0.2), different_concept_values=(0.5, 0.9)
        )
        == 1.0
    )
    # 等しい組は、半分。
    assert (
        calculate_lower_value_same_concept_auc(
            same_concept_values=(1.0,), different_concept_values=(1.0, 2.0)
        )
        == 0.75
    )
    # 有限の値が、片方にない。
    assert (
        calculate_lower_value_same_concept_auc(
            same_concept_values=(inf,), different_concept_values=(1.0,)
        )
        is None
    )
    assert (
        calculate_lower_value_same_concept_auc(
            same_concept_values=(1.0,), different_concept_values=()
        )
        is None
    )


@pytest.mark.parametrize(
    ("same_concept_values", "different_concept_values", "error_type"),
    [
        ([1.0], (1.0,), TypeError),
        ((1.0,), [1.0], TypeError),
        ((1,), (1.0,), TypeError),
        ((1.0,), (True,), TypeError),
        ((nan,), (1.0,), ValueError),
        ((1.0,), (nan,), ValueError),
    ],
)
def test_lower_value_same_concept_auc_rejects_invalid_values(
    same_concept_values, different_concept_values, error_type
):
    with pytest.raises(error_type):
        calculate_lower_value_same_concept_auc(
            same_concept_values=same_concept_values,
            different_concept_values=different_concept_values,
        )


def summarize_agreement_like_legacy(assigned_to_same_cluster_flags, true_concepts_match_flags):
    """旧の`clustering_oracle_diagnostic_summary`と同じ式。"""
    observations = list(zip(assigned_to_same_cluster_flags, true_concepts_match_flags, strict=True))
    tp = sum(same_cluster and same_concept for same_cluster, same_concept in observations)
    fp = sum(same_cluster and not same_concept for same_cluster, same_concept in observations)
    fn = sum(not same_cluster and same_concept for same_cluster, same_concept in observations)
    tn = len(observations) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return ClusterAssignmentTrueConceptAgreement(
        same_cluster_same_concept_pair_count=tp,
        same_cluster_different_concept_pair_count=fp,
        different_cluster_same_concept_pair_count=fn,
        different_cluster_different_concept_pair_count=tn,
        same_cluster_precision=precision,
        same_cluster_recall=recall,
        same_cluster_f1=(
            2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
    )


@pytest.mark.parametrize("random_seed", range(30))
def test_cluster_assignment_agreement_matches_legacy_formula(random_seed):
    random_generator = random.Random(random_seed)
    pair_count = random_generator.randrange(0, 12)
    assigned_to_same_cluster_flags = tuple(
        random_generator.random() < 0.4 for _ in range(pair_count)
    )
    true_concepts_match_flags = tuple(random_generator.random() < 0.5 for _ in range(pair_count))
    agreement = calculate_cluster_assignment_true_concept_agreement(
        assigned_to_same_cluster_flags=assigned_to_same_cluster_flags,
        true_concepts_match_flags=true_concepts_match_flags,
    )
    assert agreement == summarize_agreement_like_legacy(
        assigned_to_same_cluster_flags, true_concepts_match_flags
    )
    for field_name in (
        "same_cluster_same_concept_pair_count",
        "same_cluster_different_concept_pair_count",
        "different_cluster_same_concept_pair_count",
        "different_cluster_different_concept_pair_count",
    ):
        assert type(getattr(agreement, field_name)) is int, field_name
    for field_name in ("same_cluster_precision", "same_cluster_recall", "same_cluster_f1"):
        assert type(getattr(agreement, field_name)) is float, field_name


def test_cluster_assignment_agreement_of_known_values():
    # 4通りが、1・2・3・4件。
    agreement = calculate_cluster_assignment_true_concept_agreement(
        assigned_to_same_cluster_flags=(True,) * 3 + (False,) * 7,
        true_concepts_match_flags=(True, False, False) + (True,) * 3 + (False,) * 4,
    )
    assert agreement == ClusterAssignmentTrueConceptAgreement(
        same_cluster_same_concept_pair_count=1,
        same_cluster_different_concept_pair_count=2,
        different_cluster_same_concept_pair_count=3,
        different_cluster_different_concept_pair_count=4,
        same_cluster_precision=1 / 3,
        same_cluster_recall=1 / 4,
        same_cluster_f1=2.0 * (1 / 3) * (1 / 4) / (1 / 3 + 1 / 4),
    )
    with pytest.raises(FrozenInstanceError):
        agreement.same_cluster_f1 = 0.0  # type: ignore[misc]
    # 対がなければ、すべて0。
    assert calculate_cluster_assignment_true_concept_agreement(
        assigned_to_same_cluster_flags=(), true_concepts_match_flags=()
    ) == ClusterAssignmentTrueConceptAgreement(
        same_cluster_same_concept_pair_count=0,
        same_cluster_different_concept_pair_count=0,
        different_cluster_same_concept_pair_count=0,
        different_cluster_different_concept_pair_count=0,
        same_cluster_precision=0.0,
        same_cluster_recall=0.0,
        same_cluster_f1=0.0,
    )


@pytest.mark.parametrize(
    ("assigned_to_same_cluster_flags", "true_concepts_match_flags", "error_type"),
    [
        ([True], (True,), TypeError),
        ((True,), [True], TypeError),
        ((1,), (True,), TypeError),
        ((True,), (None,), TypeError),
        ((True, False), (True,), ValueError),
    ],
)
def test_cluster_assignment_agreement_rejects_invalid_values(
    assigned_to_same_cluster_flags, true_concepts_match_flags, error_type
):
    with pytest.raises(error_type):
        calculate_cluster_assignment_true_concept_agreement(
            assigned_to_same_cluster_flags=assigned_to_same_cluster_flags,
            true_concepts_match_flags=true_concepts_match_flags,
        )
