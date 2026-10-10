"""区間ごとの正解の割合と、クラスの間のまとめを、実旧の関数・旧と同じ式と照合する。"""

import random
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from federated_drift_experiment import experiment as legacy_experiment
from federated_learning_experiments.evaluation.prediction_diagnostic_metric_calculations import (
    ClassMacroPredictionMetrics,
    PredictionCorrectCounts,
    StableAndRecoveryPeriodAccuracies,
    calculate_stable_and_recovery_period_accuracies,
    summarize_class_prediction_metrics,
)

PERIOD_CASES = [
    (random_seed, recovery_window_sample_count, use_inclusion)
    for random_seed in range(12)
    for recovery_window_sample_count in (0, 3, 15)
    for use_inclusion in (False, True)
]


@pytest.mark.parametrize(
    ("random_seed", "recovery_window_sample_count", "use_inclusion"), PERIOD_CASES
)
def test_period_accuracies_match_legacy_function(
    random_seed, recovery_window_sample_count, use_inclusion
):
    """乱数で作った列（変更位置が、標本の列より後にある場合、区間が重なる場合を含む）で、実旧と一致する。"""
    random_generator = random.Random(random_seed)
    client_count = random_generator.randrange(1, 4)
    prediction_correctness_by_client = []
    concept_change_sample_indices_by_client = []
    sample_inclusion_by_client = []
    for _ in range(client_count):
        sample_count = random_generator.randrange(0, 40)
        prediction_correctness_by_client.append(
            tuple(random_generator.random() < 0.6 for _ in range(sample_count))
        )
        # 変更位置は、標本の列の範囲の外（処理されなかった末尾）にもある。
        concept_change_sample_indices_by_client.append(
            tuple(
                sorted(
                    random_generator.randrange(0, 50)
                    for _ in range(random_generator.randrange(0, 5))
                )
            )
        )
        sample_inclusion_by_client.append(
            tuple(random_generator.random() < 0.7 for _ in range(sample_count))
        )
    legacy_accuracies = legacy_experiment._routing_window_accuracies(
        [list(map(int, correctness)) for correctness in prediction_correctness_by_client],
        {
            client_id: list(sample_indices)
            for client_id, sample_indices in enumerate(concept_change_sample_indices_by_client)
        },
        recovery_window_sample_count,
        observation_masks=(
            [list(inclusion) for inclusion in sample_inclusion_by_client] if use_inclusion else None
        ),
    )
    accuracies = calculate_stable_and_recovery_period_accuracies(
        prediction_correctness_by_client=tuple(prediction_correctness_by_client),
        concept_change_sample_indices_by_client=tuple(concept_change_sample_indices_by_client),
        recovery_window_sample_count=recovery_window_sample_count,
        sample_inclusion_by_client=(tuple(sample_inclusion_by_client) if use_inclusion else None),
    )
    assert type(accuracies) is StableAndRecoveryPeriodAccuracies
    assert type(accuracies.stable_period_accuracy) is float
    assert type(accuracies.recovery_period_accuracy) is float
    assert accuracies.stable_period_accuracy == legacy_accuracies["stable_accuracy"]
    assert accuracies.recovery_period_accuracy == legacy_accuracies["recovery_accuracy"]


def test_period_accuracies_of_known_values():
    # 変更位置2、窓3: 位置2・3・4が、回復の区間。
    accuracies = calculate_stable_and_recovery_period_accuracies(
        prediction_correctness_by_client=((True, False, True, True, False, True, True),),
        concept_change_sample_indices_by_client=((2,),),
        recovery_window_sample_count=3,
    )
    assert accuracies == StableAndRecoveryPeriodAccuracies(
        stable_period_accuracy=3 / 4, recovery_period_accuracy=2 / 3
    )
    with pytest.raises(FrozenInstanceError):
        accuracies.stable_period_accuracy = 0.0  # type: ignore[misc]
    # 数える標本を与えると、その標本だけを使う（位置3と、位置0・6を外す）。
    assert calculate_stable_and_recovery_period_accuracies(
        prediction_correctness_by_client=((True, False, True, True, False, True, True),),
        concept_change_sample_indices_by_client=((2,),),
        recovery_window_sample_count=3,
        sample_inclusion_by_client=((False, True, True, False, True, True, False),),
    ) == StableAndRecoveryPeriodAccuracies(
        stable_period_accuracy=1 / 2, recovery_period_accuracy=1 / 2
    )
    # その区間に標本がなければ、0。
    assert calculate_stable_and_recovery_period_accuracies(
        prediction_correctness_by_client=((True, True),),
        concept_change_sample_indices_by_client=((),),
        recovery_window_sample_count=3,
    ) == StableAndRecoveryPeriodAccuracies(stable_period_accuracy=1.0, recovery_period_accuracy=0.0)
    assert calculate_stable_and_recovery_period_accuracies(
        prediction_correctness_by_client=(),
        concept_change_sample_indices_by_client=(),
        recovery_window_sample_count=3,
    ) == StableAndRecoveryPeriodAccuracies(stable_period_accuracy=0.0, recovery_period_accuracy=0.0)


@pytest.mark.parametrize(
    ("argument_name", "invalid_value", "error_type"),
    [
        ("prediction_correctness_by_client", [(True,)], TypeError),
        ("prediction_correctness_by_client", ([True],), TypeError),
        ("prediction_correctness_by_client", ((1,),), TypeError),
        ("prediction_correctness_by_client", ((True,), (True,)), ValueError),
        ("concept_change_sample_indices_by_client", [(0,)], TypeError),
        ("concept_change_sample_indices_by_client", ([0],), TypeError),
        ("concept_change_sample_indices_by_client", ((True,),), TypeError),
        ("concept_change_sample_indices_by_client", ((-1,),), ValueError),
        ("recovery_window_sample_count", 1.0, TypeError),
        ("recovery_window_sample_count", True, TypeError),
        ("recovery_window_sample_count", -1, ValueError),
        ("sample_inclusion_by_client", [(True,)], TypeError),
        ("sample_inclusion_by_client", ((1,),), TypeError),
        ("sample_inclusion_by_client", ((True, True),), ValueError),
        ("sample_inclusion_by_client", ((True,), (True,)), ValueError),
    ],
)
def test_period_accuracies_reject_invalid_arguments(argument_name, invalid_value, error_type):
    arguments = dict(
        prediction_correctness_by_client=((True,),),
        concept_change_sample_indices_by_client=((0,),),
        recovery_window_sample_count=1,
        sample_inclusion_by_client=((True,),),
    )
    arguments[argument_name] = invalid_value
    with pytest.raises(error_type):
        calculate_stable_and_recovery_period_accuracies(**arguments)


def make_counts(sample_count, any_correct_count, combined, maximum_weight, highest_confidence):
    return PredictionCorrectCounts(
        predicted_sample_count=sample_count,
        combined_prediction_correct_count=combined,
        any_model_or_combined_prediction_correct_count=any_correct_count,
        maximum_weight_model_prediction_correct_count=maximum_weight,
        highest_confidence_model_prediction_correct_count=highest_confidence,
    )


def summarize_classes_like_legacy(class_prediction_correct_counts):
    """旧の`_add_model_diagnostic_results`の、クラスごとの集計と同じ式。"""
    oracle_accuracies, mixture_accuracies, leader_accuracies = [], [], []
    confidence_leader_accuracies, oracle_recovery_rates, oracle_gaps = [], [], []
    for counts in class_prediction_correct_counts:
        sample_count = counts.predicted_sample_count
        class_oracle_correct = counts.any_model_or_combined_prediction_correct_count
        if not sample_count:
            continue
        oracle_accuracy = class_oracle_correct / sample_count
        mixture_accuracy = counts.combined_prediction_correct_count / sample_count
        oracle_accuracies.append(oracle_accuracy)
        mixture_accuracies.append(mixture_accuracy)
        leader_accuracies.append(
            counts.maximum_weight_model_prediction_correct_count / sample_count
        )
        confidence_leader_accuracies.append(
            counts.highest_confidence_model_prediction_correct_count / sample_count
        )
        oracle_gaps.append(oracle_accuracy - mixture_accuracy)
        if class_oracle_correct:
            oracle_recovery_rates.append(
                counts.combined_prediction_correct_count / class_oracle_correct
            )

    def class_mean(values):
        return float(np.mean(values)) if values else 0.0

    return ClassMacroPredictionMetrics(
        macro_combined_prediction_accuracy=class_mean(mixture_accuracies),
        macro_any_model_or_combined_prediction_accuracy=class_mean(oracle_accuracies),
        macro_maximum_weight_model_prediction_accuracy=class_mean(leader_accuracies),
        macro_highest_confidence_model_prediction_accuracy=class_mean(confidence_leader_accuracies),
        mean_any_model_or_combined_accuracy_gap=class_mean(oracle_gaps),
        any_model_or_combined_accuracy_gap_standard_deviation=(
            float(np.std(oracle_gaps)) if oracle_gaps else 0.0
        ),
        mean_combined_to_any_model_or_combined_correct_ratio=class_mean(oracle_recovery_rates),
        minimum_combined_to_any_model_or_combined_correct_ratio=(
            min(oracle_recovery_rates) if oracle_recovery_rates else 0.0
        ),
    )


def make_random_class_counts(random_generator):
    sample_count = random_generator.choice((0, random_generator.randrange(1, 60)))
    combined = random_generator.randrange(0, sample_count + 1)
    # どれかのモデルか実際の予測が正解の数は、実際の予測の正解の数以上。
    any_correct_count = random_generator.randrange(combined, sample_count + 1)
    return make_counts(
        sample_count,
        any_correct_count,
        combined,
        random_generator.randrange(0, any_correct_count + 1),
        random_generator.randrange(0, any_correct_count + 1),
    )


@pytest.mark.parametrize("random_seed", range(40))
def test_class_macro_metrics_match_legacy_formula(random_seed):
    """乱数で作ったクラスごとの数（標本のないクラス、上限の正解が0のクラスを含む）で、旧と同じ式の値と一致する。"""
    random_generator = random.Random(random_seed)
    class_prediction_correct_counts = tuple(
        make_random_class_counts(random_generator) for _ in range(random_generator.randrange(0, 8))
    )
    metrics = summarize_class_prediction_metrics(
        class_prediction_correct_counts=class_prediction_correct_counts
    )
    assert type(metrics) is ClassMacroPredictionMetrics
    assert metrics == summarize_classes_like_legacy(class_prediction_correct_counts)
    for field_name in ClassMacroPredictionMetrics.__dataclass_fields__:
        assert type(getattr(metrics, field_name)) is float, field_name


def test_class_macro_metrics_of_known_values():
    # クラス0: 標本10、上限8、実際6、最大重み5、確信度4。クラス1: 標本4、上限2、実際2、最大重み1、確信度2。
    metrics = summarize_class_prediction_metrics(
        class_prediction_correct_counts=(
            make_counts(10, 8, 6, 5, 4),
            make_counts(0, 0, 0, 0, 0),
            make_counts(4, 2, 2, 1, 2),
        )
    )
    assert metrics == ClassMacroPredictionMetrics(
        macro_combined_prediction_accuracy=float(np.mean([0.6, 0.5])),
        macro_any_model_or_combined_prediction_accuracy=float(np.mean([0.8, 0.5])),
        macro_maximum_weight_model_prediction_accuracy=float(np.mean([0.5, 0.25])),
        macro_highest_confidence_model_prediction_accuracy=float(np.mean([0.4, 0.5])),
        mean_any_model_or_combined_accuracy_gap=float(np.mean([0.8 - 0.6, 0.0])),
        any_model_or_combined_accuracy_gap_standard_deviation=float(np.std([0.8 - 0.6, 0.0])),
        mean_combined_to_any_model_or_combined_correct_ratio=float(np.mean([6 / 8, 1.0])),
        minimum_combined_to_any_model_or_combined_correct_ratio=6 / 8,
    )
    with pytest.raises(FrozenInstanceError):
        metrics.macro_combined_prediction_accuracy = 0.0  # type: ignore[misc]
    zero_metrics = summarize_class_prediction_metrics(class_prediction_correct_counts=())
    assert set(
        getattr(zero_metrics, field_name)
        for field_name in ClassMacroPredictionMetrics.__dataclass_fields__
    ) == {0.0}
    assert (
        summarize_class_prediction_metrics(
            class_prediction_correct_counts=(make_counts(0, 0, 0, 0, 0),)
        )
        == zero_metrics
    )


def test_prediction_correct_counts_reject_invalid_values():
    with pytest.raises(TypeError):
        make_counts(1.0, 0, 0, 0, 0)
    with pytest.raises(TypeError):
        make_counts(1, True, 0, 0, 0)
    with pytest.raises(ValueError):
        make_counts(1, -1, 0, 0, 0)
    # 正解の数は、標本数以下。
    with pytest.raises(ValueError):
        make_counts(1, 2, 0, 0, 0)
    # 実際の予測・単体のモデルの正解は、どれかのモデルか実際の予測の正解に含まれる。
    with pytest.raises(ValueError):
        make_counts(5, 2, 3, 0, 0)
    with pytest.raises(ValueError):
        make_counts(5, 2, 0, 3, 0)
    with pytest.raises(ValueError):
        make_counts(5, 2, 0, 0, 3)


def test_class_macro_metrics_reject_invalid_arguments():
    counts = make_counts(2, 1, 1, 1, 1)
    with pytest.raises(TypeError):
        summarize_class_prediction_metrics(class_prediction_correct_counts=[counts])
    with pytest.raises(TypeError):
        summarize_class_prediction_metrics(class_prediction_correct_counts=(counts, None))
    # 手で壊した数は、値を使う前に検査し直す。
    object.__setattr__(counts, "combined_prediction_correct_count", 5)
    with pytest.raises(ValueError):
        summarize_class_prediction_metrics(class_prediction_correct_counts=(counts,))
