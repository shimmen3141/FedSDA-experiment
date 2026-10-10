"""モデル対の予測の相補性の計算を、実旧の`pair_diagnostic_summary`と照合する。"""

import random
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from federated_drift_experiment.servers.clustering import CrossEvaluationClusteringServer
from federated_learning_experiments.evaluation.cross_evaluation_metric_calculations import (
    ModelPairPredictionComplementarityMetrics,
    summarize_model_pair_prediction_complementarity,
)
from federated_learning_experiments.evaluation.cross_evaluation_record_store import (
    ClientCrossEvaluationRecord,
)


def make_record(*, correctness_counts, candidate_model_id=0, target_model_id=1):
    """正誤の4つの数（なければNone）を持つ、clientの評価の記録。"""
    return ClientCrossEvaluationRecord(
        round_index=0,
        client_id=0,
        candidate_model_id=candidate_model_id,
        target_model_id=target_model_id,
        evaluated_sample_count=0 if correctness_counts is None else sum(correctness_counts),
        bounded_loss_sum=0.0,
        squared_bounded_loss_sum=0.0,
        correctness_counts=correctness_counts,
        class_correctness_counts=(),
    )


def summarize_with_legacy_function(records):
    """正誤を比べた記録を、旧の診断の一覧の形にして、実旧の`pair_diagnostic_summary`を呼ぶ。"""
    legacy_records = [
        dict(
            n=record.evaluated_sample_count,
            candidate_only_correct=record.correctness_counts[0],
            target_only_correct=record.correctness_counts[1],
            both_correct=record.correctness_counts[2],
            both_wrong=record.correctness_counts[3],
        )
        for record in records
        if record.correctness_counts is not None
    ]
    return CrossEvaluationClusteringServer.pair_diagnostic_summary(
        SimpleNamespace(pair_prediction_diagnostics=legacy_records)
    )


@pytest.mark.parametrize("random_seed", range(40))
def test_complementarity_metrics_match_legacy_summary(random_seed):
    """乱数で作った記録の列（正誤を比べなかった評価、標本数が0の評価を含む）で、実旧の集計と一致する。"""
    random_generator = random.Random(random_seed)
    records = []
    for _ in range(random_generator.randrange(0, 9)):
        kind = random_generator.random()
        if kind < 0.25:
            correctness_counts = None
        elif kind < 0.4 or random_seed % 8 == 0:
            # 標本数が0の評価（seedが8の倍数の条件は、正誤を比べた評価の全部が、これ）。
            correctness_counts = (0, 0, 0, 0)
        else:
            correctness_counts = tuple(random_generator.randrange(0, 30) for _ in range(4))
        records.append(make_record(correctness_counts=correctness_counts))
    metrics = summarize_model_pair_prediction_complementarity(
        client_cross_evaluation_records=tuple(records)
    )
    assert type(metrics) is ModelPairPredictionComplementarityMetrics
    legacy_summary = summarize_with_legacy_function(records)
    assert (
        metrics.correctness_compared_evaluation_count
        == (legacy_summary["model_pair_evaluation_count"])
    )
    assert metrics.correctness_compared_sample_count == legacy_summary["model_pair_sample_count"]
    assert (
        metrics.correctness_disagreement_rate
        == (legacy_summary["model_pair_correctness_disagreement_rate"])
    )
    assert metrics.per_sample_selection_gain_rate == legacy_summary["model_pair_oracle_gain_rate"]
    assert metrics.both_correct_rate == legacy_summary["model_pair_both_correct_rate"]
    for count_name in (
        "correctness_compared_evaluation_count",
        "correctness_compared_sample_count",
    ):
        assert type(getattr(metrics, count_name)) is int, count_name
    for rate_name in (
        "correctness_disagreement_rate",
        "per_sample_selection_gain_rate",
        "both_correct_rate",
    ):
        assert type(getattr(metrics, rate_name)) is float, rate_name


def test_complementarity_metrics_of_known_records():
    # 正誤の数は（候補だけ正解、対象だけ正解、両方正解、両方不正解）。
    metrics = summarize_model_pair_prediction_complementarity(
        client_cross_evaluation_records=(
            make_record(correctness_counts=(3, 1, 4, 2)),
            make_record(correctness_counts=None),
            make_record(correctness_counts=(0, 5, 5, 0)),
        )
    )
    assert metrics == ModelPairPredictionComplementarityMetrics(
        correctness_compared_evaluation_count=2,
        correctness_compared_sample_count=20,
        correctness_disagreement_rate=9 / 20,
        # 評価ごとの、min(候補だけ正解, 対象だけ正解)の和: min(3, 1) + min(0, 5)。
        per_sample_selection_gain_rate=1 / 20,
        both_correct_rate=9 / 20,
    )
    with pytest.raises(FrozenInstanceError):
        metrics.both_correct_rate = 0.0  # type: ignore[misc]
    zero_metrics = ModelPairPredictionComplementarityMetrics(
        correctness_compared_evaluation_count=0,
        correctness_compared_sample_count=0,
        correctness_disagreement_rate=0.0,
        per_sample_selection_gain_rate=0.0,
        both_correct_rate=0.0,
    )
    assert summarize_model_pair_prediction_complementarity(client_cross_evaluation_records=()) == (
        zero_metrics
    )
    # 正誤を比べた評価の、標本数の合計が0なら、評価の数も0（旧と同じ）。
    assert (
        summarize_model_pair_prediction_complementarity(
            client_cross_evaluation_records=(
                make_record(correctness_counts=(0, 0, 0, 0)),
                make_record(correctness_counts=None),
            )
        )
        == zero_metrics
    )


def test_complementarity_metrics_reject_invalid_arguments():
    record = make_record(correctness_counts=(1, 1, 1, 1))
    with pytest.raises(TypeError):
        summarize_model_pair_prediction_complementarity(client_cross_evaluation_records=[record])
    with pytest.raises(TypeError):
        summarize_model_pair_prediction_complementarity(
            client_cross_evaluation_records=(record, SimpleNamespace(correctness_counts=None))
        )
    # 手で壊した記録は、値を使う前に検査し直す。
    object.__setattr__(record, "correctness_counts", (1, 1, 1, 2))
    with pytest.raises(ValueError):
        summarize_model_pair_prediction_complementarity(client_cross_evaluation_records=(record,))
