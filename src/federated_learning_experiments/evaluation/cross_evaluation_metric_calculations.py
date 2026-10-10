"""クロス評価の、clientの評価の記録から、モデル対の予測の相補性の指標を計算する。"""

from dataclasses import dataclass

from federated_learning_experiments.evaluation.cross_evaluation_record_store import (
    ClientCrossEvaluationRecord,
)


@dataclass(frozen=True, kw_only=True)
class ModelPairPredictionComplementarityMetrics:
    """候補のモデルと、対象のモデルの、標本ごとの正誤を比べた評価についての、相補性のまとめ。

    割合の分母は、正誤を比べた評価の、標本数の合計。合計が0なら、評価の数を含めて、すべて0。
    """

    correctness_compared_evaluation_count: int
    correctness_compared_sample_count: int
    # 片方のモデルだけが正解した標本の割合。
    correctness_disagreement_rate: float
    # 評価ごとの、min(候補だけ正解, 対象だけ正解)の和の割合（2つのモデルのうち、良いほうに対して、
    # 標本ごとに正しいほうを選べたら得られる、正解の割合の上限の改善）。
    per_sample_selection_gain_rate: float
    both_correct_rate: float


def summarize_model_pair_prediction_complementarity(
    *, client_cross_evaluation_records: tuple[ClientCrossEvaluationRecord, ...]
) -> ModelPairPredictionComplementarityMetrics:
    """正誤を比べた記録（`correctness_counts`を持つもの）について、相補性をまとめる。"""
    if type(client_cross_evaluation_records) is not tuple:
        raise TypeError("client_cross_evaluation_records must be exact tuple")
    compared_evaluation_count = 0
    compared_sample_count = 0
    disagreement_sample_count = 0
    per_sample_selection_gain_sample_count = 0
    both_correct_sample_count = 0
    for client_cross_evaluation_record in client_cross_evaluation_records:
        if type(client_cross_evaluation_record) is not ClientCrossEvaluationRecord:
            raise TypeError(
                "client_cross_evaluation_records must hold exact ClientCrossEvaluationRecord"
            )
        # 手で壊した記録も、値を使う前に検査し直す。
        client_cross_evaluation_record.__post_init__()
        correctness_counts = client_cross_evaluation_record.correctness_counts
        if correctness_counts is None:
            continue
        candidate_only_correct_count, target_only_correct_count, both_correct_count, _ = (
            correctness_counts
        )
        compared_evaluation_count += 1
        compared_sample_count += client_cross_evaluation_record.evaluated_sample_count
        disagreement_sample_count += candidate_only_correct_count + target_only_correct_count
        per_sample_selection_gain_sample_count += min(
            candidate_only_correct_count, target_only_correct_count
        )
        both_correct_sample_count += both_correct_count
    if compared_sample_count == 0:
        return ModelPairPredictionComplementarityMetrics(
            correctness_compared_evaluation_count=0,
            correctness_compared_sample_count=0,
            correctness_disagreement_rate=0.0,
            per_sample_selection_gain_rate=0.0,
            both_correct_rate=0.0,
        )
    return ModelPairPredictionComplementarityMetrics(
        correctness_compared_evaluation_count=compared_evaluation_count,
        correctness_compared_sample_count=compared_sample_count,
        correctness_disagreement_rate=disagreement_sample_count / compared_sample_count,
        per_sample_selection_gain_rate=(
            per_sample_selection_gain_sample_count / compared_sample_count
        ),
        both_correct_rate=both_correct_sample_count / compared_sample_count,
    )
