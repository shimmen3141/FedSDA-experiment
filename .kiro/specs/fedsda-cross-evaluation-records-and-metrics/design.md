# Design Document: fedsda-cross-evaluation-records-and-metrics

## Overview

**Purpose**: クロス評価の、clientの評価ごとの記録と、モデル対の予測の相補性の指標を、新実装で導出・保存し、旧実装と照合する。

**Impact**: 計算のmoduleを1つ足す。指標の型へ、まとまりを1つ足す。記録の表へ、表を2つ足す。全体runの実行は、変えない。

## Boundary Commitments

### This Spec Owns

- `evaluation/cross_evaluation_metric_calculations.py`（新規）。
- `runtime/fedsda_run_metric_derivation.py`（`FedsdaRunMetrics`へ、fieldを1つ足す）。
- `runtime/fedsda_run_record_tables.py`（表の追加）。

### Out of Boundary

- `CrossEvaluationRecordStore`（読取りだけ）。記録の型（fieldを足さない）。既存の指標の計算。

## Components

### evaluation: cross_evaluation_metric_calculations

| 名前 | 内容 |
|---|---|
| `ModelPairPredictionComplementarityMetrics` | `correctness_compared_evaluation_count`（int）、`correctness_compared_sample_count`（int）、`correctness_disagreement_rate`、`per_sample_selection_gain_rate`、`both_correct_rate`（float） |
| `summarize_model_pair_prediction_complementarity(*, client_cross_evaluation_records)` | `ClientCrossEvaluationRecord`のtupleから、正誤を比べた記録（`correctness_counts`がNoneでないもの）について、上を作る |

- `correctness_disagreement_rate`は、（候補だけ正解＋対象だけ正解）の合計÷標本数の合計。
- `per_sample_selection_gain_rate`は、評価ごとの、min(候補だけ正解, 対象だけ正解)の和÷標本数の合計（2つのモデルのうち、良いほうに対して、標本ごとに正しいほうを選べたら得られる、上限の改善。旧の「oracle gain」）。
- `both_correct_rate`は、両方正解の合計÷標本数の合計。
- 標本数の合計が0なら、評価の数を含めて、すべて0（旧と同じ）。

`FedsdaRunMetrics`へ、`model_pair_prediction_complementarity_metrics: ModelPairPredictionComplementarityMetrics`を足す（サーバの`CrossEvaluationRecordStore`の記録から作る）。

### runtime: fedsda_run_record_tables（表の追加）

| 表 | 列 |
|---|---|
| `client_cross_evaluation`（新。clientの評価ごと、評価した順） | `server_round_index`、`client_id`、`candidate_model_id`、`target_model_id`、`evaluated_sample_count`、`bounded_loss_sum`、`squared_bounded_loss_sum`、`candidate_only_correct_sample_count`、`target_only_correct_sample_count`、`both_correct_sample_count`、`both_wrong_sample_count`（正誤を比べなかった評価では、4つとも−1） |
| `client_cross_evaluation_class`（新。clientの評価の、クラスごと。評価した順、クラスの昇順） | `server_round_index`、`client_id`、`candidate_model_id`、`target_model_id`、`class_id`、`class_evaluated_sample_count`（そのクラスの、評価した標本数。評価の全体の標本数と、同じ名前にしない——独立レビューの指摘）、上の4つの数 |

- 「候補（candidate）」は評価する側のモデル、「対象（target）」は標本を持つ側のモデル（記録の型の語）。
- 列の名前は、記録の型のfieldの名前から作る。ラウンドは、ほかの表とそろえて`server_round_index`。正誤の数は、何の数かが分かるように`*_sample_count`。
- 旧の`model_pair_*`の配列に当たる表は、作らない（`client_cross_evaluation`の、正誤を比べた行——`both_correct_sample_count`が0以上——と同じ内容）。

## 旧と違う点

- 旧は、標本ごとの正誤と、クラスごとの正誤を、設定によって集める（サーバの`collect_pair_diagnostics`と、クラスタリングの判定の方式。クラスごとは、`class_functional_confidence`のときだけ）。新実装は、最終構成だけを実装していて、いつも集める。表が旧の配列と一致するのは、最終構成の実行（goldenの条件と、testの小さい条件）。

## 検証の方針

- 計算: 乱数で作った記録の列で、実旧の`pair_diagnostic_summary`と照合する。拒否のtest。
- 指標: goldenの3ケースの実旧の実行結果と、小さい条件の実旧の`pair_diagnostic_summary`の値と照合する。新→旧の対応を、testの`derive_legacy_model_pair_metric_values`にする。対応表の取り違えの検査は、前のspecと同じ形。
- 表: goldenの3ケースと、小さい条件（統合を通る）の、実旧が保存した配列と照合する。通った場合（正誤を比べなかった評価、クラスごとの行、標本数が0の評価）を観測する。
