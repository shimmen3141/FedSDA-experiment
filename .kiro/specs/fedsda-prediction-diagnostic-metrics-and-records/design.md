# Design Document: fedsda-prediction-diagnostic-metrics-and-records

## Overview

**Purpose**: 予測の診断（実際の予測と、比較用の予測の、正否の集計）の指標と、標本ごと・clientごとの記録を、新実装で導出・保存し、旧実装と照合する。

**Impact**: 計算のmoduleを1つ、導出のmoduleを1つ足す。指標の型へ、まとまりを1つ足す。記録の表へ、列と、表を1つ足す。全体runの実行は、変えない。

## Boundary Commitments

### This Spec Owns

- `evaluation/prediction_diagnostic_metric_calculations.py`（新規）。
- `runtime/fedsda_prediction_diagnostic_metric_derivation.py`（新規）。
- `runtime/fedsda_run_metric_derivation.py`（`FedsdaRunMetrics`へ、fieldを1つ足す）。
- `runtime/fedsda_run_record_tables.py`（列と表の追加）。

### Out of Boundary

- 各owner（読取りだけ）。記録の型（fieldを足さない）。既存の指標の計算。

## 予測の種類（語）

| 語（新） | 旧の語 | 内容 |
|---|---|---|
| combined prediction（実際の予測） | `mixture`・`switching`・`accuracy` | Fixed-Shareの予測重みで、全保有モデルの出力を結合した予測。最終構成では、これが、実際の予測 |
| any model or combined prediction | `oracle` | 保有モデルのどれか、または、実際の予測が正解（標本ごとに、正しいモデルを選べたら、という上限） |
| maximum weight model prediction | `leader` | 予測重みが最大のモデルだけの予測 |
| highest confidence model prediction | `confidence_leader` | 確信度が最大のモデルだけの予測 |
| global diagnostic prediction | `global` | globalの診断重み（AdaHedge）で結合した予測。予測と学習の判断には使わない |
| true concept diagnostic prediction | `oracle_concept` | 真の概念別の診断重みで結合した予測。同上 |

語は、既存の`SamplePredictionRecord`のfieldの名前のまま。

## Components

### evaluation: prediction_diagnostic_metric_calculations

| 名前 | 内容 |
|---|---|
| `StableAndRecoveryPeriodAccuracies` | `stable_period_accuracy`、`recovery_period_accuracy`（float。その区間に標本がなければ0.0） |
| `calculate_stable_and_recovery_period_accuracies(*, prediction_correctness_by_client, concept_change_sample_indices_by_client, recovery_window_sample_count, sample_inclusion_by_client=None)` | 回復の区間は、各変更位置から、窓の件数の標本（標本の列の範囲に切り詰める）。定常の区間は、それ以外。`sample_inclusion_by_client`（標本ごとのbool）を与えたときは、真の標本だけを数える |
| `PredictionCorrectCounts` | `predicted_sample_count`、`combined_prediction_correct_count`、`any_model_or_combined_prediction_correct_count`、`maximum_weight_model_prediction_correct_count`、`highest_confidence_model_prediction_correct_count`（int） |
| `ClassMacroPredictionMetrics` | `macro_combined_prediction_accuracy`、`macro_any_model_or_combined_prediction_accuracy`、`macro_maximum_weight_model_prediction_accuracy`、`macro_highest_confidence_model_prediction_accuracy`、`mean_any_model_or_combined_accuracy_gap`、`any_model_or_combined_accuracy_gap_standard_deviation`、`mean_combined_to_any_model_or_combined_correct_ratio`、`minimum_combined_to_any_model_or_combined_correct_ratio`（float） |
| `summarize_class_prediction_metrics(*, class_prediction_correct_counts)` | クラスごとの`PredictionCorrectCounts`のtuple（与えられた順）から、上を作る。標本のないクラスは、数えない。差は、（どれかのモデルか実際の予測の正解の割合）−（実際の予測の正解の割合）。比は、実際の予測の正解の数÷どれかのモデルか実際の予測の正解の数（後者が0のクラスは、比に入れない）。標準偏差は、母標準偏差。値がなければ0.0 |

- 既存の`calculate_stable_period_prediction_accuracy`（定常精度）は、同じ定常の区間を使うが、標本がないときNaNを返す（旧の`compute_metrics`と同じ）。この関数は、旧の`_routing_window_accuracies`と同じく、0を返す。既存の関数は、変えない。
- macroの平均と標準偏差は、旧と同じく、NumPy（`mean`・`std`）で計算する。

### runtime: fedsda_prediction_diagnostic_metric_derivation

| 名前 | 内容 |
|---|---|
| `FedsdaPredictionDiagnosticMetrics` | 下のfield |
| `derive_fedsda_prediction_diagnostic_metrics(*, run_clients, concept_change_sample_indices_by_client, recovery_window_sample_count)` | 参加者の記録から、上を作る |

`FedsdaPredictionDiagnosticMetrics`のfield:

- `prediction_correct_counts: PredictionCorrectCounts`（全clientの合計）。
- `global_diagnostic_prediction_correct_count: int`。
- `true_concept_known_sample_count: int`、`true_concept_diagnostic_prediction_correct_count: int`（真の概念が分かる標本について）。
- `combined_prediction_period_accuracies`、`any_model_or_combined_prediction_period_accuracies`、`maximum_weight_model_prediction_period_accuracies`、`true_concept_diagnostic_prediction_period_accuracies`（どれも`StableAndRecoveryPeriodAccuracies`。最後のものは、真の概念が分かる標本だけを数える）。
- `class_macro_prediction_metrics: ClassMacroPredictionMetrics`（クラスごとの数は、全clientの合計。クラスの順は、clientの順・標本の順に、最初に現れた順——旧と同じ）。
- `mean_effective_model_count: float`（標本ごとの実効モデル数の、全clientの平均。clientごとに、標本の順に足し、clientの順に足す——旧と同じ順）。予測した標本がなければ0.0。
- `prediction_weight_leader_switch_count: int`、`prediction_weight_model_pool_reset_count: int`、`global_diagnostic_aggregation_recalibration_count: int`（全clientの合計）。

割合（正解の数÷標本数）、差、比は、数から計算できるので、指標には持たない（旧の値との照合は、testの対応が計算する）。

`FedsdaRunMetrics`へ、`prediction_diagnostic_metrics: FedsdaPredictionDiagnosticMetrics`を足す。

### runtime: fedsda_run_record_tables（列と表の追加）

| 表 | 足す列 |
|---|---|
| `sample_prediction`（client×処理した標本） | `observed_class_id`、`observed_concept_id`（分からなければ−1）、`maximum_prediction_weight`、`effective_model_count`、`any_model_or_combined_prediction_is_correct`、`maximum_weight_model_prediction_is_correct`、`highest_confidence_model_prediction_is_correct`、`global_diagnostic_prediction_is_correct`、`true_concept_diagnostic_prediction_is_correct`（正解1、不正解0、真の概念が分からなければ−1。整数） |
| `client_weight_update_count`（新。clientごと。実行の終わりの値） | `client_id`、`global_diagnostic_model_pool_reset_count`、`global_diagnostic_concept_operation_restart_count`、`global_diagnostic_aggregation_recalibration_count`、`global_diagnostic_aggregation_recalibration_replayed_sample_count`、`prediction_weight_model_pool_reset_count`、`prediction_weight_leader_switch_count`、`prediction_weight_aggregation_recalibration_count`、`prediction_weight_aggregation_recalibration_replayed_sample_count` |

- 列の名前は、記録の型のfieldと、ownerのpropertyの名前から作る。再較正で再生した標本数は、既存の指標（`prediction_weight_recalibration_replayed_sample_count`）の語にそろえて`replayed_sample_count`。
- 旧の、clientとクラスごとの配列（`routing_class_*`）に当たる表は、作らない（`sample_prediction`の、`observed_class_id`と正否の列から、数えられる）。

## 旧と違う点

- 旧の指標のうち、最終構成では、いつも0になるもの（`routing_switching_gain_rate`・`routing_switching_stable_gain_rate`・`routing_switching_recovery_gain_rate`: Fixed-Shareの結合と、実際の予測の差。最終構成では、同じもの）は、持たない。`routing_switching_accuracy`と`routing_mixture_accuracy`は、同じ値（実際の予測の正解の割合）。
- 真の概念が分からない標本: 旧は、想定していない（実験では、いつも分かる）。新は、その標本を、真の概念別の診断の集計から外す。

## 検証の方針

- 計算: 乱数で作った列で、実旧の`_routing_window_accuracies`、旧と同じ式（クラスの間のまとめ）と照合する。拒否のtest。
- 指標: goldenの3ケースの実旧の実行結果と、小さい条件の実旧の`_add_model_diagnostic_results`の値（回復の窓を、条件の値にして呼ぶ）と、旧の40の指標の全部で照合する。新→旧の対応を、testの`derive_legacy_prediction_diagnostic_metric_values`にする。対応表の取り違えの検査は、前のspecと同じ形。
- 表: goldenの3ケースと、小さい条件（統合を通る）の、実旧が保存した配列と照合する。観測した真の概念の列は、全体runの結果の、評価用の概念列と照合する（旧は、この列を保存しない）。
