# Design Document: fedsda-model-diagnostic-metrics-and-records

## Overview

**Purpose**: モデル別の診断（モデル数の推移のまとめ、モデル別の学習量、クラスタリングの判定と真の概念の一致の比較）の指標と、登録・クラスタリング・保有モデルの記録を、新実装で導出・保存し、旧実装と照合する。

**Impact**: 計算のmoduleを1つ、導出のmoduleを1つ足す。指標の型へ、まとまりを1つ足す。記録の表へ、列と、表を1つ足す。全体runの実行は、変えない。

## Boundary Commitments

### This Spec Owns

- `evaluation/model_diagnostic_metric_calculations.py`（新規）。
- `runtime/fedsda_model_diagnostic_metric_derivation.py`（新規。参加者の記録から、指標のまとまりを作る。サーバの同期のラウンドごとのモデル数の数え方も、ここへ置き、表と共用する）。
- `runtime/fedsda_run_metric_derivation.py`（`FedsdaRunMetrics`へ、fieldを1つ足す）。
- `runtime/fedsda_run_record_tables.py`（列と表の追加）。

### Out of Boundary

- 各owner（読取りだけ）。既存の指標の計算。記録の型（fieldを足さない）。

## Components

### evaluation: model_diagnostic_metric_calculations

| 名前 | 内容 |
|---|---|
| `ValueDistributionSummary` | `total_value`、`mean_value`、`minimum_value`、`coefficient_of_variation`（すべてfloat） |
| `summarize_value_distribution(*, values)` | 値（intまたはfloatのtuple）の、合計・平均・最小・変動係数。空なら、すべて0.0。変動係数は、母標準偏差÷平均（平均が0以下なら0.0）。旧と同じく、NumPyのfloat64の配列で計算する |
| `calculate_lower_value_same_concept_auc(*, same_concept_values, different_concept_values)` | 小さい値を、同じ概念の証拠とみなしたROC-AUC。有限でない値は除く。どちらかが空になれば、NaN |
| `ClusterAssignmentTrueConceptAgreement` | `same_cluster_same_concept_pair_count`、`same_cluster_different_concept_pair_count`、`different_cluster_same_concept_pair_count`、`different_cluster_different_concept_pair_count`、`same_cluster_precision`、`same_cluster_recall`、`same_cluster_f1` |
| `calculate_cluster_assignment_true_concept_agreement(*, assigned_to_same_cluster_flags, true_concepts_match_flags)` | 同じ長さの、boolの2つのtupleから、上を作る |

「same cluster precision」は、同じクラスタにした対のうち、真の概念が一致する対の割合。「recall」は、真の概念が一致する対のうち、同じクラスタにした対の割合。

### runtime: fedsda_model_diagnostic_metric_derivation

| 名前 | 内容 |
|---|---|
| `ServerRoundModelCounts` | `global_model_counts`（ラウンドごと）、`client_held_model_counts`（ラウンドごと、clientごと） |
| `count_models_after_server_round_synchronizations(*, run_clients, run_server)` | サーバの同期のラウンドごとの、配布の後のモデル数。前のspecで、表のmoduleの中にあった数え方を、ここへ移す（表と、指標が使う） |
| `GlobalModelCountMetrics` | `mean_global_model_count_after_synchronization`、`maximum_global_model_count_after_synchronization`、`total_global_model_count_over_server_rounds`（ラウンドがなければ、すべて0.0） |
| `HeldModelTrainingAmountMetrics` | `stored_training_sample_count_summary`、`trained_sample_count_summary`、`parameter_update_step_count_summary`（どれも`ValueDistributionSummary`。モデルごとに、全clientで足した値の、モデルの間のまとめ） |
| `ClusteringTrueConceptAgreementMetrics` | `true_concept_known_pair_count`、`same_true_concept_pair_count`、`cluster_assignment_agreement`（`ClusterAssignmentTrueConceptAgreement`）、`loss_increase_distance_same_concept_auc`、`concept_specific_parameter_distance_same_concept_auc` |
| `FedsdaModelDiagnosticMetrics` | 上の3つ |
| `ClientHeldModelTrainingAmount` | `client_id`、`model_id`、`stored_training_sample_count`、`trained_sample_count`、`parameter_update_step_count` |
| `collect_client_held_model_training_amounts(*, run_clients)` | clientの順、モデルIDの昇順で、実行の終わりの保有モデルごとの学習量（表と、指標が使う） |
| `derive_fedsda_model_diagnostic_metrics(*, run_clients, run_server)` | 上のまとまりを作る |

- 保有モデルは、`HeldModelTrainingStateRegistry`の保有モデル（まだ登録していない一時IDのモデルを含む。旧の`client.models`と同じ）。
- 保持している学習標本の数は、`ModelTrainingSampleStore`の、そのモデルの標本の数（なければ0）。学習に使った標本の延べ数と、パラメータ更新の回数は、`ModelTrainingAndAssignmentCountsStore`の値（なければ0）。
- モデルの間のまとめは、全clientの保有モデルのIDの集まりについて、最初に現れた順に、値を並べて計算する（旧と同じ順。合計・平均・最小・変動係数は、順によらないが、浮動小数点の和の順を、旧とそろえる）。
- クラスタリングの比較は、`ModelClusteringRecordStore`の対の観測のうち、`true_concepts_match`がNoneでないものを使う。

`FedsdaRunMetrics`へ、`model_diagnostic_metrics: FedsdaModelDiagnosticMetrics`を足す。

### runtime: fedsda_run_record_tables（列と表の追加）

| 表 | 足す列 |
|---|---|
| `model_registration` | `registering_client_id`（初期モデルは−1）、`maximum_weight_prediction_sample_count`（全clientの、処理した標本のうち、そのモデルが最大重みだった標本の数） |
| `model_clustering` | `nearest_model_id`（なければ−1）、`nearest_loss_increase_distance`（なければNaN）、`cluster_model_count`、`maximum_within_cluster_distance`（なければNaN）、`evaluated_within_cluster_pair_count`、`possible_within_cluster_pair_count` |
| `model_pair_clustering` | `loss_increase_distance`、`decision_score`、`true_concepts_match`（一致1、不一致0、分からなければ−1。整数）、`concept_specific_parameter_distance` |
| `client_held_model`（新） | `client_id`、`model_id`、`stored_training_sample_count`、`trained_sample_count`、`parameter_update_step_count`（実行の終わりの値） |

列の名前は、記録の型のfieldの名前のまま（どれも、何の値かが、名前から分かる）。`true_concepts_match`は、記録では`bool | None`で、表では、ほかの整数の欠損（−1）とそろえて、整数にする。

## 旧と違う点

- 旧は、対の、概念固有部のパラメータの距離がないとき、NaNを入れる。新の記録は、この距離を、いつも持つ（NaNを許さない）。最終構成では、旧も、いつも持つ。
- 旧の`model_count_auc`は、ラウンドごとのモデル数の合計（面積ではない）。新の名前は、`total_global_model_count_over_server_rounds`。

## 検証の方針

- 計算: 乱数で作った列で、NumPyでの旧と同じ式、実旧の`_lower_score_auc`と照合する。拒否のtest。
- 指標: goldenの3ケースの実旧の実行結果（263の指標の辞書）と、全部の項目で照合する。モデル別の学習量とクラスタリングの比較は、小さい条件の実旧の`_add_model_diagnostic_results`の戻り値とも照合する（モデル数のまとめは、小さい条件の実旧の実行が、ラウンドごとの値を記録しないので、goldenだけ）。新→旧の対応を、testの`derive_legacy_model_diagnostic_metric_values`にする。
- 対応表の取り違え: 前のspecと同じ形（実行した全部の条件で、旧の値が同じになる対と、値が変わらない項目を数える）。
- 表: goldenの3ケースと、小さい条件の、実旧が保存した配列と照合する（小さい条件は、統合を通る）。
