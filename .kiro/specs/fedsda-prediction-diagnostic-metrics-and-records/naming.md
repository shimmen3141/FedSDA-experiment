# 命名表: fedsda-prediction-diagnostic-metrics-and-records

sourceの公開する名前（module、class、公開の関数）と、保存する表の名前を載せる。fieldと列の名前は、design.md。確認は、全taskの後の独立レビューで受ける。

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `prediction_diagnostic_metric_calculations` | `evaluation/` | 区間ごとの正解の割合、クラスの間のまとめ |
| `StableAndRecoveryPeriodAccuracies` / `calculate_stable_and_recovery_period_accuracies` | 同上 | 変更の後の回復の区間と、それ以外の定常の区間の、正解の割合 |
| `PredictionCorrectCounts` | 同上 | 標本数と、予測の種類ごとの正解の数 |
| `ClassMacroPredictionMetrics` / `summarize_class_prediction_metrics` | 同上 | クラスごとの正解の割合の、クラスの間のまとめ |
| `fedsda_prediction_diagnostic_metric_derivation` / `derive_fedsda_prediction_diagnostic_metrics` | `runtime/` | 参加者の記録から、予測の診断の指標を作る |
| `FedsdaPredictionDiagnosticMetrics` | 同上 | 指標のまとまり |
| `FedsdaRunMetrics.prediction_diagnostic_metrics` | `runtime/fedsda_run_metric_derivation.py` | 上のまとまり |

## 保存する表の名前

追加: `client_weight_update_counts`。

## 判断が必要な点

- 「prediction diagnostic」は、実際の予測と、比較用の予測（どれかのモデル、最大重みのモデルほか）の、正否の集計の総称。
- 「stable period」「recovery period」は、既存の`stable_period_prediction_accuracy`・`post_change_recovery_window_sample_count`の語。
- 「macro」は、クラスごとの値の、クラスの間の平均（標本数で重みづけしない）。
- 「gap」は、どれかのモデルか実際の予測の正解の割合と、実際の予測の正解の割合の差。「ratio」は、正解の数の比。
- 表の名前`client_weight_update_counts`は、複数の回数の列を持つので、複数形にした（独立レビューの指摘）。
- 表の`sample_prediction.true_concept_diagnostic_prediction_is_correct`は、整数（正解1、不正解0、真の概念が分からなければ−1）。ほかの正否の列（真偽）と、型が違う（記録のfieldが`bool | None`のため。ほかの表の`true_concepts_match`と同じ表し方）。
- 「weight update count」は、予測重みと、診断重みの、更新に関わる出来事（モデルの集まりの変更、再始動、再較正、最大重みのモデルの交代）の回数。
