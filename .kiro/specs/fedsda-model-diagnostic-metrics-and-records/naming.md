# 命名表: fedsda-model-diagnostic-metrics-and-records

sourceの公開する名前（module、class、公開の関数）と、保存する表の名前を載せる。fieldと列の名前は、design.md。確認は、全taskの後の独立レビューで受ける。

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `model_diagnostic_metric_calculations` | `evaluation/` | 値のまとめ、同じ概念のAUC、クラスタの割当と真の概念の一致の比較 |
| `ValueDistributionSummary` / `summarize_value_distribution` | 同上 | 値の列の、合計・平均・最小・変動係数 |
| `calculate_lower_value_same_concept_auc` | 同上 | 小さい値を同じ概念の証拠とみなしたROC-AUC |
| `ClusterAssignmentTrueConceptAgreement` / `calculate_cluster_assignment_true_concept_agreement` | 同上 | 同じクラスタにしたか、と、真の概念の一致の、4通りの数と、適合率・再現率・F1 |
| `fedsda_model_diagnostic_metric_derivation` / `derive_fedsda_model_diagnostic_metrics` | `runtime/` | 参加者の記録から、モデル別の診断の指標を作る |
| `FedsdaModelDiagnosticMetrics` / `GlobalModelCountMetrics` / `HeldModelTrainingAmountMetrics` / `ClusteringTrueConceptAgreementMetrics` | 同上 | 指標のまとまりと、その部分 |
| `ServerRoundModelCounts` / `count_models_after_server_round_synchronizations` | 同上 | サーバの同期のラウンドごとの、配布の後のモデル数 |
| `ClientHeldModelTrainingAmount` / `collect_client_held_model_training_amounts` | 同上 | clientの保有モデルごとの、実行の終わりの学習量 |
| `FedsdaRunMetrics.model_diagnostic_metrics` | `runtime/fedsda_run_metric_derivation.py` | 上のまとまり |

## 保存する表の名前

追加: `client_held_model`。

## 判断が必要な点

- 「model diagnostic」は、モデル数・モデル別の学習量・クラスタリングの比較の総称（旧の`_add_model_diagnostic_results`の語）。
- 「training amount」は、保持している学習標本の数、学習に使った標本の延べ数、パラメータ更新の回数の総称。
- 「true concept」は、既存の`true_concepts_match`の語（旧の「oracle」）。
- 「stored training sample」は、`ModelTrainingSampleStore`が持つ標本（旧の「assigned samples」。学習帰属で割り当てられ、保持している標本）。
