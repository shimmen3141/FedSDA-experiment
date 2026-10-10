# 命名表: fedsda-change-response-metrics-and-records

sourceの公開する名前（module、class、公開の関数）と、保存する表の名前を載せる。fieldと列の名前は、design.md。確認は、全taskの後の独立レビューで受ける。

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `change_response_metric_calculations` | `evaluation/` | 変更位置とイベントの対応づけ、対応づかなかったイベントの分類、変化点の推定の誤差 |
| `ConceptChangeEventMatching` / `UnmatchedEventCounts` | 同上 | 対応づけの結果、対応づかなかったイベントの、時機ごとの数 |
| `match_events_to_concept_changes` / `count_unmatched_events_by_timing` / `calculate_change_point_estimation_errors` | 同上 | 上の計算 |
| `fedsda_change_response_metric_derivation` / `derive_fedsda_change_response_metrics` | `runtime/` | 参加者の記録から、応答の指標を作る |
| `FedsdaChangeResponseMetrics` / `TrainingModelSwitchTimingMetrics` / `ChangePointEstimationMetrics` / `AdaptationActionMetrics` / `CandidateValidationDecisionMetrics` | 同上 | 応答の指標のまとまりと、その部分 |
| `FedsdaRunMetrics.change_response_metrics` | `runtime/fedsda_run_metric_derivation.py` | 上のまとまり |

## 保存する表の名前

変更: `round`ほか3つ→`server_round`。追加: `loss_change_alarm`、`sample_loss_monitoring`。

## 判断が必要な点

- 「change response」は、概念の変更に対する、検出・警報・適応・候補の判定の総称（このspecで決める語）。
- 「server round」は、既存の`ServerRoundSynchronization`・`synchronize_models_in_server_round`の語。
- 「timing」は、変更位置に対する、イベントの時機（重複・早期・遅延・単独）。
