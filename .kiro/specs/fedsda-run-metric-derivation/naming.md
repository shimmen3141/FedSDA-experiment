# 命名表: fedsda-run-metric-derivation

sourceの公開する名前（module、class、公開の関数・メソッド）だけを載せる。確認は、全taskの後の独立レビューで受ける。

## module

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `run_metric_calculations` | `evaluation/` | 手法に依存しない、指標の計算の部品と、指標の設定 |
| `fedsda_run_metric_derivation` | `runtime/` | FedSDAの全体runの結果と参加者から、指標を導出する |

## class

| 名前 | 役割 |
|---|---|
| `RunMetricSettings` | 指標の計算の設定（検出の許容遅延、変更後の回復の窓） |
| `EventMatchCounts` | 変更位置とイベントの対応づけの結果（対応したイベントの数、対応した変更位置の数） |
| `DetectionMetrics` | 検出の適合率・再現率・F1と、その計数 |
| `FedsdaRunMetrics` | FedSDAの全体run 1回の指標 |

## 関数

| 名前 | 役割 |
|---|---|
| `extract_concept_change_sample_indices` | 概念列から、概念が変わった標本位置を取り出す |
| `count_events_matched_to_concept_changes` | 変更位置とイベントを、順に1対1で対応づけて数える |
| `calculate_prediction_accuracy` | 全clientの全標本の、正解の割合 |
| `calculate_stable_period_prediction_accuracy` | 変更の直後の回復の窓を除いた、正解の割合 |
| `calculate_detection_metrics` | clientごとに対応づけて、全clientの合計で、検出の指標を計算する |
| `derive_fedsda_run_metrics` | 全体runの結果と参加者から、`FedsdaRunMetrics`を作る |

## 判断が必要な点

- 「concept change」は、真の概念の変更（旧の「true drift」）。新実装は、検出の事象を「alarm」「training model switch」と呼び、「drift」を、真の変更と検出の両方に使わない。
- 「stable period」は、旧の`stable_accuracy`（回復の窓を除いた精度）。
- `training_model_switch_detection_metrics`: 検出として数える事象が、学習帰属の切替であることを、field名で示す（警報を事象とする指標は、このspecでは作らない）。
- 「derive」は、記録から値を作ること（状態を変えない）。既存の`derive_…`（testのhelper）と同じ語感。
