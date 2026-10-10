# 命名表: model-computation-measurement

sourceの公開する名前（module、class、公開の関数・メソッド）だけを載せる。確認は、全taskの後の独立レビューで受ける。

## module

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `model_computation_measurement` | `learning/models/` | モデルの計算（順伝播の標本数、optimizerの更新回数）を、実行中に外側から数える |
| `loss_monitoring_computation_count_store` | `evaluation/` | 損失の監視（検出器）の計算の計数を持つ |
| `fedsda_measured_run_execution` | `runtime/` | FedSDAの全体runを、モデルの計算の計測つきで実行する |

## class

| 名前 | 役割 |
|---|---|
| `ModelComputationCounts` | 共有部・概念固有部を通った標本数（学習と推論の別）と、optimizerの更新回数 |
| `ModelComputationMeter` | 計測の区間の間、計数を持つ。途中の計数を読める |
| `LossMonitoringComputationCounts` | 検出器の部品の更新回数と、評価した候補×賭け率の数 |
| `LossMonitoringComputationCountStore` | 上の計数のowner |
| `FedsdaMeasuredRun` | 計測つきの全体runの結果（実行の枠の結果、参加者、計数） |

## 関数・メソッド

| 名前 | 役割 |
|---|---|
| `measure_model_computation` | 計測の区間（context manager）。meterを渡す |
| `ModelComputationMeter.get_model_computation_counts` | その時点までの計数 |
| `subtract_model_computation_counts` | 2つの時点の計数の差 |
| `LossMonitoringComputationCountStore.record_loss_monitoring_computation` | 標本1件ぶんの計数を足す |
| `LossMonitoringComputationCountStore.get_loss_monitoring_computation_counts` | 合計を読む |
| `validate_classifier_bounded_loss_inputs` | 有界損失の評価の入力の検査（順伝播なし）。既存の非公開の検査を、公開にする |
| `evaluate_classifier_per_sample_bounded_losses_and_outputs` | 1回の順伝播から、標本ごとの損失と、分類器の出力を返す |
| `evaluate_classifiers_per_sample_bounded_losses_from_shared_features` | 共有部の特徴を1回だけ計算して、複数の分類器の、標本ごとの損失を返す |
| `execute_fedsda_stream_protocol_run_with_computation_measurement` | 計測つきの全体run |

## 判断が必要な点

- 「shared part」「concept-specific part」は、新実装の既存の語（`shared_parameter_optimizer`、`concept_specific_parameter_optimizer_state`、`split_shared_and_concept_specific_parameters`）に合わせる。旧の「backbone」「head」を使わない。
- 「training」「inference」は、順伝播のときの勾配の有無（学習か、それ以外か）。旧の用途名（prediction、detectionほか）は使わない。
- 「example」は、モデルへ入力した標本（旧の`examples`）。観測標本（sample）と区別する: 1つの観測標本が、複数のモデル・複数の処理で、何度も入力される。
- 「computation」は、計数で表す計算量。実行時間・FLOPsではない。
