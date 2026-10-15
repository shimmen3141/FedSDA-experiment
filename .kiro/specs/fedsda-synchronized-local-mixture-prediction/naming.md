# 命名と役割

## 新しい公開名

| 公開名 | 場所 | 役割 |
|---|---|---|
| `mixture_of_synchronized_and_local_parameters` | `PredictionCombinationSettings.prediction_model_parameter_source`の値 | 同期の時点のモデルの結合予測と、局所更新したモデルの結合予測を、clientごとの重みで混ぜて予測する |
| `SynchronizedAndLocalPredictionMixture` | `methods/fedsda/prediction_combination/synchronized_and_local_prediction_mixture.py` | clientごとの2つの候補（同期の時点・局所）の重みを持ち、混ぜ、更新するowner |
| `mix_prediction_probabilities`・`update_weights_after_loss_observation`・`snapshot_weights` | 同上のmethod | 混ぜる、損失を見た後に更新する（`FixedShareWeightController`と同じ語）、重みを読む |
| `true_class_loss_of_prediction_probabilities` | 同上のmoduleの関数 | 結合予測の確率から、真のクラスの損失（1−確率、倍精度）を求める |
| `LEARNING_RATE` | 同上のクラスの定数 | 重みの更新の学習率（2.0） |
| `fedsda-synchronized-local-mixture-prediction` | 構成名 | 提案構成の予測を、上の混ぜ方で行う構成 |
| `SYNCHRONIZED_LOCAL_MIXTURE_PREDICTION_CONFIGURATION_NAME` | `sweep/experiment_configurations.py` | 上の構成名の定数 |

## 似た名前との違い

- `prediction_mixture_activation_policy`（値は`always`）: モデルの予測を重みで結合するかの方針。新しい混ぜ方は、結合した後の2つの予測（同期の時点・局所）を混ぜるもので、別の段。
- 移植していない回復期の混合予測（UNPORTED-003）: 変化の直後に、別のモデルの予測を混ぜる旧の仕組み。新しい混ぜ方は、時期に依らず、同じモデルの集合の、同期の時点と局所の2つの値を混ぜる。

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `predict_observed_sample_and_update_prediction_weights`・`process_observed_sample` | 写しのownerを受け取れる | 混ぜる重みのowner（省略可能）も受け取り、受け取ったときは、混ぜた確率で予測する |
| 予測の記録の`combined_prediction_is_correct` | 予測に使う重みでの結合予測の正否 | 混ぜる値のときは、混ぜた確率の予測の正否（ほかの値では今と同じ） |
