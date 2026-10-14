# 命名と役割

## 新しい公開名

| 公開名 | 場所 | 役割 |
|---|---|---|
| `prediction_model_parameter_source` | `PredictionCombinationSettings`の項目 | 予測に使うパラメータ。`current_local_parameters`（局所更新した現在の値。既定）、`parameters_at_last_synchronization`（最後の同期の時点の値） |
| `SynchronizedPredictionModelSnapshot` | `methods/fedsda/prediction_combination/synchronized_prediction_model_snapshot.py` | 同期の時点の保有モデルの写しを持ち、予測に使う分類器を選ぶowner。既存の`snapshot_classifier_parameters`（パラメータのtensorの辞書を返すだけ）や`snapshot_*`（読取り専用の一覧）と違い、分類器を丸ごと写して持ち、同期ごとに書き換える |
| `capture_held_models` | 同上のmethod | 保有モデル全部を写す（前の写しは捨てる） |
| `select_prediction_classifiers` | 同上のmethod | 共有特徴を計算する分類器と、モデルごとの予測の分類器を返す |
| `fedsda-synchronized-model-prediction` | 構成名 | 提案構成の予測を、最後の同期の時点のモデルで行う構成 |
| `SYNCHRONIZED_MODEL_PREDICTION_CONFIGURATION_NAME` | `sweep/experiment_configurations.py` | 上の構成名の定数 |

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `predict_observed_sample_and_update_prediction_weights` | 保有モデルの今の分類器で予測する | 省略可能な引数`synchronized_prediction_model_snapshot`を受けたとき、そのownerが選んだ分類器で予測する。省略かNoneなら今と同じ |
| `process_observed_sample`（標本1件の処理） | 写しを扱わない | 同じ省略可能な引数を受け、予測へ渡すだけ |
