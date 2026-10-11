# 命名表: fedsda-single-model-prediction

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `current_training_model_prediction` | `PredictionCombinationSettings.prediction_combination_strategy`の値 | 結合せず、現在の学習帰属のモデルだけで予測する |
| `_predict_with_current_training_model_only` | `runtime/observed_sample_prediction.py` | 上の方式の、標本1件の予測 |
| `evidence_is_updated_by_prediction` | `AdaHedgeWeightEvidenceCollection`の引数と読み取り | 予測の処理が、この集まりの証拠を更新する構成か |
| `prediction_combination_strategy` | `FedsdaRunClient`の読み取り | このclientの、予測の結合の方式 |
| `SINGLE_MODEL_PREDICTION_CONFIGURATION_NAME`（値`fedsda-single-model-prediction`） | `sweep/experiment_configurations.py` | 掃引のコマンドで選ぶ、構成の名前 |
