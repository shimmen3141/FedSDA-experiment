# 命名表: fedsda-without-prediction-weight-recalibration

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `keep_weights_without_recalibration` | `PredictionCombinationSettings.prediction_weight_recalibration_after_aggregation_policy`の値 | 集約の後、予測重みと診断重みを、作り直さずに、そのまま使う |
| `PREDICTION_WEIGHT_RECALIBRATION_ABLATION_CONFIGURATION_NAME`（値`fedsda-without-prediction-weight-recalibration`） | `sweep/experiment_configurations.py` | 掃引のコマンドで選ぶ、構成の名前 |
