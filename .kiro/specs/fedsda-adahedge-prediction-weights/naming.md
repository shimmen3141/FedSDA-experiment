# 命名表: fedsda-adahedge-prediction-weights

## 改めた名前（値は変えない）

| 前 | 後 |
|---|---|
| `AdaHedgeDiagnosticEvidence`（`evaluation/adahedge_diagnostic_evidence.py`） | `AdaHedgeWeightEvidence`（`evaluation/adahedge_weight_evidence.py`） |
| `AdaHedgeDiagnosticEvidenceCollection`（`evaluation/adahedge_diagnostic_evidence_collection.py`） | `AdaHedgeWeightEvidenceCollection`（`evaluation/adahedge_weight_evidence_collection.py`） |
| `diagnostic_evidence_collection`（引数、ownerのfield） | `adahedge_weight_evidence_collection` |
| `global_diagnostic_evidence`（集まりの中の、全標本の証拠） | `all_sample_weight_evidence` |
| `get_diagnostic_weights_before_loss_observation`、引数`diagnostic_weights_by_model_id` | `get_weights_before_loss_observation`、`weights_by_model_id` |
| `global_diagnostic_weights_by_model_id` | `adahedge_weights_by_model_id` |
| `global_diagnostic_prediction_is_correct`（記録、表の列） | `adahedge_weighted_prediction_is_correct` |
| `global_diagnostic_prediction_correct_count`（指標） | `adahedge_weighted_prediction_correct_count` |
| `global_diagnostic_aggregation_recalibration_count`（指標、表の列） | `adahedge_weight_aggregation_recalibration_count` |
| `global_diagnostic_recalibration_replayed_sample_count`（指標） | `adahedge_weight_recalibration_replayed_sample_count` |
| 表`client_weight_update_counts`の`global_diagnostic_*`の列 | `adahedge_weight_*` |
| `notify_diagnostics_of_training_assignment_change`（`runtime/training_assignment_diagnostic_notification.py`） | `restart_adahedge_weight_evidence_after_training_assignment_change`（`runtime/training_assignment_adahedge_weight_evidence_restart.py`） |
| `FixedSharePredictionWeightController`（`…/fixed_share_prediction_weights.py`） | `FixedShareWeightController`（`…/fixed_share_weights.py`） |
| `fixed_share_prediction_weight_controller`（引数、ownerのfield） | `fixed_share_weight_controller` |
| `get_prediction_weights_before_label_observation` | `get_weights_before_label_observation` |
| 保持側の`prediction_weight_leader_switch_count` | `leader_switch_count` |
| `prediction_weight_leader_switch_count`・`prediction_weight_model_pool_reset_count`（指標） | `fixed_share_weight_leader_switch_count`・`fixed_share_weight_model_pool_reset_count` |
| `prediction_weight_recalibration_replayed_sample_count`（指標） | `fixed_share_weight_recalibration_replayed_sample_count` |
| 表`client_weight_update_counts`の`prediction_weight_*`の列 | `fixed_share_weight_*` |

## 変えなかった名前（理由）

- `true_concept_diagnostic_…`: 真の概念を使う重みで、どの構成でも比較用。
- `combined_prediction_…`、`maximum_weight_model_id`、`maximum_prediction_weight`、`effective_model_count`、`prediction_weights_by_model_id`: 予測に使う重みでの値。方式に依らない名前のまま。
- `prediction_weight_recalibration_after_aggregation_policy`（設定）: 集約の後の、重みの作り直し全体の方針（Fixed-ShareとAdaHedgeの両方）。
- `restart_adahedge_preserve_fixed_share_prediction_state`（設定の値）: 方式の名前で書かれている。
- Fixed-Shareの重みの保持の、引数`prediction_weights_by_model_id`（更新と、最大重みのモデルの選択）: 最大重みのモデルの選択は、どの重みにも使う関数。更新の引数は、この保持の重みを渡す場所で、名前を変えると、多くの呼出しが変わるので、残した。
- 指標の束の名前（`FedsdaPredictionDiagnosticMetrics`ほかの`diagnostic`）: 予測そのものでなく、予測の内訳の指標、という意味で使っている。

## 足した名前

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `adahedge_weighted_prediction` | `PredictionCombinationSettings.prediction_combination_strategy`の値 | AdaHedgeの重みで結合して予測する |
| `prediction_combination_strategy` | 標本1件の予測・処理の引数 | 予測に使う重みの選び方 |
| `fixed_share_weights_by_model_id` | `ObservedSamplePrediction` | Fixed-Shareの重み（正規化後） |
| `fixed_share_weighted_prediction_is_correct`・`fixed_share_maximum_weight_model_id`・`fixed_share_effective_model_count` | `SamplePredictionRecord`、表`sample_prediction`の列 | Fixed-Shareの重みでの、結合の正否・最大重みのモデル・実効モデル数 |
| `fixed_share_weighted_prediction_correct_count`・`fixed_share_weighted_prediction_period_accuracies`・`mean_fixed_share_effective_model_count` | `FedsdaPredictionDiagnosticMetrics` | 上の3項目の、集計 |
| `ADAHEDGE_PREDICTION_WEIGHTS_CONFIGURATION_NAME`（値`fedsda-adahedge-prediction-weights`） | `sweep/experiment_configurations.py` | 掃引のコマンドで選ぶ、構成の名前 |
