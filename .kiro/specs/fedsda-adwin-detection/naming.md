# 命名表: fedsda-adwin-detection

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `AdwinLossWindowDetector`・`AdwinLossWindowObservation`・`AdwinLossWindowState` | `methods/fedsda/loss_change_detection/adwin_loss_window_detection.py` | 単一の損失系列の、ADWINの検出器と、観測の結果、状態の写し |
| `adwin` | `LossChangeDetectionSettings.drift_detector_name`の値 | ADWINで監視する |
| `adwin_confidence_delta`・`adwin_minimum_test_window_sample_count` | `LossChangeDetectionSettings` | ADWINの信頼度のパラメータと、検定を始める窓の標本数 |
| `overall_adwin_window_sample_count` | `LossMonitoringObservation` | ADWINの、全体の窓の標本数（e-SRではNone） |
| `overall_adwin_state`・`class_adwin_states_by_class_id` | `OverallAndTrueClassLossMonitoringState` | ADWINの窓の状態の写し |
| `_current_model_loss_worsened_over_adwin_window` | `runtime/observed_sample_processing.py` | 補助の判定（旧`_forced_drift_check`） |
| `capacity_sample_count` | `PendingTrainingAssignmentBuffer` | 保留の容量 |
| `CLASSWISE_ADWIN_DETECTION_CONFIGURATION_NAME`・`OVERALL_LOSS_ADWIN_DETECTION_CONFIGURATION_NAME` | `sweep/experiment_configurations.py` | 掃引のコマンドで選ぶ、構成の名前 |
