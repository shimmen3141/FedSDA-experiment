# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `LossChangeDetectionSettings.short_change_interval_alarm_policy` | `methods/fedsda/loss_change_detection/loss_change_detection_settings.py` | 検出器の合図があり、推定変化区間が最小件数より短いときに、警報とするかどうかの方針（保存するrun設定に入る） | `TrainingDataAssignmentSettings.pending_sample_retention_after_too_short_change_interval_policy`は、短い警報になった後の、保留の扱い（待つ扱いでは、短い警報が起きないので、使われない） |
| `SHORT_CHANGE_INTERVAL_ALARM_POLICIES` | 同上 | 方針の、宣言した値の一覧 | 個々の値の定数（下の2つ）を並べたもの |
| `RAISE_ALARM_IMMEDIATELY_POLICY_NAME` | 同上 | 値`raise_alarm_immediately`。合図があれば、すぐに警報とする（短い警報として処理する。既定。固定旧と同じ） | もう一方は、待つ |
| `WAIT_UNTIL_CHANGE_INTERVAL_REACHES_MINIMUM_POLICY_NAME` | 同上 | 値`wait_until_change_interval_reaches_minimum`。推定変化区間が最小件数に達するまで、警報としない（検出器がe-SRのときだけ選べる） | — |
| `FedsdaRunClient.short_change_interval_alarm_policy` | `runtime/fedsda_run_client.py` | このclientの設定の、方針の値（読取り） | 既存の`prediction_combination_strategy`と同じ形の、設定の読取り |

## 既存の公開名で、役割が広がるもの

| 公開名 | これまで | これから |
|---|---|---|
| `process_observed_sample` | 検出器の合図（または、ADWINの補助の判定）があれば、警報の処理を行う | 方針（新しい引数`short_change_interval_alarm_policy`）が「待つ」なら、推定変化区間が最小件数に達するまで、警報の処理を行わない |

検出器（`OverallAndTrueClassLossMonitor`）、警報の処理（`handle_alarm_occurrence`）、完了処理は、名前と役割を変えない。
