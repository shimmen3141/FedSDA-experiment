# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `TrainingDataAssignmentSettings.pending_sample_retention_after_too_short_change_interval_policy` | `methods/fedsda/training_data_assignment/training_data_assignment_settings.py` | 短い警報（変化区間が最小件数より少ない警報）のあと、保留に何を残すかの方針（保存するrun設定に入る） | 容量（`pending_assignment_buffer_capacity_samples`）は、警報のないときの確定の条件 |
| `PENDING_SAMPLE_RETENTION_AFTER_TOO_SHORT_CHANGE_INTERVAL_POLICIES` | 同上 | 方針の、宣言した値の一覧 | 個々の値の定数（下の2つ）を並べたもの |
| `RELEASE_EARLIER_SAMPLES_ASSIGNED_TO_CURRENT_MODEL_POLICY_NAME` | 同上 | 値`release_earlier_samples_assigned_to_current_model`。現行モデルへ取り込んだ前区間を保留から外し、変化区間を残す（既定） | もう一方は、全部残す |
| `KEEP_ALL_PENDING_SAMPLES_POLICY_NAME` | 同上 | 値`keep_all_pending_samples`。保留を全部残す（固定旧と同じ。取り込んだ前区間も残る） | 「legacy」としない（動作を言う名前にする。LEGACY-016のspecと同じ） |
| `PendingTrainingAssignmentBuffer.releases_earlier_samples_after_too_short_change_interval` | `methods/fedsda/training_data_assignment/pending_training_assignment_buffer.py` | このownerの設定の方針が、「前区間を外す」かどうか（読取りの真偽） | 設定の項目は、方針の値そのもの。完了処理は、この真偽だけを読む |
| `PendingTrainingAssignmentBuffer.release_earliest_pending_sample_indices` | 同上 | 指定した位置（保留の先頭の並びと一致すること）だけを、保留から外す | `release_sample_indices_exceeding_capacity`は、容量を超えた分を外す。`drain_pending_sample_indices`は、全部を外す |

## 既存の公開名で、役割が広がるもの

| 公開名 | これまで | これから |
|---|---|---|
| `complete_alarm_buffer_response` | 短い警報では、保留の位置を消費しない | 短い警報では、ownerの方針に応じて、前区間の位置を外すか、消費しない |
| `AlarmResponseCompletion.drained_pending_sample_indices` | 短い警報では、必ず空 | 短い警報では、空か、前区間の位置 |
| `AlarmBufferResponse.pending_assignment_buffer_should_be_cleared` | 保留を全部消費するか（短い警報では偽で、保留は変わらない） | 保留を**全部**消費するか（短い警報では偽のまま。短い警報で前区間だけを外すかどうかは、この値ではなく、保留の位置のownerの方針で決まる） |

`PendingSampleObservationStore`の操作、`prepare_alarm_training_intervals`は、名前と役割を変えない。
