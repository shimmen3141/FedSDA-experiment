# 変異とfresh CPUの証拠

Windowsの基準環境（Python 3.13、torch 2.12.1+cpu）、source/test commit `e80b368`。証拠ファイルは元checkoutの`venv/refactoring-tests/`（Git管理外、このPCだけ）。

## 実sourceの変異

script: `venv/refactoring-tests/alarm-occurrence-handling-mutations.py`、証拠: `venv/refactoring-tests/alarm-occurrence-handling-mutation-evidence/`（変異ごとのlogと`report.json`）。対象は`src/federated_learning_experiments/runtime/alarm_occurrence_handling.py`。1種ずつ書き換えて対象test（`tests/refactoring/test_alarm_occurrence_handling.py`、40件）を実行し、毎回元byteへ戻した。変異後のsourceは実行前にcompileし、収集失敗は検出に数えない。

結果: 41/41検出。復元後は40 passed（exit 0）。元のsourceのbyteのSHA-256 `0b8ba8901a37778c6b1757250aa8eb48103b97eeb3b79d532bce9ae1e28f0cb7`、復元後も同じ値。

「失敗したtest」の名前は先頭の`test_`を省いている。

| # | 変異 | 何を変えたか | 結果 | 失敗したtest |
| --- | --- | --- | --- | --- |
| 1 | `move_all_validation_after_response` | 3つの検査を応答の呼出しの後へ移す | 検出（22 failed, 18 passed） | `alarm_occurrence_rejects_invalid_input_before_any_step` |
| 2 | `move_owner_type_checks_after_response` | ownerの型検査だけを応答の後へ移す | 検出（10 failed, 30 passed） | `alarm_occurrence_rejects_invalid_input_before_any_step` |
| 3 | `move_record_rule_check_after_response` | 記録の規則による検査だけを応答の後へ移す | 検出（9 failed, 31 passed） | `alarm_occurrence_rejects_invalid_input_before_any_step` |
| 4 | `move_last_observed_check_after_response` | 最終観測位置の検査だけを応答の後へ移す | 検出（2 failed, 38 passed） | `alarm_occurrence_rejects_invalid_input_before_any_step` |
| 5 | `skip_owner_type_checks` | ownerの型検査を全て削除 | 検出（10 failed, 30 passed） | `alarm_occurrence_rejects_invalid_input_before_any_step` |
| 6 | `owner_type_checks_accept_subclasses` | exact型検査をisinstanceへ緩める | 検出（5 failed, 35 passed） | `alarm_occurrence_rejects_invalid_input_before_any_step` |
| 7 | `skip_owner_type_check_validation_session_holder` | validation_session_holderの型検査を削除 | 検出（2 failed, 38 passed） | `alarm_occurrence_rejects_invalid_input_before_any_step` |
| 8 | `skip_owner_type_check_adaptation_record_store` | adaptation_record_storeの型検査を削除 | 検出（2 failed, 38 passed） | `alarm_occurrence_rejects_invalid_input_before_any_step` |
| 9 | `skip_owner_type_check_diagnostic_evidence_collection` | diagnostic_evidence_collectionの型検査を削除 | 検出（2 failed, 38 passed） | `alarm_occurrence_rejects_invalid_input_before_any_step` |
| 10 | `skip_owner_type_check_loss_change_monitor` | loss_change_monitorの型検査を削除 | 検出（2 failed, 38 passed） | `alarm_occurrence_rejects_invalid_input_before_any_step` |
| 11 | `skip_owner_type_check_pending_training_assignment_buffer` | pending_training_assignment_bufferの型検査を削除 | 検出（2 failed, 38 passed） | `alarm_occurrence_rejects_invalid_input_before_any_step` |
| 12 | `skip_record_rule_check` | 記録の規則による検査を削除 | 検出（9 failed, 31 passed） | `alarm_occurrence_rejects_invalid_input_before_any_step` |
| 13 | `record_rule_check_ignores_alarm_index` | 記録の規則による検査から警報位置を外す | 検出（2 failed, 38 passed） | `alarm_occurrence_rejects_invalid_input_before_any_step` |
| 14 | `record_rule_check_ignores_detector_name` | 記録の規則による検査から検出器名を外す | 検出（3 failed, 37 passed） | `alarm_occurrence_rejects_invalid_input_before_any_step` |
| 15 | `record_rule_check_ignores_change_point` | 記録の規則による検査から推定変化点を外す | 検出（2 failed, 38 passed） | `alarm_occurrence_rejects_invalid_input_before_any_step` |
| 16 | `record_rule_check_ignores_episode_id` | 記録の規則による検査からepisode IDを外す | 検出（2 failed, 38 passed） | `alarm_occurrence_rejects_invalid_input_before_any_step` |
| 17 | `skip_last_observed_check` | 最終観測位置の検査を削除 | 検出（2 failed, 38 passed） | `alarm_occurrence_rejects_invalid_input_before_any_step` |
| 18 | `last_observed_check_accepts_earlier_alarm_index` | 最終観測位置より前の警報位置を受理 | 検出（1 failed, 39 passed） | `alarm_occurrence_rejects_invalid_input_before_any_step` |
| 19 | `last_observed_check_accepts_later_alarm_index` | 最終観測位置より後の警報位置を受理 | 検出（1 failed, 39 passed） | `alarm_occurrence_rejects_invalid_input_before_any_step` |
| 20 | `respond_without_held_session` | 保持を読まず、進行中のsessionなしで応答する | 検出（2 failed, 38 passed） | `alarm_occurrence_matches_real_legacy_alarm` |
| 21 | `respond_with_zero_proposal_index` | 提案位置を警報位置でない値にする | 検出（5 failed, 35 passed） | `alarm_occurrence_matches_real_legacy_alarm`、`alarm_occurrence_runs_each_step_once_in_order` |
| 22 | `respond_without_episode_id` | 応答へepisode IDを渡さない | 検出（3 failed, 37 passed） | `alarm_occurrence_matches_real_legacy_alarm`、`alarm_occurrence_runs_each_step_once_in_order` |
| 23 | `respond_without_change_point` | 応答へ推定変化点を渡さない | 検出（3 failed, 37 passed） | `alarm_occurrence_matches_real_legacy_alarm`、`alarm_occurrence_runs_each_step_once_in_order` |
| 24 | `complete_twice` | 完了処理を2回実行 | 検出（12 failed, 28 passed） | `alarm_occurrence_matches_real_legacy_alarm`、`alarm_occurrence_runs_each_step_once_in_order`、`failed_step_stops_alarm_occurrence_before_later_steps` |
| 25 | `complete_without_episode_id` | 完了処理へepisode IDを渡さない | 検出（11 failed, 29 passed） | `alarm_occurrence_matches_real_legacy_alarm`、`alarm_occurrence_runs_each_step_once_in_order` |
| 26 | `complete_without_change_point` | 完了処理へ推定変化点を渡さない | 検出（11 failed, 29 passed） | `alarm_occurrence_matches_real_legacy_alarm`、`alarm_occurrence_runs_each_step_once_in_order` |
| 27 | `record_twice` | 適応記録を2回追加 | 検出（13 failed, 27 passed） | `alarm_occurrence_matches_real_legacy_alarm`、`alarm_occurrence_runs_each_step_once_in_order`、`failed_step_stops_alarm_occurrence_before_later_steps` |
| 28 | `record_constant_detector_name` | 記録へ受け取った検出器名を渡さない | 検出（11 failed, 29 passed） | `alarm_occurrence_matches_real_legacy_alarm`、`alarm_occurrence_runs_each_step_once_in_order` |
| 29 | `skip_holder_update` | 保持への反映を省略 | 検出（6 failed, 34 passed） | `alarm_occurrence_matches_real_legacy_alarm`、`alarm_occurrence_runs_each_step_once_in_order`、`failed_step_stops_alarm_occurrence_before_later_steps` |
| 30 | `holder_update_twice` | 保持への反映を2回実行 | 検出（5 failed, 35 passed） | `alarm_occurrence_matches_real_legacy_alarm`、`alarm_occurrence_runs_each_step_once_in_order` |
| 31 | `holder_update_before_recording` | 保持への反映を記録の前へ移す | 検出（3 failed, 37 passed） | `alarm_occurrence_runs_each_step_once_in_order`、`failed_step_stops_alarm_occurrence_before_later_steps` |
| 32 | `holder_update_before_completion` | 保持への反映を完了処理の前へ移す | 検出（4 failed, 36 passed） | `alarm_occurrence_runs_each_step_once_in_order`、`failed_step_stops_alarm_occurrence_before_later_steps` |
| 33 | `skip_notification` | 診断通知を省略 | 検出（4 failed, 36 passed） | `alarm_occurrence_matches_real_legacy_alarm`、`alarm_occurrence_runs_each_step_once_in_order`、`failed_step_stops_alarm_occurrence_before_later_steps` |
| 34 | `notification_twice` | 診断通知を2回実行 | 検出（3 failed, 37 passed） | `alarm_occurrence_matches_real_legacy_alarm`、`alarm_occurrence_runs_each_step_once_in_order` |
| 35 | `notification_before_holder_update` | 診断通知を保持への反映の前へ移す | 検出（2 failed, 38 passed） | `alarm_occurrence_runs_each_step_once_in_order`、`failed_step_stops_alarm_occurrence_before_later_steps` |
| 36 | `notification_before_recording` | 診断通知を記録の前へ移す | 検出（4 failed, 36 passed） | `alarm_occurrence_runs_each_step_once_in_order`、`failed_step_stops_alarm_occurrence_before_later_steps` |
| 37 | `notification_without_assignment_change` | 通知へ帰属変更を渡さない | 検出（3 failed, 37 passed） | `alarm_occurrence_matches_real_legacy_alarm`、`alarm_occurrence_runs_each_step_once_in_order` |
| 38 | `result_record_not_frozen` | 結果recordを可変にする | 検出（1 failed, 39 passed） | `alarm_occurrence_handling_record_is_frozen_and_keyword_only` |
| 39 | `result_record_accepts_positional_fields` | 結果recordを位置引数で作れるようにする | 検出（1 failed, 39 passed） | `alarm_occurrence_handling_record_is_frozen_and_keyword_only` |
| 40 | `consume_torch_random_number` | 応答の前にtorch乱数を消費 | 検出（10 failed, 30 passed） | `alarm_occurrence_matches_real_legacy_alarm` |
| 41 | `consume_python_random_number` | 完了処理の前に渡されたPython乱数を消費 | 検出（8 failed, 32 passed） | `alarm_occurrence_matches_real_legacy_alarm` |

検査を応答の後へ移す変異は1〜4（3つの検査をまとめて、および1つずつ）。どれも、拒否のtestで「呼ばれたら失敗するmock」に差し替えた応答が呼ばれて検出される。

## 新実装だけのfresh CPU

script: `venv/refactoring-tests/alarm-occurrence-handling-fresh-cpu.py`、log: `venv/refactoring-tests/alarm-occurrence-handling-fresh-cpu.log`。旧実装とtest moduleをimportしないprocessで、1つの保持・適応記録・診断証拠・損失監視・保留位置のownerを使い、`handle_alarm_occurrence`と既存の`advance_held_candidate_validation`を実行した。2/4class×5つの流れの10条件が成功し、警報応答の5種類の結果をすべて観測した。各条件で、戻り値と記録・保持・監視・保留位置の状態が互いに合うこと、診断の再始動が他モデルの再利用のときだけ1回であること、旧実装とtestのmoduleがimportされていないことを確かめている。

```text
PASS 2 classes: too_short -> alarm_change_interval_too_short ; restarts 0 ; legacy imports=0
PASS 2 classes: reuse_at_alarm -> alarm_interval_held_model_reused ; restarts 1 ; legacy imports=0
PASS 2 classes: maintain_at_alarm -> alarm_interval_current_model_maintained ; restarts 0 ; legacy imports=0
PASS 2 classes: start_then_alarm_during_validation -> alarm_interval_candidate_validation_started / alarm_during_candidate_validation ; restarts 0 ; legacy imports=0
PASS 2 classes: start_then_complete_then_alarm -> alarm_interval_candidate_validation_started / post_alarm_validation_candidate_adopted / alarm_interval_current_model_maintained ; restarts 0 ; legacy imports=0
PASS 4 classes: too_short -> alarm_change_interval_too_short ; restarts 0 ; legacy imports=0
PASS 4 classes: reuse_at_alarm -> alarm_interval_held_model_reused ; restarts 1 ; legacy imports=0
PASS 4 classes: maintain_at_alarm -> alarm_interval_current_model_maintained ; restarts 0 ; legacy imports=0
PASS 4 classes: start_then_alarm_during_validation -> alarm_interval_candidate_validation_started / alarm_during_candidate_validation ; restarts 0 ; legacy imports=0
PASS 4 classes: start_then_complete_then_alarm -> alarm_interval_candidate_validation_started / post_alarm_validation_candidate_adopted / alarm_interval_current_model_maintained ; restarts 0 ; legacy imports=0
ALL 5 ALARM OUTCOMES OBSERVED
```

流れ: 不足（保留を消費しない）／警報で他モデルを再利用（再始動1回）／警報で現行を維持／候補検証の開始→未到達のまま次の警報（保持中のsessionが応答へ渡り、保持は変わらない）／候補検証の開始→4標本で確定（採用）→次の警報（保持が空なので進行中のsessionなしで応答）。
