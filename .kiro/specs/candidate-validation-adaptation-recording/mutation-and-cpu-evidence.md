# 候補検証の適応記録 — 検出力と新実装単独の動作（Task 2）

対象: Task 1の実装commit（このファイルと同じcommit）のsource・test。主担当Claude Code、2026-10-08、Windows CPUの固定venv。

## 実source変異

新module（名前が`r_`で始まる変異）と記録owner（`s_`）を1種ずつ書き換え、対象test2件（新test 63件、既存の警報適応記録test 40件）を実行した。各回finallyで元byteへ戻し、戻した後のbyteが元と同じことをassertしている。収集失敗・構文失敗は検出に数えず、pytestがexit 1で「failed」を報告した場合だけを検出とした。

結果: **42/42検出**。復元後は103 passed。

- `runtime/candidate_validation_adaptation_recording.py`: LF sha256 `9fcedd0b51f559dd8318b6ca2573437e1611d8270bb540f5fdfe4f7787bbaeb2`
- `evaluation/adaptation_record_store.py`: LF sha256 `ed98ed07e8f860f279f7e10d0791d01947081098359ef36500755c74177cdd96`

| 変異 | 何を変えたか | 対象testの結果 | 失敗したtest（`test_`を省略） |
| --- | --- | --- | --- |
| r_skip_completion_type_check | 完了情報のexact型検査を削除 | 1 failed, 102 passed | completed_validation_recording_rejects_invalid_input_before_updating_store |
| r_skip_store_type_check_completed | 到達時の記録ownerのexact型検査を削除 | 20 failed, 83 passed | completed_validation_recording_rejects_invalid_input_before_updating_store |
| r_skip_decision_record_type_check | 判定記録のexact型検査を削除 | 1 failed, 102 passed | completed_validation_recording_rejects_invalid_input_before_updating_store |
| r_skip_resolution_type_check | 確定結果のexact型検査を削除 | 1 failed, 102 passed | completed_validation_recording_rejects_invalid_input_before_updating_store |
| r_skip_outcome_str_check | 結果種別のstr検査を削除 | 1 failed, 102 passed | completed_validation_recording_rejects_invalid_input_before_updating_store |
| r_skip_outcome_membership_check | 未知の結果種別を棄却として記録 | 1 failed, 102 passed | completed_validation_recording_rejects_invalid_input_before_updating_store |
| r_skip_assigned_id_type_check | 帰属先IDの型検査を削除 | 1 failed, 102 passed | completed_validation_recording_rejects_invalid_input_before_updating_store |
| r_skip_change_type_check | 変更記録のexact型検査を削除 | 1 failed, 102 passed | completed_validation_recording_rejects_invalid_input_before_updating_store |
| r_skip_change_id_type_check | 変更記録のIDの型検査を削除 | 2 failed, 101 passed | completed_validation_recording_rejects_invalid_input_before_updating_store |
| r_skip_change_previous_match_check | 変更記録の変更前IDの一致検査を削除 | 1 failed, 102 passed | completed_validation_recording_rejects_invalid_input_before_updating_store |
| r_skip_change_differs_check | 変更記録の前後IDが異なることの検査を削除（設計r1の穴） | 1 failed, 102 passed | completed_validation_recording_rejects_invalid_input_before_updating_store |
| r_skip_assigned_match_check | 帰属先IDと確定後の学習帰属の一致検査を削除 | 1 failed, 102 passed | completed_validation_recording_rejects_invalid_input_before_updating_store |
| r_move_assigned_check_after_append | 検査を更新の後へ移す | 1 failed, 102 passed | completed_validation_recording_rejects_invalid_input_before_updating_store |
| r_position_from_proposal_index | 到達時の位置を提案位置にする | 18 failed, 85 passed | completed_validation_record_matches_real_legacy_event, completed_validation_recording_rejects_invalid_input_before_updating_store |
| r_swap_previous_and_current_ids | 変更前後IDの入替え | 8 failed, 95 passed | completed_validation_record_matches_real_legacy_event |
| r_current_id_from_assigned_model | 変更記録があっても変更後IDを使わない | 22 failed, 81 passed | completed_validation_record_matches_real_legacy_event, completed_validation_recording_rejects_invalid_input_before_updating_store |
| r_swap_adopted_and_reused_outcomes | 採用と再利用の対応を入替え | 9 failed, 94 passed | adaptation_outcomes_cover_alarm_and_validation_results_exactly, completed_validation_record_matches_real_legacy_event |
| r_swap_maintained_and_rejected_outcomes | 維持と棄却の対応を入替え | 9 failed, 94 passed | adaptation_outcomes_cover_alarm_and_validation_results_exactly, completed_validation_record_matches_real_legacy_event |
| r_drop_change_point_completed | 到達時の推定変化点を落とす | 17 failed, 86 passed | completed_validation_record_matches_real_legacy_event, completed_validation_recording_rejects_invalid_input_before_updating_store |
| r_drop_episode_completed | 到達時のepisode IDを落とす | 1 failed, 102 passed | completed_validation_recording_rejects_invalid_input_before_updating_store |
| r_detector_name_stripped | 到達時の検出器名を変える | 16 failed, 87 passed | completed_validation_record_matches_real_legacy_event |
| r_skip_append_completed | 到達時の記録の追加を省略 | 16 failed, 87 passed | completed_validation_record_matches_real_legacy_event |
| r_append_twice_completed | 到達時の記録を2回追加 | 16 failed, 87 passed | completed_validation_record_matches_real_legacy_event |
| r_consume_python_random | Python乱数を消費 | 16 failed, 87 passed | completed_validation_record_matches_real_legacy_event |
| r_consume_torch_random | 終端の記録でtorch乱数を消費 | 8 failed, 95 passed | incomplete_validation_record_matches_real_legacy_event |
| r_skip_finalization_type_check | 終端回収の結果のexact型検査を削除 | 1 failed, 102 passed | incomplete_validation_recording_rejects_invalid_input_before_updating_store |
| r_skip_incomplete_decision_record_type_check | 終端の判定記録のexact型検査を削除 | 1 failed, 102 passed | incomplete_validation_recording_rejects_invalid_input_before_updating_store |
| r_incomplete_position_from_proposal_index | 終端の位置を提案位置にする | 6 failed, 97 passed | incomplete_validation_record_matches_real_legacy_event, incomplete_validation_recording_rejects_invalid_input_before_updating_store |
| r_incomplete_recorded_as_rejected | 未完了を到達時の棄却として記録 | 8 failed, 95 passed | incomplete_validation_record_matches_real_legacy_event |
| r_drop_change_point_incomplete | 終端の推定変化点を落とす | 9 failed, 94 passed | incomplete_validation_record_matches_real_legacy_event, incomplete_validation_recording_rejects_invalid_input_before_updating_store |
| r_drop_episode_incomplete | 終端のepisode IDを落とす | 1 failed, 102 passed | incomplete_validation_recording_rejects_invalid_input_before_updating_store |
| r_skip_append_incomplete | 終端の記録の追加を省略 | 8 failed, 95 passed | incomplete_validation_record_matches_real_legacy_event |
| s_switch_outcomes_without_adopted | 切替結果の定数から採用を除く | 19 failed, 84 passed | completed_validation_record_matches_real_legacy_event, completed_validation_recording_rejects_invalid_input_before_updating_store, store_applies_switch_index_and_count_rules_for_every_outcome |
| s_switch_outcomes_without_validation_reuse | 切替結果の定数から検証後の再利用を除く | 5 failed, 98 passed | completed_validation_record_matches_real_legacy_event, store_applies_switch_index_and_count_rules_for_every_outcome |
| s_switch_outcomes_with_validation_maintained | 切替結果の定数へ検証後の維持を加える | 8 failed, 95 passed | completed_validation_record_matches_real_legacy_event, completed_validation_recording_rejects_invalid_input_before_updating_store, store_applies_switch_index_and_count_rules_for_every_outcome |
| s_switch_index_only_for_alarm_reuse | 切替位置を警報時の再利用だけに戻す | 10 failed, 93 passed | completed_validation_record_matches_real_legacy_event, store_applies_switch_index_and_count_rules_for_every_outcome |
| s_reuse_count_for_every_switch | 再利用件数を全切替結果で数える | 10 failed, 93 passed | completed_validation_record_matches_real_legacy_event, store_applies_switch_index_and_count_rules_for_every_outcome |
| s_fit_count_for_validation_maintained | 現行適合件数を検証後の維持でも数える | 5 failed, 98 passed | completed_validation_record_matches_real_legacy_event, store_applies_switch_index_and_count_rules_for_every_outcome |
| s_id_rule_only_for_alarm_reuse | ID整合の規則を警報時の再利用だけに戻す | 24 failed, 79 passed | completed_validation_record_matches_real_legacy_event, completed_validation_recording_rejects_invalid_input_before_updating_store, store_applies_switch_index_and_count_rules_for_every_outcome |
| s_skip_outcome_membership_check | 適応結果の値の検査を削除 | 20 failed, 83 passed | record_constructor_and_store_reject_invalid_fields_before_any_update, recording_rejects_invalid_input_before_updating_populated_store |
| s_switch_index_from_zero | 切替位置に記録の位置を使わない | 14 failed, 89 passed | completed_alarm_record_matches_all_real_legacy_event_fields, completed_validation_record_matches_real_legacy_event, record_store_keeps_input_order_equal_positions_and_old_immutable_snapshot, store_applies_switch_index_and_count_rules_for_every_outcome |
| s_reorder_new_outcomes | 適応結果の順序を入替え | 1 failed, 102 passed | adaptation_outcomes_cover_alarm_and_validation_results_exactly |

経過: 1回目の実行で`r_skip_outcome_membership_check`（未知の結果種別を棄却として記録する）が未検出だった。拒否条件のtestが採用の完了情報を元にしており、変更記録との不対応で同じ例外になっていたためで、変更記録を持たない棄却を元にするよう直した（結果種別がstrでない条件も同じ）。

入れていない変異: 完了情報の変更前IDの型検査の削除。この検査を消しても、`AdaptationRecord`のconstructorが同じTypeErrorを更新前に出すので、外から観測できる挙動が変わらない（比較より前に型を見るという設計の順序を保つために残している）。

script: 元checkoutの`venv/refactoring-tests/candidate-validation-adaptation-recording-mutations.py`（実装ファイルを書き換えるので、レビュー担当は実行せず内容を読む）。各変異のpytest出力と`report.json`は`venv/refactoring-tests/candidate-validation-adaptation-recording-mutation-evidence/`。

## 新実装だけのfresh CPU

旧実装とtest moduleをimportしない別processで、2/4class×5シナリオを実行した。各シナリオは、警報応答→完了処理で候補検証を開始し、警報応答の記録を追加した後、検証標本を観測して到達時の確定（採用・棄却・現行維持・他の保有モデルの再利用）または途中での終端回収を行い、同じ記録ownerへ候補検証の記録を追加する。全10条件が成功し、5種類の適応結果がすべて現れた（exit 0）。

確認したこと: 適応結果、記録の位置（到達時は確定位置、終端は終端位置）、検出器名、変更前後のID（採用では新しい一時ID、再利用では他の保有モデル）、推定変化点とepisode ID、記録ownerの履歴の順（警報応答の記録→候補検証の記録）、切替位置（採用と再利用だけ）、2種類の件数が0のままであること、記録の前後で学習標本・損失統計・3種の乱数の状態全体が不変、`sys.modules`に旧実装・test moduleがないこと。

script: `venv/refactoring-tests/candidate-validation-adaptation-recording-fresh-cpu.py`（worktreeルートで実行）。出力は`candidate-validation-adaptation-recording-fresh-cpu.log`。

```text
PASS 2 classes: adopt post_alarm_validation_candidate_adopted 2->-105 at 14 ; legacy imports=0
PASS 2 classes: reject post_alarm_validation_candidate_rejected 2->2 at 14 ; legacy imports=0
PASS 2 classes: maintain post_alarm_validation_current_model_maintained 2->2 at 14 ; legacy imports=0
PASS 2 classes: reuse_other post_alarm_validation_held_model_reused 2->-103 at 14 ; legacy imports=0
PASS 2 classes: incomplete post_alarm_validation_incomplete_candidate_rejected 2->2 at 12 ; legacy imports=0
PASS 4 classes: adopt post_alarm_validation_candidate_adopted 2->-105 at 14 ; legacy imports=0
PASS 4 classes: reject post_alarm_validation_candidate_rejected 2->2 at 14 ; legacy imports=0
PASS 4 classes: maintain post_alarm_validation_current_model_maintained 2->2 at 14 ; legacy imports=0
PASS 4 classes: reuse_other post_alarm_validation_held_model_reused 2->-103 at 14 ; legacy imports=0
PASS 4 classes: incomplete post_alarm_validation_incomplete_candidate_rejected 2->2 at 12 ; legacy imports=0
ALL 5 OUTCOMES OBSERVED
```

## 依存のexact一致

新moduleのimportは9 symbol、記録ownerは`typing.get_args`を追加した4 symbol（設計5節）。`dependency_is_allowed`の分岐、両resolverのtupleへの登録、注入契約test 70条件。対象103＋依存境界2681＝2784 passed。
