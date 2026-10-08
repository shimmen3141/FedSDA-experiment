# 変異とfresh CPUの証拠

Windowsの基準環境（Python 3.13、torch 2.12.1+cpu）、source/test commit `ef1be82`。証拠ファイルは元checkoutの`venv/refactoring-tests/`（Git管理外、このPCだけ）。

## 実sourceの変異

script: `venv/refactoring-tests/held-candidate-validation-diagnostic-notification-mutations.py`、証拠: `venv/refactoring-tests/held-candidate-validation-diagnostic-notification-mutation-evidence/`（変異ごとのlogと`report.json`）。対象は`runtime/held_candidate_validation_progress.py`と`runtime/candidate_validation_session_holder.py`。1種ずつ書き換えて対象test（`tests/refactoring/test_held_candidate_validation_progress.py`、57件）を実行し、毎回元byteへ戻した。変異後のsourceは実行前にcompileし、収集失敗は検出に数えない。

結果: 41/41検出。復元後は57 passed（exit 0）。元のsourceのbyteのSHA-256は`candidate_validation_session_holder.py` `845f062e5df2439ee0a6b1ea6411a39e102f404d6d5cd0433a90337803b7bfd5`、`held_candidate_validation_progress.py` `9f1fa240d212742e5cd1c7f6b395c62548bd58b6b473ad61b019294b4fd88a91`で、復元後も同じ値。

`n_`で始まる11種が本specで足した変異（診断のownerの型検査と通知）。`h_`・`p_`で始まる30種は、held-candidate-validation-progressの変異を変更後のsourceへ合わせて再実行したもの（進行のownerの型検査と上流の呼出しの間に診断の型検査が入ったこと、拒否の文言を直したことに合わせて、置換する文字列だけを改めた。変異の内容は同じ）。「失敗したtest」の名前は先頭の`test_`を省いている。

| # | 変異 | 何を変えたか | 結果 | 失敗したtest |
| --- | --- | --- | --- | --- |
| 1 | `n_move_diagnostic_type_check_after_advance` | 進行: 診断のownerの型検査を上流の呼出しの後へ移す | 検出（4 failed, 53 passed） | `owner_types_are_rejected_before_upstream_updates` |
| 2 | `n_skip_diagnostic_type_check` | 進行: 診断のownerの型検査を削除 | 検出（4 failed, 53 passed） | `owner_types_are_rejected_before_upstream_updates` |
| 3 | `n_diagnostic_type_check_accepts_subclasses` | 進行: 診断のownerのexact型検査をisinstanceへ緩める | 検出（2 failed, 55 passed） | `owner_types_are_rejected_before_upstream_updates` |
| 4 | `n_diagnostic_type_check_only_with_held_session` | 進行: 診断のownerの型検査を、保持があるときだけ行う | 検出（2 failed, 55 passed） | `owner_types_are_rejected_before_upstream_updates` |
| 5 | `n_skip_notification` | 進行: 確定後に診断へ通知しない | 検出（8 failed, 49 passed） | `completed_held_validation_is_recorded_then_released_like_legacy` |
| 6 | `n_notify_twice` | 進行: 診断へ2回通知する | 検出（8 failed, 49 passed） | `completed_held_validation_is_recorded_then_released_like_legacy` |
| 7 | `n_notify_before_release` | 進行: 保持の解除より前に通知する | 検出（8 failed, 49 passed） | `completed_held_validation_is_recorded_then_released_like_legacy` |
| 8 | `n_notify_before_record` | 進行: 記録より前に通知する | 検出（8 failed, 49 passed） | `completed_held_validation_is_recorded_then_released_like_legacy` |
| 9 | `n_notify_before_unfinished_return` | 進行: 未到達かどうかを見る前に通知する | 検出（10 failed, 47 passed） | `completed_held_validation_is_recorded_then_released_like_legacy`、`operations_without_held_session_change_nothing`、`unfinished_held_validation_keeps_session_and_adds_no_record` |
| 10 | `n_notify_without_assignment_change` | 進行: 通知へ帰属変更を渡さない | 検出（4 failed, 53 passed） | `completed_held_validation_is_recorded_then_released_like_legacy` |
| 11 | `n_notify_other_collection` | 進行: 渡された診断証拠でないものへ通知する | 検出（8 failed, 49 passed） | `completed_held_validation_is_recorded_then_released_like_legacy` |
| 12 | `h_skip_session_type_check` | holder: sessionのexact型検査を削除 | 検出（1 failed, 56 passed） | `session_holder_holds_one_session_and_rejects_before_changing` |
| 13 | `h_replace_held_session_silently` | holder: 保持中でも黙って置き換える | 検出（1 failed, 56 passed） | `session_holder_holds_one_session_and_rejects_before_changing` |
| 14 | `h_release_keeps_session` | holder: 解除してもsessionを残す | 検出（13 failed, 44 passed） | `completed_held_validation_is_recorded_then_released_like_legacy`、`incomplete_held_validation_is_recorded_then_released_like_legacy`、`session_holder_holds_one_session_and_rejects_before_changing` |
| 15 | `h_release_empty_returns_none` | holder: 空の解除を拒否しない | 検出（1 failed, 56 passed） | `session_holder_holds_one_session_and_rejects_before_changing` |
| 16 | `p_skip_response_type_check` | 反映: 応答のexact型検査を削除 | 検出（1 failed, 56 passed） | `alarm_response_is_rejected_before_changing_holder` |
| 17 | `p_skip_holder_type_check_in_apply` | 反映: holderのexact型検査を削除 | 検出（12 failed, 45 passed） | `alarm_response_is_rejected_before_changing_holder` |
| 18 | `p_skip_response_revalidation` | 反映: 応答の再検査を削除 | 検出（1 failed, 56 passed） | `alarm_response_is_rejected_before_changing_holder` |
| 19 | `p_skip_outcome_session_correspondence_check` | 反映: 結果種別とsessionの対応の検査を削除 | 検出（2 failed, 55 passed） | `alarm_response_is_rejected_before_changing_holder` |
| 20 | `p_skip_held_session_identity_check` | 反映: 候補検証中の応答のsession同一性の検査を削除 | 検出（2 failed, 55 passed） | `alarm_response_is_rejected_before_changing_holder` |
| 21 | `p_identity_check_only_requires_any_session` | 反映: 候補検証中の応答で、別のsessionを保持していても受理 | 検出（1 failed, 56 passed） | `alarm_response_is_rejected_before_changing_holder` |
| 22 | `p_skip_empty_holder_precondition` | 反映: 候補検証中でない応答で保持が空であることの検査を削除 | 検出（2 failed, 55 passed） | `alarm_response_is_rejected_before_changing_holder` |
| 23 | `p_never_hold_started_session` | 反映: 開始の応答でも保持しない | 検出（2 failed, 55 passed） | `alarm_response_updates_holder_like_legacy_session_attribute` |
| 24 | `p_release_on_validation_alarm` | 反映: 候補検証中の応答で保持を解除する | 検出（2 failed, 55 passed） | `alarm_response_updates_holder_like_legacy_session_attribute` |
| 25 | `p_skip_owner_validation_in_advance` | 進行: ownerの型検査の呼出しを削除 | 検出（8 failed, 49 passed） | `owner_types_are_rejected_before_upstream_updates` |
| 26 | `p_skip_owner_validation_in_finalize` | 終端: ownerの型検査の呼出しを削除 | 検出（8 failed, 49 passed） | `owner_types_are_rejected_before_upstream_updates` |
| 27 | `p_move_owner_validation_after_advance` | 進行: ownerの型検査を上流の呼出しの後へ移す | 検出（8 failed, 49 passed） | `owner_types_are_rejected_before_upstream_updates` |
| 28 | `p_move_owner_validation_after_finalize` | 終端: ownerの型検査を上流の呼出しの後へ移す | 検出（8 failed, 49 passed） | `owner_types_are_rejected_before_upstream_updates` |
| 29 | `p_skip_holder_type_check_in_helper` | 進行・終端: holderのexact型検査を削除 | 検出（8 failed, 49 passed） | `owner_types_are_rejected_before_upstream_updates` |
| 30 | `p_skip_record_store_type_check_in_helper` | 進行・終端: 適応記録ownerのexact型検査を削除 | 検出（8 failed, 49 passed） | `owner_types_are_rejected_before_upstream_updates` |
| 31 | `p_skip_release_after_completion` | 進行: 確定後に保持を解除しない | 検出（8 failed, 49 passed） | `completed_held_validation_is_recorded_then_released_like_legacy` |
| 32 | `p_release_before_record_on_completion` | 進行: 記録より前に保持を解除する | 検出（8 failed, 49 passed） | `completed_held_validation_is_recorded_then_released_like_legacy` |
| 33 | `p_record_completion_to_other_store` | 進行: 確定の記録を渡された記録ownerへ追加しない | 検出（8 failed, 49 passed） | `completed_held_validation_is_recorded_then_released_like_legacy` |
| 34 | `p_release_unfinished_session` | 進行: 未到達でも保持を解除する | 検出（1 failed, 56 passed） | `unfinished_held_validation_keeps_session_and_adds_no_record` |
| 35 | `p_drop_record_from_advance_result` | 進行: 結果へ記録を入れない | 検出（8 failed, 49 passed） | `completed_held_validation_is_recorded_then_released_like_legacy` |
| 36 | `p_skip_release_after_incomplete_finalization` | 終端: 回収後に保持を解除しない | 検出（4 failed, 53 passed） | `incomplete_held_validation_is_recorded_then_released_like_legacy` |
| 37 | `p_record_incomplete_to_other_store` | 終端: 記録を渡された記録ownerへ追加しない | 検出（4 failed, 53 passed） | `incomplete_held_validation_is_recorded_then_released_like_legacy` |
| 38 | `p_finalize_ignores_held_session` | 終端: 保持中のsessionを上流へ渡さない | 検出（4 failed, 53 passed） | `incomplete_held_validation_is_recorded_then_released_like_legacy` |
| 39 | `p_advance_ignores_held_session` | 進行: 保持中のsessionを上流へ渡さない | 検出（9 failed, 48 passed） | `completed_held_validation_is_recorded_then_released_like_legacy`、`unfinished_held_validation_keeps_session_and_adds_no_record` |
| 40 | `p_consume_python_random_without_session` | 進行: 保持がなくてもPython乱数を消費 | 検出（1 failed, 56 passed） | `operations_without_held_session_change_nothing` |
| 41 | `p_consume_torch_random_in_finalize` | 終端: 保持がなくてもtorch乱数を消費 | 検出（1 failed, 56 passed） | `operations_without_held_session_change_nothing` |

検査を上流の後へ移す変異は1（診断）と、既存の`p_move_owner_validation_after_advance`・`p_move_owner_validation_after_finalize`（保持と記録）。`n_notify_before_unfinished_return`は、未到達かどうかを見る前に通知を呼ぶ変異で、未到達のとき確定の結果がないため例外になって検出される（未到達で通知しないことは、未到達のtestの診断証拠の不変でも確かめている）。

拒否の文言（NEW-001）の変更は、挙動を変えないので変異の対象にしていない。

## 新実装だけのfresh CPU

script: `venv/refactoring-tests/held-candidate-validation-diagnostic-notification-fresh-cpu.py`、log: `venv/refactoring-tests/held-candidate-validation-diagnostic-notification-fresh-cpu.log`。旧実装とtest moduleをimportしないprocessで、1つの保持・適応記録・診断証拠のownerを使い、`handle_alarm_occurrence`（候補検証の開始）→`advance_held_candidate_validation`（4標本の観測と確定）を実行した。2/4class×確定4種類の8条件が成功した。各条件で、開始と未到達の間は再始動が0回、確定の後は採用と他モデルの再利用で1回・棄却と現行の維持で0回、確定の後の標本（保持なし）では何も変わらないこと、旧実装とtestのmoduleがimportされていないことを確かめている。

```text
PASS 2 classes: adopt post_alarm_validation_candidate_adopted 2->-105 ; restarts 1 ; legacy imports=0
PASS 2 classes: reject post_alarm_validation_candidate_rejected 2->2 ; restarts 0 ; legacy imports=0
PASS 2 classes: maintain post_alarm_validation_current_model_maintained 2->2 ; restarts 0 ; legacy imports=0
PASS 2 classes: reuse_other post_alarm_validation_held_model_reused 2->-103 ; restarts 1 ; legacy imports=0
PASS 4 classes: adopt post_alarm_validation_candidate_adopted 2->-105 ; restarts 1 ; legacy imports=0
PASS 4 classes: reject post_alarm_validation_candidate_rejected 2->2 ; restarts 0 ; legacy imports=0
PASS 4 classes: maintain post_alarm_validation_current_model_maintained 2->2 ; restarts 0 ; legacy imports=0
PASS 4 classes: reuse_other post_alarm_validation_held_model_reused 2->-103 ; restarts 1 ; legacy imports=0
ALL 4 COMPLETION OUTCOMES OBSERVED
```
