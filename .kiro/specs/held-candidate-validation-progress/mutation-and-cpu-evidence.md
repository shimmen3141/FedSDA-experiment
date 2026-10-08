# 候補検証sessionの保持と進行 — 検出力と新実装単独の動作（Task 2）

対象: このファイルと同じcommitのsource・test。主担当Claude Code、2026-10-08。

**実行環境: WSL 2 Ubuntu、Python 3.14.4、torch 2.12.1+cpu、NumPy 2.4.6、pytest 9.1.1（リポジトリ直下の`.venv`）。** Windowsの基準環境（固定venv、Python 3.13）は、torchの読込みがスマートアプリコントロールでブロックされており使えない。以下はWSLでの結果で、Windows基準での検証ではない。

## 実source変異

保持のowner（名前が`h_`で始まる変異）と接続（`p_`）を1種ずつ書き換え、対象test（38件）を実行した。各回finallyで元byteへ戻し、戻した後のbyteが元と同じことをassertしている。収集失敗・構文失敗は検出に数えず、pytestがexit 1で「failed」を報告した場合だけを検出とした。

結果: **28/28検出**（1回目の実行で未検出なし）。復元後は38 passed。

- `runtime/candidate_validation_session_holder.py`: LF sha256 `845f062e5df2439ee0a6b1ea6411a39e102f404d6d5cd0433a90337803b7bfd5`
- `runtime/held_candidate_validation_progress.py`: LF sha256 `f1dfd19a919e4bc9361956538dd0f3a079d03247c8ac92e9a0c1e2d6246e2bc4`

| 変異 | 何を変えたか | 対象testの結果 | 失敗したtest（`test_`を省略） |
| --- | --- | --- | --- |
| h_skip_session_type_check | holder: sessionのexact型検査を削除 | 1 failed, 37 passed | session_holder_holds_one_session_and_rejects_before_changing |
| h_replace_held_session_silently | holder: 保持中でも黙って置き換える | 1 failed, 37 passed | session_holder_holds_one_session_and_rejects_before_changing |
| h_release_keeps_session | holder: 解除してもsessionを残す | 13 failed, 25 passed | completed_held_validation_is_recorded_then_released_like_legacy, incomplete_held_validation_is_recorded_then_released_like_legacy, session_holder_holds_one_session_and_rejects_before_changing |
| h_release_empty_returns_none | holder: 空の解除を拒否しない | 1 failed, 37 passed | session_holder_holds_one_session_and_rejects_before_changing |
| p_skip_response_type_check | 反映: 応答のexact型検査を削除 | 1 failed, 37 passed | alarm_response_is_rejected_before_changing_holder |
| p_skip_holder_type_check_in_apply | 反映: holderのexact型検査を削除 | 9 failed, 29 passed | alarm_response_is_rejected_before_changing_holder |
| p_skip_response_revalidation | 反映: 応答の再検査を削除 | 2 failed, 36 passed | alarm_response_is_rejected_before_changing_holder |
| p_skip_held_session_identity_check | 反映: 候補検証中の応答のsession同一性の検査を削除 | 2 failed, 36 passed | alarm_response_is_rejected_before_changing_holder |
| p_identity_check_only_requires_any_session | 反映: 候補検証中の応答で、別のsessionを保持していても受理 | 1 failed, 37 passed | alarm_response_is_rejected_before_changing_holder |
| p_skip_empty_holder_precondition | 反映: 候補検証中でない応答で保持が空であることの検査を削除 | 2 failed, 36 passed | alarm_response_is_rejected_before_changing_holder |
| p_never_hold_started_session | 反映: 開始の応答でも保持しない | 2 failed, 36 passed | alarm_response_updates_holder_like_legacy_session_attribute |
| p_release_on_validation_alarm | 反映: 候補検証中の応答で保持を解除する | 2 failed, 36 passed | alarm_response_updates_holder_like_legacy_session_attribute |
| p_skip_owner_validation_in_advance | 進行: ownerの型検査の呼出しを削除 | 2 failed, 36 passed | owner_types_are_rejected_before_upstream_updates |
| p_skip_owner_validation_in_finalize | 終端: ownerの型検査の呼出しを削除 | 2 failed, 36 passed | owner_types_are_rejected_before_upstream_updates |
| p_move_owner_validation_after_advance | 進行: ownerの型検査を上流の呼出しの後へ移す | 2 failed, 36 passed | owner_types_are_rejected_before_upstream_updates |
| p_skip_holder_type_check_in_helper | 進行・終端: holderのexact型検査を削除 | 2 failed, 36 passed | owner_types_are_rejected_before_upstream_updates |
| p_skip_record_store_type_check_in_helper | 進行・終端: 適応記録ownerのexact型検査を削除 | 2 failed, 36 passed | owner_types_are_rejected_before_upstream_updates |
| p_skip_release_after_completion | 進行: 確定後に保持を解除しない | 8 failed, 30 passed | completed_held_validation_is_recorded_then_released_like_legacy |
| p_release_before_record_on_completion | 進行: 記録より前に保持を解除する | 8 failed, 30 passed | completed_held_validation_is_recorded_then_released_like_legacy |
| p_record_completion_to_other_store | 進行: 確定の記録を渡された記録ownerへ追加しない | 8 failed, 30 passed | completed_held_validation_is_recorded_then_released_like_legacy |
| p_release_unfinished_session | 進行: 未到達でも保持を解除する | 1 failed, 37 passed | unfinished_held_validation_keeps_session_and_adds_no_record |
| p_drop_record_from_advance_result | 進行: 結果へ記録を入れない | 8 failed, 30 passed | completed_held_validation_is_recorded_then_released_like_legacy |
| p_skip_release_after_incomplete_finalization | 終端: 回収後に保持を解除しない | 4 failed, 34 passed | incomplete_held_validation_is_recorded_then_released_like_legacy |
| p_record_incomplete_to_other_store | 終端: 記録を渡された記録ownerへ追加しない | 4 failed, 34 passed | incomplete_held_validation_is_recorded_then_released_like_legacy |
| p_finalize_ignores_held_session | 終端: 保持中のsessionを上流へ渡さない | 4 failed, 34 passed | incomplete_held_validation_is_recorded_then_released_like_legacy |
| p_advance_ignores_held_session | 進行: 保持中のsessionを上流へ渡さない | 9 failed, 29 passed | completed_held_validation_is_recorded_then_released_like_legacy, unfinished_held_validation_keeps_session_and_adds_no_record |
| p_consume_python_random_without_session | 進行: 保持がなくてもPython乱数を消費 | 1 failed, 37 passed | operations_without_held_session_change_nothing |
| p_consume_torch_random_in_finalize | 終端: 保持がなくてもtorch乱数を消費 | 1 failed, 37 passed | operations_without_held_session_change_nothing |

「検査を更新の後へ移す」変異は`p_move_owner_validation_after_advance`（進行のownerの型検査を、上流の進行の呼出しの後へ移す）。

script: 元checkoutの`venv/refactoring-tests/held-candidate-validation-progress-mutations.py`（実装ファイルを書き換えるので、レビュー担当は実行せず内容を読む）。各変異のpytest出力と`report.json`は`venv/refactoring-tests/held-candidate-validation-progress-mutation-evidence-wsl/`。

## 新実装だけのfresh CPU

旧実装とtest moduleをimportしない別processで、2/4class×5シナリオを実行した。各シナリオは1つの保持ownerと1つの適応記録ownerを使い、警報応答（候補検証の開始）→完了処理→警報応答の記録→保持への反映→標本の観測を行い、到達時の確定（採用・棄却・現行維持・他の保有モデルの再利用）、または途中の警報（候補検証中の応答の反映）と終端回収で終わる。全10条件が成功し、5種類の適応結果がすべて現れた（exit 0）。

確認したこと: 開始の応答の反映で応答のsessionが保持されること、未到達の間は保持が続き記録が増えないこと、候補検証中の応答の反映で保持が変わらないこと、確定・終端回収の後に保持が空になり記録が警報応答の記録の次に並ぶこと、その後の進行と終端回収が記録・学習標本・3種の乱数を変えないこと、空の保持が次の開始の応答を受け入れること、適応結果・位置・ID・切替位置（前のspecと同じ確認）、`sys.modules`に旧実装・test moduleがないこと。途中の警報の応答は、公開constructorで組み立てた`AlarmBufferResponse`（候補検証中、sessionは保持中のもの）を使い、警報応答そのものは再実行していない。

script: `venv/refactoring-tests/held-candidate-validation-progress-fresh-cpu.py`（worktreeルートで実行）。WSLでの出力は`held-candidate-validation-progress-fresh-cpu-wsl.log`。

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

## 依存のexact一致とWSLでの対象test

保持のownerのimportは1 symbol、接続は20 symbol（設計5節）。`dependency_is_allowed`の2つの分岐、両resolverのtupleへの登録、注入契約test 118条件。WSLでは対象38＋依存境界2799のうち2836 passed・1 failed。失敗の1件は本specと無関係の既存test（`test_temporary_model_id_allocation_rejects_every_import_except_annotations[from __future__ import *-False]`）で、Python 3.14が`from __future__ import *`を構文解析の時点でSyntaxErrorにするため（基準のPython 3.13では成功していた）。Windows側のPythonで実行できる静的検査（Ruff check/format 175 files、Pyright、`spec_checks.py names`）は成功。
