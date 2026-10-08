# 警報応答の完了処理 — 検出力と新実装単独の動作（Task 2）

対象commit `def37e4`（source・testはこのcommitの内容）。主担当Claude Code、2026-10-08、Windows CPUの固定venv。

## 実source変異

`src/federated_learning_experiments/runtime/alarm_response_completion.py`を1種ずつ書き換え、`tests/refactoring/test_alarm_response_completion.py`（57件）を実行した。各回finallyで元byteへ戻し、戻した後のbyteが元と同じであることをassertしている。収集失敗・構文失敗は検出に数えず、pytestがexit 1で「failed」を報告した場合だけを検出とした（変異後のsourceは事前にcompileで構文を確認）。

結果: **33/33検出**。復元後は57 passed in 3.81s。sourceのLF sha256は`d4903fda4b898e48dd7cb57e29cb8a3c63ad424dd260cc28fdb3d4a006837bc6`（実行前後のbyte sha256は`d4903fda4b898e48dd7cb57e29cb8a3c63ad424dd260cc28fdb3d4a006837bc6`で同じ）。

| 変異 | 対象testの結果 |
| --- | --- |
| skip_monitor_reset | 34 failed, 23 passed in 4.67s |
| reset_with_fixed_baseline | 31 failed, 26 passed in 5.22s |
| baseline_from_previous_model | 5 failed, 52 passed in 4.53s |
| baseline_ignores_missing_statistics_rule | 2 failed, 55 passed in 4.26s |
| skip_drain | 18 failed, 39 passed in 4.50s |
| drain_even_when_too_short | 12 failed, 45 passed in 4.49s |
| record_reports_no_drained_indices | 14 failed, 43 passed in 4.67s |
| build_record_after_updates | 2 failed, 55 passed in 4.33s |
| skip_assignment_change_type_check | 1 failed, 56 passed in 4.41s |
| skip_assignment_change_id_type_check | 1 failed, 56 passed in 4.75s |
| drain_before_reset | 1 failed, 56 passed in 4.58s |
| swap_previous_and_current_ids | 4 failed, 53 passed in 4.78s |
| previous_id_never_differs | 7 failed, 50 passed in 4.62s |
| switch_index_always_reported | 29 failed, 28 passed in 5.08s |
| episode_operation_always_required | 29 failed, 28 passed in 5.04s |
| drop_change_point_from_record | 28 failed, 29 passed in 4.67s |
| drop_episode_from_record | 28 failed, 29 passed in 5.77s |
| skip_last_observed_index_check | 2 failed, 55 passed in 4.20s |
| accept_buffer_without_observed_sample | 1 failed, 56 passed in 3.95s |
| skip_prepared_interval_check | 2 failed, 55 passed in 4.11s |
| skip_assignment_consistency_check | 1 failed, 56 passed in 4.72s |
| accept_bool_alarm_index | 1 failed, 56 passed in 4.16s |
| accept_any_monitor_type | 1 failed, 56 passed in 3.93s |
| accept_any_statistics_store_type | 1 failed, 56 passed in 3.87s |
| consume_torch_random | 25 failed, 32 passed in 4.32s |
| consume_python_random | 25 failed, 32 passed in 4.00s |
| consume_numpy_random_array | 25 failed, 32 passed in 4.03s |
| change_numpy_gaussian_cache_only | 1 failed, 56 passed in 3.84s |
| validate_after_reset | 11 failed, 46 passed in 4.05s |
| accept_negative_optional_index | 3 failed, 54 passed in 3.83s |
| record_not_frozen | 1 failed, 56 passed in 3.89s |
| record_accepts_inconsistent_ids | 8 failed, 49 passed in 4.01s |
| record_accepts_drained_indices_for_too_short | 5 failed, 52 passed in 3.91s |

変異の内容（上の名前の順）: 監視のreset省略／基準を固定値に／基準を変更前モデルの統計から選ぶ／統計の件数条件を変える／drain省略／不足でもdrain／recordの消費位置を常に空に／recordの組立を更新の後へ戻す（設計r2までの順序）／変更記録のexact型検査の削除／変更記録のIDの型検査の削除／drainをresetより先に／変更前後IDの入替え／変更前IDを常に現行IDに／切替位置を常に返す／episode操作を常に要求／推定変化点をrecordから落とす／episode IDをrecordから落とす／最終観測位置の検査の削除／未観測FIFOの受理へ戻す（設計r4までの挙動）／準備済み区間の検査の削除／応答と現在の帰属の対応検査の削除／警報位置のbool受理／監視のexact型検査の削除／統計storeのexact型検査の削除／torch乱数の消費／Python乱数の消費／NumPy乱数の配列の消費／NumPyのGaussian cacheだけの変更／検査の前にresetを実行／任意indexの負値受理／recordのfrozen解除／recordのID整合検査の削除／不足の応答で消費位置を持つrecordの受理。

経過: 設計r4の時点の1回目の実行で「応答と現在の帰属の対応検査の削除」が未検出だった。testが応答後の帰属を変更前のモデルへ戻しており、recordの検査でも同じ例外になっていたためで、変更前後のどちらでもないIDへ変えるようtestを直した（review.md）。Task 2・3の独立レビュー1回目が、testとfresh CPU scriptのNumPy乱数の比較が状態の配列だけで、位置とGaussian cacheを比べていないことを指摘した。testとscriptを状態全体（種別・配列・位置・Gaussian cache）の比較へ改め、乱数消費の変異4種（torch、Python global、NumPyの配列、NumPyのGaussian cacheだけ）を加えた。当初この変異を省いていた判断（依存にtorch・randomがないので不要）は取り下げた。Gaussian cacheだけの変異は、配列だけの比較では検出できない。

script: 元checkoutの`venv/refactoring-tests/alarm-response-completion-mutations.py`（実装ファイルを書き換えるので、レビュー担当は実行せず内容を読む）。各変異のpytest出力と`report.json`は`venv/refactoring-tests/alarm-response-completion-mutation-evidence-r6/`。

## 新実装だけのfresh CPU

旧実装とtest moduleをimportしない別processで、2/4class×5経路（不足・維持・再利用・候補検証開始・候補検証中の2回目の警報）の10条件を、`respond_to_alarm_with_buffered_samples`→`complete_alarm_buffer_response`の順に実行した。全10条件が成功（exit 0）。

各条件で確認したこと: 応答結果、完了処理の前後で学習標本・損失統計・3種の乱数（torch・Python global・NumPyの状態全体）と明示Randomが不変、変更前後の帰属ID・切替位置・episode操作の要否、監視の状態が「応答後の現行モデルの統計から選んだ基準で作り直した監視」のsnapshotと等しいこと、保留位置の保持（不足）と全件消費（その他）、最終観測位置の維持、再開した監視が次の位置の観測を受け付けること、`sys.modules`に旧実装・test moduleがないこと。

script: `venv/refactoring-tests/alarm-response-completion-fresh-cpu.py`（worktreeルートで実行）。出力は`alarm-response-completion-fresh-cpu.log`。

```text
PASS 2 classes: too_short alarm_change_interval_too_short baseline 0.5033514499664307 drained 0 ; legacy imports=0
PASS 2 classes: maintained alarm_interval_current_model_maintained baseline 0.5033514159066336 drained 11 ; legacy imports=0
PASS 2 classes: reused alarm_interval_held_model_reused baseline 0.5193137356213161 drained 11 ; legacy imports=0
PASS 2 classes: started alarm_interval_candidate_validation_started baseline 0.12583786249160767 drained 11 ; legacy imports=0
PASS 2 classes: during_validation alarm_during_candidate_validation baseline 0.32112041115760803 drained 3 ; legacy imports=0
PASS 4 classes: too_short alarm_change_interval_too_short baseline 0.7563734650611877 drained 0 ; legacy imports=0
PASS 4 classes: maintained alarm_interval_current_model_maintained baseline 0.7563734139714922 drained 11 ; legacy imports=0
PASS 4 classes: reused alarm_interval_held_model_reused baseline 0.7403211508478436 drained 11 ; legacy imports=0
PASS 4 classes: started alarm_interval_candidate_validation_started baseline 0.18909336626529694 drained 11 ; legacy imports=0
PASS 4 classes: during_validation alarm_during_candidate_validation baseline 0.4762001906832059 drained 3 ; legacy imports=0
```

## 依存のexact一致

新moduleのimportは8 symbol（設計r5の5節）。`tests/refactoring/test_single_run_dependency_boundaries.py`の`dependency_is_allowed`の分岐（8 symbol）、両resolverのtupleへの登録、注入契約test 87条件。対象57＋依存境界2542＝2599 passed（Task 1の実測）。
