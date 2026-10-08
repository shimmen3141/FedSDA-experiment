# 警報1回ぶんの処理の接続 — 調査記録

2026-10-09、主担当Claude Code（Codexから引継ぎ）。基点commit `d764474`。固定旧基準`748c3aa`。実行環境はWindowsの基準環境（Python 3.13、torch 2.12.1+cpu。開始時にtorchの読込みを確認した）。

## 旧処理の事実（`federated_drift_experiment/clients/fedsda.py`）

`process_one_step`（461〜534行）の、警報が起きた標本での処理は次の順。

1. `flush_pending_updates()`（485行。保留中の学習更新の消化）。
2. `detected_event_positions`・`estimated_drift_start_positions`・`detector_candidate_start_positions`への追加と、その間の`_on_drift_alarm(idx)`（486〜492行）。`_on_drift_alarm`は最終構成のclassでは予測側の再生（1444行）。
3. `detection_episodes.observe_detection(idx)`（493行）。無効なら常に`(True, None)`を返す。
4. `_resolve_drift(sample_idx=idx, estimated_start=..., episode_id=...)`（495行）。戻り値が1か2なら`mark_operation()`（無効なら何もしない）。
5. `_forward_validation`がNoneなら`_on_drift_resolution(idx)`（508行。予測側）。

本specが対応するのは4の`_resolve_drift`の呼出し1回。`_resolve_drift`（737〜907行）の中の処理は、完了済みの部品が実旧との対照つきで移植している: 区間の準備と旧側の吸収・再利用評価・切替・候補検証の開始（alarm-buffer-response以前）、検出器resetとFIFOの消費（alarm-response-completion）、イベント・切替位置・再利用件数（alarm-adaptation-recording）、sessionの代入（held-candidate-validation-progress）、`_set_local_current_model`から呼ばれる`_on_local_model_change`の再始動（held-adahedge-diagnostic-notification）。

`_resolve_drift`の中の順は、(候補検証中)全件吸収→イベント→reset→clear。(通常)旧側の吸収→[不足ならイベント→resetで終了]→再利用評価→[切替位置の追加→`_set_local_current_model`（hook）→吸収]または[候補の学習→sessionの代入]→イベント→reset→clear。

## 検出episodeを当面移植しない判断

- `FEDSDA_DETECTION_EPISODES_ENABLED`の既定はFalse（`config.py` 94行）。最終3goldenの設定もfalse（`tests/proposed_regression_golden.json`）。旧11goldenの回帰testと、`tools/baselines/studies/`の比較条件にも有効な条件はない（リポジトリ全体を`detection_episodes`・`DETECTION_EPISODES`でgrepして確認）。
- 無効のとき、`observe_detection`は`(True, None)`を返し、`mark_operation`は何もしない。したがって最終構成では、episode IDは常にNoneで、「同一episodeの追加検出」（`_resolve_episode_duplicate`、718行）へは入らない。
- 既存の部品（警報応答、完了処理、適応記録、候補検証session）は、episode IDを`int | None`で受け取って記録まで運ぶ形になっている。完了処理の結果には「episodeの操作が必要か」を返すpropertyもある。後でepisodeの制御を足す場合、本関数の呼出し側で、警報の前に許可とIDを求め、戻り値を見て操作を記録すればよく、本関数と既存部品の変更は要らない。
- 以上から、制御（旧`DetectionEpisodeController`）と追加検出の経路は当面移植しない。本specはepisode IDを受け取って渡すだけにする。これは主担当の判断で、ユーザーが必要と判断すれば別specで足せる（再開案内の「未移植として残している細目」へ記録する）。

## oracleの実行可能性

- 応答と完了: `test_alarm_response_completion.py`の`build_response_completion_oracle`が、新の引数（応答用・完了用）と実旧clientを返し、`run_legacy_alarm_with_real_completion`が実旧`_resolve_drift`を、イベント記録・検出器reset・FIFO clearも差し替えずに実行する。新側は、応答用の引数から進行中のsessionと提案位置を除き、完了用の監視と警報位置、保持・記録・診断のownerを足して本関数へ渡す。応答用の検出器名は実旧の`_detector_label()`と同じ値（下書きのtestで確認）。
- 候補検証中: 上流の記録oracle（`build_completed_recording_oracle`）と同じ形で、1回目の警報（候補検証の開始）の後、保留位置を消費した同じ位置で、保留標本なしの2回目の警報を受ける。新側は1回目で保持させたsessionを2回目が保持から読む。
- 再始動: 上流のoracleの実旧clientは`SharedBackboneClassConditionalESRFedSDAClient`で、`_on_local_model_change`は帰属変更を記録するだけのhookへ差し替えられている（上流の照合helperがその記録を読む）。本specは、その記録を残したまま、実旧`RestartingSoftRoutingClassConditionalESRFedSDAClient._on_local_model_change`（2052行）を同じclientに対して呼ぶhookを置く。このメソッドが読む属性は`expert_router`・`context_expert_routers`・`shadow_meta_routers`・`routing_active_set`だけで、held-adahedge-diagnostic-notificationのoracleと同じ値（実`AdaHedgeRouter`、空、空、None）を与える。再始動が観測できるよう、警報の前に新旧へ同じ損失を1回与える。リポジトリ外の下書きを作業ツリーの複製で実行し、10条件（2/4class×5種類）が成功することを確かめた。

## 判断

- **進行中のsessionを保持から読む**: 既存の応答は進行中のsessionを引数で受ける。呼出し側が保持と別の値を渡す余地をなくすため、本関数は保持だけを受け取る。これで、保持への反映（段4）の前提（候補検証中の応答は保持中のsessionを指す、それ以外は保持が空）が、応答の契約から常に満たされる。
- **提案位置を警報位置にする**: 旧は同じ`sample_idx`を両方に使う。引数を1つにして、食い違う入力をなくす。
- **応答より後の段だけが検査する値を先に確かめる**: 応答は多くのownerを更新する。検出器名・推定変化点・episode IDは、候補検証中と不足の経路では応答が検査せず、完了処理と記録が応答の更新の後で検査する。適応記録の規則を複製せず、同じ値で記録を1つ組み立てて検査させる（保存しない）。
- **警報位置と最終観測位置の一致を先に確かめる**: 完了処理の検査を、応答の更新より前へ持ってくる。
- **通知を最後に置く**: 診断証拠は他の段が読まない。旧は切替の直後に再始動するが、最終状態は同じ（対照testで確認）。応答が返した帰属変更を使うので、応答より後であればよい。記録・保持より後にしたのは、学習と記録の状態を先に確定させ、診断だけを最後にまとめるため。
- **結果は完了情報と記録の2つ**: 応答は完了情報の中にある。旧の戻り値（`drift_type` 0/1/2）は、完了情報の「帰属が変わったか」と結果種別から決まるので持たない。標本ごとの履歴への記録は標本1件の処理全体のspecで扱う。

## 手順上の事実

- 命名の事前登録のため、sourceとtestをリポジトリ外で下書きし、`git archive HEAD`で作った作業ツリーの複製へ置いて、Windowsの基準環境のPythonで実行した（worktreeは変更していない。対象40件が成功、Ruff・Pyright成功）。`spec_checks.py names`は、下書きを置いた複製で報告なし。
- 下書きの途中で、応答がどの経路でも自分で検査する2つのowner（現在の学習帰属、損失統計）の型検査を本関数から外した（重複のため）。
