# 候補検証の適応記録 — 調査記録

2026-10-08、主担当Claude Code。基点commit `c9fd737`。固定旧基準`748c3aa`。

## 旧処理の事実（`federated_drift_experiment/clients/fedsda.py`）

- `_finalize_forward_validation`（193行〜）は、到達時の確定の最後に`_record_adaptation_event(position=sample_idx, detector=session.detector, action, old_model_id, new_model_id=current_model_id, estimated_change_point=session.estimated_change_point, episode_id=session.episode_id)`を1件追加する。actionは、採用が`create`、参照モデルへの切替が`reuse`、現行維持が`maintain`、棄却が`create_rejected`。
- `local_switch_positions.append(sample_idx)`と`detection_episodes.mark_operation()`は、採用（313–314行）と、切替先が確定前と異なる再利用（324–325、335–336行）だけで実行する。
- `finalize_incomplete_forward_validation`（359行〜）は、位置を`max(session.proposal_position, processed_samples - 1)`、actionを`create_rejected`、old/newをどちらも現在のモデルIDとしてイベントを1件追加する。切替位置は追加しない。
- `reuse_selection_counts`を加算するのは`_resolve_drift`の中の2箇所（823、833行）だけで、上の2つのメソッドは加算しない（ファイル全体のgrepで確認）。
- 新実装の完了情報は、これらの値をすでに持つ: `PostAlarmCandidateValidationCompletion`（判定記録の`resolution_sample_index`・`detector_name`、確定結果、`previous_training_model_id`、推定変化点、episode ID）と`IncompletePostAlarmCandidateValidationFinalization`（判定記録の`finalization_sample_index`・`detector_name`、`current_training_model_id`、推定変化点、episode ID）。完了済みspecのtestが、これらを実旧イベントの各fieldと照合している（`test_post_alarm_candidate_validation_progress.py`の`assert_validation_progress_matches_legacy`、`test_incomplete_post_alarm_candidate_validation_finalization.py`の`assert_incomplete_validation_finalization_matches_legacy`）。
- 上流の4つの型（完了情報、判定記録2種、終端回収の結果）はfieldを検査しない。確定結果のconstructorは結果種別が4値に属することだけを見る。変更記録は検査を持たない。

## oracleの実行可能性

既存`build_validation_progress_oracle`＋`set_scripted_validation_losses`で実旧`_observe_forward_validation`を、既存`build_incomplete_validation_finalization_oracle`で実旧`finalize_incomplete_forward_validation`を実行する。どちらも完了済みspecのtestが同じ形で実行しており、実旧clientは実`adaptation_events`（`AdaptationEvent`）と`local_switch_positions`を持つ。新しい旧メソッドは使わない。旧4確定条件は新の4確定結果と一対一に対応する（下書きの試行で、16条件すべてが期待した適応結果になった）。

## 判断

- **位置fieldの改名**: 設計2節。候補検証の記録を警報位置の名前のfieldへ入れない。旧名のaliasは残さない。
- **未完了の棄却を別の結果値にする**: 旧actionは同じ`create_rejected`だが、新は判定記録の型も理由も別で、履歴から区別できる方が後の集計で情報を落とさない。旧actionへの対応は多対一で照合できる。
- **確定後の学習帰属IDを完了情報から導く**: 学習帰属ownerを引数に取らない。記録は確定時点の事実を写し、依存を記録ownerと完了情報だけに保つ。
- **切替結果の集合を1つの定数にする**: recordの規則とstoreの切替位置の追加が同じ3値を使うので、別々の列挙にしない。testは定数を使わず値を書き下して期待値にする。
- **通知・session・episode**: 本specでは扱わない。通知が保存診断のために必要であることは直前specのresearch.mdにある。

## 手順上の事実

命名を事前登録するため、sourceとtestをリポジトリ外で下書きし、作業ツリーの一時複製へ適用して実行した（worktreeは変更していない。対象102件が成功: 新test 62件と、field名を置換した既存test 40件）。worktreeへは、命名承認の後にtest→RED→srcの順で追加する。
