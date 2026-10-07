# 命名と役割 revision3

## Source

| 名前 | 役割・型・所有 |
| --- | --- |
| post_alarm_candidate_validation_decision_record.py | methodsの不変判定情報、runtime/sessionを持たない |
| PostAlarmCandidateValidationDecisionRecord | metadata＋既存不変評価。再分析情報、状態変更なし |
| candidate_training_interval_sample_count | 候補を最初に学習した区間の標本件数。延べ学習件数とは別 |
| validation_completion_delay_sample_count | 提案から確定までの標本位置差 |
| full_validation_mean_loss_advantage | 全検証区間の参照平均−候補平均。正で候補優位 |
| second_segment_mean_loss_advantage | 検証後半の参照平均−候補平均 |
| reference_mean_loss_difference_from_history | 参照全体平均−履歴平均、履歴なしNone |
| post_alarm_candidate_validation_progress.py | 1標本の観測と到達時確定の組立 |
| advance_post_alarm_candidate_validation | optional sessionを1標本進めて未到達/完了を返す |
| PostAlarmCandidateValidationProgress | 継続session/完了情報の排他的返却、active ownerではない |
| session_to_continue | 次の標本でも呼出側が保持する同session、完了/非activeはNone |
| completed_validation | 完了情報、未到達/非activeはNone |
| PostAlarmCandidateValidationCompletion | 判定記録・適用結果・呼出側記録/通知に必要な値 |
| decision_record | methodsの不変判定record |
| validation_resolution | 状態適用済みの既存resolution record、評価とは別 |
| previous_training_model_id | 確定適用直前の帰属ID、開始時IDとは別 |
| training_model_switch_sample_index | 実帰属変更時の確定位置、それ以外None |
| detection_episode_operation_required | 実帰属変更があったか。episode自体は更新しない |
| validation_session | 進行対象optional既存session |
| _validate_validation_progress_inputs | session/settings/threshold/registry/currentの観測前検査 |
| _validate_validation_progress_threshold | builtin有限非負数の検査とfloat値返却 |

proposal_sample_index/resolution_sample_index/detector_name/estimated_change_point_sample_index/detection_episode_id/post_alarm_candidate_loss_evaluation、全既存型・function/owner引数、sample_index/input_features/observed_class_labels、pending_assignment_sample_concept_ids/upload_delay_round_count/2閾値は既存同義を再利用。

局所名collection_state_snapshot/post_alarm_candidate_loss_collection_state、available_reference_model_ids、reference_losses_by_model_id、held_model_training_state、parameter_name/specified_value、candidate_classifier、candidate_epoch_training_result、assignment_changeは既存同義。validation_progressは返却progressの局所変数、validation_completionは完了recordの局所変数。

## Tests / Evidence

- test_post_alarm_candidate_validation_progress.py: 正常/拒否/実NN接続。
- test_validation_decision_record_matches_legacy / test_validation_decision_record_is_immutable: Task1の不変判定record単独対照・binding固定。
- build_validation_progress_oracle: 既存resolution oracleを開始sessionと進行引数へ結合する。
- assert_validation_progress_matches_legacy: 判定/適応情報/全stateを既存対照helperで照合する。
- test_validation_progress_matches_legacy / test_validation_progress_waits_without_resolution / test_validation_progress_without_session_is_noop: 完了/未到達/非active。
- test_validation_progress_rejects_before_observation / test_validation_progress_records_are_immutable: 事前拒否/不変record。
- test_validation_progress_connects_actual_training_and_resolution: 実NNと後続共同更新。
- test_validation_progress_uses_current_reference_availability: 確定時ID可用性/現在ID。
- test_validation_progress_dependency_contract / test_validation_decision_record_dependency_contract: 2module exact依存契約。
- progress_arguments: 進行関数の明示keyword dict、既存resolution_argumentsとは区別。
- legacy_decision / legacy_adaptation_event: 実旧が生成した判定/適応記録。
- set_scripted_validation_losses / scripted_validation_losses_by_classifier_identity: 経路検証で分類器identityごとに指定した損失を供給するtest fixtureとそのmap。実NN接続とは区別。
- validation_progress_cpu_smoke.py / validation_progress_mutation_evidence.py: Git管理外の起動/変異証拠script。新コード名は必要なら事前追加レビュー。

既存testのclass_count/legacy_resolution_case/pending_sample_count/optimizer_variant/monkeypatch/source_text/expected_acceptance、snapshot_adoption_state/assert_adoption_state_unchanged等のhelper、initial_rng_state/expected_rng_state/invalid_case、field_name/field_value等は同義再利用。新しいhelper/fixture文字列型名を含め必要な追加名は実装前にレビューへ戻す。
