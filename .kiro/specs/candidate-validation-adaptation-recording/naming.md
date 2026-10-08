# 候補検証の適応記録 — 命名 revision1

sourceとtestの実装前一覧。リポジトリ外の下書きを`.kiro/settings/scripts/spec_checks.py names`で命名表と照合した（下書きを適用した作業ツリーの複製で実行。未登録・役割の再利用とも報告なし）。既存名は定義元と同じ役割で再利用する。承認状態はspec.json。

## source

| 名前 | 役割と似た名前との違い |
| --- | --- |
| candidate_validation_adaptation_recording.py | runtimeの新module。候補検証の到達時の確定と未完了の終端回収を適応記録へ写す。既存`alarm_adaptation_recording.py`は警報応答の完了だけを写す。 |
| `record_completed_candidate_validation` | 到達時に確定した候補検証の完了情報（`PostAlarmCandidateValidationCompletion`）を検査して`AdaptationRecord`へ変換し、storeへ1件追加して返す。確定そのものは実行しない。既存`record_completed_alarm_response`と対になる。 |
| `record_incomplete_candidate_validation_finalization` | 未完了の終端回収の結果（`IncompletePostAlarmCandidateValidationFinalization`）を同じく記録する。回収そのものは実行しない。 |
| `ADAPTATION_OUTCOME_BY_VALIDATION_RESOLUTION_OUTCOME` | 確定結果の4値（`POST_ALARM_CANDIDATE_VALIDATION_RESOLUTION_OUTCOMES`）から適応結果への対応表（dict、公開定数）。旧actionとの対応ではない。 |
| `TRAINING_MODEL_SWITCH_ADAPTATION_OUTCOMES` | evaluationの公開定数（tuple）。学習帰属IDが変わる適応結果3値。recordのID整合の規則とstoreの切替位置の追加が共用する。既存property `training_model_switch_sample_index`と同じ「学習帰属の切替」の語を使う。 |
| `adaptation_sample_index` | `AdaptationRecord`のfield。その適応が確定した標本位置（0始まり、単位は標本）。警報応答では警報位置、到達時の確定では確定位置、終端回収では終端位置。旧`AdaptationEvent.position`。既存`alarm_sample_index`（`AlarmResponseCompletion`のfield、警報位置）からの改名で、後者は変更しない。 |
| `validation_completion` | 引数。到達時の完了情報。既存`advance_post_alarm_candidate_validation`内の同名の局所変数と同じ対象。 |
| `incomplete_validation_finalization` | 引数。終端回収の結果。既存testの同名の変数と同じ対象。 |
| `resolution_outcome` | 局所名。確定結果の結果種別（既存fieldと同名・同じ値）。 |

適応結果の新しい値（文字列）: `post_alarm_validation_candidate_adopted`（候補を新モデルとして採用）、`post_alarm_validation_held_model_reused`（検証後に参照モデルを再利用）、`post_alarm_validation_current_model_maintained`（現行モデルを維持）、`post_alarm_validation_candidate_rejected`（要求件数に到達して候補を棄却）、`post_alarm_validation_incomplete_candidate_rejected`（要求件数に届かず終端で棄却）。既存の警報時の値は`alarm_`で始まり、候補検証の値は`post_alarm_validation_`で始まる。警報時の`alarm_interval_held_model_reused`・`alarm_interval_current_model_maintained`とは契機（区間評価か、候補検証の確定か）が違う。

変更する既存module `adaptation_record_store.py` の既存名（`AdaptationOutcome`、`AdaptationRecord`、`AdaptationRecordSnapshot`、`AdaptationRecordStore`、`adaptation_records`、`training_model_switch_sample_indices`、`alternative_model_reuse_count`、`current_model_fit_count`、`append_adaptation_record`、`get_state_snapshot`）は、alarm-adaptation-recordingの命名表が正本で、役割を変えない。

局所名`decision_record`、`validation_resolution`、`previous_training_model_id`、`current_training_model_id`、`training_model_assignment_change`、`adaptation_record`、`adaptation_record_store`、`validated_adaptation_record`、`model_id`、`parameter_name`、`specified_value`は既存の同名と同じ役割。新moduleがimportする9 symbolと、evaluationへ追加する`get_args`（`typing.get_args`）は定義元と同じ役割。

## test

| 名前 | 役割 |
| --- | --- |
| test_candidate_validation_adaptation_recording.py | 新test module。2つの記録関数を実旧のイベント・切替位置と照合する。 |
| `LEGACY_ACTION_BY_VALIDATION_ADAPTATION_OUTCOME` | 候補検証の適応結果5値から旧イベントのaction文字列への対応。照合専用。既存`LEGACY_ACTION_BY_RESPONSE_OUTCOME`（警報応答5値）と合わせて全結果を覆う。 |
| `ADAPTATION_OUTCOME_BY_LEGACY_RESOLUTION_CASE` | 上流oracleの旧確定条件名（create・reuse・maintain・create_rejected）から、期待する適応結果への対応。照合専用。 |
| `INVALID_VALIDATION_COMPLETION_CASES` | 拒否条件名から（元にする旧確定条件、完了情報を不正にする操作、期待する例外）への対応。条件名はこのdictを正本とする。 |
| `INVALID_INCOMPLETE_FINALIZATION_CASES` | 拒否条件名から（終端回収の結果を不正にする操作、期待する例外）への対応。 |
| `snapshot_random_states` | torch・Python global・NumPyの乱数状態を取る。 |
| `assert_random_states_unchanged` | 上の状態と現在の状態全体（NumPyは種別・配列・位置・Gaussian cache）が同じことを確かめる。 |
| `make_populated_adaptation_record_store` | 警報時の再利用1件を先に持つstoreを作る。既存`make_adaptation_record`を使う。 |
| `assert_recorded_event_matches_legacy` | 返された記録とstoreの増分を、実旧の最後のイベント・切替位置と照合する。 |
| `complete_validation_in_both_implementations` | 実旧の標本観測（到達時の確定を含む）と新の候補検証の進行を同じ状態から実行する。記録は行わない。 |
| `finalize_incomplete_validation_in_both_implementations` | 実旧と新の未完了の終端回収を同じ状態から実行する。記録は行わない。 |
| `replace_frozen_fields` | constructorの検査を通さずにfieldだけを差し替えたcopyを作る（手で壊した入力の再現）。 |
| `replace_validation_resolution_fields` | 完了情報の確定結果のfieldを差し替えたcopyを作る。 |
| `replace_decision_record_fields` | 完了情報または終端回収の結果の、判定記録のfieldを差し替えたcopyを作る。 |
| `test_completed_validation_record_matches_real_legacy_event` | 到達時16条件の実旧対照と、記録後の上流状態の再照合。 |
| `test_incomplete_validation_record_matches_real_legacy_event` | 終端8条件の実旧対照。 |
| `test_adaptation_outcomes_cover_alarm_and_validation_results_exactly` | 適応結果の値と順序、対応表、切替結果の定数の整合。 |
| `test_store_applies_switch_index_and_count_rules_for_every_outcome` | 全10結果の切替位置・件数の規則と、ID整合の双方向の拒否。 |
| `test_completed_validation_recording_rejects_invalid_input_before_updating_store` | 到達時の全拒否条件でstoreが不変。 |
| `test_incomplete_validation_recording_rejects_invalid_input_before_updating_store` | 終端の全拒否条件でstoreが不変。 |
| `test_candidate_validation_adaptation_recording_dependency_contract` | 依存境界test内の注入契約test。新moduleのexact 9 symbolの許可・拒否。既存の同種testと同じ形。 |
| `random_states` | 上のhelperが返す3種の乱数状態。helperの引数名としても使う。 |
| `previous_state_snapshot` | 記録の前のstoreのsnapshot。 |
| `validation_adaptation_outcomes` | 候補検証の適応結果5値のtuple。 |
| `training_model_switches` | その適応結果で学習帰属が変わるか（bool）。testは定数を使わず値を書き下して期待値にする。 |
| `record_fields` | `make_adaptation_record`へ渡すfieldのdict。 |
| `frozen_record` / `replaced_record` | 差し替え元の不変recordと、差し替え後のcopy。 |
| `field_values` / `field_name` / `field_value` | 差し替えるfieldのdictと、その1項目。 |
| `completion_or_finalization` | 判定記録を持つ完了情報または終端回収の結果。 |
| `make_invalid_completion` / `make_invalid_finalization` | 拒否条件の、入力を不正にする操作（callable）。 |
| `invalid_validation_completion` | 不正にした完了情報。 |
| `completion` / `finalization` | 拒否条件のlambdaの引数（正常な完了情報、正常な終端回収の結果）。`completion`は既存testの同名の変数と同じ対象。 |
| `torch_random_state` / `python_random_state` / `numpy_random_state` | 乱数状態の各要素。既存testの同名と同じ役割。 |

testが再利用する既存名（`progress_arguments`、`resolution_arguments`、`finalization_arguments`、`shared_optimizer_owners`、`legacy_client`、`legacy_drift_type`、`previous_model_id`、`validation_progress`、`state_snapshot`、`class_count`、`legacy_resolution_case`、`pending_sample_count`、`observed_validation_sample_count`、`processed_sample_count`、`invalid_case`、`expected_exception`、`adaptation_outcome`、`monkeypatch`）は、`test_post_alarm_candidate_validation_progress.py`、`test_incomplete_post_alarm_candidate_validation_finalization.py`、`test_alarm_adaptation_recording.py`の同名と同じ役割。importする既存helper（`make_adaptation_record`、`LEGACY_ACTION_BY_RESPONSE_OUTCOME`、`build_validation_progress_oracle`、`set_scripted_validation_losses`、`assert_validation_progress_matches_legacy`、`build_incomplete_validation_finalization_oracle`、`assert_incomplete_validation_finalization_matches_legacy`）も定義元と同じ役割。条件名（dictのkey）は個別に登録しない。
