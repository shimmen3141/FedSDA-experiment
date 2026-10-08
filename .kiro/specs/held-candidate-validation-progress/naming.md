# 候補検証sessionの保持と進行 — 命名 revision1

sourceとtestの実装前一覧。リポジトリ外の下書きを、作業ツリーの複製へ置いて`.kiro/settings/scripts/spec_checks.py names`で照合した（未登録・役割の再利用とも報告なし）。既存名は定義元と同じ役割で再利用する。承認状態はspec.json。

## source

| 名前 | 役割と似た名前との違い |
| --- | --- |
| candidate_validation_session_holder.py | runtimeの新module。進行中の候補検証sessionを高々1つ保持するowner。sessionを作る既存`post_alarm_candidate_validation_session_start.py`、進める既存`post_alarm_candidate_validation_progress.py`とは別で、保持と解除だけを行う。 |
| `CandidateValidationSessionHolder` | 上のowner。旧`_forward_validation`属性に対応する。開始・進行・確定の判断は行わない。 |
| `held_validation_session` | holderのreadonly property。保持中の`PostAlarmCandidateValidationSession`またはNone。応答の既存field `active_validation_session`（その応答の時点で進行中のsession）と違い、ownerが現在持っている値。 |
| `hold_validation_session` | holderの操作。空のときだけ引数`validation_session`を保持する。 |
| `release_validation_session` | holderの操作。保持中のsessionを外して返す。 |
| `released_validation_session` | 局所名。外して返すsession。 |
| held_candidate_validation_progress.py | runtimeの新module。保持中のsessionについて、警報応答の反映・進行・終端回収を、適応記録と保持の更新へつなぐ。既存`post_alarm_candidate_validation_progress.py`はsessionを引数で受けて進めるだけで、保持と記録を扱わない。 |
| `apply_alarm_response_to_validation_session_holder` | 完了した`AlarmBufferResponse`の結果種別とsessionをholderへ反映する。応答そのものは実行しない。 |
| `advance_held_candidate_validation` | 保持中のsessionへ標本1件の進行を行い、確定したら適応記録を追加して保持を解除する。既存`advance_post_alarm_candidate_validation`の「held」版で、sessionを引数でなくholderから取る。 |
| `finalize_held_incomplete_candidate_validation` | 実験の終端で保持中の未完了sessionを回収し、適応記録を追加して保持を解除する。既存`finalize_incomplete_post_alarm_candidate_validation`の「held」版。 |
| `HeldCandidateValidationAdvance` | `advance_held_candidate_validation`の不変結果（frozen/kw_only）。field `validation_progress`（既存`PostAlarmCandidateValidationProgress`）と`adaptation_record`（確定時だけ。既存`AdaptationRecord`）。 |
| `HeldIncompleteCandidateValidationFinalization` | `finalize_held_incomplete_candidate_validation`の不変結果。field `incomplete_validation_finalization`（既存の終端回収の結果）と`adaptation_record`。 |
| `validation_progress` | 上のrecordのfieldと局所名。既存testの同名の変数と同じ対象（進行の戻り値）。 |
| `validation_session_holder` | 引数。holderのexact instance。 |
| `_validate_holder_and_record_store` | module内の検査helper。holderと適応記録のownerのexact型を確かめる。進行と終端回収が共用する。 |

引数`adaptation_record_store`、`alarm_buffer_response`、`validation_session`、`sample_index`、`input_features`、`observed_class_labels`、`candidate_model_training_and_acceptance_settings`、`maximum_reference_mean_loss_increase`、`minimum_candidate_mean_loss_improvement`、`pending_assignment_sample_concept_ids`、`temporary_model_id_allocator`、`upload_delay_round_count`、`held_model_training_state_registry`、`loss_statistics_store`、`training_sample_store`、`model_training_and_assignment_counts_store`、`current_training_model_assignment`、`pending_model_upload_state`、`processed_sample_count`と、局所名`adaptation_record`、`incomplete_validation_finalization`は、既存の進行・終端回収・記録の同名と同じ役割。importする既存symbol（設計5節の1＋20）も定義元と同じ役割。

## test

| 名前 | 役割 |
| --- | --- |
| test_held_candidate_validation_progress.py | 新test module。保持と3つの関数を、実旧のsession保持・適応イベントと照合する。 |
| `INVALID_ALARM_RESPONSE_HOLDER_CASES` | 拒否条件名から（元にする警報応答の条件、応答の前に保持させるsessionの種類、応答を不正にする操作、期待する例外）への対応。条件名はこのdictを正本とする。 |
| `make_started_validation_session` | 実際の候補・参照・損失収集を持つ開始済みsessionを1つ作る（既存oracleから取り出す）。 |
| `make_holder_holding` | 指定のsessionを保持した（Noneなら空の）holderを作る。 |
| `make_record_store_with_alarm_record` | 警報時の記録1件を先に持つ適応記録ownerを作る。 |
| `make_placeholder_arguments` | holderと記録owner以外を`object()`にした引数dictを作る。保持がない経路と、型の拒否が上流より前であることの確認に使う。 |
| `record_release_of_validation_session` | 記録用wrapper。解除の時点の適応記録の件数を記録して、実`release_validation_session`へ委譲する。 |
| `test_session_holder_holds_one_session_and_rejects_before_changing` | holderの保持・解除・拒否と状態不変。 |
| `test_alarm_response_updates_holder_like_legacy_session_attribute` | 2/4class×警報応答5種類で、反映後の保持を実旧の`_forward_validation`と照合。 |
| `test_alarm_response_is_rejected_before_changing_holder` | 反映の全拒否条件で保持が不変。 |
| `test_completed_held_validation_is_recorded_then_released_like_legacy` | 到達時の確定: 記録→解除の順、実旧イベントと進行結果の照合。 |
| `test_unfinished_held_validation_keeps_session_and_adds_no_record` | 未到達の3標本で保持が続き、記録が増えないこと。 |
| `test_operations_without_held_session_change_nothing` | 保持が空のとき、進行と終端回収が何も更新しないこと。 |
| `test_owner_types_are_rejected_before_upstream_updates` | 2関数×2ownerで、型の拒否が上流の呼出しより前であること。 |
| `test_incomplete_held_validation_is_recorded_then_released_like_legacy` | 終端回収: 記録と解除、解除後の再回収が何も変えないこと。 |
| `test_held_candidate_validation_progress_dependency_contract` | 依存境界test内の注入契約test。2 moduleのexact symbolの許可・拒否。既存の同種testと同じ形。 |
| `held_progress_module` | 新moduleのimport別名（test専用）。上流の関数を差し替えるために使う。既存testの`progress_module`と同じ使い方。 |
| `held_validation_advance` | `advance_held_candidate_validation`の戻り値。 |
| `held_incomplete_finalization` | `finalize_held_incomplete_candidate_validation`の戻り値。 |
| `held_session_source` | 拒否条件で、応答の前に保持させるsessionの種類（"none"・"response"・"other"）。 |
| `make_invalid_response` | 拒否条件の、応答を不正にする操作（callable）。 |
| `invalid_validation_session` / `second_validation_session` | holderが拒否するべき値（sessionでない値、保持中に渡す2つ目のsession）。 |
| `record_counts_at_release` | 解除が呼ばれた時点の適応記録の件数のlist。記録だけで判定を持たない。 |
| `legacy_event_count` | 実旧の適応イベントの件数（再回収で増えないことの確認）。 |
| `upstream_operation_name` / `invalid_owner_name` | parametrize引数。差し替える上流の関数名、不正にするownerの引数名。 |
| `operation` | parametrize引数とhelperの引数。対象の関数（進行または終端回収）。 |
| `argument_name` / `argument` | 引数dictの1項目。 |

testが再利用する既存名（`progress_arguments`、`resolution_arguments`、`finalization_arguments`、`shared_optimizer_owners`、`legacy_client`、`legacy_session`、`legacy_drift_type`、`previous_model_id`、`previous_state_snapshot`、`state_snapshot`、`random_states`、`alarm_response_completion`、`recording_oracle_case`、`response_outcome`、`class_count`、`legacy_resolution_case`、`observed_validation_sample_count`、`invalid_case`、`expected_exception`、`monkeypatch`）と、importする既存helper・定数（`RECORDING_ORACLE_CASES`、`build_completed_recording_oracle`、`make_adaptation_record`、`assert_random_states_unchanged`、`assert_recorded_event_matches_legacy`、`replace_frozen_fields`、`snapshot_random_states`、`build_validation_progress_oracle`、`set_scripted_validation_losses`、`assert_validation_progress_matches_legacy`、`build_incomplete_validation_finalization_oracle`、`assert_incomplete_validation_finalization_matches_legacy`）は、定義元と同じ役割。条件名（dictのkey）は個別に登録しない。
