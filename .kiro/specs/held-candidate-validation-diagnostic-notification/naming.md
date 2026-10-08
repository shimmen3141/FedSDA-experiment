# 候補検証の確定に伴う診断通知 — 命名 revision1

変更するsourceとtestの、追加する名前の実装前一覧。リポジトリ外の下書きを、作業ツリーの複製へ置いて`.kiro/settings/scripts/spec_checks.py names`で照合した。新しいmodule・関数・class・結果recordはsourceに追加しない。承認状態はspec.json。

## source（`runtime/held_candidate_validation_progress.py`）

| 名前 | 役割と似た名前との違い |
| --- | --- |
| `diagnostic_evidence_collection` | `advance_held_candidate_validation`へ足す引数。exact `AdaHedgeDiagnosticEvidenceCollection`。既存の通知`notify_diagnostics_of_training_assignment_change`と`handle_alarm_occurrence`の同名の引数と同じ役割（通知の受け手）。 |

関数名`advance_held_candidate_validation`は変えない。確定時に通知まで行うようになるが、「保持中の候補検証を標本1件ぶん進める」という役割は同じで、通知は確定の後始末の1段（記録、保持の解除と並ぶ）として扱う。追加するimport（`AdaHedgeDiagnosticEvidenceCollection`、`notify_diagnostics_of_training_assignment_change`）は定義元と同じ役割。

## test（`tests/refactoring/test_held_candidate_validation_progress.py`）

| 名前 | 役割 |
| --- | --- |
| `DIAGNOSTIC_LOSSES_BEFORE_VALIDATION` | 再始動を観測できるよう、確定の前に新旧の診断証拠へ与える損失（モデルIDから損失）。 |
| `make_diagnostics_observing_losses` | 上の損失を1回観測済みの診断証拠の保持集合を作る。実旧clientを渡すと、同じ損失を観測済みの実旧routerと、実旧の再始動hookを呼ぶhookをそのclientへ置く。 |
| `record_local_model_change` | 上流のoracleが実旧clientへ置いている、帰属変更を記録するだけのhook（元の値）。alarm-occurrence-handlingのtestの同名と同じ役割。 |
| `record_change_then_restart_legacy_routers` | 実旧clientへ置くhook。上の記録を行ってから、実旧の再始動hookを同じclientに対して実行する。記録と委譲だけで判定を持たない。alarm-occurrence-handlingのtestの同名と同じ役割。 |
| `notification_calls` | 記録用wrapperが記録した（通知の引数、その時点の保持中のsession、その時点の適応記録の件数）のlist。 |
| `notify_diagnostics` | 差し替える前の実`notify_diagnostics_of_training_assignment_change`。 |
| `record_notification_call` | 記録用wrapper。通知の時点の状態を記録して実関数へ委譲する。判定を持たない。 |
| `notification_arguments` | 通知へ渡ったkeyword引数。 |
| `held_session_at_notification` / `record_count_at_notification` | 通知の時点の、保持中のsessionと適応記録の件数。 |
| `diagnostic_snapshot` | 操作の前の診断証拠の保持集合の状態（既存`get_diagnostic_collection_snapshot`の戻り値）。 |
| `placeholder_arguments` | 既存`make_placeholder_arguments`の戻り値（保持・記録・診断のowner以外を`object()`にした引数dict）。 |
| `invalid_owner_kind` | parametrize引数。不正なownerの種類（"other_type"・"subclass"）。 |
| `invalid_owner` | 拒否されるべきownerの値。 |
| `owner_subclass` | 正しいownerのclassの派生class（初期化しないinstanceを作る）。alarm-occurrence-handlingのtestの同名と同じ役割。 |
| `global_diagnostic_evidence` | 局所名。診断証拠の保持集合の同名のpropertyが返す値。 |
| `validation_resolution` | 局所名。確定の結果（既存の完了情報の同名のfieldの値）。 |
| `test_held_candidate_validation_diagnostic_notification_dependency_contract` | 依存境界test内の注入契約test。追加する2 symbolの許可・拒否。 |

既存の名前は、held-candidate-validation-progressの命名revision2で承認済みのまま、同じ役割で使う: `HeldCandidateValidationAdvance`、`HeldIncompleteCandidateValidationFinalization`、`_validate_holder_and_record_store`、`advance_held_candidate_validation`、`apply_alarm_response_to_validation_session_holder`、`finalize_held_incomplete_candidate_validation`、`held_validation_session`、`release_validation_session`、`INJECTED_SESSION_MARKER`、`INVALID_ALARM_RESPONSE_HOLDER_CASES`、`held_incomplete_finalization`、`held_progress_module`、`held_session_source`、`held_validation_advance`、`invalid_validation_session`、`legacy_event_count`、`make_holder_holding`、`make_invalid_response`、`make_placeholder_arguments`、`make_record_store_with_alarm_record`、`make_started_validation_session`、`record_counts_at_release`、`record_release_of_validation_session`、`second_validation_session`、`upstream_operation_name`、`invalid_owner_name`、`operation`と、既存のtest関数8つ（`test_alarm_response_is_rejected_before_changing_holder`、`test_alarm_response_updates_holder_like_legacy_session_attribute`、`test_completed_held_validation_is_recorded_then_released_like_legacy`、`test_incomplete_held_validation_is_recorded_then_released_like_legacy`、`test_operations_without_held_session_change_nothing`、`test_owner_types_are_rejected_before_upstream_updates`、`test_session_holder_holds_one_session_and_rejects_before_changing`、`test_unfinished_held_validation_keeps_session_and_adds_no_record`）。`test_completed_held_validation_is_recorded_then_released_like_legacy`は、診断の照合と通知の時点の確認を足すが、名前は変えない（記録→解除の順の確認が引き続き中心で、通知はその後の1段）。importする既存helper（`assert_adahedge_matches_legacy`、`get_diagnostic_collection_snapshot`）と実旧のclass（`RestartingSoftRoutingClassConditionalESRFedSDAClient`、`AdaHedgeRouter`）は定義元と同じ役割。
