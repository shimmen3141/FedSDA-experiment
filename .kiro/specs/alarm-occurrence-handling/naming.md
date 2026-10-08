# 警報1回ぶんの処理の接続 — 命名 revision1

sourceとtestの実装前一覧。リポジトリ外の下書きを、作業ツリーの複製へ置いて`.kiro/settings/scripts/spec_checks.py names`で照合した。既存名は定義元と同じ役割で再利用する。承認状態はspec.json。

## source

| 名前 | 役割と似た名前との違い |
| --- | --- |
| alarm_occurrence_handling.py | runtimeの新module。警報が起きた標本での処理を、既存の5つの部品の呼出しとして並べる。既存`alarm_buffer_response.py`（保留標本への応答だけ）、`alarm_response_completion.py`（応答の後始末だけ）、`alarm_adaptation_recording.py`（記録だけ）を呼ぶ側。 |
| `handle_alarm_occurrence` | 警報1回ぶんの処理。応答→完了→記録→保持への反映→診断通知を各1回実行する。既存`respond_to_alarm_with_buffered_samples`は最初の段だけを行い、進行中のsessionを引数で受ける。本関数はsessionを保持から読み、後の4段まで行う。警報の検出は行わない。 |
| `AlarmOccurrenceHandling` | `handle_alarm_occurrence`の不変結果（frozen/kw_only）。field `alarm_response_completion`（既存`AlarmResponseCompletion`。既存の同名の引数と同じ対象）と`adaptation_record`（既存`AdaptationRecord`。追加した記録）。 |
| `owner_argument_name` | 局所名。exact型を確かめるownerの引数名（例外の文言に使う）。 |
| `owner` | 局所名。exact型を確かめるownerの値。既存testの同名の変数と同じく、状態を所有するobjectを指す。 |
| `required_owner_type` | 局所名。そのownerに要求するclass。 |

引数`validation_session_holder`、`adaptation_record_store`、`diagnostic_evidence_collection`、`loss_change_monitor`、`alarm_sample_index`と、既存の応答の引数（`pending_training_assignment_buffer`、`pending_sample_observations`、`estimated_change_span_sample_count`、`minimum_change_interval_sample_count`、`model_evaluation_sample_store`、`python_random_generator`、`maximum_alarm_interval_mean_loss_increase`、`held_model_training_state_registry`、`loss_statistics_store`、`training_sample_store`、`model_training_and_assignment_counts_store`、`current_training_model_assignment`、`candidate_parameter_initialization_settings`、`architecture_reference_classifier`、`parameter_optimizer_settings`、`candidate_epoch_training_settings`、`candidate_model_training_and_acceptance_settings`、`estimated_change_point_sample_index`、`detection_episode_id`、`detector_name`）、局所名`alarm_buffer_response`、`alarm_response_completion`、`adaptation_record`、`change_interval_resolution`は、既存の5つの部品の同名と同じ役割。importする既存symbol（設計5節の27）も定義元と同じ役割。

## test

| 名前 | 役割 |
| --- | --- |
| test_alarm_occurrence_handling.py | 新test module。接続を、実旧の警報解決・session属性・帰属変更hookと照合する。 |
| `occurrence_module` | 新moduleのimport別名（test専用）。5つの段を差し替えるために使う。既存testの`held_progress_module`と同じ使い方。 |
| `ALARM_OCCURRENCE_STEP_NAMES` | 接続する5つの段の、新moduleが参照する名前（実行順）。 |
| `ARGUMENT_NAMES_ADDED_TO_RESPONSE` | 応答の引数に対して本関数が追加する5つの引数名。 |
| `ARGUMENT_NAMES_DERIVED_FOR_RESPONSE` | 応答の引数のうち、本関数が受け取らずに自分で与える2つの引数名。 |
| `DIAGNOSTIC_LOSSES_BEFORE_ALARM` | 再始動を観測できるよう、警報の前に新旧の診断証拠へ与える損失（モデルIDから損失）。 |
| `REUSED_HELD_MODEL_ORACLE_CASE` | 既存`RECORDING_ORACLE_CASES`のうち、他の保有モデルを再利用する条件。順序・拒否・失敗のtestの既定に使う。 |
| `OWNER_ARGUMENT_NAMES_VALIDATED_BEFORE_RESPONSE` | 応答より前にexact型を確かめる5つのownerの引数名。 |
| `INVALID_ALARM_OCCURRENCE_INPUT_CASES` | 拒否条件名から（不正にする引数名、正常な引数から不正な値を作る操作、期待する例外）への対応。条件名はこのdictを正本とする。 |
| `build_alarm_occurrence_oracle` | 既存`build_response_completion_oracle`へ、保持・記録・診断のownerと、実旧の再始動hookを加える。新関数の引数dictと、既存helperの引数・実旧clientを返す。 |
| `record_local_model_change` | 上流のoracleが実旧clientへ置いている、帰属変更を記録するだけのhook（元の値）。 |
| `record_change_then_restart_legacy_routers` | 実旧clientへ置くhook。上の記録を行ってから、実旧の再始動hookを同じclientに対して実行する。記録と委譲だけで判定を持たない。 |
| `snapshot_owners_updated_after_response` | 応答より後の段が更新するowner（監視、保留位置、記録、保持、診断）の状態をまとめて読む。 |
| `make_uninitialized_subclass_instance` | exact型の検査が拒否するべき、ownerと同じclassの派生型の値を、初期化せずに作る。 |
| `owner_subclass` | 上で作る派生class。 |
| `get_last_observed_sample_index` | 引数dictの保留位置ownerから、最終観測位置を読む。 |
| `test_alarm_occurrence_signature_extends_response_arguments` | 引数の集合が「応答の引数−2＋5」で、全てkeyword-only・既定値なしであること。 |
| `test_alarm_occurrence_matches_real_legacy_alarm` | 2/4class×警報応答5種類で、応答・完了・記録・保持・診断・乱数を実旧と照合。 |
| `test_alarm_occurrence_runs_each_step_once_in_order` | 5段の順と回数、段の間で渡る値の同一性。 |
| `test_alarm_occurrence_rejects_invalid_input_before_any_step` | 全拒否条件で、どの段も呼ばれず状態が不変。 |
| `test_failed_step_stops_alarm_occurrence_before_later_steps` | 各段の失敗で、後の段が呼ばれないこと。 |
| `test_alarm_occurrence_handling_record_is_frozen_and_keyword_only` | 結果recordの形。 |
| `test_alarm_occurrence_handling_dependency_contract` | 依存境界test内の注入契約test。新moduleのexact symbolの許可・拒否。既存の同種testと同じ形。 |
| `handling_arguments` / `invalid_handling_arguments` | 新関数の引数dictと、その1項目を不正な値へ替えたもの。 |
| `alarm_occurrence_handling` | 新関数の戻り値。 |
| `handling_parameters` / `response_parameters` | 新関数と既存の応答の、signatureの引数。 |
| `global_diagnostic_evidence` | 局所名。診断証拠の保持集合の同名のpropertyが返す値（global用の診断証拠）。 |
| `pending_sample_indices_before_alarm` | 新関数を呼ぶ直前の保留位置。 |
| `expected_record_count` | 記録ownerに期待する記録の件数（候補検証中の条件は2回の警報で2）。 |
| `step_calls` | 記録用wrapperが記録した（段の名前、引数、結果）のlist。拒否のtestでは、段の名前から「呼ばれたら失敗するmock」へのdict。 |
| `real_steps` | 段の名前から、差し替える前の実関数へのdict。 |
| `record_step_call` | 記録用wrapper。実関数へ委譲し、名前・引数・結果を記録する。判定を持たない。 |
| `step_name` / `step_arguments` / `step_result` / `step_call` | 1つの段の名前、keyword引数、戻り値、差し替えたmock。 |
| `response_step_arguments` / `completion_step_arguments` / `recording_step_arguments` / `holder_step_arguments` / `notification_step_arguments` | 5つの段それぞれへ渡ったkeyword引数。 |
| `make_invalid_argument` | 拒否条件の、不正な値を作る操作（callable）。 |
| `failing_step_name` / `failing_step_position` | parametrize引数（失敗させる段の名前）と、5段の中での位置。 |
| `later_step_calls` | 失敗させた段より後の段の名前から、差し替えたmockへのdict。 |
| `adaptation_record_count` | 記録ownerの記録の件数。 |

testが再利用する既存名（`response_arguments`、`completion_arguments`、`preparation_arguments`、`resolution_arguments`、`shared_optimizer_owners`、`legacy_client`、`legacy_result`、`legacy_weights`、`diagnostic_weights`、`diagnostic_evidence_collection`、`validation_session_holder`、`adaptation_record_store`、`pending_training_assignment_buffer`、`alarm_buffer_response`、`alarm_response_completion`、`adaptation_record`、`adaptation_record_snapshot`、`started_validation_session`、`training_model_assignment_change`、`initial_training_model_id`、`initial_torch_random_state`、`expected_torch_random_state`、`torch_random_state_before_second_alarm`、`expected_response_outcome`、`recording_oracle_case`、`alarm_interval_resolution_case`、`estimated_change_span_sample_count`、`class_count`、`state_snapshot`、`invalid_case`、`expected_exception`、`argument_name`、`argument`、`parameter`、`record_field`、`previous_model_id`、`current_model_id`、`monkeypatch`）と、importする既存helper・定数（`RECORDING_ORACLE_CASES`、`LEGACY_ACTION_BY_RESPONSE_OUTCOME`、`build_response_completion_oracle`、`run_legacy_alarm_with_real_completion`、`assert_response_completion_matches_legacy`、`assert_buffer_response_matches_legacy`、`assert_adahedge_matches_legacy`、`get_diagnostic_collection_snapshot`）は、定義元と同じ役割。条件名（dictのkey）は個別に登録しない。
