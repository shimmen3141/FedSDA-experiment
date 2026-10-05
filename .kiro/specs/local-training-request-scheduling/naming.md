# 命名 revision 1

意味と単位を名前で区別する正式名の正本。Luna承認前は実装しない。

| 名前 | 役割/型/単位/状態と区別 |
|---|---|
| local_training_schedule_settings.py / LocalTrainingScheduleSettings | 学習要求間隔と一要求予算だけの不変設定。更新方式LocalTrainingSettingsと別 |
| training_requests_per_update_interval | int>=1、実行間隔一つに累積する要求件数。ストア標本数ではない |
| joint_update_iterations_per_training_request | int>=0、一要求あたり共同更新試行回数。optimizer実stepやepoch数ではない |
| local_training_request_schedule.py / LocalTrainingRequestSchedule | 保留要求counterの単一所有者。学習実体は所有しない |
| local_training_schedule_settings / _local_training_schedule_settings | constructorの借用不変設定/保持参照 |
| pending_training_request_count / _pending_training_request_count | 読取専用property/内部counter。未完了要求件数、FIFOや実更新件数ではない |
| record_training_request | 一要求追加し、間隔到達時の試行回数intを返す。学習・clearはしない |
| calculate_pending_joint_update_iteration_count | 間隔を無視した全保留×一要求予算を返す無変更照会、flush用 |
| acknowledge_completed_training_requests | 現在の保留全件の正常完了を確認してclear。自動callbackではない |
| completed_training_request_count | 正のexact int、成功した現在batchの要求件数snapshot。loss数ではない |
| __post_init__ | 2field型/値の検査。既存公開validate_settings_field_valuesを再利用 |
| parameter_name / parameter_value | 設定2fieldの型検査用一時名。入力指定を変更しない |
| test_local_training_request_scheduling.py | 要求列/旧比較/拒否/NN明示接続の検証 |
| test_training_request_schedule_matches_legacy | 旧要求/flushのevent列・counter/callback予算と比較 |
| test_training_request_schedule_rejects_invalid_settings | 型/値域/派生整数と改変済み設定を拒否 |
| test_training_request_schedule_rejects_invalid_acknowledgement | 不正/空/不一致ackとcounter保持 |
| test_training_request_schedule_retains_pending_requests_on_failure | 実行失敗でclearしない旧動作/再試行 |
| test_training_schedule_settings_are_frozen_and_explicit | frozen/kwonly/defaultなし、readonly状態 |
| test_training_request_schedule_integrates_with_joint_training | 同初期旧/新NNの要求列接続比較 |
| test_training_request_schedule_consumes_ineligible_attempts | 不参加正常returnでも要求消化 |
| test_training_request_schedule_supports_large_counts | arbitrary precisionの純回数計算、NN反復には渡さない |
| build_training_schedule_oracle_pair | 旧BaseClient操作をnamespaceへbindし比較用counter/callbackを構成 |
| execute_scheduled_training_request | test-onlyの一要求/明示flushとNN実行後ackを接続。production wrapperではない |
| interval_count / iteration_budget / class_count / optimizer_variant / update_shared_features | parameterized検証値、要求間隔/一要求予算/既存NN条件 |
| schedule / schedule_settings / legacy_client | 新counter所有者/設定と実旧namespace |
| training_event / training_events / event_index | requestまたはflushというtest-only外側event順 |
| pending_count_before / pending_count_after / pending_request_count | event前後状態と成功ack用snapshot |
| requested_iteration_count / expected_iteration_count | 今回の実学習へ渡す試行回数/旧予算 |
| legacy_training_calls / new_training_calls | eventごとの旧multiplier/new試行回数履歴 |
| fail_training / failure_armed | test callbackの失敗有無/一回失敗の制御。production状態ではない |
| invalid_settings / invalid_value / invalid_ack_count / invalid_field | 拒否入力 |
| python_random_generator / expected_random_state / initial_random_state | 前回同名の借用Random/状態 |
| participating_training_batches / shared_parameter_optimizer / held_model_training_bindings / ordered_model_training_samples | 前回公開記録/借用学習参照を再利用 |
| global_python_random_state / global_torch_random_state | 旧のglobal乱数をfinally復元する開始state |
| actual_losses / expected_losses / classifier_parameters_before / optimizer_state_before | 実測/対照loss、無操作確認state |
| record_field / initial_parameters / keyword_arguments / binding / training_sample / model_training_samples | 既存意味と同じfield/検証snapshot/入力dict/記録の局所名 |
| original_training_function / sampling_calls / update_calls | 実関数をwrapするtest-only観測 |

