# 警報時の保留標本への応答 — 命名 revision5

sourceとtestの実装前一覧。既存公開型/関数/引数は定義元と同じ役割で再利用する。全体の承認状態はspec.json。

## 依存注入検査の補足

## 状態照合の補足（実装前）

| 名前 | 役割 |
| --- | --- |
| `np` | testだけで使う標準的なnumpy import alias。グローバルNumPy乱数の不消費を確認する。 |
| `numpy_random_state` | 通常・active応答直前のNumPy RNG状態。アルゴリズムの状態所有者ではない。 |
| `session_gradient_snapshot` | active応答直前の候補と固定参照parameterのgradのcloneまたはNone。値だけでなくgradも変わらないことを確認する。 |
| `session_reference_history_snapshot` | 固定参照のモデルIDごとの履歴平均損失dictのコピー。不変性を確認する。 |

r4の命名は維持し、testの検証を強める4名だけを補足する。

`test_alarm_buffer_response_dependency_contract`はこのmoduleのexact symbol依存と両resolverの許可・拒否を検証する。引数`source_text`と`expected_acceptance`、局所`dependency_boundary_violations`、既存`collect_dependency_boundary_violations`は既存の同名依存契約testと同じ役割で再利用する。record段階は実際の5symbolだけ、runtime段階で実23symbolへ拡張し、未来の依存を先行許可しない。alias名`AcceptedDependency`も既存注入fixtureと同じ役割。r2の全命名は変更せず、この関数名と役割のみを追加する。

| 名前 | 役割と違い |
| --- | --- |
| alarm_buffer_response.py | runtimeの応答選択と不変結果。既存alarm_change_interval_resolutionは切出し済み区間だけ、本moduleはactive/FIFO/最小件数の選択も担う。 |
| test_alarm_buffer_response.py | 実旧警報の5経路と公開操作接続の検証。 |
| `ALARM_BUFFER_RESPONSE_OUTCOMES` | 進行中検証、不足、既存3解決結果を合わせた正式5値。旧actionではない。 |
| `ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES` | `federated_learning_experiments.runtime.alarm_change_interval_resolution.ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `AdamParameterOptimizerSettings` | `federated_learning_experiments.learning.training.parameter_optimizer_settings.AdamParameterOptimizerSettings`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `AlarmBufferResponse` | 警報への応答結果。frozen/kw_only。準備と解決とsessionを借用し、後始末は実行しない。 |
| `AlarmChangeIntervalResolution` | `federated_learning_experiments.runtime.alarm_change_interval_resolution.AlarmChangeIntervalResolution`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `CandidateEpochTrainingSettings` | `federated_learning_experiments.learning.training.candidate_epoch_training_settings.CandidateEpochTrainingSettings`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `CandidateModelTrainingAndAcceptanceSettings` | `federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.CandidateModelTrainingAndAcceptanceSettings`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `CandidateParameterInitializationSettings` | `federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings.CandidateParameterInitializationSettings`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `CurrentTrainingModelAssignment` | `federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `FrozenInstanceError` | `dataclasses.FrozenInstanceError`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `HeldModelTrainingStateRegistry` | `federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `IndexedObservedTrainingSample` | `federated_learning_experiments.learning.training.indexed_observed_training_sample.IndexedObservedTrainingSample`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `LocalTrainingSettings` | `federated_learning_experiments.learning.training.local_training_settings.LocalTrainingSettings`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `ModelAndClassLossStatisticsStore` | `federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `ModelEvaluationSampleStore` | `federated_learning_experiments.evaluation.model_evaluation_sample_store.ModelEvaluationSampleStore`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `ModelTrainingAndAssignmentCountsStore` | `federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `ModelTrainingSampleStore` | `federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `ParticipatingModelTrainingBatch` | `federated_learning_experiments.learning.training.participating_model_training_batch.ParticipatingModelTrainingBatch`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `PendingTrainingAssignmentBuffer` | `federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer.PendingTrainingAssignmentBuffer`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `PostAlarmCandidateValidationSession` | `federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start.PostAlarmCandidateValidationSession`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `PreparedAlarmTrainingIntervals` | `federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals.PreparedAlarmTrainingIntervals`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `Random` | `random.Random`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `ResidualAdapterClassifier` | `federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `SgdParameterOptimizerSettings` | `federated_learning_experiments.learning.training.parameter_optimizer_settings.SgdParameterOptimizerSettings`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `_` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `__post_init__` | 正式値とactive/不足/解決ごとのfield組合せの不整合を拒否するdataclass hook。 |
| `_validate_buffered_alarm_observations` | FIFO owner、tuple/record/非負位置、FIFOとの完全一致を更新前に確認する私的helper。 |
| `absorb_assigned_training_samples_into_held_model` | `federated_learning_experiments.runtime.assigned_training_sample_absorption.absorb_assigned_training_samples_into_held_model`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `active_validation_session` | 入力では進行中sessionかNone、結果では継続または新規開始したsession。状態は呼出側が保持。 |
| `actual_joint_loss` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `alarm_interval_resolution_case` | `test helper/case入力`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `architecture_reference_classifier` | `ResidualAdapterClassifier`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `argument_name` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `arguments` | `test helper/case入力`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `assert_absorption_matches_legacy` | `test_assigned_training_sample_absorption.assert_absorption_matches_legacy`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `assert_alarm_change_interval_resolution_matches_legacy` | `test_alarm_change_interval_resolution.assert_alarm_change_interval_resolution_matches_legacy`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `assert_alarm_change_interval_resolution_state_unchanged` | `test_alarm_change_interval_resolution.assert_alarm_change_interval_resolution_state_unchanged`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `assert_buffer_response_matches_legacy` | 実旧と新応答の状態/結果/乱数/後始末指示を照合するhelper。判断の正解を再実装しない。 |
| `assert_collected_losses_match_legacy` | `test_post_alarm_candidate_validation_sample_observation.assert_collected_losses_match_legacy`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `assert_evaluation_samples_match_legacy` | `test_alarm_training_interval_preparation.assert_evaluation_samples_match_legacy`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `assert_held_model_states_match_legacy` | `test_adopted_candidate_initial_local_registration.assert_held_model_states_match_legacy`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `assert_nested_state_equal` | `test_joint_model_parameter_update.assert_nested_state_equal`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `build_alarm_preparation_oracle` | `test_alarm_training_interval_preparation.build_alarm_preparation_oracle`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `build_buffer_response_oracle` | 既存実NNの準備/解決oracleを再利用し、新応答の入力と実旧clientを作るtest helper。 |
| `candidate_epoch_training_settings` | `CandidateEpochTrainingSettings`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `candidate_model_training_and_acceptance_settings` | `CandidateModelTrainingAndAcceptanceSettings`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `candidate_parameter_initialization_settings` | `CandidateParameterInitializationSettings`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `change_interval_input_features` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `change_interval_resolution` | 十分な変化区間で得た実解決結果。選択やsession開始を再実行しない。 |
| `class_count` | `test helper/case入力`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `classifier` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `collection` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `completed_preparations` | resolver拒否時にも準備更新が残る保証範囲を示す実結果の記録列。 |
| `config` | `federated_drift_experiment.config`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `continue_joint_training_after_buffer_response` | 応答後の保存標本を使った実旧/新の共同更新を同条件で実行し照合するtest helper。 |
| `counts_store` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `current_training_model_assignment` | `CurrentTrainingModelAssignment`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `dataclass` | `dataclasses.dataclass`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `deepcopy` | `copy.deepcopy`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `defaultdict` | `collections.defaultdict`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `deque` | `collections.deque`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `detection_episode_id` | `int | None`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `detector_name` | `str`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `detector_reset_calls` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `earlier_sample_count` | `test helper/case入力`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `empty_buffer` | `test helper/case入力`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `estimated_change_point_sample_index` | `int | None`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `estimated_change_span_sample_count` | 検出側が供給する正の推定変化区間件数。activeでは使用しない。 |
| `evaluation_snapshot` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `event_fields` | `test helper/case入力`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `expected_exception` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `expected_joint_loss` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `expected_torch_random_state` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `forwarded_arguments` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `global_python_random_state` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `held_model_training_state_registry` | `HeldModelTrainingStateRegistry`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `indexed_observation` | 位置/標本/診断概念IDの1record。分割・吸収へ位置順で渡す。 |
| `initial_loss_count` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `initial_torch_random_state` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `initial_training_model_id` | `test helper/case入力`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `interval_resolution` | 公開resolveの返却を一度だけ保持する局所参照。 |
| `invalid_case` | `test helper/case入力`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `legacy_adaptation_events` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `legacy_client` | `test helper/case入力`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `legacy_drift_type` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `legacy_epoch_training_calls` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `legacy_python_random_state` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `legacy_result` | `test helper/case入力`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `legacy_session` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `legacy_training_batches` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `legacy_training_sample` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `legacy_training_samples` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `local_training_settings` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `loss_statistics_store` | `ModelAndClassLossStatisticsStore`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `maximum_alarm_interval_mean_loss_increase` | `float`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `minimum_change_interval_sample_count` | 変化区間を評価する最小標本件数。正のbuiltin int。activeでは使用しない。 |
| `model_evaluation_sample_store` | `ModelEvaluationSampleStore`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `model_id` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `model_training_and_assignment_counts_store` | `ModelTrainingAndAssignmentCountsStore`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `model_training_sample_collections` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `monkeypatch` | `test helper/case入力`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `negative_training_model_id` | `test helper/case入力`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `observations` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `observe_post_alarm_candidate_validation_sample` | `federated_learning_experiments.runtime.post_alarm_candidate_validation_sample_observation.observe_post_alarm_candidate_validation_sample`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `operation_call` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `operation_calls` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `optimizer_owner` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `original_prepare` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `original_resolve` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `parameter` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `parameter_optimizer_settings` | `AdamParameterOptimizerSettings | SgdParameterOptimizerSettings`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `participating_training_batches` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `patch` | `unittest.mock.patch`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `pending_assignment_buffer_should_be_cleared` | 呼出側の後始末判断。不足だけFalse。他はTrue。実際のdrainは行わないreadonly property。 |
| `pending_assignment_snapshot` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `pending_sample_observations` | FIFOと完全一致する位置順の借用record tuple。payloadの意味上の対応は供給側の保証。 |
| `pending_training_assignment_buffer` | `PendingTrainingAssignmentBuffer`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `perform_joint_model_parameter_update` | `federated_learning_experiments.learning.training.joint_model_parameter_update.perform_joint_model_parameter_update`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `preparation_arguments` | `test helper/case入力`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `prepare_alarm_training_intervals` | `federated_learning_experiments.runtime.alarm_training_interval_preparation.prepare_alarm_training_intervals`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `prepared_alarm_training_intervals` | active以外で得た実準備結果。前区間は既に保存/吸収済み。 |
| `prepared_intervals` | 公開prepareの返却を一度だけ保持する局所参照。 |
| `previous_parameter` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `previous_snapshot` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `proposal_sample_index` | `int`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `provide_estimated_change_span` | 旧oracleへ試験指定spanを供給し、active経路で未呼出しを確認するcallback。 |
| `pytest` | `pytest`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `python_random_generator` | `Random`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `python_random_state` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `random` | `random`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `record_legacy_adaptation_event` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `recorded_prepare` | 元prepareを実行し、実結果と呼出し順を保存するwrapper。stubではない。 |
| `recorded_resolve` | 元resolveを実行し、転送引数を記録するwrapper。旧判断を再実装しない。 |
| `registry` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `replace` | `dataclasses.replace`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `resolution_arguments` | `test helper/case入力`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `resolve_alarm_change_interval` | `federated_learning_experiments.runtime.alarm_change_interval_resolution.resolve_alarm_change_interval`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `resolve_alarm_change_interval_in_legacy_client` | `test_alarm_change_interval_resolution.resolve_alarm_change_interval_in_legacy_client`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `respond_to_alarm_with_buffered_samples` | 位置付きFIFOとsession有無から吸収/準備/件数判定/解決を順に組み立てる公開操作。 |
| `response` | `test helper/case入力`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `response_arguments` | `test helper/case入力`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `response_module` | `federated_learning_experiments.runtime.alarm_buffer_response`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `response_outcome` | 正式5値の応答種別。旧の戻り値0だけでは区別できない結果を示す。 |
| `run_legacy_buffer_response` | 実旧_resolve_driftを実行するtest helper。span/reset/eventを境界記録用に差し替え、乱数を保存/復元する。 |
| `run_legacy_joint_update` | `test_joint_model_parameter_update.run_legacy_joint_update`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `sample_index` | `test helper/case入力`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `session` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `session_loss_snapshot` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `session_optimizer_snapshot` | 候補の共有部と概念固有部の両optimizerのdeepcopy。吸収で変更しないことを実Tensorごと比較する。 |
| `session_parameter_snapshot` | 候補と固定参照の全parameter値のclone。active分岐の不変性を確認する。 |
| `session_parameters` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `session_pending_samples` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `shared_optimizer_owners` | `test helper/case入力`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `snapshot_alarm_change_interval_resolution_state` | `test_alarm_change_interval_resolution.snapshot_alarm_change_interval_resolution_state`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `span_calls` | 推定span callbackの実呼出し位置を記録する列。 |
| `target_model_id` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `test_active_candidate_alarm_absorbs_all_and_preserves_session` | 進行中の実候補検証を持つ新旧でFIFO全吸収とsession/parameter/optimizer/履歴不変を確認。 |
| `test_buffer_response_matches_real_legacy` | 2/4classと3解決条件、span/minimum境界で実旧5経路・共同更新・候補観測を照合するtest。 |
| `test_buffer_response_rejects_owned_inputs_before_updates` | 位置/型/件数の不正を、全状態/評価保存/乱数不変で拒否するtest。 |
| `test_empty_buffer_is_insufficient_without_candidate_work` | 空FIFOの不足応答が状態/乱数を変更しないことを検証。 |
| `test_failure_after_preparation_keeps_completed_earlier_updates` | 準備後resolver拒否では前区間更新が残るという明示保証境界のtest。 |
| `test_prepare_precedes_resolution_and_preserves_original_metadata` | 実prepare→実resolveの1回順序と元metadata/準備tupleの転送を検証。 |
| `test_response_record_rejects_inconsistent_fields_and_is_frozen` | 正式値/fieldの組/keyword専用/凍結を検証するtest。 |
| `torch` | `torch`。既存公開契約または標準/検証ライブラリを同じ意味で再利用。旧/test importはtestのみ。 |
| `training_batch` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `training_binding` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `training_bindings` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `training_sample` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `training_sample_store` | `ModelTrainingSampleStore`。既存の区間準備/解決または検証helperの同名入力と同じ値・単位・役割。 |
| `update_shared_features` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |
| `valid_response_arguments` | testの条件/結果/状態記録。実旧との照合と不変性確認に使用し、本番の状態所有者を追加しない。 |

## exact source import

- `dataclasses.dataclass`
- `random.Random`
- `federated_learning_experiments.evaluation.model_evaluation_sample_store.ModelEvaluationSampleStore`
- `federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore`
- `federated_learning_experiments.learning.models.residual_adapter_classifier.ResidualAdapterClassifier`
- `federated_learning_experiments.learning.training.candidate_epoch_training_settings.CandidateEpochTrainingSettings`
- `federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment`
- `federated_learning_experiments.learning.training.held_model_training_state_registry.HeldModelTrainingStateRegistry`
- `federated_learning_experiments.learning.training.indexed_observed_training_sample.IndexedObservedTrainingSample`
- `federated_learning_experiments.learning.training.model_training_and_assignment_counts.ModelTrainingAndAssignmentCountsStore`
- `federated_learning_experiments.learning.training.model_training_sample_store.ModelTrainingSampleStore`
- `federated_learning_experiments.learning.training.parameter_optimizer_settings.AdamParameterOptimizerSettings`
- `federated_learning_experiments.learning.training.parameter_optimizer_settings.SgdParameterOptimizerSettings`
- `federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings.CandidateModelTrainingAndAcceptanceSettings`
- `federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings.CandidateParameterInitializationSettings`
- `federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer.PendingTrainingAssignmentBuffer`
- `federated_learning_experiments.methods.fedsda.training_data_assignment.prepared_alarm_training_intervals.PreparedAlarmTrainingIntervals`
- `federated_learning_experiments.runtime.alarm_change_interval_resolution.ALARM_CHANGE_INTERVAL_RESOLUTION_OUTCOMES`
- `federated_learning_experiments.runtime.alarm_change_interval_resolution.AlarmChangeIntervalResolution`
- `federated_learning_experiments.runtime.alarm_change_interval_resolution.resolve_alarm_change_interval`
- `federated_learning_experiments.runtime.alarm_training_interval_preparation.prepare_alarm_training_intervals`
- `federated_learning_experiments.runtime.assigned_training_sample_absorption.absorb_assigned_training_samples_into_held_model`
- `federated_learning_experiments.runtime.post_alarm_candidate_validation_session_start.PostAlarmCandidateValidationSession`

標準builtin/selfは固有命名対象外。動的condition名・parametrize idのdict keyはtestを正本とする。test helper/argument/代入/内包target/import bindingをASTから抽出し追加表へ含めた。

`ObservedTrainingSample`は位置付きrecordのfield型として定義元で解決される。新moduleのannotation/実参照には不要なためimportしない。exact source importは実ASTの23symbol。新定義ALARM_BUFFER_RESPONSE_OUTCOMESはimportではなく、借用した既存3結果定数にactive/不足2値を加える。
