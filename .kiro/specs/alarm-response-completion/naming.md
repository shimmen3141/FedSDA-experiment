# 警報応答の完了処理 — 命名 revision5

sourceとtestの実装前一覧。リポジトリ外の下書きをASTで走査し（関数・class・引数・代入・内包表記・keyword・属性）、既存のsrc/tests/旧実装に現れない名前を機械的に洗い出して全件を登録した。既存名は定義元と同じ役割で再利用する。全体の承認状態はspec.json。

## 新source

| 名前 | 役割と似た名前との違い |
| --- | --- |
| `alarm_response_completion.py` | runtimeの新module。完了した警報応答を受け、損失監視の再開と保留位置の消費を行い、記録用の情報を返す。既存`alarm_buffer_response.py`は標本・統計・帰属・sessionまでを扱い、監視と保留位置を変更しない。 |
| `AlarmResponseCompletion` | 完了処理の不変結果（frozen/kw_only）。応答、警報位置、変更前後の学習帰属ID、推定変化点、episode ID、監視の再開に使った基準平均、消費した保留位置を持つ。既存`PostAlarmCandidateValidationCompletion`は候補検証の到達時の完了情報で、警報時点の完了情報である本recordとは契機が違う。イベント一覧・切替位置一覧・再利用計数は所有しない。 |
| `complete_alarm_buffer_response` | 完了した`AlarmBufferResponse`に対する後処理の公開関数。全入力を検査し、現行モデルの統計から基準平均を選び、監視のreset→保留位置のdrain（不足の応答ではdrainしない）の順に更新して`AlarmResponseCompletion`を返す。応答そのもの（吸収・解決）は実行しない。 |
| `_validate_optional_nonnegative_index` | module内の検査helper。Noneまたはboolを除くbuiltin intの非負値だけを受理する。推定変化点とepisode IDの2引数・2fieldで共用する。 |
| `alarm_sample_index` | 警報を観測した標本のclient内の通し位置（0始まり、単位は標本）。旧イベントの`position`に対応する。応答へ渡した既存`proposal_sample_index`と同じ値を呼出側が渡す。候補検証の文脈を持たない完了処理では「候補の提案位置」ではなく「警報位置」と呼ぶ。 |
| `previous_training_model_id` | 応答前の学習帰属ID。旧イベントの`old_model_id`。既存`PostAlarmCandidateValidationCompletion.previous_training_model_id`と同じ役割で再利用する。 |
| `current_training_model_id` | 応答後の学習帰属ID。旧イベントの`new_model_id`。既存`CurrentTrainingModelAssignment.current_training_model_id`と同じ値。 |
| `loss_change_monitor` | 引数。既存`OverallAndTrueClassLossMonitor`のexact instance。本処理がresetする唯一の監視owner。 |
| `loss_monitoring_baseline_mean_loss` | 監視の再開に使った基準平均（有界損失、0より大きく1未満）。既存`select_loss_monitoring_baseline_mean_loss`の戻り値で、応答後の現行モデルの履歴統計から選ぶ。recordのfieldと関数内の局所名で同じ値を指す。 |
| `drained_pending_sample_indices` | 本処理が保留FIFOから消費した標本位置（保留順のtuple）。不足の応答では空tuple。呼出側が位置に対応するpayloadを破棄するために使う。既存`drain_pending_sample_indices`の戻り値。 |
| `current_model_loss_statistics` | 局所名。現行モデルの`ModelAndClassLossStatistics`またはNone（統計なし）。 |
| `pending_assignment_state` | 局所名。完了処理の開始時点の`PendingTrainingAssignmentState`（保留位置列と最終観測位置）。応答との対応検査だけに使う。 |
| `training_model_changed` | recordの検査内の局所bool。変更前後の学習帰属IDが異なるか。 |
| `training_model_switch_sample_index` | readonly property。帰属IDが変わった応答では警報位置、変わらなければNone。旧`local_switch_positions`へ追加する位置。既存`PostAlarmCandidateValidationCompletion`の同名propertyと同じ役割。 |
| `detection_episode_operation_required` | readonly property。帰属IDが変わった応答でTrue（旧の戻り値1または2に対応し、警報応答では再利用の1だけが起こる）。既存`PostAlarmCandidateValidationCompletion`の同名propertyと同じ役割。 |

recordのfield `alarm_buffer_response`、`estimated_change_point_sample_index`、`detection_episode_id`と、引数`current_training_model_assignment`、`loss_statistics_store`、`pending_training_assignment_buffer`は、既存`respond_to_alarm_with_buffered_samples`・`AlarmBufferResponse`と同じ名前・同じ役割で再利用する。局所名`prepared_alarm_training_intervals`、`change_interval_resolution`、`training_model_assignment_change`、`indexed_observation`、`sample_index`、`model_id`、`specified_value`、`parameter_name`も既存の同名と同じ役割。

## 新sourceのimport（exact集合、8 symbol）

| symbol | 定義元 |
| --- | --- |
| `dataclass` | `dataclasses.dataclass` |
| `ModelAndClassLossStatisticsStore` | `federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics.ModelAndClassLossStatisticsStore` |
| `CurrentTrainingModelAssignment` | `federated_learning_experiments.learning.training.current_training_model_assignment.CurrentTrainingModelAssignment` |
| `TrainingModelAssignmentChange` | `federated_learning_experiments.learning.training.current_training_model_assignment.TrainingModelAssignmentChange` |
| `OverallAndTrueClassLossMonitor` | `federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring.OverallAndTrueClassLossMonitor` |
| `select_loss_monitoring_baseline_mean_loss` | `federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection.select_loss_monitoring_baseline_mean_loss` |
| `PendingTrainingAssignmentBuffer` | `federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer.PendingTrainingAssignmentBuffer` |
| `AlarmBufferResponse` | `federated_learning_experiments.runtime.alarm_buffer_response.AlarmBufferResponse` |

## 新testと依存境界test

| 名前 | 役割 |
| --- | --- |
| `test_alarm_response_completion.py` | 新test module。完了処理を実旧`_resolve_drift`（イベント記録・検出器reset・FIFO clearを差し替えない）と照合する。 |
| `LEGACY_ACTION_BY_RESPONSE_OUTCOME` | 応答結果5値から旧イベントのaction文字列への対応。照合専用。 |
| `LEGACY_REUSE_SELECTION_COUNTS_BY_RESPONSE_OUTCOME` | 応答結果5値から、1回の警報で旧`reuse_selection_counts`に加わる内容（空、`alternative_fit`1件、`current_fit`1件）への対応。照合専用。 |
| `LOSS_MONITOR_BETTING_FRACTIONS` | 新監視へ渡す賭け率の固定tuple。既存監視testが使う値と同じ。 |
| `MAXIMUM_RETAINED_CANDIDATE_COUNT` | 新監視と実旧検出器へ共通に与える保持候補数（test用の小さい値）。既存引数`maximum_retained_candidate_count`へ渡す。 |
| `MONITORED_LOSSES_BEFORE_ALARM` | 警報前に新旧の監視へ与える有界損失列。reset前にclass別検出器と候補を持たせる。 |
| `MONITORED_LOSSES_AFTER_COMPLETION` | 完了処理の後に新旧の監視へ与える有界損失列。再開後の継続を照合する。 |
| `INVALID_COMPLETION_INPUT_CASES` | 拒否条件名から（入力を不正にする操作、期待する例外型）への対応。条件名はこのdictを正本とする。 |
| `select_current_model_monitoring_baseline` | test helper。新ownerの現行モデル統計から、公開の基準選択で監視の基準平均を得る。productionの完了処理を呼ばない。 |
| `observe_monitored_losses_in_both_implementations` | test helper。同じ損失列を新監視と実旧ClassESRへ1件ずつ与え、各観測後のe値・警報・推定区間長・全状態を照合する。 |
| `build_response_completion_oracle` | test helper。既存`build_buffer_response_oracle`へ、警報前の損失を観測済みの新監視と実旧の検出状態、完了処理の引数を加える。上流oracleが差し替えたイベント記録・検出器resetを実旧メソッドへ戻す。 |
| `run_legacy_alarm_with_real_completion` | test helper。実旧`_resolve_drift`を、推定区間長の供給だけを差し替えて実行する（イベント記録・検出器reset・FIFO clear・吸収・評価・候補学習は実処理）。既存`run_legacy_buffer_response`はイベントとresetを記録用に差し替える点が違う。 |
| `assert_response_completion_matches_legacy` | test helper。完了recordのfield・保留位置・監視状態を、実旧のイベント・切替位置・再利用計数・FIFO・検出器と照合する。 |
| `assign_other_model_after_response` | 拒否条件の操作。応答の後に学習帰属を別の保有モデルへ変える。 |
| `append_sample_index_after_response` | 拒否条件の操作。応答の後に保留位置を1件追加し、警報位置も合わせて進める（準備済み区間と保留位置の不一致だけを起こす）。 |
| `test_response_completion_matches_real_legacy` | 2/4class×3解決条件×区間長・最小件数4組で、応答＋完了処理を実旧と照合し、完了後の監視の継続も照合する。 |
| `test_second_alarm_during_started_validation_completes_like_legacy` | 1回目の警報で候補検証を開始して完了処理を行い、続く2回目の警報（候補検証中、保留0件/3件）の応答＋完了処理を実旧と照合する。 |
| `test_completion_baseline_follows_current_model_statistics_like_legacy` | 現行モデルの統計なし・0件・下限未満・中間・上限の5条件で、再開の基準平均を実旧のresetと照合する。不足の応答のrecordの性質も確認する。 |
| `test_completion_rejects_invalid_inputs_before_updates` | 全拒否条件で、監視・保留位置・統計・帰属・乱数が不変であることを確認する。 |
| `test_repeated_completion_of_consumed_response_is_rejected` | 保留位置を消費済みの応答へ完了処理を再適用すると、監視を変更せず拒否することを確認する。 |
| `test_completion_resets_monitor_with_selected_baseline_before_draining` | reset→drainの順と、resetへ渡す基準平均が切替・吸収後の現行モデル統計から選ばれることを確認する。 |
| `test_completion_record_is_frozen_and_rejects_inconsistent_fields` | recordのfrozen/kw_only、field検査、property、応答との不整合の拒否を確認する。 |
| `test_alarm_response_completion_dependency_contract` | 依存境界test module内の注入契約test。新moduleのexact symbol依存の許可・拒否を両resolverで確認する。既存の同種testと同じ形。 |
| `completion_arguments` | 完了処理へ渡すkeyword引数のdict（test内の受け渡し用）。helperの引数名としても使う。 |
| `valid_completion_arguments` | 不正化する前の完了処理の引数dictのcopy。不変確認でownerを参照する。 |
| `response_completion` | 完了処理の戻り値。helperの引数名としても使う。 |
| `first_response_completion` | 1回目の警報の完了処理の戻り値。 |
| `second_response_completion` | 2回目の警報の完了処理の戻り値。 |
| `reused_response_completion` | 再利用の応答に対する完了処理の戻り値（record検査の元にする）。 |
| `second_alarm_response` | 2回目の警報に対する`AlarmBufferResponse`。 |
| `second_alarm_sample_count` | parametrize引数。1回目の完了後、2回目の警報までに保留する標本数。 |
| `second_alarm_sample_index` | 2回目の警報位置。 |
| `second_alarm_observations` | 2回目の警報時点の位置付き保留標本のtuple。 |
| `first_alarm_observations` | 1回目の警報時点の位置付き保留標本のtuple（2回目の標本の元にする）。 |
| `pending_sample_indices_before_completion` | 完了処理の直前の保留位置列。消費結果の期待値。 |
| `torch_random_state_before_completion` | 完了処理の直前のtorch乱数状態。不消費の確認用。 |
| `torch_random_state_before_second_alarm` | 2回目の警報の直前のtorch乱数状態。候補検証中の応答と完了処理の不消費の確認用。 |
| `monitored_losses` | helper引数。監視へ与える有界損失列。 |
| `monitored_loss` | 損失列の1要素。 |
| `monitoring_observation` | 新監視の1観測の戻り値（既存`LossMonitoringObservation`）。 |
| `monitoring_state` | 新監視の`get_state_snapshot`の戻り値。不変確認用。 |
| `loss_statistics_state` | 統計storeの`get_state_snapshot`の戻り値。不変確認用。 |
| `legacy_drift_detected` | 実旧`_update_drift_detectors`の戻り値（警報の有無）。 |
| `legacy_model_statistics` | parametrize引数。実旧`model_stats`の現行モデルへ置く統計dict、またはNone（統計なし）。 |
| `expected_baseline_mean_loss` | 期待する再開の基準平均。 |
| `recorded_event_count` | 実旧警報処理の直前の`adaptation_events`の件数。今回追加されたイベントを取り出す。 |
| `replaced_method_name` | 上流oracleがinstance属性で差し替えた実旧メソッド名（loop変数）。 |
| `apply_invalid_input` | 拒否条件の、入力を不正にする操作（callable）。 |
| `invalid_fields` | record検査で`replace`へ渡す不正なfieldのdict。 |
| `completion_operation_calls` | reset/drainの呼出しを順に記録するlist。記録だけで判定を持たない。 |
| `record_monitor_reset` | 記録用wrapper。呼出しを記録して実`reset`へ委譲する。 |
| `record_pending_index_drain` | 記録用wrapper。呼出しを記録して実`drain_pending_sample_indices`へ委譲する。 |
| `reset_loss_change_monitor` | 差し替え前の実`reset`のbound method。 |

条件名（parametrizeのidになるdictのkey）は`INVALID_COMPLETION_INPUT_CASES`を正本とし、個別には登録しない。依存境界testの引数`source_text`・`expected_acceptance`、局所`dependency_boundary_violations`、alias `AcceptedDependency`は既存の注入契約testと同じ役割で再利用する。

## 新testのimport

| symbol | 定義元 |
| --- | --- |
| `random` | `random` |
| `defaultdict` | `collections.defaultdict` |
| `deque` | `collections.deque` |
| `FrozenInstanceError` | `dataclasses.FrozenInstanceError` |
| `asdict` | `dataclasses.asdict` |
| `replace` | `dataclasses.replace` |
| `patch` | `unittest.mock.patch` |
| `np` | `numpy` |
| `pytest` | `pytest` |
| `torch` | `torch` |
| `assert_buffer_response_matches_legacy` | `test_alarm_buffer_response.assert_buffer_response_matches_legacy` |
| `build_buffer_response_oracle` | `test_alarm_buffer_response.build_buffer_response_oracle` |
| `assert_alarm_change_interval_resolution_state_unchanged` | `test_alarm_change_interval_resolution.assert_alarm_change_interval_resolution_state_unchanged` |
| `snapshot_alarm_change_interval_resolution_state` | `test_alarm_change_interval_resolution.snapshot_alarm_change_interval_resolution_state` |
| `assert_class_monitor_matches_reference` | `test_loss_change_monitoring.assert_class_monitor_matches_reference` |
| `valid_run_settings_mapping` | `test_run_settings_validation.valid_run_settings_mapping` |
| `config` | `federated_drift_experiment.config` |
| `AdaptationEvent` | `federated_drift_experiment.adaptation_events.AdaptationEvent` |
| `BoundedMeanEDetector` | `federated_drift_experiment.drift_detectors.e_detector.BoundedMeanEDetector` |
| `TrainingModelAssignmentChange` | `federated_learning_experiments.learning.training.current_training_model_assignment.TrainingModelAssignmentChange` |
| `IndexedObservedTrainingSample` | `federated_learning_experiments.learning.training.indexed_observed_training_sample.IndexedObservedTrainingSample` |
| `OverallAndTrueClassLossMonitor` | `federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring.OverallAndTrueClassLossMonitor` |
| `select_loss_monitoring_baseline_mean_loss` | `federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection.select_loss_monitoring_baseline_mean_loss` |
| `PendingTrainingAssignmentBuffer` | `federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer.PendingTrainingAssignmentBuffer` |
| `TrainingDataAssignmentSettings` | `federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings.TrainingDataAssignmentSettings` |
| `AlarmBufferResponse` | `federated_learning_experiments.runtime.alarm_buffer_response.AlarmBufferResponse` |
| `respond_to_alarm_with_buffered_samples` | `federated_learning_experiments.runtime.alarm_buffer_response.respond_to_alarm_with_buffered_samples` |
| `AlarmResponseCompletion` | `federated_learning_experiments.runtime.alarm_response_completion.AlarmResponseCompletion` |
| `complete_alarm_buffer_response` | `federated_learning_experiments.runtime.alarm_response_completion.complete_alarm_buffer_response` |

testが再利用する既存の局所名（`response_arguments`、`preparation_arguments`、`resolution_arguments`、`shared_optimizer_owners`、`legacy_client`、`legacy_result`、`legacy_drift_type`、`legacy_adaptation_events`、`legacy_epoch_training_calls`、`legacy_python_random_state`、`global_python_random_state`、`numpy_random_state`、`initial_torch_random_state`、`expected_torch_random_state`、`initial_training_model_id`、`alarm_buffer_response`、`started_validation_session`、`legacy_session`、`indexed_observation`、`sample_offset`、`sample_index`、`observed_class_id`、`class_count`、`alarm_interval_resolution_case`、`earlier_sample_count`、`estimated_change_span_sample_count`、`minimum_change_interval_sample_count`、`invalid_case`、`expected_exception`、`false_alarm_control_alpha`、`loss_change_detection_settings`、`first_sample_index`、`current_training_model_id`、`loss_statistics_store`、`pending_training_assignment_buffer`、`python_random_state`、`torch_random_state`、`response_outcome`、`legacy_adaptation_event`）は、既存のsrc・tests/refactoring（主にtest_alarm_buffer_response.py、test_alarm_change_interval_resolution.py、test_loss_change_monitoring.py）の同名と同じ役割。

## revision2の補足（Task 1の独立レビューの反映）

r1の名前と役割は変更しない。testで次の既存名を同じ役割で再利用することを明記する。

| 名前 | 役割 |
| --- | --- |
| `snapshot_alarm_change_interval_resolution_state` | `test_alarm_change_interval_resolution.py`の既存helper。保有モデルの値とgrad・optimizer・統計・学習標本・計数・帰属・3乱数の処理前snapshotを作る。本testでは完了処理（成功時と拒否時）の前に使う。 |
| `assert_alarm_change_interval_resolution_state_unchanged` | 同じmoduleの既存helper。上のsnapshotと現在の状態が同じであることを確かめる。 |
| `previous_snapshot` | 上のhelperが返す処理前snapshotを受ける局所名。`test_alarm_buffer_response.py`の同名と同じ役割。 |
| `drain_pending_sample_indices`（test内の局所名） | 差し替え前の実`PendingTrainingAssignmentBuffer.drain_pending_sample_indices`のbound method。既存メソッドと同じ対象を指す。記録用wrapper `record_pending_index_drain`が委譲する先で、`reset_loss_change_monitor`と対になる。r1のAST照合は既存srcに同じ語があるため新規名として検出せず、独立レビューが局所束縛の未記載を指摘した。 |

## revision3の補足（Task 1の再レビューの反映）

r2までの名前と役割は変更しない。設計r3でrecordの組立を更新の前へ移すため、source内でも局所名`response_completion`（完了処理の戻り値になる`AlarmResponseCompletion`。testの同名と同じ役割）を使う。新しい名前は追加しない。拒否条件の追加はdictのkey（条件名）とlambdaだけで行う。

## revision4の補足（設計r4の反映）

r3（未承認のまま改訂）までの名前と役割は変更しない。設計r4に合わせて次を改める。

- 「新sourceのimport」は8 symbolになる。追加は`TrainingModelAssignmentChange`（`learning/training/current_training_model_assignment.py`の既存の帰属変更record。完了処理が応答内の変更記録のexact型を更新前に確かめるために使う。役割は定義元と同じ）。上の表は改訂後の下書きから生成しており、この行を含む。節の見出しの「7 symbol」は「8 symbol」へ改めた。
- recordのfield `drained_pending_sample_indices`の値は、消費指示のある応答では完了処理の開始時に読んだ保留位置列（その後のdrainが消費する内容と同じ）、不足の応答では空tuple。役割（本処理が消費した保留位置）はr1と同じで、source内の同名の局所変数は使わなくなる。
- testでは、既存の`TrainingModelAssignmentChange`を拒否条件の入力（boolのIDを持つ変更記録）を作るためにimportする。
- testへ新しいhelper名を1つ追加する: `replace_assignment_change_in_response`（拒否条件の操作。再利用の応答の帰属変更記録だけを差し替えた応答を作り、完了処理の引数dictへ入れる。引数`completion_arguments`と`training_model_assignment_change`は既存の同名と同じ役割。応答や完了処理を実行しない）。

## revision5の補足（設計r5の反映）

r4までの名前と役割は変更しない。設計r5（一度も観測していない保留FIFOの拒否）のtest入力を作るため、testで既存の`PendingTrainingAssignmentBuffer`、`TrainingDataAssignmentSettings`、`AlarmBufferResponse`を定義元と同じ役割でimportする（上の「新testのimport」表は改訂後の下書きから生成しており、この3行を含む）。新しい名前は追加しない。拒否条件の追加はdictのkey（条件名）とlambdaだけで行う。
