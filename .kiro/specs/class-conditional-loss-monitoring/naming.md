# 命名正本

revision: 1。承認hashとrevisionはspec.json。旧名はtest oracleのみ。ESRは既存e-SRのアルゴリズム名を保持する。

## ファイル・型・API

| 名前 | 役割・入出力 |
|---|---|
| bounded_loss_e_sr_detection.py | 単一有界損失系列のe-SR数値状態 |
| overall_and_true_class_loss_monitoring.py | 全体/正解classの混合・global位置対応 |
| BoundedLossESRDetector | explicit baseline/alpha/capacity/betsで単一系列を監視 |
| BoundedLossESRObservation | 1回の単一系列更新後のimmutable結果 |
| BoundedLossESRState | capital等をcopyしたimmutable診断state |
| OverallAndTrueClassLossMonitor | 全体と正解class成分の更新・混合・候補位置 |
| LossMonitoringObservation | 混合警報・global候補・その回の計算件数 |
| OverallAndTrueClassLossMonitoringState | 全体/class/位置/最後の観測の診断copy |
| observe_loss | observed_loss→単一系列結果、内部候補のみ更新 |
| observe_loss_after_label_observation | loss/class/index/baseline→混合結果 |
| reset | explicit baselineでepisode状態を初期化 |
| last_observation | readonly最新結果、単一は初期結果、混合は初期None |
| get_state_snapshot | tupleとfrozen結果から診断stateをcopy |

## 引数・結果・状態・局所名

| 名前 | 意味・型・単位 |
|---|---|
| baseline_loss_mean / initial_baseline_loss_mean / current_model_baseline_loss_mean | 指定baseline / overall初期 / class初出時の現行モデルbaseline。finite[0,1]、無次元 |
| false_alarm_control_alpha / loss_change_detection_settings | 単体alpha / 既存immutable監視固定条件 |
| maximum_retained_candidate_count | 保持する候補開始点の最大件数、int≥1 |
| betting_fractions / betting_fraction | tuple賭け率列 / 個別finite(0,1)、無次元 |
| class_count / observed_class_id | class総数 / 観測後正解class、int |
| observed_loss / sample_index | 現行モデル損失 / 外部global位置（0始まり） |
| observed_loss_count | reset後の観測件数、int |
| log_e_value / log_alarm_threshold / drift_detected | 対数e値 / 対数閾値 / 今回警報bool |
| candidate_start_observation_number(s) / oldest_retained_candidate_observation_number | best/全候補番号 / 保持最古番号、内部1始まり |
| estimated_change_span_sample_count | 現在から推定候補までの標本数、global基準/単体は内部観測数 |
| evaluated_candidate_bet_count / component_update_count | その回の候補×bet評価件数 / 成分更新件数 |
| detector_candidate_start_sample_index | FIFO切詰め前のglobal候補位置 |
| candidate_log_capitals / log_betting_weights | 候補×betの対数capital表 / bet等重み対数、float64 |
| overall_esr_state / class_esr_states_by_class_id / class_sample_indices_by_class_id | overall診断 / class別診断tupleペア / class位置tupleペア |
| last_sample_index | episode内の最後のglobal位置、未観測None |
| _baseline_loss_mean / _maximum_retained_candidate_count / _betting_fractions / _log_betting_weights / _log_alarm_threshold | 单体固定/episodebaselineの内部値、arraysは非公開 |
| _candidate_log_capitals / _candidate_start_observation_numbers / _observed_loss_count / _last_observation | 単一e-SRの内部更新状態 |
| _loss_change_detection_settings / _class_count / _overall_component_weight / _class_component_weight | 混合monitor固定条件と導出重み |
| _overall_detector / _class_detectors_by_class_id / _class_sample_indices_by_class_id / _last_sample_index | overall単体 / 遅延class単体 / class位置deque / global順検査 |
| _validate_bounded_number | specified_value/parameter_nameをfinite builtin数値[0,1]検査しfloat返却 |
| _log_sum_exp | log_valuesを旧NumPy順で対数合計 |
| _validate_monitoring_observation | 4入力の型・域・位置連続性を更新前に検証 |
| specified_value / parameter_name / log_values / maximum_log_value | 検査値/項目名/log列/有限最大log |
| loss_increments / log_loss_increments / new_candidate_log_capitals / excess_candidate_count | 更新の増分/log / zero新候補 / 古い削除件数 |
| weighted_candidate_logs / row_maximum_logs / candidate_logs / best_candidate_index | bet混合前表 / 行max / 候補ごとのlog寄与 / 最初argmax |
| overall_observation / class_observation / class_detector / class_sample_indices | 今回2成分結果 / 該当単体 / そのclass位置deque |
| component_logs / finite_component_logs / combined_log_e_value / best_component | 旧優先順成分dict / finite列 / 混合log / 勝者キー |
| other_class_id / other_class_detector / candidate_position_offset | 他class / その単体 / 保持最古からの候補offset |

内部1始まり候補番号とglobalのsample_indexを区別する。baselineは推定処理ではなく指定値、candidateは変化点候補で新モデル候補とは別。snapshotは診断用copyでreset/状態更新APIではない。

## 検証名

test_loss_change_monitoring.py。test名はtest_loss_monitoring_接頭辞に以下を続ける。

- single_series_matches_reference_after_each_observation
- candidate_ties_and_retention_match_reference
- reset_and_baseline_limits_match_reference
- invalid_inputs_preserve_complete_state
- overall_and_class_series_match_reference
- first_class_observation_freezes_current_baseline
- component_ties_preserve_reference_priority
- class_positions_survive_retention_and_reset
- snapshots_and_instances_are_independent
- current_model_observed_loss_connects_without_prediction_state
- public_calls_preserve_random_states
- public_functions_require_explicit_keyword_arguments

test helper: make_reference_class_monitor（旧__new__fixture）、assert_single_series_matches_reference、assert_class_monitor_matches_reference。局所名: detector、monitor、reference_detector、reference_monitor、reference_observation、reference_detected、reference_span、reference_start_sample_index、state_before_call、state_snapshot、reference_state、observation_index、model_outputs_by_model_id、prediction_probabilities_by_model_id、observed_losses_by_model_id、current_training_model_id、observed_class_labels、global_python_random_state、global_numpy_random_state、global_torch_random_state、invalid_parameter_name、invalid_parameter_value、exception_info、operation、monkeypatch、valid_run_settings_mapping。既承認のfixture・AST checker名を再利用する。
