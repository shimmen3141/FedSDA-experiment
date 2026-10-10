# 旧実装の1 runの出力と、新実装への移行の一覧

- 記録日: 2026-10-11。作成: 主担当（Claude Code）。
- 対象: 旧実装`federated_drift_experiment.run_random_drift_experiment`の、最終構成（Residual Adapter＋ClassESR＋Switching）での、
  1 runの戻り値（指標）と、`raw_path`へ保存するNPZ（配列）。旧の回帰testのsine2の条件で実行して、全部の名前を列挙した。
- 目的: 新実装で保存する項目の範囲を決め、旧の結果（既存の`results/`のCSV・NPZ）を、新の名前へ変換するための対応を作る。
- ユーザーの決定（2026-10-11）: 新実装の保存は、新実装の名前で行う。旧の結果を、新の名前へ変換する（新→旧の変換は作らない）。
  「記録から導く（未照合）」の区分は、全部、新実装で導出して、旧の値と照合する。

区分は、名前からの機械的な分類である。「記録から導く（未照合）」の中に、新の記録だけでは導けない項目が見つかった場合は、
その項目を、理由とともに、この文書へ書き足す。新の名前は、各specで決めて、ここへ書き足す。

## 件数

| 区分 | 指標 | 配列 |
|---|---:|---:|
| 照合済み | 33 | 31 |
| 記録から導く（未照合） | 124 | 86 |
| 用途別の計算の計数 | 16 | 21 |
| 未移植の診断 | 29 | 29 |
| 所要時間 | 6 | 3 |
| 最終構成では使わない選択肢の項目 | 53 | 27 |
| 条件・来歴 | 2 | 36 |
| 合計 | 263 | 233 |

## 照合済み

新の記録からの導出が、実旧の値と、goldenの3ケース（3環境）で一致している。

**指標（33）**

`accuracy`、`stable_accuracy`、`recall`、`precision`、`f1`、`total_detect`、`provisional_proposal_count`、`provisional_forward_count`、`final_model_count`、`compute_training_examples_total`、`compute_optimizer_steps_total`、`compute_backbone_examples_total`、`compute_head_examples_total`、`compute_drift_detector_updates_total`、`compute_drift_detector_hypotheses_total`、`compute_inference_examples_total`、`routing_soft_prediction_sample_count`、`routing_switching_recalibration_sample_count`、`routing_aggregation_recalibration_sample_count`、`comm_models_up`、`comm_models_down`、`comm_models_total`、`comm_messages_up`、`comm_messages_down`、`comm_messages_total`、`comm_parameter_values_up`、`comm_parameter_values_down`、`comm_parameter_values_total`、`comm_bytes_up`、`comm_bytes_down`、`comm_bytes_total`、`final_parameter_values`、`final_parameter_bytes`

**配列（31）**

`history_accuracy`、`drift_client_ids`、`drift_positions`、`history_model_id`、`switch_client_ids`、`switch_positions`、`adaptation_client_ids`、`adaptation_positions`、`adaptation_actions`、`adaptation_old_model_ids`、`adaptation_new_model_ids`、`provisional_client_ids`、`provisional_positions`、`provisional_accepted`、`provisional_reasons`、`provisional_resolution_positions`、`model_registration_ids`、`model_registration_rounds`、`model_registration_final_active`、`clustering_rounds`、`clustering_model_ids`、`clustering_representative_model_ids`、`clustering_participated_in_merge`、`clustering_absorbed`、`clustering_pair_rounds`、`clustering_pair_left_model_ids`、`clustering_pair_right_model_ids`、`clustering_pair_same_cluster`、`history_routing_soft_active`、`history_routing_switching_correct`、`history_routing_switching_leader_id`

## 記録から導く（未照合）

新の記録（予測、警報、適応、候補の判定、登録、クラスタリング、クロス評価、保有モデル数ほか）から導けると見込む。導出と、実旧の値との照合を、(A2)の一連のspecで行う（2026-10-11ユーザー決定: この区分は、全部入れる）。

**指標（124）**

`miss_rate`、`fdr`、`avg_delay`、`change_point_mae`、`change_point_bias`、`change_point_estimate_count`、`total_true`、`tp`、`fp`、`fn`、`alarm_precision`、`alarm_recall`、`alarm_f1`、`alarm_total`、`switch_fp_early`、`switch_fp_late`、`switch_fp_duplicate`、`switch_fp_isolated`、`adaptation_reuse_count`、`adaptation_reuse_precision`、`adaptation_create_count`、`adaptation_create_precision`、`adaptation_create_rejected_count`、`model_reuse_current_fit_count`、`model_reuse_alternative_fit_count`、`provisional_acceptance_rate`、`provisional_matched_true_count`、`provisional_accepted_matched_true_count`、`provisional_rejected_matched_true_count`、`provisional_accepted_precision`、`provisional_interval_count_mean`、`provisional_training_count_mean`、`provisional_validation_count_mean`、`provisional_resolution_delay_mean`、`provisional_accepted_full_margin_mean`、`provisional_accepted_recent_margin_mean`、`provisional_rejected_full_margin_mean`、`provisional_rejected_recent_margin_mean`、`provisional_matched_full_margin_mean`、`provisional_matched_recent_margin_mean`、`provisional_unmatched_full_margin_mean`、`provisional_unmatched_recent_margin_mean`、`provisional_reject_insufficient_forward_data_count`、`provisional_reject_first_interval_count`、`provisional_reject_second_interval_count`、`provisional_reject_first_and_second_count`、`provisional_reject_reference_refit_count`、`provisional_reject_current_refit_count`、`provisional_reject_alternative_refit_count`、`provisional_reference_excess_mean`、`adaptation_maintain_count`、`server_mapping_change_count`、`mean_model_count`、`max_model_count`、`model_count_auc`、`e_detector_max_log_e`、`model_assigned_samples_total`、`model_assigned_samples_mean`、`model_assigned_samples_min`、`model_assigned_samples_cv`、`model_training_examples_total`、`model_training_examples_mean`、`model_training_examples_min`、`model_training_examples_cv`、`model_optimizer_steps_total`、`model_optimizer_steps_mean`、`model_optimizer_steps_min`、`model_optimizer_steps_cv`、`model_pair_evaluation_count`、`model_pair_sample_count`、`model_pair_correctness_disagreement_rate`、`model_pair_oracle_gain_rate`、`model_pair_both_correct_rate`、`clustering_oracle_pair_count`、`clustering_oracle_same_pair_count`、`clustering_oracle_merge_tp`、`clustering_oracle_merge_fp`、`clustering_oracle_merge_fn`、`clustering_oracle_merge_tn`、`clustering_oracle_merge_precision`、`clustering_oracle_merge_recall`、`clustering_oracle_merge_f1`、`clustering_oracle_loss_distance_auc`、`clustering_oracle_parameter_distance_auc`、`routing_sample_count`、`routing_oracle_accuracy`、`routing_mixture_accuracy`、`routing_leader_accuracy`、`routing_confidence_leader_accuracy`、`routing_oracle_gain_rate`、`routing_oracle_recovery_rate`、`routing_oracle_stable_accuracy`、`routing_oracle_recovery_accuracy`、`routing_oracle_concept_accuracy`、`routing_oracle_concept_stable_accuracy`、`routing_oracle_concept_recovery_accuracy`、`routing_oracle_concept_stable_gain_rate`、`routing_oracle_concept_recovery_gain_rate`、`routing_leader_stable_accuracy`、`routing_leader_recovery_accuracy`、`routing_stable_oracle_gap`、`routing_recovery_oracle_gap`、`routing_missed_oracle_count`、`routing_confidence_leader_oracle_recovery_rate`、`routing_confidence_leader_missed_oracle_count`、`routing_class_macro_oracle_accuracy`、`routing_class_macro_mixture_accuracy`、`routing_class_macro_leader_accuracy`、`routing_class_macro_confidence_leader_accuracy`、`routing_class_oracle_gap_mean`、`routing_class_oracle_gap_std`、`routing_class_oracle_recovery_rate_mean`、`routing_class_oracle_recovery_rate_min`、`routing_switching_accuracy`、`routing_switching_gain_rate`、`routing_switching_global_gain_rate`、`routing_switching_stable_accuracy`、`routing_switching_stable_gain_rate`、`routing_switching_recovery_accuracy`、`routing_switching_recovery_gain_rate`、`routing_switching_effective_experts_mean`、`routing_switching_leader_switch_count`、`routing_switching_pool_reset_count`、`routing_aggregation_recalibration_count`

**配列（86）**

`estimated_drift_start_client_ids`、`estimated_drift_alarm_positions`、`estimated_drift_start_positions`、`detector_candidate_start_positions`、`adaptation_detectors`、`adaptation_estimated_change_points`、`provisional_detectors`、`provisional_interval_counts`、`provisional_training_counts`、`provisional_validation_counts`、`provisional_reference_model_ids`、`provisional_candidate_mean_losses`、`provisional_reference_mean_losses`、`provisional_reference_historical_means`、`provisional_candidate_recent_losses`、`provisional_reference_recent_losses`、`provisional_validation_sources`、`provisional_matched_true`、`model_registration_client_ids`、`model_registration_selection_counts`、`clustering_nearest_model_ids`、`clustering_nearest_distances`、`clustering_cluster_sizes`、`clustering_cluster_max_distances`、`clustering_evaluated_pair_counts`、`clustering_possible_pair_counts`、`clustering_pair_distances`、`clustering_pair_decision_scores`、`clustering_pair_oracle_same_concept`、`clustering_pair_personalized_parameter_distances`、`round_global_model_count`、`round_client_held_model_count`、`round_pending_clustering_count`、`history_detector_log_e`、`history_routing_effective_experts`、`history_routing_max_weight`、`routing_pool_reset_counts`、`routing_concept_restart_counts`、`routing_aggregation_recalibration_counts`、`routing_aggregation_recalibration_sample_counts`、`history_routing_oracle_correct`、`history_routing_leader_correct`、`history_routing_oracle_concept_correct`、`history_routing_switching_effective_experts`、`routing_class_client_ids`、`routing_class_ids`、`routing_class_sample_counts`、`routing_class_oracle_correct_counts`、`routing_class_mixture_correct_counts`、`routing_class_leader_correct_counts`、`routing_class_confidence_leader_correct_counts`、`routing_class_missed_oracle_counts`、`routing_class_confidence_leader_missed_oracle_counts`、`model_diagnostic_client_ids`、`model_diagnostic_model_ids`、`model_diagnostic_assigned_samples`、`model_diagnostic_training_examples`、`model_diagnostic_optimizer_steps`、`model_pair_candidate_model_id`、`model_pair_target_model_id`、`model_pair_n`、`model_pair_candidate_only_correct`、`model_pair_target_only_correct`、`model_pair_both_correct`、`model_pair_both_wrong`、`cross_evaluation_round_index`、`cross_evaluation_client_id`、`cross_evaluation_candidate_model_id`、`cross_evaluation_target_model_id`、`cross_evaluation_n`、`cross_evaluation_candidate_only_correct`、`cross_evaluation_target_only_correct`、`cross_evaluation_both_correct`、`cross_evaluation_both_wrong`、`cross_evaluation_sum`、`cross_evaluation_sum_sq`、`cross_evaluation_class_round_index`、`cross_evaluation_class_client_id`、`cross_evaluation_class_candidate_model_id`、`cross_evaluation_class_target_model_id`、`cross_evaluation_class_class_id`、`cross_evaluation_class_n`、`cross_evaluation_class_candidate_only_correct`、`cross_evaluation_class_target_only_correct`、`cross_evaluation_class_both_correct`、`cross_evaluation_class_both_wrong`

## 用途別の計算の計数

旧は、モデルへ入力した標本の数を、用途（予測・検出・統計・クロス評価・初期化・再較正・学習）別に数える。新は、用途別には数えず、共有部・概念固有部、学習・推論、ローカルの処理・同期の別に、標本数と積和演算を数える（旧の用途別の値との一致は、testで確かめてある。NEW-004）。新の保存項目は、新の区分にする。旧の結果からの変換は、合計へ足し合わせる形になる。

**指標（16）**

`compute_prediction_forward_calls_total`、`compute_prediction_examples_total`、`compute_detection_forward_calls_total`、`compute_detection_examples_total`、`compute_statistics_forward_calls_total`、`compute_statistics_examples_total`、`compute_cross_evaluation_forward_calls_total`、`compute_cross_evaluation_examples_total`、`compute_initialization_forward_calls_total`、`compute_initialization_examples_total`、`compute_routing_recalibration_forward_calls_total`、`compute_routing_recalibration_examples_total`、`compute_training_forward_calls_total`、`compute_backbone_optimizer_steps_total`、`compute_head_optimizer_steps_total`、`compute_model_examples_total`

**配列（21）**

`round_client_prediction_forward_calls`、`round_client_prediction_examples`、`round_client_detection_forward_calls`、`round_client_detection_examples`、`round_client_statistics_forward_calls`、`round_client_statistics_examples`、`round_client_cross_evaluation_forward_calls`、`round_client_cross_evaluation_examples`、`round_client_initialization_forward_calls`、`round_client_initialization_examples`、`round_client_routing_recalibration_forward_calls`、`round_client_routing_recalibration_examples`、`round_client_training_forward_calls`、`round_client_training_examples`、`round_client_optimizer_steps`、`round_client_backbone_optimizer_steps`、`round_client_head_optimizer_steps`、`round_client_backbone_examples`、`round_client_head_examples`、`round_client_drift_detector_updates`、`round_client_drift_detector_hypotheses`

## 未移植の診断

モデル別の除外寄与（LOO）と、共有部の勾配の診断。新実装に、まだない（順序(C)）。

**指標（29）**

`routing_loo_evaluation_count`、`routing_loo_bounded_delta_mean`、`routing_loo_zero_one_delta_mean`、`routing_loo_positive_rate`、`routing_loo_fallback_count`、`routing_loo_active_model_count`、`routing_loo_active_evaluable_model_count`、`routing_loo_active_joint_nonpositive_model_count`、`routing_loo_active_joint_nonpositive_rate`、`routing_loo_active_unassigned_model_count`、`routing_loo_active_unassigned_evaluable_model_count`、`routing_loo_active_unassigned_nonpositive_model_count`、`routing_loo_active_unassigned_nonpositive_rate`、`routing_loo_active_unassigned_joint_nonpositive_model_count`、`routing_loo_active_unassigned_joint_nonpositive_rate`、`backbone_gradient_pair_count`、`backbone_gradient_conflict_count`、`backbone_gradient_conflict_rate`、`backbone_gradient_cosine_mean`、`backbone_gradient_negative_cosine_mean`、`backbone_gradient_applied_pair_count`、`backbone_gradient_applied_conflict_count`、`backbone_gradient_applied_conflict_rate`、`backbone_gradient_applied_cosine_mean`、`backbone_gradient_applied_negative_cosine_mean`、`backbone_gradient_update_comparison_count`、`backbone_gradient_update_cosine_mean`、`backbone_gradient_update_norm_ratio_mean`、`backbone_gradient_update_delta_ratio_mean`

**配列（29）**

`routing_loo_client_ids`、`routing_loo_pool_epochs`、`routing_loo_block_indices`、`routing_loo_model_ids`、`routing_loo_sample_counts`、`routing_loo_positive_counts`、`routing_loo_negative_counts`、`routing_loo_hard_assignment_counts`、`routing_loo_fallback_counts`、`routing_loo_probability_sums`、`routing_loo_bounded_delta_sums`、`routing_loo_bounded_delta_squared_sums`、`routing_loo_zero_one_delta_sums`、`routing_loo_is_active_final`、`routing_loo_is_assigned_final`、`routing_loo_final_active_model_ids`、`routing_loo_final_assigned_model_ids`、`backbone_gradient_pair_counts`、`backbone_gradient_conflict_counts`、`backbone_gradient_cosine_sums`、`backbone_gradient_negative_cosine_sums`、`backbone_gradient_applied_pair_counts`、`backbone_gradient_applied_conflict_counts`、`backbone_gradient_applied_cosine_sums`、`backbone_gradient_applied_negative_cosine_sums`、`backbone_gradient_update_comparison_counts`、`backbone_gradient_update_cosine_sums`、`backbone_gradient_update_norm_ratio_sums`、`backbone_gradient_update_delta_ratio_sums`

## 所要時間

実時間。再現しない値なので、照合の対象にしない。新での計測は、順序(C)。

**指標（6）**

`runtime_seconds`、`client_online_seconds_sum`、`client_training_seconds_sum`、`client_cross_evaluation_seconds_sum`、`client_compute_seconds_sum`、`client_compute_seconds_max`

**配列（3）**

`round_client_online_seconds`、`round_client_training_seconds`、`round_client_cross_evaluation_seconds`

## 最終構成では使わない選択肢の項目

最終構成で無効の選択肢（Meta-switching、archive shadow、active set、非劣性merge、検出episode、回復期だけの混合予測、旧の候補検証の方針）の項目。最終構成では、値が0・空・一定になる。新の保存項目に入れない。

**指標（53）**

`provisional_reject_insufficient_data_count`、`provisional_reject_full_interval_count`、`provisional_reject_recent_interval_count`、`provisional_reject_full_and_recent_count`、`adaptation_episode_suppressed_count`、`clustering_noninferiority_candidate_count`、`clustering_noninferiority_accepted_count`、`clustering_noninferiority_rejected_count`、`clustering_noninferiority_comparison_count`、`clustering_noninferiority_sample_count`、`clustering_noninferiority_acceptance_rate`、`routing_hard_prediction_sample_count`、`routing_soft_prediction_rate`、`routing_activation_count`、`routing_deactivation_count`、`routing_activation_exit_deferred_sample_count`、`routing_activation_replay_sample_count`、`routing_meta_accuracy`、`routing_meta_gain_rate`、`routing_meta_global_accuracy`、`routing_meta_global_stable_accuracy`、`routing_meta_global_recovery_accuracy`、`routing_meta_stable_accuracy`、`routing_meta_recovery_accuracy`、`routing_meta_context_leader_accuracy`、`routing_meta_context_mixture_accuracy`、`routing_meta_best_candidate_gain_rate`、`routing_meta_context_leader_weight_mean`、`routing_meta_context_leader_preferred_rate`、`routing_class_macro_meta_accuracy`、`routing_class_macro_meta_global_accuracy`、`routing_class_macro_meta_context_mixture_accuracy`、`routing_class_macro_meta_context_leader_accuracy`、`routing_meta_switching_accuracy`、`routing_meta_switching_meta_gain_rate`、`routing_meta_switching_switching_gain_rate`、`routing_meta_switching_selected_switching_rate`、`routing_meta_switching_leader_switch_count`、`routing_aggregation_restart_count`、`routing_aggregation_recalibration_check_count`、`routing_aggregation_recalibration_skip_count`、`routing_archive_shadow_sample_count`、`routing_archive_shadow_accuracy`、`routing_archive_shadow_accuracy_delta`、`routing_archive_shadow_bounded_delta_mean`、`routing_archive_shadow_retained_global_model_rate`、`routing_archive_shadow_reconfiguration_count`、`routing_active_set_sample_count`、`routing_active_set_probe_sample_count`、`routing_active_set_retained_global_model_rate`、`routing_active_set_apply_retained_global_model_rate`、`routing_active_set_reconfiguration_count`、`routing_active_set_failure_probe_count`

**配列（27）**

`adaptation_episode_ids`、`clustering_noninferiority_rounds`、`clustering_noninferiority_representative_model_ids`、`clustering_noninferiority_target_model_ids`、`clustering_noninferiority_candidate_model_ids`、`clustering_noninferiority_sample_counts`、`clustering_noninferiority_mean_differences`、`clustering_noninferiority_upper_bounds`、`clustering_noninferiority_margins`、`clustering_noninferiority_accepted`、`clustering_noninferiority_target_in_accepted_cluster`、`routing_aggregation_restart_counts`、`routing_aggregation_recalibration_check_counts`、`routing_aggregation_recalibration_skip_counts`、`history_routing_gate_open`、`history_routing_meta_correct`、`history_routing_meta_global_correct`、`history_routing_meta_context_mixture_correct`、`history_routing_meta_context_leader_correct`、`history_routing_meta_context_leader_weight`、`history_routing_meta_switching_correct`、`history_routing_meta_switching_selected_switching`、`routing_class_meta_sample_counts`、`routing_class_meta_correct_counts`、`routing_class_meta_global_correct_counts`、`routing_class_meta_context_mixture_correct_counts`、`routing_class_meta_context_leader_correct_counts`

## 条件・来歴

条件と来歴。新は、完全なrun設定の辞書（`convert_settings_to_plain_mapping`）と、実行の来歴で持つ。

**指標（2）**

`concept_schedule`、`e_detector_alpha`

**配列（36）**

`dataset`、`concept_schedule`、`mode`、`label`、`parameter_schema_version`、`sweep_parameter`、`sweep_value`、`seed`、`min_stable`、`aggregation_interval`、`feddrift_detection_batch_size`、`fedsda_distance_threshold`、`feddrift_distance_threshold`、`adwin_delta`、`clustering_policy`、`clustering_decision`、`clustering_consolidation`、`merge_noninferiority_margin`、`cluster_linkage`、`shared_backbone_training`、`shared_backbone_gradient_strategy`、`shared_backbone_routing_recalibration`、`soft_routing_context`、`soft_routing_activation_policy`、`soft_routing_top_combination`、`soft_routing_meta_loss`、`routing_active_set_policy`、`shared_adapter_rank`、`detection_episodes`、`new_model_creation_policy`、`fifo_size`、`new_model_validation_fraction`、`new_model_forward_validation_samples`、`result_metrics_json`、`total_data`、`e_detector_alpha`
