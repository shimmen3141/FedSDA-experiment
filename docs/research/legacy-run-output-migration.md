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
| 照合済み（最初から。goldenが比べる項目） | 33 | 31 |
| 記録から導く（未照合）——2026-10-11に、全部、照合済みになった | 124 | 85 |
| 用途別の計算の計数 | 16 | 21 |
| 未移植の診断 | 29 | 29 |
| 所要時間 | 6 | 3 |
| 最終構成では使わない選択肢の項目 | 53 | 28 |
| 条件・来歴 | 2 | 36 |
| 合計 | 263 | 233 |

## 照合済み

新の記録からの導出が、実旧の値と、goldenの3ケース（3環境）で一致している。

**指標（33）**

`accuracy`、`stable_accuracy`、`recall`、`precision`、`f1`、`total_detect`、`provisional_proposal_count`、`provisional_forward_count`、`final_model_count`、`compute_training_examples_total`、`compute_optimizer_steps_total`、`compute_backbone_examples_total`、`compute_head_examples_total`、`compute_drift_detector_updates_total`、`compute_drift_detector_hypotheses_total`、`compute_inference_examples_total`、`routing_soft_prediction_sample_count`、`routing_switching_recalibration_sample_count`、`routing_aggregation_recalibration_sample_count`、`comm_models_up`、`comm_models_down`、`comm_models_total`、`comm_messages_up`、`comm_messages_down`、`comm_messages_total`、`comm_parameter_values_up`、`comm_parameter_values_down`、`comm_parameter_values_total`、`comm_bytes_up`、`comm_bytes_down`、`comm_bytes_total`、`final_parameter_values`、`final_parameter_bytes`

**配列（31）**

`history_accuracy`、`drift_client_ids`、`drift_positions`、`history_model_id`、`switch_client_ids`、`switch_positions`、`adaptation_client_ids`、`adaptation_positions`、`adaptation_actions`、`adaptation_old_model_ids`、`adaptation_new_model_ids`、`provisional_client_ids`、`provisional_positions`、`provisional_accepted`、`provisional_reasons`、`provisional_resolution_positions`、`model_registration_ids`、`model_registration_rounds`、`model_registration_final_active`、`clustering_rounds`、`clustering_model_ids`、`clustering_representative_model_ids`、`clustering_participated_in_merge`、`clustering_absorbed`、`clustering_pair_rounds`、`clustering_pair_left_model_ids`、`clustering_pair_right_model_ids`、`clustering_pair_same_cluster`、`history_routing_soft_active`、`history_routing_switching_correct`、`history_routing_switching_leader_id`

## 記録から導く（未照合）——全部、照合済みになった

区分の名前は、分類したときのもの（ほかの文書が、この名前で指す）。新の記録（予測、警報、適応、候補の判定、登録、クラスタリング、クロス評価、保有モデル数ほか）から導けると見込み、導出と、実旧の値との照合を、(A2)の一連のspecで行った（2026-10-11ユーザー決定: この区分は、全部入れる）。2026-10-11に、全部の項目の照合が済んだ（内訳と、旧→新の対応は、下の「新実装の保存の形式と、対応」と「残り」）。新の記録だけでは導けなかった項目は、ない。

**指標（124）**

`miss_rate`、`fdr`、`avg_delay`、`change_point_mae`、`change_point_bias`、`change_point_estimate_count`、`total_true`、`tp`、`fp`、`fn`、`alarm_precision`、`alarm_recall`、`alarm_f1`、`alarm_total`、`switch_fp_early`、`switch_fp_late`、`switch_fp_duplicate`、`switch_fp_isolated`、`adaptation_reuse_count`、`adaptation_reuse_precision`、`adaptation_create_count`、`adaptation_create_precision`、`adaptation_create_rejected_count`、`model_reuse_current_fit_count`、`model_reuse_alternative_fit_count`、`provisional_acceptance_rate`、`provisional_matched_true_count`、`provisional_accepted_matched_true_count`、`provisional_rejected_matched_true_count`、`provisional_accepted_precision`、`provisional_interval_count_mean`、`provisional_training_count_mean`、`provisional_validation_count_mean`、`provisional_resolution_delay_mean`、`provisional_accepted_full_margin_mean`、`provisional_accepted_recent_margin_mean`、`provisional_rejected_full_margin_mean`、`provisional_rejected_recent_margin_mean`、`provisional_matched_full_margin_mean`、`provisional_matched_recent_margin_mean`、`provisional_unmatched_full_margin_mean`、`provisional_unmatched_recent_margin_mean`、`provisional_reject_insufficient_forward_data_count`、`provisional_reject_first_interval_count`、`provisional_reject_second_interval_count`、`provisional_reject_first_and_second_count`、`provisional_reject_reference_refit_count`、`provisional_reject_current_refit_count`、`provisional_reject_alternative_refit_count`、`provisional_reference_excess_mean`、`adaptation_maintain_count`、`server_mapping_change_count`、`mean_model_count`、`max_model_count`、`model_count_auc`、`e_detector_max_log_e`、`model_assigned_samples_total`、`model_assigned_samples_mean`、`model_assigned_samples_min`、`model_assigned_samples_cv`、`model_training_examples_total`、`model_training_examples_mean`、`model_training_examples_min`、`model_training_examples_cv`、`model_optimizer_steps_total`、`model_optimizer_steps_mean`、`model_optimizer_steps_min`、`model_optimizer_steps_cv`、`model_pair_evaluation_count`、`model_pair_sample_count`、`model_pair_correctness_disagreement_rate`、`model_pair_oracle_gain_rate`、`model_pair_both_correct_rate`、`clustering_oracle_pair_count`、`clustering_oracle_same_pair_count`、`clustering_oracle_merge_tp`、`clustering_oracle_merge_fp`、`clustering_oracle_merge_fn`、`clustering_oracle_merge_tn`、`clustering_oracle_merge_precision`、`clustering_oracle_merge_recall`、`clustering_oracle_merge_f1`、`clustering_oracle_loss_distance_auc`、`clustering_oracle_parameter_distance_auc`、`routing_sample_count`、`routing_oracle_accuracy`、`routing_mixture_accuracy`、`routing_leader_accuracy`、`routing_confidence_leader_accuracy`、`routing_oracle_gain_rate`、`routing_oracle_recovery_rate`、`routing_oracle_stable_accuracy`、`routing_oracle_recovery_accuracy`、`routing_oracle_concept_accuracy`、`routing_oracle_concept_stable_accuracy`、`routing_oracle_concept_recovery_accuracy`、`routing_oracle_concept_stable_gain_rate`、`routing_oracle_concept_recovery_gain_rate`、`routing_leader_stable_accuracy`、`routing_leader_recovery_accuracy`、`routing_stable_oracle_gap`、`routing_recovery_oracle_gap`、`routing_missed_oracle_count`、`routing_confidence_leader_oracle_recovery_rate`、`routing_confidence_leader_missed_oracle_count`、`routing_class_macro_oracle_accuracy`、`routing_class_macro_mixture_accuracy`、`routing_class_macro_leader_accuracy`、`routing_class_macro_confidence_leader_accuracy`、`routing_class_oracle_gap_mean`、`routing_class_oracle_gap_std`、`routing_class_oracle_recovery_rate_mean`、`routing_class_oracle_recovery_rate_min`、`routing_switching_accuracy`、`routing_switching_gain_rate`、`routing_switching_global_gain_rate`、`routing_switching_stable_accuracy`、`routing_switching_stable_gain_rate`、`routing_switching_recovery_accuracy`、`routing_switching_recovery_gain_rate`、`routing_switching_effective_experts_mean`、`routing_switching_leader_switch_count`、`routing_switching_pool_reset_count`、`routing_aggregation_recalibration_count`

**配列（85）**

`estimated_drift_start_client_ids`、`estimated_drift_alarm_positions`、`estimated_drift_start_positions`、`detector_candidate_start_positions`、`adaptation_detectors`、`adaptation_estimated_change_points`、`provisional_detectors`、`provisional_interval_counts`、`provisional_training_counts`、`provisional_validation_counts`、`provisional_reference_model_ids`、`provisional_candidate_mean_losses`、`provisional_reference_mean_losses`、`provisional_reference_historical_means`、`provisional_candidate_recent_losses`、`provisional_reference_recent_losses`、`provisional_validation_sources`、`provisional_matched_true`、`model_registration_client_ids`、`model_registration_selection_counts`、`clustering_nearest_model_ids`、`clustering_nearest_distances`、`clustering_cluster_sizes`、`clustering_cluster_max_distances`、`clustering_evaluated_pair_counts`、`clustering_possible_pair_counts`、`clustering_pair_distances`、`clustering_pair_decision_scores`、`clustering_pair_oracle_same_concept`、`clustering_pair_personalized_parameter_distances`、`round_global_model_count`、`round_client_held_model_count`、`history_detector_log_e`、`history_routing_effective_experts`、`history_routing_max_weight`、`routing_pool_reset_counts`、`routing_concept_restart_counts`、`routing_aggregation_recalibration_counts`、`routing_aggregation_recalibration_sample_counts`、`history_routing_oracle_correct`、`history_routing_leader_correct`、`history_routing_oracle_concept_correct`、`history_routing_switching_effective_experts`、`routing_class_client_ids`、`routing_class_ids`、`routing_class_sample_counts`、`routing_class_oracle_correct_counts`、`routing_class_mixture_correct_counts`、`routing_class_leader_correct_counts`、`routing_class_confidence_leader_correct_counts`、`routing_class_missed_oracle_counts`、`routing_class_confidence_leader_missed_oracle_counts`、`model_diagnostic_client_ids`、`model_diagnostic_model_ids`、`model_diagnostic_assigned_samples`、`model_diagnostic_training_examples`、`model_diagnostic_optimizer_steps`、`model_pair_candidate_model_id`、`model_pair_target_model_id`、`model_pair_n`、`model_pair_candidate_only_correct`、`model_pair_target_only_correct`、`model_pair_both_correct`、`model_pair_both_wrong`、`cross_evaluation_round_index`、`cross_evaluation_client_id`、`cross_evaluation_candidate_model_id`、`cross_evaluation_target_model_id`、`cross_evaluation_n`、`cross_evaluation_candidate_only_correct`、`cross_evaluation_target_only_correct`、`cross_evaluation_both_correct`、`cross_evaluation_both_wrong`、`cross_evaluation_sum`、`cross_evaluation_sum_sq`、`cross_evaluation_class_round_index`、`cross_evaluation_class_client_id`、`cross_evaluation_class_candidate_model_id`、`cross_evaluation_class_target_model_id`、`cross_evaluation_class_class_id`、`cross_evaluation_class_n`、`cross_evaluation_class_candidate_only_correct`、`cross_evaluation_class_target_only_correct`、`cross_evaluation_class_both_correct`、`cross_evaluation_class_both_wrong`

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

**配列（28）**

`adaptation_episode_ids`、`clustering_noninferiority_rounds`、`clustering_noninferiority_representative_model_ids`、`clustering_noninferiority_target_model_ids`、`clustering_noninferiority_candidate_model_ids`、`clustering_noninferiority_sample_counts`、`clustering_noninferiority_mean_differences`、`clustering_noninferiority_upper_bounds`、`clustering_noninferiority_margins`、`clustering_noninferiority_accepted`、`clustering_noninferiority_target_in_accepted_cluster`、`routing_aggregation_restart_counts`、`routing_aggregation_recalibration_check_counts`、`routing_aggregation_recalibration_skip_counts`、`history_routing_gate_open`、`round_pending_clustering_count`、`history_routing_meta_correct`、`history_routing_meta_global_correct`、`history_routing_meta_context_mixture_correct`、`history_routing_meta_context_leader_correct`、`history_routing_meta_context_leader_weight`、`history_routing_meta_switching_correct`、`history_routing_meta_switching_selected_switching`、`routing_class_meta_sample_counts`、`routing_class_meta_correct_counts`、`routing_class_meta_global_correct_counts`、`routing_class_meta_context_mixture_correct_counts`、`routing_class_meta_context_leader_correct_counts`

## 条件・来歴

条件と来歴。新は、完全なrun設定の辞書（`convert_settings_to_plain_mapping`）と、実行の来歴で持つ。

**指標（2）**

`concept_schedule`、`e_detector_alpha`

**配列（36）**

`dataset`、`concept_schedule`、`mode`、`label`、`parameter_schema_version`、`sweep_parameter`、`sweep_value`、`seed`、`min_stable`、`aggregation_interval`、`feddrift_detection_batch_size`、`fedsda_distance_threshold`、`feddrift_distance_threshold`、`adwin_delta`、`clustering_policy`、`clustering_decision`、`clustering_consolidation`、`merge_noninferiority_margin`、`cluster_linkage`、`shared_backbone_training`、`shared_backbone_gradient_strategy`、`shared_backbone_routing_recalibration`、`soft_routing_context`、`soft_routing_activation_policy`、`soft_routing_top_combination`、`soft_routing_meta_loss`、`routing_active_set_policy`、`shared_adapter_rank`、`detection_episodes`、`new_model_creation_policy`、`fifo_size`、`new_model_validation_fraction`、`new_model_forward_validation_samples`、`result_metrics_json`、`total_data`、`e_detector_alpha`

## 新実装の保存の形式と、対応（決まった分）

新実装は、1 runにつき、要約（JSON: 完全なrun設定、指標、実行環境）と、記録の表（NPZ: `表の名前.列の名前`の配列）を保存する
（[fedsda-experiment-run-artifacts](../../.kiro/specs/fedsda-experiment-run-artifacts/design.md)）。

### 配列（旧→新）

| 旧 | 新 |
|---|---|
| `history_accuracy` | `sample_prediction.combined_prediction_is_correct`（予測に使う重みでの結合） |
| `history_routing_switching_correct` | `sample_prediction.fixed_share_weighted_prediction_is_correct`（Fixed-Shareの重みでの結合。Fixed-Shareの重みで予測する構成では、上と同じ値） |
| `history_model_id` | `sample_prediction.maximum_weight_model_id`（予測に使う重みでの、最大重みのモデル） |
| `history_routing_switching_leader_id` | `sample_prediction.fixed_share_maximum_weight_model_id` |
| `history_routing_soft_active` | なし（最終構成では、常に真） |
| `drift_client_ids`・`drift_positions` | `concept_change.client_id`・`.first_new_concept_sample_index` |
| `switch_client_ids`・`switch_positions` | `training_model_switch.client_id`・`.switch_sample_index` |
| `adaptation_client_ids`・`adaptation_positions`・`adaptation_old_model_ids`・`adaptation_new_model_ids` | `adaptation.client_id`・`.adaptation_sample_index`・`.previous_training_model_id`・`.current_training_model_id` |
| `adaptation_actions` | `adaptation.adaptation_action`（値の名前が変わる。下の表）。新は、より細かい`adaptation.adaptation_outcome`も持つ（旧の結果からは、作れない） |
| `provisional_client_ids`・`provisional_positions`・`provisional_accepted`・`provisional_resolution_positions` | `candidate_validation_decision.client_id`・`.proposal_sample_index`・`.candidate_accepted`・`.resolution_sample_index` |
| `provisional_reasons` | `candidate_validation_decision.decision_reason`と`.is_incomplete_at_run_end`（値の名前が変わる。下の表） |
| `model_registration_ids`・`model_registration_rounds`・`model_registration_final_active` | `model_registration.model_id`・`.registered_server_round_index`（初期モデルは−1）・`.is_final_global_model` |
| `clustering_rounds`・`clustering_model_ids`・`clustering_representative_model_ids`・`clustering_participated_in_merge`・`clustering_absorbed` | `model_clustering.server_round_index`・`.model_id`・`.representative_model_id`・`.merged_with_other_models`・`.absorbed_into_representative` |
| `clustering_pair_rounds`・`clustering_pair_left_model_ids`・`clustering_pair_right_model_ids`・`clustering_pair_same_cluster` | `model_pair_clustering.server_round_index`・`.lower_model_id`・`.higher_model_id`・`.assigned_to_same_cluster` |
| `round_global_model_count`・`round_client_held_model_count` | `server_round.global_model_count_after_synchronization`・`server_round.client_held_model_count_after_synchronization` |
| `estimated_drift_start_client_ids`・`estimated_drift_alarm_positions`・`estimated_drift_start_positions`・`detector_candidate_start_positions` | `loss_change_alarm.client_id`・`.alarm_sample_index`・`.estimated_change_point_sample_index`・`.detector_candidate_start_sample_index` |
| `history_detector_log_e` | `sample_loss_monitoring.monitored_log_e_value` |
| `adaptation_detectors`・`adaptation_estimated_change_points` | `adaptation.detector_name`・`.estimated_change_point_sample_index`（ないときは、新旧とも−1） |
| `provisional_detectors`・`provisional_validation_counts`・`provisional_reference_model_ids`・`provisional_matched_true` | `candidate_validation_decision.detector_name`・`.validation_sample_count`・`.comparison_reference_model_id`（ないときは、新旧とも−1）・`.matched_to_concept_change` |
| `provisional_interval_counts`、`provisional_training_counts` | `candidate_validation_decision.candidate_training_interval_sample_count`（最終構成では、旧の2つは同じ値） |
| `provisional_candidate_mean_losses`・`provisional_reference_mean_losses`・`provisional_candidate_recent_losses`・`provisional_reference_recent_losses`・`provisional_reference_historical_means` | `candidate_validation_decision.candidate_full_interval_mean_loss`・`.reference_full_interval_mean_loss`・`.candidate_second_segment_mean_loss`・`.reference_second_segment_mean_loss`・`.reference_historical_mean_loss`（ないときは、新旧ともNaN） |
| `provisional_validation_sources` | なし（最終構成では、常に`forward`） |
| `model_registration_client_ids`・`model_registration_selection_counts` | `model_registration.registering_client_id`（初期モデルは、新旧とも−1）・`.maximum_weight_prediction_sample_count` |
| `clustering_nearest_model_ids`・`clustering_nearest_distances` | `model_clustering.nearest_model_id`・`.nearest_loss_increase_distance`（距離を持つ対がなければ、新旧とも−1とNaN） |
| `clustering_cluster_sizes`・`clustering_cluster_max_distances`・`clustering_evaluated_pair_counts`・`clustering_possible_pair_counts` | `model_clustering.cluster_model_count`・`.maximum_within_cluster_distance`（なければ、新旧ともNaN）・`.evaluated_within_cluster_pair_count`・`.possible_within_cluster_pair_count` |
| `clustering_pair_distances`・`clustering_pair_decision_scores`・`clustering_pair_oracle_same_concept`・`clustering_pair_personalized_parameter_distances` | `model_pair_clustering.loss_increase_distance`・`.decision_score`・`.true_concepts_match`（一致1、不一致0、分からなければ、新旧とも−1）・`.concept_specific_parameter_distance` |
| `history_routing_effective_experts` | `sample_prediction.effective_model_count`（予測に使う重みでの値） |
| `history_routing_switching_effective_experts` | `sample_prediction.fixed_share_effective_model_count`（Fixed-Shareの重みで予測する構成では、上と同じ値） |
| `history_routing_max_weight` | `sample_prediction.maximum_prediction_weight` |
| `history_routing_oracle_correct`・`history_routing_leader_correct` | `sample_prediction.any_model_or_combined_prediction_is_correct`・`.maximum_weight_model_prediction_is_correct` |
| `history_routing_oracle_concept_correct` | `sample_prediction.true_concept_diagnostic_prediction_is_correct`（新は整数: 正解1、不正解0、真の概念が分からなければ−1。旧は真偽） |
| `routing_pool_reset_counts`・`routing_concept_restart_counts`・`routing_aggregation_recalibration_counts`・`routing_aggregation_recalibration_sample_counts` | `client_weight_update_counts.adahedge_weight_model_pool_reset_count`・`.adahedge_weight_concept_operation_restart_count`・`.adahedge_weight_aggregation_recalibration_count`・`.adahedge_weight_aggregation_recalibration_replayed_sample_count` |
| `routing_class_client_ids`・`routing_class_ids`・`routing_class_sample_counts`・`routing_class_oracle_correct_counts`・`routing_class_mixture_correct_counts`・`routing_class_leader_correct_counts`・`routing_class_confidence_leader_correct_counts`・`routing_class_missed_oracle_counts`・`routing_class_confidence_leader_missed_oracle_counts` | 表は持たない。`sample_prediction`の、`observed_class_id`と、正否の列（`combined_prediction_is_correct`・`any_model_or_combined_prediction_is_correct`・`maximum_weight_model_prediction_is_correct`・`highest_confidence_model_prediction_is_correct`）から、clientとクラスごとに数える（「missed」は、どれかのモデルか実際の予測が正解で、その予測が不正解の標本の数） |
| `cross_evaluation_round_index`・`_client_id`・`_candidate_model_id`・`_target_model_id`・`_n`・`_sum`・`_sum_sq` | `client_cross_evaluation.server_round_index`・`.client_id`・`.candidate_model_id`・`.target_model_id`・`.evaluated_sample_count`・`.bounded_loss_sum`・`.squared_bounded_loss_sum` |
| `cross_evaluation_candidate_only_correct`・`_target_only_correct`・`_both_correct`・`_both_wrong` | `client_cross_evaluation.candidate_only_correct_sample_count`・`.target_only_correct_sample_count`・`.both_correct_sample_count`・`.both_wrong_sample_count`（正誤を比べなかった評価では、新旧とも−1） |
| `cross_evaluation_class_*`（10） | `client_cross_evaluation_class`の、同じ名前の対応の列と、`.class_id`（クラスごとの標本数は、`.class_evaluated_sample_count`） |
| `model_pair_candidate_model_id`・`_target_model_id`・`_n`・`_candidate_only_correct`・`_target_only_correct`・`_both_correct`・`_both_wrong` | 表は持たない。`client_cross_evaluation`の、正誤を比べた行（`both_correct_sample_count`が0以上）の、対応する列と同じ |
| `model_diagnostic_client_ids`・`model_diagnostic_model_ids`・`model_diagnostic_assigned_samples`・`model_diagnostic_training_examples`・`model_diagnostic_optimizer_steps` | `client_held_model.client_id`・`.model_id`・`.stored_training_sample_count`・`.trained_sample_count`・`.parameter_update_step_count`（実行の終わりの値。clientの順、モデルIDの昇順） |
| （なし） | `sample_prediction.observed_class_id`・`.observed_concept_id`（観測したクラスと、真の概念）、`.highest_confidence_model_prediction_is_correct`・`.adahedge_weighted_prediction_is_correct`（旧は、合計だけを持つ）、`client_weight_update_counts.prediction_weight_*`（予測重みの側の、clientごとの回数。旧は、合計だけを指標に持つ） |
| （なし） | `sample_prediction.held_model_count_at_prediction`（その標本の予測に使った保有モデルの数）、`server_round.local_processing_*`・`server_round.synchronization_*`（サーバの同期のラウンドごとの、モデルの計算） |

整数は、新では`int64`（旧は、列により`int32`・`int8`・`float64`）。

### 値の名前（旧→新）

| 列 | 旧の値 | 新の値 |
|---|---|---|
| 適応の区分 | `maintain` | `maintain_current_model` |
| | `reuse` | `reuse_held_model` |
| | `create_pending` | `start_candidate_validation` |
| | `create` | `adopt_candidate_model` |
| | `create_rejected` | `reject_candidate_model` |
| | `forward_validation_pending` | `defer_during_candidate_validation` |
| | `insufficient_data` | `skip_too_short_change_interval` |
| | `server_merge` | `remap_by_server_consolidation` |
| 候補の判定の理由 | `accepted` | `both_segment_margins_passed` |
| | `first_interval` | `first_segment_margin_failed` |
| | `second_interval` | `second_segment_margin_failed` |
| | `first_and_second` | `both_segment_margins_failed` |
| | `current_reference_refit` | `current_reference_within_historical_loss_tolerance` |
| | `alternative_reference_refit` | `alternative_reference_within_historical_loss_tolerance` |
| | `insufficient_forward_data` | 空文字（`is_incomplete_at_run_end`が真） |

旧の`action`には、最終構成では現れない値もある: `episode_suppressed`（検出episodeの統合。未移植——UNPORTED-001）と、前向きの検証を使わない新規作成の方針での`create`・`create_rejected`（意味が、最終構成のものと違う）。最終構成でない旧の結果を変換するときは、これらの値の扱いを、(A2-変換)で決める。

### 指標（旧→新）

照合済みの33指標は、要約の`run_metrics`（`FedsdaRunMetrics`の辞書）にある。対応は、
`tests/refactoring/test_fedsda_run_metric_derivation.py`の`derive_legacy_metric_values`。

概念の変更への応答の指標（旧の`compute_metrics`が返す指標の残りと、監視の値の最大。
[fedsda-change-response-metrics-and-records](../../.kiro/specs/fedsda-change-response-metrics-and-records/design.md)）は、
`run_metrics.change_response_metrics`にある。対応は、同じtestの`derive_legacy_change_response_metric_values`。

| 旧 | 新（`change_response_metrics`の下） |
|---|---|
| `avg_delay` | `training_model_switch_timing_metrics.mean_matched_switch_delay_sample_count` |
| `switch_fp_duplicate`・`switch_fp_early`・`switch_fp_late`・`switch_fp_isolated` | `training_model_switch_timing_metrics.unmatched_switch_counts`の`duplicate_event_count`・`early_event_count`・`late_event_count`・`isolated_event_count` |
| `alarm_precision`・`alarm_recall`・`alarm_f1`・`alarm_total` | `loss_change_alarm_detection_metrics`の`detection_precision`・`detection_recall`・`detection_f1`・`detection_count` |
| `change_point_mae`・`change_point_bias`・`change_point_estimate_count` | `change_point_estimation_metrics`の`mean_absolute_error_sample_count`・`mean_signed_error_sample_count`・`matched_alarm_count` |
| `adaptation_reuse_count`・`adaptation_create_count`・`adaptation_create_rejected_count`・`adaptation_maintain_count`・`server_mapping_change_count` | `adaptation_action_metrics`の`reuse_held_model_count`・`adopt_candidate_model_count`・`reject_candidate_model_count`・`maintain_current_model_count`・`remap_by_server_consolidation_count` |
| `adaptation_reuse_precision`・`adaptation_create_precision` | `adaptation_action_metrics`の`reuse_held_model_precision`・`adopt_candidate_model_precision` |
| `model_reuse_current_fit_count`・`model_reuse_alternative_fit_count` | `adaptation_action_metrics`の`current_model_fit_selection_count`・`alternative_model_reuse_selection_count` |
| `provisional_acceptance_rate`・`provisional_accepted_precision` | `candidate_validation_decision_metrics`の`accepted_decision_rate`・`accepted_decision_precision` |
| `provisional_matched_true_count`・`provisional_accepted_matched_true_count`・`provisional_rejected_matched_true_count` | 同、`matched_decision_count`・`matched_accepted_decision_count`・`matched_rejected_decision_count` |
| `provisional_interval_count_mean`、`provisional_training_count_mean` | 同、`mean_candidate_training_interval_sample_count`（最終構成では、旧の2つは同じ値） |
| `provisional_validation_count_mean`・`provisional_resolution_delay_mean` | 同、`mean_validation_sample_count`・`mean_resolution_delay_sample_count` |
| `provisional_{accepted,rejected,matched,unmatched}_full_margin_mean` | 同、`{accepted,rejected,matched,unmatched}_mean_full_interval_loss_margin` |
| `provisional_{accepted,rejected,matched,unmatched}_recent_margin_mean` | 同、`{accepted,rejected,matched,unmatched}_mean_second_segment_loss_margin` |
| `provisional_reject_insufficient_forward_data_count` | 同、`incomplete_at_run_end_decision_count` |
| `provisional_reject_first_interval_count`・`provisional_reject_second_interval_count`・`provisional_reject_first_and_second_count` | 同、`first_segment_margin_failed_decision_count`・`second_segment_margin_failed_decision_count`・`both_segment_margins_failed_decision_count` |
| `provisional_reject_current_refit_count`・`provisional_reject_alternative_refit_count` | 同、`current_reference_within_historical_loss_tolerance_decision_count`・`alternative_reference_within_historical_loss_tolerance_decision_count` |
| `provisional_reference_excess_mean` | 同、`mean_reference_loss_excess_over_historical_mean` |
| `e_detector_max_log_e` | `maximum_monitored_log_e_value` |
| `total_true`・`tp`・`fp`・`fn`・`miss_rate`・`fdr`、`provisional_reject_reference_refit_count` | 持たない（ほかの値から計算できる: 検出の指標の計数と、2つの理由の数の和） |

新は、適応の区分ごとの件数を、8つとも持つ（旧が返すのは、上の5つ）。

モデル別の診断の指標（[fedsda-model-diagnostic-metrics-and-records](../../.kiro/specs/fedsda-model-diagnostic-metrics-and-records/design.md)）は、
`run_metrics.model_diagnostic_metrics`にある。対応は、同じtestの`derive_legacy_model_diagnostic_metric_values`。

| 旧 | 新（`model_diagnostic_metrics`の下） |
|---|---|
| `mean_model_count`・`max_model_count`・`model_count_auc` | `global_model_count_metrics`の`mean_global_model_count_after_synchronization`・`maximum_global_model_count_after_synchronization`・`total_global_model_count_over_server_rounds`（旧の「auc」は、ラウンドごとのモデル数の合計） |
| `model_assigned_samples_{total,mean,min,cv}` | `held_model_training_amount_metrics.stored_training_sample_count_summary`の`total_value`・`mean_value`・`minimum_value`・`coefficient_of_variation` |
| `model_training_examples_{total,mean,min,cv}` | 同、`trained_sample_count_summary`の4つ |
| `model_optimizer_steps_{total,mean,min,cv}` | 同、`parameter_update_step_count_summary`の4つ |
| `clustering_oracle_pair_count`・`clustering_oracle_same_pair_count` | `clustering_true_concept_agreement_metrics`の`true_concept_known_pair_count`・`same_true_concept_pair_count` |
| `clustering_oracle_merge_tp`・`_fp`・`_fn`・`_tn` | 同、`cluster_assignment_agreement`の`same_cluster_same_concept_pair_count`・`same_cluster_different_concept_pair_count`・`different_cluster_same_concept_pair_count`・`different_cluster_different_concept_pair_count` |
| `clustering_oracle_merge_precision`・`_recall`・`_f1` | 同、`cluster_assignment_agreement`の`same_cluster_precision`・`same_cluster_recall`・`same_cluster_f1` |
| `clustering_oracle_loss_distance_auc`・`clustering_oracle_parameter_distance_auc` | 同、`loss_increase_distance_same_concept_auc`・`concept_specific_parameter_distance_same_concept_auc`（定義できないとき、旧はNaN、新は`None`。保存した要約では、`null`） |

モデル対の予測の相補性の指標は、`run_metrics.model_pair_prediction_complementarity_metrics`にある。

| 旧 | 新（`model_pair_prediction_complementarity_metrics`の下） |
|---|---|
| `model_pair_evaluation_count`・`model_pair_sample_count`・`model_pair_correctness_disagreement_rate`・`model_pair_oracle_gain_rate`・`model_pair_both_correct_rate` | `correctness_compared_evaluation_count`・`correctness_compared_sample_count`・`correctness_disagreement_rate`・`per_sample_selection_gain_rate`・`both_correct_rate`（[fedsda-cross-evaluation-records-and-metrics](../../.kiro/specs/fedsda-cross-evaluation-records-and-metrics/design.md)。対応は、同じtestの`derive_legacy_model_pair_metric_values`） |

予測の診断の指標（[fedsda-prediction-diagnostic-metrics-and-records](../../.kiro/specs/fedsda-prediction-diagnostic-metrics-and-records/design.md)）は、
`run_metrics.prediction_diagnostic_metrics`にある。新は、数（標本数、正解の数）を持ち、旧の割合・差・比は、数から計算する。
対応と、計算の式は、同じtestの`derive_legacy_prediction_diagnostic_metric_values`。下の表で、n＝`prediction_correct_counts.predicted_sample_count`、
combined・any・maxw・conf＝`prediction_correct_counts`の`combined_prediction_correct_count`・`any_model_or_combined_prediction_correct_count`・
`maximum_weight_model_prediction_correct_count`・`highest_confidence_model_prediction_correct_count`。分母が0なら、0。

| 旧 | 新（`prediction_diagnostic_metrics`の下）と、計算 |
|---|---|
| `routing_sample_count` | n |
| `routing_mixture_accuracy` | combined÷n（`run_metrics.prediction_accuracy`と同じ） |
| `routing_switching_accuracy` | `fixed_share_weighted_prediction_correct_count`÷n（Fixed-Shareの重みで予測する構成では、上と同じ値） |
| `routing_oracle_accuracy`・`routing_leader_accuracy`・`routing_confidence_leader_accuracy` | any÷n・maxw÷n・conf÷n |
| `routing_oracle_gain_rate`・`routing_missed_oracle_count` | （any−combined）÷n・any−combined |
| `routing_oracle_recovery_rate`・`routing_confidence_leader_oracle_recovery_rate` | combined÷any・conf÷any |
| `routing_confidence_leader_missed_oracle_count` | any−conf |
| `routing_oracle_concept_accuracy` | `true_concept_diagnostic_prediction_correct_count`÷`true_concept_known_sample_count` |
| `routing_switching_global_gain_rate` | （`fixed_share_weighted_prediction_correct_count`−`adahedge_weighted_prediction_correct_count`）÷n |
| `routing_switching_stable_accuracy`・`routing_switching_recovery_accuracy` | `fixed_share_weighted_prediction_period_accuracies`の`stable_period_accuracy`・`recovery_period_accuracy` |
| `routing_oracle_stable_accuracy`・`routing_oracle_recovery_accuracy` | `any_model_or_combined_prediction_period_accuracies`の2つ |
| `routing_leader_stable_accuracy`・`routing_leader_recovery_accuracy` | `maximum_weight_model_prediction_period_accuracies`の2つ |
| `routing_oracle_concept_stable_accuracy`・`routing_oracle_concept_recovery_accuracy` | `true_concept_diagnostic_prediction_period_accuracies`の2つ |
| `routing_stable_oracle_gap`・`routing_recovery_oracle_gap` | 「どれかのモデルか実際の予測」の区間の値−実際の予測の区間の値 |
| `routing_oracle_concept_stable_gain_rate`・`routing_oracle_concept_recovery_gain_rate` | 真の概念別の診断の区間の値−実際の予測の区間の値 |
| `routing_class_macro_mixture_accuracy`・`routing_class_macro_oracle_accuracy`・`routing_class_macro_leader_accuracy`・`routing_class_macro_confidence_leader_accuracy` | `class_macro_prediction_metrics`の`macro_combined_prediction_accuracy`・`macro_any_model_or_combined_prediction_accuracy`・`macro_maximum_weight_model_prediction_accuracy`・`macro_highest_confidence_model_prediction_accuracy` |
| `routing_class_oracle_gap_mean`・`routing_class_oracle_gap_std` | 同、`mean_any_model_or_combined_accuracy_gap`・`any_model_or_combined_accuracy_gap_standard_deviation` |
| `routing_class_oracle_recovery_rate_mean`・`routing_class_oracle_recovery_rate_min` | 同、`mean_combined_to_any_model_or_combined_correct_ratio`・`minimum_combined_to_any_model_or_combined_correct_ratio` |
| `routing_switching_effective_experts_mean` | `mean_fixed_share_effective_model_count` |
| `routing_switching_leader_switch_count`・`routing_switching_pool_reset_count` | `fixed_share_weight_leader_switch_count`・`fixed_share_weight_model_pool_reset_count` |
| `routing_aggregation_recalibration_count` | `adahedge_weight_aggregation_recalibration_count` |
| `routing_switching_gain_rate`・`routing_switching_stable_gain_rate`・`routing_switching_recovery_gain_rate` | 持たない。Fixed-Shareの結合と、実際の予測の差で、上の数と割合から計算できる（（`fixed_share_weighted_prediction_correct_count`−combined）÷n、区間の正解の割合の差）。Fixed-Shareの重みで予測する構成では、いつも0 |

### 残り（2026-10-11時点）

「記録から導く（未照合）」（指標124・配列85）は、全部、照合が済んだ: 指標124（応答の指標52、`e_detector_max_log_e`、モデル別の診断の指標26、モデル対の予測の相補性の指標5、予測の診断の指標40）と、配列85（ラウンドごとのモデル数の2つ、応答の記録の、値を照合した18、常に`forward`であることを確かめた`provisional_validation_sources`、登録・クラスタリング・保有モデルの17、クロス評価の28、予測の診断の19）。
新の記録だけでは導けなかった項目は、ない。残るのは、ほかの区分（上の区分の表: 用途別の計算の計数、最終構成では使わない選択肢の項目、未移植の診断、所要時間）。
