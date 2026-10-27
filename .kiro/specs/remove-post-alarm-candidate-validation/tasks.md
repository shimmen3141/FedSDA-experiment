# Implementation Plan

依存: task 1 → 2 → 3 → 4 → 5 → 6。実装はGPT-6.1-sol（task 1〜5）。版15のgoldenの生成、全pytest、WSL、独立レビューは主担当（task 6）。

各taskの完了条件に共通:
- `tests/`の全体が通る（壊れたtestを次のtaskへ送らない。そのtaskの変更で成り立たなくなったtestは、そのtaskの中で要求4.5の規則に従って直すか消し、記録に書く）。
  sandboxの制約で実行できないtestは、名前と理由を記録に書き、主担当が通常の環境で確かめる。
- 版1〜14の残すcaseの数値の照合（`tests/refactoring/test_important_behavior_regression_goldens.py`）と、要求4.7の乱数の状態の照合が通る。
- そのtaskで消した名前について、要求5.3の参照の検査（`git grep`）で残りが無い。

- [x] 1. 残す比較と共有の部品を確定する（まだ何も消さない）
  - 候補の学習を新しいモジュールへ移し、呼び出し側と、調査の報告の5の表で挙げた研究用の道具のpatchの先（`tools/research_s3_contribution/`、`tools/research_output_layer_learner/`ほか）を直す。
  - goldenの照合に「外すcase」「除く項目」の仕組みを足す（外すcaseは要求4.6、除く項目はこの段では空）。外すcaseの一覧のtest（要求4.6）。
  - 要求4.7の乱数の状態のfixtureを、このtaskの開始時のHEADのコードで作り、照合のtestを足す。
  - _Requirements: 2.2, 4.1, 4.2, 4.6, 4.7_

- [ ] 2. 構成と設定の入口を閉じる
  - 構成・方針の値・設定の項目・制約を消す。検証ありの設定を組み立てていたtest・helper・fixtureを、要求4.5の規則で扱う。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 4.5_

- [ ] 3. 実行と状態を消す
  - _Requirements: 2.1, 2.3, 4.5_

- [ ] 4. 記録・指標・保存の形を消す
  - goldenの除く項目の一覧を確定する（一覧にない差は失敗）。
  - _Requirements: 3.1, 3.2, 3.3, 4.2, 5.1, 5.2_

- [ ] 5. 研究用の道具・文書・版15
  - 研究用の道具の候補検証のモード・入口を外し、調査の報告の5の表の道具ごとに、残るモードの専用の検証（`checks*.py`）が通ることを確かめて記録する。
  - 文書（要求6.2・6.3）。`BEHAVIOR_BASELINE_VERSION = 15`と版15のcaseの一覧（要求4.3）、版15と版14の比較のtest（版15のファイルが無い間は失敗してよい。記録に書く）。
  - 要求5.3の参照の検査を、消した全部の名前で行う。
  - _Requirements: 4.3, 4.4, 5.3, 6.1, 6.2, 6.3_

- [ ] 6. 確認（主担当）
  - 版15のgolden（Windows・Linux）を作り、4.4のtestを通す。全pytest、WSLのgolden、pyright、ruff。独立の実装レビュー1回。


## Implementation Notes

### task 1（2026-10-28、実装担当）

（実装担当の記録の前半は、書込みの文字化けで読めなくなったため、主担当が実装担当の最終の報告（`results/_sol_prompts/improve023-task1-output.txt`、gitの管理外）から書き直した。）

- 開始時のHEADは`6adb3ee77ae57af4bcef612916fc6389d075ab80`（branch `refactor/architecture`）。task 2以降は未着手、stage・commitはしていない。
- 移したもの: `TrainedAlarmIntervalCandidate`、`train_candidate_on_alarm_change_interval`と、その前の入力検査の2関数を、新しい`runtime/alarm_interval_candidate_training.py`へ移した。
  非公開名の変更を除き、移す前とのASTの一致を確かめた。呼び出し側は`alarm_change_interval_resolution.py`、`post_alarm_candidate_validation_session_start.py`。
- 研究用の道具: `tools/research_s3_contribution/diagnostic.py`のpatchの先を直した。`tools/research_output_layer_learner/policy.py`は呼び出し側をpatchしているので変更不要。接続のtestは17 passed。
- goldenの照合: `EXCLUDED_CANDIDATE_VALIDATION_CASE_NAMES_BY_BASELINE_VERSION`、`REMOVED_CANDIDATE_VALIDATION_ITEM_PATHS = ()`、`without_removed_candidate_validation_items`を足した。保存済みのJSONは変えていない。
- 外すcase（保存済みの設定と各版のbuilderの両方と集合が一致することを確かめた）:
  - 版1〜6: `fedsda`、`fedsda-without-prediction-weight-recalibration`、`fedsda-loss-distance-clustering`、`fedsda-overall-loss-esr-detection`、`fedsda-adahedge-prediction-weights`、
    `fedsda-single-model-prediction`、`fedsda-classwise-adwin-detection`、`fedsda-overall-loss-adwin-detection`、`fedsda-without-concept-specific-adapter`、`fedsda-independent-models`
  - 版1〜14: `fedsda-sine2-events`、`fedsda-mnist2-decisions`。版2〜14: `fedsda-sine2-reference-reuse`。版4〜14: `fedsda-sine2-alternative-reference-reuse`
  - 残す数値goldenは170件で、全部一致した（外した数値照合は112件）。
- 乱数の状態のfixture: `tests/refactoring/golden/fedsda_without_candidate_validation_random_states.json`。移す前のHEAD `6adb3ee7`（srcの変更なし）で、FedSDAの14 caseを実行して作った。
  全client・通信の終端の処理の後、torchの状態の復元の前に、runが借りるPython `Random`、NumPy `RandomState`、torch CPUの全状態を採った。移した後も14 caseとも完全一致した。
  作る道具は、開始時のcommitとsrcの無変更を検査し、既存のfixtureを上書きしない（移した後の再生成は拒否する）。
- 新しいtest（`tests/refactoring/test_candidate_validation_removal_regression.py`）は30 passed（要求4.6の14の一覧と、要求4.7の14 caseの照合を含む）。依存境界のtestは3127 passed。
- Pyright: src3ファイルと新しいhelper・testは0 errors。変更した9ファイル全体の33件は、開始時のHEADにもある既存の診断で、新しい診断は0。Ruff: 対象8ファイルのcheck・formatが通った。

- 全pytest（最終のsource/test、開始時HEADからの未コミット差分）: `PYTHONPATH="src;.;.tmp"`（最後の要素は検証用pluginの読込みだけ）、OMP/MKL各1、MNIST固定入力、固定venvで `python -m pytest tests -q -p no:cacheprovider -p task1_sandbox_plugin --tb=short --junitxml=.tmp/task1-full.xml`。**14646 passed / 432 skipped / 2 failed、全15080件、exit 1、2149.20秒**。全体成功とは記録しない。収集数は開始時相当15162件から、数値照合112件を外し、新規30件を足した差−82と一致する。
- JUnitで確認した結果: 版1〜14の残す数値goldenは170 passed。goldenのファイル全体は414 passed / 19 skipped。新規test 30 passed、依存境界3127 passed。`tests/test_regression.py::test_regression`はpassed。`tests/test_proposed_regression.py::test_proposed_regression`は下記の権限で失敗し、最終3ケースの全照合は未完了。JUnitは`.tmp/task1-full.xml`、集計は`.tmp/task1-results.json`、標準出力は`.tmp/task1-full.log`。
- 全体で失敗した2件は、両方ともsandboxの一時領域制約で実行不能。再試行やsandbox解除はしていない。通常環境で主担当が確認する。
  - `tests/refactoring/test_fresh_process_smoke.py::test_fresh_process_smoke_runs_without_legacy_or_test_modules`: 子process内の`fresh_process_smoke.py`がFedDrift成果物を`TemporaryDirectory`へ保存する箇所で、`feddrift.npz.<uuid>.partial`への`PermissionError`。後始末のディレクトリ走査もWinError 5。候補学習の実装の例外ではない。
  - `tests/test_proposed_regression.py::test_proposed_regression`: 旧実装の`compute_all()`内の`TemporaryDirectory(prefix="fedsda-regression-")`へ`sine2.npz`を保存する箇所で`PermissionError`。後始末もWinError 5。旧実装・旧testは変更していない。
- skip 432件の内訳: 一時領域fixtureの事前除外406件、環境条件・網羅前提未成立による26件。後者には、前提となる一時領域testを実行しなかったための網羅確認のskipも含む。全件の関数名と理由を下へ記録する（完全なパラメータ付き名はJUnit）。
- 最終のRuff check・format（対象8ファイル）、`git diff --check`はexit 0。Pyrightの再現用設定は`.tmp/task1-pyright.json`（変更9ファイル）と`.tmp/task1-pyright-new-code.json`（src3ファイル＋新helper/test）。前者の既存33診断・新規0、後者の0 errorsを確認。通常のvenv自動探索は権限制約で機能しておらず、通常環境での標準コマンドの再確認は残る。
- 完了の判断: task 1の実装・数値golden・乱数・依存境界の照合は完了。共通完了条件が許すsandbox例外として、上記2件と下記406件および網羅確認は主担当の通常環境での確認待ち。`kiro-verify-completion`では、実装・数値・乱数はVERIFIED、環境制約を含む全確認はMANUAL_VERIFY_REQUIRED。WSLでの数値実行、全task後の独立実装レビューは本担当では未実施（task 6）。主担当がtask 1を確認するまでtask 2へ進まない。コミット・stageはしていない。

<details>
<summary>通常環境で再実行するtestの名前（一時領域fixtureによる406件）</summary>

理由はすべて、事前確認したsandboxでの一時ディレクトリ内の書込み／アクセス拒否。各関数のパラメータは全件が対象で、括弧は除外した件数。完全なパラメータ付き名と理由はJUnitへ保存した。

| testファイル | test関数（パラメータ全件） |
|---|---|
| `tests/refactoring/test_behavior_fix_comparison_scripts.py` | `test_analysis_reports_absolute_values_and_differences_per_dataset_only`（1件）<br>`test_analysis_tables_are_unchanged_from_the_previous_script`（1件）<br>`test_analysis_warns_right_after_the_title_when_baseline_versions_are_mixed`（1件）<br>`test_analysis_without_input_tables_returns_one_and_writes_nothing`（1件）<br>`test_analysis_writes_a_file_with_provenance_in_the_input_directory`（1件）<br>`test_comparison_record_holds_code_revision_and_behavior_baseline`（1件）<br>`test_comparison_record_is_not_overwritten_within_the_same_second`（1件）<br>`test_comparison_requires_purpose_and_places_results_under_its_study`（1件）<br>`test_comparison_stops_with_a_warning_when_existing_runs_have_another_baseline`（1件）<br>`test_short_alarm_axis_rejects_a_registration_level`（1件） |
| `tests/refactoring/test_cross_evaluation_download_deduplication.py` | `test_independent_run_saves_corrected_communication_volume`（1件） |
| `tests/refactoring/test_environment_proposed_regression.py` | `test_damaged_golden_files_are_reported_with_their_file_name`（4件）<br>`test_golden_collection_may_be_empty_and_detects_misplaced_files`（1件）<br>`test_golden_is_selected_only_by_exactly_matching_environment`（1件）<br>`test_goldens_with_the_same_environment_are_rejected`（1件）<br>`test_missing_golden_reason_tells_the_environment_and_how_to_add_one`（1件）<br>`test_writing_a_golden_follows_the_environment_and_refuses_unreviewed_overwrites`（1件）<br>`test_writing_a_golden_is_always_refused_in_the_fixed_golden_environment`（1件） |
| `tests/refactoring/test_experiment_sweep.py` | `test_code_revision_counts_only_tracked_files_as_uncommitted_changes`（1件）<br>`test_code_revision_describes_commit_or_reason`（1件）<br>`test_command_creates_no_readme_when_existing_runs_have_another_baseline`（1件）<br>`test_command_keeps_an_existing_readme_and_records_each_purpose`（1件）<br>`test_command_places_results_under_the_study_with_a_readme`（1件）<br>`test_command_rejects_invalid_plans_before_running`（5件）<br>`test_command_rejects_names_that_are_not_plain_directory_names`（6件）<br>`test_command_rejects_study_with_output_directory`（1件）<br>`test_command_requires_a_single_line_purpose_to_run`（3件）<br>`test_command_returns_nonzero_when_a_run_fails`（1件）<br>`test_command_runs_sweep_and_writes_table_and_record`（1件）<br>`test_command_stops_before_running_when_mnist_files_are_missing`（1件）<br>`test_command_stops_with_a_warning_when_existing_runs_have_another_baseline`（1件）<br>`test_command_writes_record_even_when_a_run_summary_is_unreadable`（1件）<br>`test_completed_runs_are_skipped_and_left_unchanged`（1件）<br>`test_conflicting_existing_files_are_reported_and_not_overwritten`（4件）<br>`test_existing_runs_with_another_baseline_stop_the_sweep_before_anything_runs`（6件）<br>`test_existing_runs_with_damaged_provenance_are_conflicting_not_stopping`（3件）<br>`test_existing_runs_with_the_same_baseline_are_skipped_even_with_another_commit`（1件）<br>`test_experiment_readme_is_written_once_and_validates_arguments`（1件）<br>`test_failed_run_is_reported_and_other_runs_continue`（1件）<br>`test_parallel_sweep_cancels_waiting_runs_when_interrupted`（1件）<br>`test_parallel_sweep_saves_the_same_content_as_sequential_sweep`（1件）<br>`test_print_plan_lists_runs_and_creates_nothing`（1件）<br>`test_repository_script_runs_parallel_sweep_in_separate_processes`（1件）<br>`test_repository_script_runs_without_installation_or_environment_variables`（1件）<br>`test_run_summary_table_holds_one_row_per_saved_run`（1件）<br>`test_run_summary_table_holds_run_provenance_columns`（1件）<br>`test_saved_summary_holds_the_given_run_provenance`（1件）<br>`test_sequential_sweep_saves_the_same_content_as_direct_run`（1件）<br>`test_sweep_execution_rejects_invalid_arguments`（1件）<br>`test_sweep_record_holds_plan_command_revision_and_outcomes`（1件） |
| `tests/refactoring/test_experiment_sweep_resumption.py` | `test_active_output_refuses_command_before_readme_or_records_change`（1件）<br>`test_cli_recovery_counts_and_preserves_saved_npz`（1件）<br>`test_completed_artifacts_are_not_changed_and_training_interrupt_resumes`（1件）<br>`test_different_or_broken_plan_is_not_offered`（1件）<br>`test_different_requested_run_set_does_not_resume_existing`（1件）<br>`test_invalid_recovery_proof_types_and_unknown_current_code_are_refused`（4件）<br>`test_output_lock_is_cross_process_and_released`（1件）<br>`test_partial_old_readme_can_be_selected_with_new_metadata_and_workers`（1件）<br>`test_plan_identity_ignores_order_but_not_settings`（1件）<br>`test_process_termination_releases_output_lock`（1件）<br>`test_same_proof_recovers_orphan_without_losing_old_bytes`（1件）<br>`test_select_new_and_explicit_new_do_not_overwrite_same_second_output`（1件）<br>`test_selection_can_finish_without_any_file_changes`（4件）<br>`test_symlink_does_not_allow_recovery_outside_output`（1件）<br>`test_unproven_interruption_is_shown_and_selection_does_not_modify_it`（1件）<br>`test_unproven_orphan_is_preserved`（7件）<br>`test_untracked_execution_source_change_disables_recovery`（3件） |
| `tests/refactoring/test_fedavg_experiment_run.py` | `test_existing_half_artifact_is_not_overwritten`（2件）<br>`test_loader_rejects_other_method_and_unsafe_reference`（3件）<br>`test_run_artifact_roundtrip`（2件）<br>`test_stale_summary_and_inconsistent_predictions_are_rejected_before_write`（1件） |
| `tests/refactoring/test_fedavg_experiment_sweep.py` | `test_print_plan_is_read_only_and_old_name_is_rejected`（1件）<br>`test_real_sweep_saves_resumes_and_refuses_conflict`（2件） |
| `tests/refactoring/test_fedavg_local_updates_per_aggregation.py` | `test_sweep_command_saves_runs_under_the_configuration_directory`（1件） |
| `tests/refactoring/test_feddrift_experiment_run_artifacts.py` | `test_existing_output_is_preserved`（2件）<br>`test_loader_rejects_unreadable_table_or_summary`（5件）<br>`test_loader_rejects_wrong_version_method_or_path`（10件）<br>`test_save_failure_cleans_its_files`（5件）<br>`test_save_load_roundtrip_and_nonfinite_summary`（2件）<br>`test_save_rejects_invalid_name_without_writing`（10件）<br>`test_save_rejects_invalid_result_or_directory`（3件） |
| `tests/refactoring/test_feddrift_experiment_sweep.py` | `test_actual_sweep_matches_single_run_and_resumes_without_execution`（2件）<br>`test_cli_rejects_fedsda_only_explicit_axis_and_handles_missing_mnist`（1件）<br>`test_cli_rejects_invalid_axis_before_outputs`（5件）<br>`test_cli_writes_threshold_columns_records_and_resumes`（1件）<br>`test_conflicting_feddrift_artifacts_are_not_overwritten`（5件）<br>`test_feddrift_cli_reports_execution_failure`（1件）<br>`test_real_cli_help_and_spawn_execution`（1件） |
| `tests/refactoring/test_fedsda_configuration_variants.py` | `test_configuration_variant_matches_real_legacy_whole_run`（70件） |
| `tests/refactoring/test_fedsda_experiment_run_artifacts.py` | `test_failed_save_leaves_no_partial_files`（1件）<br>`test_golden_condition_change_response_columns_match_legacy_saved_arrays`（3件）<br>`test_golden_condition_experiment_run_matches_separately_executed_run`（3件）<br>`test_golden_condition_model_diagnostic_columns_match_legacy_saved_arrays`（3件）<br>`test_golden_condition_record_tables_rebuild_legacy_saved_arrays`（3件）<br>`test_golden_condition_round_model_counts_match_legacy_saved_arrays`（3件）<br>`test_golden_condition_saved_artifacts_match_real_legacy_and_golden`（3件）<br>`test_loading_accepts_previous_format_without_run_provenance`（1件）<br>`test_loading_rejects_unsupported_or_damaged_summaries`（1件）<br>`test_round_model_counts_match_real_legacy_whole_run`（3件）<br>`test_saved_artifacts_round_trip`（1件）<br>`test_saving_refuses_to_overwrite_and_writes_nothing`（1件）<br>`test_saving_rejects_invalid_arguments`（1件）<br>`test_saving_rejects_invalid_destinations`（10件） |
| `tests/refactoring/test_fedsda_fixed_share_time_scale_axis.py` | `test_command_rejects_invalid_time_scales_before_running`（1件）<br>`test_command_saves_runs_with_time_scale_column_and_record_format_version`（1件） |
| `tests/refactoring/test_fedsda_pending_assignment_fifo_capacity_axis.py` | `test_command_rejects_invalid_capacities_before_running`（1件）<br>`test_command_saves_runs_with_capacity_column_and_record_format_version`（1件） |
| `tests/refactoring/test_fedsda_run_metric_derivation.py` | `test_derivation_does_not_change_participant_state_and_is_repeatable`（3件）<br>`test_derivation_rejects_invalid_arguments`（3件）<br>`test_golden_condition_comparison_computation_metrics_match_real_legacy_counts`（3件）<br>`test_golden_condition_decision_kinds_are_reported`（3件）<br>`test_golden_condition_metrics_and_traces_match_environment_golden`（3件）<br>`test_golden_condition_metrics_match_real_legacy_experiment_result`（3件）<br>`test_golden_condition_trace_arrays_match_real_legacy_saved_arrays`（3件）<br>`test_small_run_metrics_match_real_legacy_metric_computation`（14件） |
| `tests/refactoring/test_fmnist_and_rotated_mnist_runs.py` | `test_command_accepts_new_datasets_without_creating_output`（1件）<br>`test_command_rejects_missing_image_files_before_output`（2件） |
| `tests/refactoring/test_fmnist_and_rotated_mnist_sample_generation.py` | `test_cache_does_not_mix_mnist_and_fashion`（1件）<br>`test_fashion_directory_resolution`（1件）<br>`test_fashion_labels_and_sampling_follow_mnist`（2件）<br>`test_fashion_missing_file_is_rejected_without_download`（2件）<br>`test_fashion_reuses_idx_validation`（1件）<br>`test_generators_belong_only_to_their_dataset`（1件）<br>`test_rotation_matches_official_numpy_operations`（2件）<br>`test_shared_directory_for_mnist_and_fashion_is_rejected`（3件） |
| `tests/refactoring/test_important_behavior_regression_goldens.py` | `test_corrupt_golden_is_rejected_before_environment_skip`（9件）<br>`test_duplicate_environment_is_rejected`（1件）<br>`test_generation_preserves_existing_output_before_starting`（1件）<br>`test_generation_rejects_unrecorded_source_before_workers`（3件）<br>`test_generation_requires_two_matching_process_results`（3件）<br>`test_unicode_dtype_is_a_valid_digest`（1件）<br>`test_unknown_environment_has_no_match_but_reads_structure`（1件） |
| `tests/refactoring/test_mnist_sample_generation.py` | `test_damaged_files_are_rejected_as_invalid_format`（8件）<br>`test_data_directory_follows_environment_variable_then_repository_default`（1件）<br>`test_differing_image_and_label_counts_are_rejected`（1件）<br>`test_invalid_file_formats_are_rejected`（6件）<br>`test_loaded_training_data_is_reused_per_resolved_directory`（1件）<br>`test_loading_requires_keyword_path_argument`（1件）<br>`test_missing_files_are_rejected_without_fetching`（3件）<br>`test_small_files_are_read_as_written`（1件） |
| `tests/refactoring/test_seed_argument_parsing.py` | `test_invalid_main_arguments_create_no_output`（3件）<br>`test_seed_syntax_does_not_change_plan_identifiers_or_legacy_resume`（1件） |
| `tests/refactoring/test_synthetic_sample_generation.py` | `test_expected_generator_types_cover_all_defined_datasets`（1件） |
| `tests/test_build_feddrift_baseline.py` | `test_build_baseline_groups_canonical_results_by_dataset`（1件）<br>`test_build_baseline_keeps_same_sweep_for_multiple_schedules`（1件）<br>`test_build_baseline_prefers_explicit_sweep_parameter`（1件）<br>`test_build_baseline_refuses_to_overwrite_existing_output`（1件）<br>`test_extend_baseline_adds_new_seed_atomically_and_keeps_backup`（1件）<br>`test_extend_baseline_adds_new_sweep_value`（1件）<br>`test_extend_baseline_rejects_conflicting_duplicate`（1件） |
| `tests/test_clustering_functional_diagnostics.py` | `test_clustering_functional_diagnostics_separates_merged_and_retained`（1件） |
| `tests/test_experiment_artifacts.py` | `test_rebuild_artifacts_recovers_pareto_metrics_from_old_raw_and_log`（1件）<br>`test_rebuild_artifacts_uses_embedded_exact_metrics`（1件） |
| `tests/test_experiment_manifests.py` | `test_backfill_existing_csv`（1件）<br>`test_backfill_infers_missing_schedule_and_total_data_from_raw`（1件）<br>`test_manifest_session_records_lifecycle_and_outputs`（1件）<br>`test_overlap_distinguishes_exact_and_different_provenance`（1件）<br>`test_preview_overlaps_does_not_create_manifest`（1件）<br>`test_provenance_changes_when_regression_golden_changes`（1件）<br>`test_recursive_backfill_separates_variants_and_ignores_analysis_csv`（1件）<br>`test_study_manifest_is_generated_from_utf8_definition`（1件） |
| `tests/test_paper_datasets.py` | `test_idx_reader_parses_mnist_files`（1件） |
| `tests/test_pareto_sweep.py` | `test_load_csv_accepts_canonical_feddrift_baseline_names`（1件）<br>`test_load_csv_rejects_legacy_parameter_columns`（1件）<br>`test_plot_pareto_can_use_overall_accuracy`（1件）<br>`test_plot_pareto_draws_baseline_standard_deviation_band`（1件）<br>`test_replot_filter_selects_interval_sweeps_and_plot_accepts_compute_x`（1件） |
| `tests/test_recovery_analysis.py` | `test_plot_recovery_skips_file_when_no_series`（1件） |
| `tests/test_shared_backbone.py` | `test_drift_recovery_activation_is_saved_without_changing_history_shape`（1件）<br>`test_shared_backbone_experiment_reports_component_metrics`（1件） |
| `tests/test_telemetry.py` | `test_experiment_records_compute_and_model_telemetry`（1件） |

</details>

環境条件・網羅前提未成立の26件（sandboxの事前除外とは別）:

- `tests.refactoring.test_environment_proposed_regression::test_legacy_final_configuration_matches_environment_golden`: 固定のgoldenの環境では、既存の回帰test（tests/test_proposed_regression.py）が照合する
- `tests.refactoring.test_feddrift_linux_core_metrics::test_linux_full_run_core_metrics_and_parameters_match_fixed_legacy[sine2]`: 固定Linux CPU基準環境専用。実旧全体run対照は各環境で別に実行する。
- `tests.refactoring.test_feddrift_linux_core_metrics::test_linux_full_run_core_metrics_and_parameters_match_fixed_legacy[sea4]`: 固定Linux CPU基準環境専用。実旧全体run対照は各環境で別に実行する。
- `tests.refactoring.test_feddrift_linux_core_metrics::test_linux_full_run_core_metrics_and_parameters_match_fixed_legacy[mnist2]`: 固定Linux CPU基準環境専用。実旧全体run対照は各環境で別に実行する。
- `tests.refactoring.test_feddrift_linux_core_metrics::test_linux_full_run_core_metrics_and_parameters_match_fixed_legacy[sine2_drift]`: 固定Linux CPU基準環境専用。実旧全体run対照は各環境で別に実行する。
- `tests.refactoring.test_fedsda_configuration_variants::test_without_prediction_weight_recalibration_takes_its_own_path[fedsda-without-prediction-weight-recalibration]`: 構成の、小さい条件の全部を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_configuration_variants::test_without_candidate_validation_takes_its_own_path[fedsda-without-candidate-validation]`: 構成の、小さい条件の全部を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_configuration_variants::test_loss_distance_clustering_takes_its_own_path[fedsda-loss-distance-clustering]`: 構成の、小さい条件の全部を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_configuration_variants::test_overall_loss_esr_detection_takes_its_own_path[fedsda-overall-loss-esr-detection]`: 構成の、小さい条件の全部を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_configuration_variants::test_adahedge_prediction_weights_take_their_own_path[fedsda-adahedge-prediction-weights]`: 構成の、小さい条件の全部を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_configuration_variants::test_single_model_prediction_takes_its_own_path[fedsda-single-model-prediction]`: 構成の、小さい条件の全部を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_configuration_variants::test_adwin_detection_takes_its_own_path[fedsda-classwise-adwin-detection-True]`: 構成の、小さい条件の全部を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_configuration_variants::test_adwin_detection_takes_its_own_path[fedsda-overall-loss-adwin-detection-False]`: 構成の、小さい条件の全部を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_configuration_variants::test_without_concept_specific_adapter_takes_its_own_path[fedsda-without-concept-specific-adapter]`: 構成の、小さい条件の全部を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_configuration_variants::test_independent_models_take_their_own_path[fedsda-independent-models]`: 構成の、小さい条件の全部を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_experiment_run_artifacts::test_cross_evaluation_table_conditions_cover_required_cases`: goldenの3ケースと、小さい条件の全部を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_experiment_run_artifacts::test_prediction_diagnostic_column_conditions_cover_required_cases`: goldenの3ケースと、小さい条件の全部を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_experiment_run_artifacts::test_model_diagnostic_column_conditions_cover_required_cases`: goldenの3ケースと、小さい条件の全部を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_run_metric_derivation::test_small_run_conditions_cover_required_cases`: 対照の全条件を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_run_metric_derivation::test_change_response_metric_correspondence_distinguishes_metric_pairs`: goldenと、対照の全条件を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_run_metric_derivation::test_model_diagnostic_metric_correspondence_distinguishes_metric_pairs`: goldenと、対照の全条件を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_run_metric_derivation::test_model_pair_metric_correspondence_distinguishes_metric_pairs`: goldenと、対照の全条件を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_run_metric_derivation::test_prediction_diagnostic_metric_correspondence_distinguishes_metric_pairs`: goldenと、対照の全条件を実行したときだけ確かめる
- `tests.test_main_ablation_suite::test_main_ablation_suite_lists_individual_variants`: POSIX bashを利用できない環境
- `tests.test_server_sweep_wrapper::test_server_sweep_wrapper_resolves_runtime_options`: POSIX bashが利用できない環境
- `tests.test_server_sweep_wrapper::test_server_sweep_wrapper_rejects_owned_options`: POSIX bashが利用できない環境
