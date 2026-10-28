# Implementation Plan

依存: task 1 → 2 → 3 → 4 → 5 → 6。実装はGPT-6.1-sol（task 1〜5）。版15のgoldenの生成、全pytest、WSL、独立レビューは主担当（task 6）。

各taskの完了条件に共通:
- `tests/`の全体が通る（壊れたtestを次のtaskへ送らない。そのtaskの変更で成り立たなくなったtestは、そのtaskの中で要求4.5の規則に従って直すか消し、記録に書く）。
  sandboxの制約で実行できないtestは、名前と理由を記録に書き、主担当が通常の環境で確かめる。
- 版1〜14の残すcaseの数値の照合（`tests/refactoring/test_important_behavior_regression_goldens.py`）と、要求4.7の乱数の状態の照合が通る。
- そのtaskで消した名前について、要求5.3の参照の検査（`git grep`）で残りが無い。
- 段の境界の小さな食い違い（そのtaskで消すものを参照する別の箇所が、後のtaskに割り当てられている場合など）は、要求を満たす最小の範囲でそのtaskに含めてよい。
  含めたものと理由を記録に書く。要求・設計と食い違う判断（消してはいけないものを消す、値が変わるなど）が要るときだけ止めて報告する。

- [x] 1. 残す比較と共有の部品を確定する（まだ何も消さない）
  - 候補の学習を新しいモジュールへ移し、呼び出し側と、調査の報告の5の表で挙げた研究用の道具のpatchの先（`tools/research_s3_contribution/`、`tools/research_output_layer_learner/`ほか）を直す。
  - goldenの照合に「外すcase」「除く項目」の仕組みを足す（外すcaseは要求4.6、除く項目はこの段では空）。外すcaseの一覧のtest（要求4.6）。
  - 要求4.7の乱数の状態のfixtureを、このtaskの開始時のHEADのコードで作り、照合のtestを足す。
  - _Requirements: 2.2, 4.1, 4.2, 4.6, 4.7_

- [x] 2. 構成の入口を閉じる
  - 構成`fedsda-with-candidate-validation`を消す（要求1.1）。この構成と、旧実装との照合の検証ありのfixtureに依るtest・helperを、要求4.5の規則で扱う。
  - 設定の型・項目・検証ありの方針の値・制約は、まだ消さない（task 3まで残す候補検証の処理が使うため。2026-10-28、実装担当の停止の報告を受けて主担当が段の境界を直した）。
  - この構成を参照する研究用の道具の候補検証のモード・入口（`run_candidate_validation_waiting_diagnostic.py`と`tools/research_candidate_validation_waiting/`の削除、
    `tools/research_prediction_initial_mass/`・`tools/research_provisional_candidate_expert/`の候補検証のモード・変種の削除と、残るモードの専用の検証）も、この段で行う（要求6.1の一部。2回目の停止の報告を受けて前倒しした）。
  - _Requirements: 1.1, 4.5, 6.1_

- [x] 3. 実行と状態、設定の型・項目・値・制約を消す
  - 候補検証の実行と状態（要求2.1・2.3）と、設定（要求1.2・1.3）を一緒に消す。共有の値（要求1.4）は残す。
    この段で成り立たなくなるtest（候補検証の部品のtest、設定の検査のtest）を、要求4.5の規則で扱う。保存の形の変わる設定の項目は、要求4.2の除く項目の一覧に足す。
  - _Requirements: 1.2, 1.3, 1.4, 2.1, 2.3, 4.2, 4.5_

- [x] 4. 記録・指標・保存の形を消す
  - goldenの除く項目の一覧を確定する（一覧にない差は失敗）。
  - _Requirements: 3.1, 3.2, 3.3, 4.2, 5.1, 5.2_

- [x] 5. 研究用の道具・文書・版15
  - 研究用の道具のうち、task 2〜4で扱っていない残り（task 3・4で消した名前に依る出力の項目など）を直し、調査の報告の5の表の道具ごとに、残るモードの専用の検証（`checks*.py`）が通ることを確かめて記録する。
  - 文書（要求6.2・6.3）。`BEHAVIOR_BASELINE_VERSION = 15`と版15のcaseの一覧（要求4.3）、版15と版14の比較のtest（版15のファイルが無い間は失敗してよい。記録に書く）。
  - 要求5.3の参照の検査を、消した全部の名前で行う。
  - _Requirements: 4.3, 4.4, 5.3, 6.1, 6.2, 6.3_

- [x] 6. 確認（主担当）
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

### task 2（2026-10-28、実装着手前の停止）

- 開始時のHEADは`51f31cb6`、branchは`refactor/architecture`。仕様書一式、task 1の記録、共通引継ぎ手順とsteeringを読んだ。開始時の未追跡`.tmp/`は変更していない。
- 段の境界の確認が必要なため、ユーザーの「仕様のとおりにできない点は止めて報告する」に従ってコード変更前に停止した。task 2は未完了。
- 要求1.2で型ごと削除する`CandidateModelTrainingAndAcceptanceSettings`は、task 3まで残す`post_alarm_candidate_loss_collection.py`・`post_alarm_candidate_loss_evaluation.py`・`post_alarm_candidate_validation_session_start.py`・`post_alarm_candidate_validation_progress.py`・`held_candidate_validation_progress.py`が実行時にimportし、引数の型検査や設定の読取りに使う。共有の`alarm_interval_candidate_training.py`も同じ型を検査する。構成の入口を閉じて到達不能にするだけでは、型の削除によるimportエラーを防げず、消した名前の参照検査も通らない。
- 候補検証の実行・状態を残したまま型を消す場合は、検証専用処理にも設定型を使わない引数・入力検査への変更が必要になる。その一時的な契約は設計・命名に定義されていないため、独断で追加しなかった。推奨は、設定削除に伴う検証専用処理の契約変更までtask 2に含めるか、設定型の完全削除をtask 3へ移すかを主担当が決め、段の境界を確定すること。
- source・test・helper・fixtureの変更・削除、goldenの除く項目の追加、naming.mdの変更、stage・commitは行っていない。pytest・golden・乱数の照合・Pyright・Ruffは未実行。静的な参照検索とGitの状態確認だけを行った。task 1の検証結果を今回の検証として流用していない。

### task 2（2026-10-28、段の境界修正後の再確認）

- 開始時のHEADは`51f31cb6`、branchは`refactor/architecture`。開始時から未コミットのtasks.mdの修正（設定の削除をtask 3へ移したもの）を読み直した。この修正で、前回報告した設定型のimportの問題は解消する。開始時の未追跡`.tmp/`は変更していない。
- 別の段の境界の衝突を確認した。共通完了条件と要求5.3は、task 2で消す構成名を`tools/`・入口のscriptからも参照しないことを求める。一方、task 5と要求6.1は、その構成を選ぶ研究用の道具のモード・入口の削除をtask 5へ置いている。ユーザーは今回task 2だけを実装し、task 3以降へ進まないよう指定している。
- `git grep -n -F 'fedsda-with-candidate-validation' -- tools run_candidate_validation_waiting_diagnostic.py`で、次の実行中の参照を確認した（過去文書や保存済みgoldenではない）:
  - `run_candidate_validation_waiting_diagnostic.py:113`、`tools/research_candidate_validation_waiting/checks.py:171`。
  - `tools/research_prediction_initial_mass/run.py:130`、同`checks.py:132`。
  - `tools/research_provisional_candidate_expert/run.py:51,53–57,65`。変種の表から構成名を読み、同`:2485–2486`で構成builderへ渡す。
- 構成だけを削除すると、これらのモード・検証は存在しない構成を要求し、参照検査にも失敗する。モードを削るとtask 5へ踏み込む。名前の分割や別名で検索を回避する変更、候補検証ありの設定を研究用の道具に直接組み直して一時的に維持する変更は、段の境界を独断で補うため行っていない。推奨は、上記の研究用の道具の構成参照と該当モード・検証の整理をtask 2へ含めるか、task 2の参照検査でtask 5まで残すこれらの参照を明示的に例外とするかを主担当が確定すること。
- ユーザーの「仕様のとおりにできない点は止めて報告する」に従い、実装前に停止した。task 2は未完了。今回変更したのは本記録だけ。source・test・helperの変更・削除はなく、要求4.5によるtestの削除・修正も未実施。goldenの外すcase・除く項目の一覧、乱数fixture、設定・保存の形は変更していない。stage・commitはしていない。
- 実行した確認はGitの状態・HEADと静的な参照検索のみ。pytestの実行件数は0件。tests/全体、版1〜14の数値golden、乱数の状態、Pyright、Ruffは実装前停止のため未実施（sandboxによる実行不能の判定ではない）。消した名前はまだ無く、要求5.3の完了確認も未実施。task 1の結果を今回の検証済みの根拠にしていない。記録の書込み後にUTF-8で再読し、今回追記した部分に文字化けがないことを確認した。

### task 2（2026-10-28、入口の削除と中断後の修正）

- `refactor/architecture`、HEAD `51f31cb6`の作業ツリーを引き継いだ。主担当が修正したtask 2と共通完了条件を読み直し、この段の入口の削除と、それに依るtest・研究用の道具の修正だけを行った。task 3以降は進めていない。stage・commitはしていない。
- `src/federated_learning_experiments/sweep/experiment_configurations.py`から`CANDIDATE_VALIDATION_CONFIGURATION_NAME`、構成の登録、組立ての分岐を削除した。設定の型・項目・検証ありの方針・制約、候補検証の実行・状態は残した。保存の形は変えていない。goldenの外すcaseの一覧、除く項目の一覧（空のまま）、保存済みgolden、乱数fixtureは変更していない。
- `run_candidate_validation_waiting_diagnostic.py`と`tools/research_candidate_validation_waiting/checks.py`を削除した。初期の予測質量の道具から`validation`モードを外し、残る3モードのjob数・対照を直した。仮の専門家の道具から`validation`、`provisional-expert`、`validation-forward30`、`provisional-expert-forward30`、`sequential-test`、`provisional-expert-sequential-test`、`validation-frozen-candidate-backbone`と、専用の方針・定数・到達不能な分岐を削除した。残るモードの出力の形の整理は、この段では行っていない。
- `test_fedsda_with_candidate_validation.py`を削除し、全構成に共通する候補の設定と実行の検査を`test_fedsda_configuration_candidate_settings.py`へ移した。設定を直接組み立てる部品のtest、MDLと検証ありの方針の制約のtestは残した。

要求4.5の扱い:

- (a) 検証の開始、参照の固定、損失の収集・評価、進行、決定、未完了の終端処理、保持と解放、記録について、検証ありの旧実装そのものを照合の対象とするtestと、その専用helperを削除した。削除した関数名は後の表に全件を記した。同じ部品の直接の入力検査・原子性・順序・乱数の検査は残した。中断時に誤って外していた`torch`、`snapshot_adoption_state`、`FrozenInstanceError`などの共有のimport・定義と、直接の部品の検査を戻した。残した検証の部品を照合するfixtureでは、旧実装の`forward_persistent`を明示し、通常runの`immediate`が漏れないようにした。
- (b) 共通の警報応答、吸収、採用、標本処理、run、分布、統合、指標・保存配列の照合は、旧の`immediate`と新の検証なしの設定へ替えた。採用の照合では旧の実際の`_spawn_new_model`の登録・現行の切替を通し、吸収では旧の実際の`_absorb_into_store`を通す。検証の拒否だけにある吸収先の重複caseを外し、保存先と現行先が異なるcaseを残した。検証ありの固定goldenとのrunの照合だけを削除し、実際に実行した旧の検証なしのrunとの数値・配列の照合は残した。
- (c) 検証なしの数値golden、共通の候補の学習・採用・訓練・分布・統合の部品と、設定を直接組み立てて検証する部品のtestを残した。全構成に共通する検査の移設は削除として扱っていない。

検証なしの条件の探索と経路の網羅:

- 小さい全runの既存条件と追加条件を実際に実行した。`(seed=1, clients=5, samples=600, interval=10, minimum_change_gap=30, probability=.05, update_interval=1)`で3モデルの統合とサーバのIDの対応付けを確認し、指標と保存の照合の条件へ追加した。これによりモデル診断の26項目はすべて区別でき、従来区別できなかったF1・precision・recallの組も、区別できない組の期待から外した。指標の対応付けの検査を弱めていない。
- 全ストリームの統合の経路は、SEA4を`seed=0, clients=5, samples=600, interval=10, gap=30, probability=.05, update=1`、MNIST2を`seed=1, clients=3, samples=500, interval=10, gap=30, probability=.05, update=2`へ替えて維持した。採用の経路はその場の採用で検査し、検証の待ち・拒否・保留中のセッションだけを網羅の対象から外した。
- 統合の部品は`class_count=4, block_size=22, seed=11`の11回目で3モデルの統合を通すfixtureへ替え、`class_count=4, block_size=30, seed=7`の条件を追加して2モデルの統合・再統合も残した。途中段での失敗のtestは、以前のラウンドの記録を保存して比較し、分布が走らないことを分類器の同一性で確かめる。
- 分布は`class_count=2, block_size=16, seed=23, rounds=10, stored_evaluation_sample_limit=1`を追加して再標本化を維持し、`class_count=2, block_size=22, seed=7`を追加して未登録の一時モデルを残す経路を維持した。登録済みと一時の現行モデルが混ざるfixtureも実際の検証なしの状態へ合わせた。
- AdaHedgeの`adahedge_weight_model_pool_reset`だけは、既存条件と追加した全runで発生しなかった。その場の採用による現行の割当ての再始動が先行するため、この全runの列の網羅から明示的に除いた。理由をtestの除外の集合にも記し、モデル集合の再始動そのものの部品のtestは残した。検証だけの26指標は旧のrunで0であることを明示的に検査する。
- 条件探索の実測は`.tmp/task2-search-small.log`、`.tmp/task2-search-datasets.log`、`.tmp/task2-search-distribution.log`、`.tmp/task2-clustering-search.log`、`.tmp/task2-more-clustering.log`。task 2から消す構成を参照する研究用の入口・専用検査の削除は、修正済みtasks.mdが前倒しした範囲に含めた。それ以外の設定・保存・出力の整理は前倒ししていない。

検証の環境と制約:

- Pythonは`../../venv/Scripts/python.exe`、`PYTHONPATH="src;.;.tmp"`（手動の照合だけtest helperの場所も追加）、OMP/MKLは各1。全体はファイルの所属を保つ7群とgoldenのcase別10群、同時8プロセスで実行した。`--collect-only`は14,664件。収集した全件を実行対象にし、slowの除外はしていない。
- 事前確認で、一時ディレクトリの作成後の書込みは`PermissionError`、pytestの通常の一時領域の走査は`WinError 5`となった。このため一時領域のfixtureを使うtestと内部の`TemporaryDirectory`へ保存する2関数を、理由付きの一時pluginで除外した。権限・sandboxの変更はしていない。後の表の各関数の全パラメータが対象で、完全なパラメータ付き名・理由はJUnitへ保存した。
- `test_experiment_sweep_resumption.py::test_same_proof_recovers_orphan_without_losing_old_bytes`は単独でも試したが、test本体に入る前の一時領域の準備が`WinError 5`となった（`.tmp/task2-resumption-alone.log`）。主担当が見た`'failed' == 'completed'`の再現・解消は確認できていない。通常の環境での単独実行が残る。

変更したPythonファイル（削除を含む。tasks.mdはこの記録の追記）:

- `run_candidate_validation_waiting_diagnostic.py`
- `src/federated_learning_experiments/sweep/experiment_configurations.py`
- `tests/refactoring/behavior_golden_cases.py`
- `tests/refactoring/test_adopted_candidate_local_adoption.py`
- `tests/refactoring/test_alarm_adaptation_recording.py`
- `tests/refactoring/test_alarm_buffer_response.py`
- `tests/refactoring/test_alarm_change_interval_resolution.py`
- `tests/refactoring/test_alarm_occurrence_handling.py`
- `tests/refactoring/test_alarm_response_completion.py`
- `tests/refactoring/test_assigned_training_sample_absorption.py`
- `tests/refactoring/test_candidate_validation_adaptation_recording.py`
- `tests/refactoring/test_fedsda_configuration_variants.py`
- `tests/refactoring/test_fedsda_cross_evaluation_clients_with_target_samples.py`
- `tests/refactoring/test_fedsda_experiment_run_artifacts.py`
- `tests/refactoring/test_fedsda_final_configuration_run_settings.py`
- `tests/refactoring/test_fedsda_loss_monitor_rebaseline_and_all_evaluating_clients.py`
- `tests/refactoring/test_fedsda_minimum_description_length_alarm_interval_assignment.py`
- `tests/refactoring/test_fedsda_run_client.py`
- `tests/refactoring/test_fedsda_run_metric_derivation.py`
- `tests/refactoring/test_fedsda_stream_protocol_run.py`
- `tests/refactoring/test_fedsda_warm_started_learner_and_cross_validated_initial_statistics.py`
- `tests/refactoring/test_fedsda_with_candidate_validation.py`
- `tests/refactoring/test_global_model_distribution.py`
- `tests/refactoring/test_held_candidate_validation_progress.py`
- `tests/refactoring/test_incomplete_post_alarm_candidate_validation_finalization.py`
- `tests/refactoring/test_loss_baseline_selection.py`
- `tests/refactoring/test_model_clustering_and_consolidation.py`
- `tests/refactoring/test_observed_sample_processing.py`
- `tests/refactoring/test_post_alarm_candidate_loss_collection.py`
- `tests/refactoring/test_post_alarm_candidate_loss_evaluation.py`
- `tests/refactoring/test_post_alarm_candidate_validation_progress.py`
- `tests/refactoring/test_post_alarm_candidate_validation_resolution.py`
- `tests/refactoring/test_post_alarm_candidate_validation_sample_observation.py`
- `tests/refactoring/test_post_alarm_candidate_validation_session_start.py`
- `tests/refactoring/test_post_alarm_reference_model_fixation.py`
- `tools/research_candidate_validation_waiting/checks.py`
- `tools/research_prediction_initial_mass/checks.py`
- `tools/research_prediction_initial_mass/run.py`
- `tools/research_provisional_candidate_expert/run.py`
- `tests/refactoring/test_fedsda_configuration_candidate_settings.py`

削除・置換したtestとhelperの名前（開始時HEADとの差分。別の場所へ移した名前は除く）:

| ファイル | 名前 | 要求4.5と理由 |
|---|---|---|
| `run_candidate_validation_waiting_diagnostic.py` | `_force_adoption_after_wait` | task 2で前倒しした、削除する構成の専用の道具・検査。残るモードの専用検査を別途実行。 |
| `tests/refactoring/test_adopted_candidate_local_adoption.py` | `finalize_adoption_in_legacy_client`<br>`test_local_adoption_matches_actual_legacy_forward_validation_acceptance` | (b) `adopt_trained_candidate_in_legacy_client`と`test_local_adoption_matches_actual_legacy_immediate_creation`へ置換。その場の採用を実際の旧実装と照合する。 |
| `tests/refactoring/test_alarm_buffer_response.py` | `test_active_candidate_alarm_absorbs_all_and_preserves_session` | (a) 候補検証そのものの旧実装との照合と、その専用helper。直接の部品の検査は残す。 |
| `tests/refactoring/test_alarm_change_interval_resolution.py` | `test_alarm_change_interval_resolution_observes_samples_after_started_validation_like_legacy`<br>`test_alarm_change_interval_resolution_starts_candidate_validation_like_legacy` | (a) 候補検証そのものの旧実装との照合と、その専用helper。直接の部品の検査は残す。 |
| `tests/refactoring/test_alarm_response_completion.py` | `test_second_alarm_during_started_validation_completes_like_legacy` | (a) 候補検証そのものの旧実装との照合と、その専用helper。直接の部品の検査は残す。 |
| `tests/refactoring/test_assigned_training_sample_absorption.py` | `test_absorption_matches_actual_legacy_non_adoption_branches_of_forward_validation` | (b) `test_absorption_matches_actual_legacy_store_update`へ置換。検証の拒否の重複caseを外し、保存先・現行先の2条件を残す。 |
| `tests/refactoring/test_candidate_validation_adaptation_recording.py` | `assert_recorded_event_matches_legacy`<br>`test_completed_validation_record_matches_real_legacy_event`<br>`test_incomplete_validation_record_matches_real_legacy_event` | (a) 候補検証そのものの旧実装との照合と、その専用helper。直接の部品の検査は残す。 |
| `tests/refactoring/test_fedsda_experiment_run_artifacts.py` | `test_golden_condition_saved_artifacts_match_real_legacy_and_golden` | (b) 検証ありの固定goldenとの照合を外したため、`test_golden_condition_saved_artifacts_match_real_legacy`へ改名。実際の旧の検証なしの指標・配列の照合は残す。 |
| `tests/refactoring/test_fedsda_run_metric_derivation.py` | `test_golden_condition_decision_kinds_are_reported`<br>`test_golden_condition_metrics_and_traces_match_environment_golden` | (a) 検証ありの固定goldenと検証の決定種別の検査。(b) 同じ条件の実旧検証なしの数値・配列の照合を残す。 |
| `tests/refactoring/test_fedsda_with_candidate_validation.py` | `test_configuration_name_is_registered_last`<br>`test_with_candidate_validation_differs_from_the_default_only_in_the_explicit_values` | (a) 廃止した構成だけの登録・差分の検査。全構成共通の検査は新しいファイルへ移設。 |
| `tests/refactoring/test_held_candidate_validation_progress.py` | `test_alarm_response_updates_holder_like_legacy_session_attribute`<br>`test_completed_held_validation_is_recorded_then_released_like_legacy`<br>`test_incomplete_held_validation_is_recorded_then_released_like_legacy`<br>`test_unfinished_held_validation_keeps_session_and_adds_no_record` | (a) 候補検証そのものの旧実装との照合と、その専用helper。直接の部品の検査は残す。 |
| `tests/refactoring/test_incomplete_post_alarm_candidate_validation_finalization.py` | `test_incomplete_validation_decision_record_matches_legacy`<br>`test_incomplete_validation_finalization_connects_actual_training`<br>`test_incomplete_validation_finalization_matches_legacy`<br>`test_incomplete_validation_finalization_uses_current_training_model` | (a) 候補検証そのものの旧実装との照合と、その専用helper。直接の部品の検査は残す。 |
| `tests/refactoring/test_loss_baseline_selection.py` | `test_loss_baseline_matches_legacy_history` | (a) 候補検証そのものの旧実装との照合と、その専用helper。直接の部品の検査は残す。 |
| `tests/refactoring/test_post_alarm_candidate_loss_collection.py` | `make_legacy_loss_session`<br>`make_legacy_observation_client`<br>`test_candidate_loss_collection_connects_to_acceptance_evaluation`<br>`test_candidate_loss_collection_invalid_observation_is_atomic_for_legacy_partial_update`<br>`test_candidate_loss_collection_matches_legacy_session`<br>`test_candidate_loss_collection_readiness_matches_legacy_client` | (a) 候補検証そのものの旧実装との照合と、その専用helper。直接の部品の検査は残す。 |
| `tests/refactoring/test_post_alarm_candidate_loss_evaluation.py` | `test_post_alarm_candidate_evaluation_matches_legacy_finalization`<br>`test_post_alarm_candidate_public_loss_connection_matches_legacy`<br>`test_post_alarm_candidate_reference_selection_matches_legacy`<br>`test_post_alarm_candidate_rounding_boundary_preserves_decision_reason` | (a) 候補検証そのものの旧実装との照合と、その専用helper。直接の部品の検査は残す。 |
| `tests/refactoring/test_post_alarm_candidate_validation_progress.py` | `test_validation_decision_record_matches_legacy`<br>`test_validation_progress_connects_actual_training_and_resolution`<br>`test_validation_progress_matches_legacy`<br>`test_validation_progress_records_are_immutable`<br>`test_validation_progress_uses_current_reference_availability`<br>`test_validation_progress_waits_without_resolution` | (a) 候補検証そのものの旧実装との照合と、その専用helper。直接の部品の検査は残す。 |
| `tests/refactoring/test_post_alarm_candidate_validation_resolution.py` | `finalize_forward_validation_in_legacy_client`<br>`test_resolution_matches_actual_legacy_forward_validation_branches`<br>`test_resolved_validation_continues_actual_joint_training` | (a) 候補検証そのものの旧実装との照合と、その専用helper。直接の部品の検査は残す。 |
| `tests/refactoring/test_post_alarm_candidate_validation_sample_observation.py` | `observe_and_resolve_in_both_implementations`<br>`test_observation_then_evaluation_and_resolution_match_actual_legacy_observation`<br>`test_observed_and_resolved_validation_continues_actual_joint_training`<br>`test_observed_loss_sequences_match_actual_legacy_observation` | (a) 候補検証そのものの旧実装との照合と、その専用helper。直接の部品の検査は残す。 |
| `tests/refactoring/test_post_alarm_candidate_validation_session_start.py` | `begin_candidate_validation_session_in_legacy_client`<br>`test_session_start_connects_to_validation_observation`<br>`test_session_start_matches_legacy`<br>`test_session_start_preserves_optional_metadata_and_empty_pending_samples`<br>`test_session_start_retains_historical_mean_losses` | (a) 候補検証そのものの旧実装との照合と、その専用helper。直接の部品の検査は残す。 |
| `tests/refactoring/test_post_alarm_reference_model_fixation.py` | `begin_forward_validation_in_legacy_client`<br>`start_validation_in_both_implementations`<br>`test_fixation_then_observation_and_resolution_match_actual_legacy_session`<br>`test_fixed_references_match_actual_legacy_reference_snapshots_and_random_state`<br>`test_historical_mean_losses_match_actual_legacy_session_start`<br>`test_references_stay_fixed_while_held_models_continue_actual_joint_training` | (a) 候補検証そのものの旧実装との照合と、その専用helper。直接の部品の検査は残す。 |
| `tools/research_candidate_validation_waiting/checks.py` | `test_forced_evaluation_retains_original_and_restores`<br>`test_incomplete_session_uses_original_finalization`<br>`test_nine_samples_wait_and_tenth_forces_real_adoption`<br>`test_rng_fingerprint_includes_gaussian_cache` | task 2で前倒しした、削除する構成の専用の道具・検査。残るモードの専用検査を別途実行。 |
| `tools/research_provisional_candidate_expert/run.py` | `_ProvisionalExpert`<br>`_provisional_expert_policy`<br>`_sequential_test_policy`<br>`_true_class_loss` | task 2で前倒しした、削除する構成の専用の道具・検査。残るモードの専用検査を別途実行。 |

移設した名前、または別moduleにも同名があるため固有名のgrepから区別した名前:

| 削除元 | 名前 | 扱い |
|---|---|---|
| `run_candidate_validation_waiting_diagnostic.py` | `_code_identity`<br>`_execute_job`<br>`_main`<br>`_rng_fingerprint`<br>`_run_with_policy` | このmoduleの定義は削除。同名の別moduleの定義は残る。削除したmodule名・入口への参照は0件。 |
| `tests/refactoring/test_fedsda_with_candidate_validation.py` | `candidate_values`<br>`fedsda_configuration_names`<br>`test_fedsda_configurations_have_the_expected_candidate_values_and_run` | 全構成の共通検査として`test_fedsda_configuration_candidate_settings.py`へ移設。 |
| `tools/research_candidate_validation_waiting/checks.py` | `diagnostic`<br>`test_control_matches_every_array_metrics_and_rng`<br>`test_failed_worker_retains_reservation_and_no_completion`<br>`test_invalid_plan_does_not_create_outputs`<br>`test_plan_is_read_only_and_rejects_existing_reservation`<br>`test_worker_reserves_output_before_training_and_writes_provenance` | このmoduleの定義は削除。同名の別moduleの定義は残る。削除したmodule名・入口への参照は0件。 |

検証結果:

- `tests/`の全収集対象: **14,223 passed / 441 skipped / 0 failed**（修正した群を再実行した結果を採用）。最初の17群の実行は14,222 passed / 441 skipped / 1 failed。失敗は統合の途中段で止まるtestの、以前のラウンドから保持していた分類器数の固定値だった。同一性で保存前後を比較するように直し、単独1件と、そのファイルを含む1,504件の群を再実行して通過した。元の失敗のXMLも上書きせず残した。
- 全体の証拠は`.tmp/task2-final-jobs.json`、`.tmp/task2-final-{files-0..6,golden-0..9}.{log,xml}`、修正後の`.tmp/task2-final-files-2-recheck.{log,xml}`。全体の重複しない収集対象は14,664件で、再実行分は通過件数へ二重加算していない。
- 版1〜14の残すcaseの数値golden: **170 passed**。除外caseの保存済み方針との一致: **14 passed**。借用したPython・NumPy・torch CPUのrun終端の全乱数状態: **14 passed**。これらは上の全体件数に含む。
- sandboxで除外した指標・保存の照合は、既存の作業領域`.tmp`を出力先にして元のtest関数を追加で直接実行した。3 datasetの実旧runの数値・配列、16条件の小さいrun、4条件のラウンドごとのモデル数、8個の対応付け・経路の網羅の検査の**計52回が通過**。最初は51 passed / 1 failedで、追加した3モデルの統合の条件により全診断指標が区別できるようになったため、区別できない組の旧期待を空に直した。保存した同じ実測値で該当の検査を再実行して通過した。証拠は`.tmp/task2-manual-coverage.json`、`.tmp/task2-manual-observations.json`、`.tmp/task2-manual-correspondence-recheck.log`。pytestの一時領域やCLIの保存先の検査を代替したとは扱わない。
- 残る研究用のモードの専用検査4ファイル: **68 passed / 23 skipped / 0 failed**、計91件。`tools/research_prediction_initial_mass/checks.py`、`tools/research_provisional_candidate_expert/checks_class_conditioned_prediction.py`、同`checks_loss_monitor_rebaseline.py`、同`checks_mixing_past_posteriors.py`を明示して実行した。証拠は`.tmp/task2-tools-final.log`・同`.xml`。スキップは下記のsandboxによる保存先の検査。
- 変更したsourceのPyrightは**0 errors / 0 warnings**。変更した37 PythonファイルのPyrightは既存の型診断を含み**5,374 errors / 0 warnings**、同じ設定の開始時HEADの36ファイルは7,074 errors。診断のファイル・規則・本文・重複数を比較した**追加診断は0件**。既存のtest・研究用の道具まで型検査が全件通ったとは扱わない。証拠は`.tmp/task2-pyright-source.log`、`.tmp/task2-pyright-final-{base-results,results}.json`、`.tmp/task2-pyright-new-final.json`。
- 変更したsource・testのRuff check/formatは通過。通常の設定で除外される研究用の道具も、同じ規則を明示して開始時と比較し、**既存10件 / 追加0件**。`git diff --check`は通過。
- 最後に、保存の照合から固定goldenとの比較を外した実態へ名前とdocstringを合わせ、`test_golden_condition_saved_artifacts_match_real_legacy_and_golden`を`test_golden_condition_saved_artifacts_match_real_legacy`へ改名した。このファイルを再実行し**6 passed / 41 skipped**。全体件数に二重加算していない。専用の研究用検査の再確認も**68 passed / 23 skipped**で同じ結果だった。
- 消した構成名・定数・入口・module名、削除・置換したtest/helperの固有名の**72名前**を、`git grep -n -w -F -f .tmp/task2-removed-symbols.txt -- src tests tools ":(top,glob)*.py" ":!tests/refactoring/golden/*.json"`で検査し、該当0件（exit 1）。移設した定義や別moduleに同じ一般名があるhelperは削除名に含めず、削除したmoduleへの参照も検査した。削除名と移設名の内訳は`.tmp/task2-symbol-inventory.json`。
- `docs/reference/options.md`、保留中の3文書、旧実装、golden JSON、乱数fixtureに差分がないことを確認した。独立CLIの実装レビューは、共通引継ぎ手順に従い全task後のtask 6で主担当が行う。このtaskだけのレビューは追加していない。

<details>
<summary>sandboxで実行できなかったtestの関数名と件数</summary>

理由はいずれも一時領域の準備・作成後の書込みのアクセス拒否。各関数の全パラメータが対象。完全な名前と理由は上記JUnitにある。手動で数値・配列の照合だけを追加実行した関数も、通常のpytestの保存先を検査できた扱いにはしない。

| test module | test関数（件数） |
|---|---|
| `tests.refactoring.test_behavior_fix_comparison_scripts` | `test_analysis_reports_absolute_values_and_differences_per_dataset_only`（1）<br>`test_analysis_tables_are_unchanged_from_the_previous_script`（1）<br>`test_analysis_warns_right_after_the_title_when_baseline_versions_are_mixed`（1）<br>`test_analysis_without_input_tables_returns_one_and_writes_nothing`（1）<br>`test_analysis_writes_a_file_with_provenance_in_the_input_directory`（1）<br>`test_comparison_record_holds_code_revision_and_behavior_baseline`（1）<br>`test_comparison_record_is_not_overwritten_within_the_same_second`（1）<br>`test_comparison_requires_purpose_and_places_results_under_its_study`（1）<br>`test_comparison_stops_with_a_warning_when_existing_runs_have_another_baseline`（1）<br>`test_short_alarm_axis_rejects_a_registration_level`（1） |
| `tests.refactoring.test_cross_evaluation_download_deduplication` | `test_independent_run_saves_corrected_communication_volume`（1） |
| `tests.refactoring.test_environment_proposed_regression` | `test_damaged_golden_files_are_reported_with_their_file_name`（4）<br>`test_golden_collection_may_be_empty_and_detects_misplaced_files`（1）<br>`test_golden_is_selected_only_by_exactly_matching_environment`（1）<br>`test_goldens_with_the_same_environment_are_rejected`（1）<br>`test_missing_golden_reason_tells_the_environment_and_how_to_add_one`（1）<br>`test_writing_a_golden_follows_the_environment_and_refuses_unreviewed_overwrites`（1）<br>`test_writing_a_golden_is_always_refused_in_the_fixed_golden_environment`（1） |
| `tests.refactoring.test_experiment_sweep` | `test_code_revision_counts_only_tracked_files_as_uncommitted_changes`（1）<br>`test_code_revision_describes_commit_or_reason`（1）<br>`test_command_creates_no_readme_when_existing_runs_have_another_baseline`（1）<br>`test_command_keeps_an_existing_readme_and_records_each_purpose`（1）<br>`test_command_places_results_under_the_study_with_a_readme`（1）<br>`test_command_rejects_invalid_plans_before_running`（5）<br>`test_command_rejects_names_that_are_not_plain_directory_names`（6）<br>`test_command_rejects_study_with_output_directory`（1）<br>`test_command_requires_a_single_line_purpose_to_run`（3）<br>`test_command_returns_nonzero_when_a_run_fails`（1）<br>`test_command_runs_sweep_and_writes_table_and_record`（1）<br>`test_command_stops_before_running_when_mnist_files_are_missing`（1）<br>`test_command_stops_with_a_warning_when_existing_runs_have_another_baseline`（1）<br>`test_command_writes_record_even_when_a_run_summary_is_unreadable`（1）<br>`test_completed_runs_are_skipped_and_left_unchanged`（1）<br>`test_conflicting_existing_files_are_reported_and_not_overwritten`（4）<br>`test_existing_runs_with_another_baseline_stop_the_sweep_before_anything_runs`（6）<br>`test_existing_runs_with_damaged_provenance_are_conflicting_not_stopping`（3）<br>`test_existing_runs_with_the_same_baseline_are_skipped_even_with_another_commit`（1）<br>`test_experiment_readme_is_written_once_and_validates_arguments`（1）<br>`test_failed_run_is_reported_and_other_runs_continue`（1）<br>`test_parallel_sweep_cancels_waiting_runs_when_interrupted`（1）<br>`test_parallel_sweep_saves_the_same_content_as_sequential_sweep`（1）<br>`test_print_plan_lists_runs_and_creates_nothing`（1）<br>`test_repository_script_runs_parallel_sweep_in_separate_processes`（1）<br>`test_repository_script_runs_without_installation_or_environment_variables`（1）<br>`test_run_summary_table_holds_one_row_per_saved_run`（1）<br>`test_run_summary_table_holds_run_provenance_columns`（1）<br>`test_saved_summary_holds_the_given_run_provenance`（1）<br>`test_sequential_sweep_saves_the_same_content_as_direct_run`（1）<br>`test_sweep_execution_rejects_invalid_arguments`（1）<br>`test_sweep_record_holds_plan_command_revision_and_outcomes`（1） |
| `tests.refactoring.test_experiment_sweep_resumption` | `test_active_output_refuses_command_before_readme_or_records_change`（1）<br>`test_cli_recovery_counts_and_preserves_saved_npz`（1）<br>`test_completed_artifacts_are_not_changed_and_training_interrupt_resumes`（1）<br>`test_different_or_broken_plan_is_not_offered`（1）<br>`test_different_requested_run_set_does_not_resume_existing`（1）<br>`test_invalid_recovery_proof_types_and_unknown_current_code_are_refused`（4）<br>`test_output_lock_is_cross_process_and_released`（1）<br>`test_partial_old_readme_can_be_selected_with_new_metadata_and_workers`（1）<br>`test_plan_identity_ignores_order_but_not_settings`（1）<br>`test_process_termination_releases_output_lock`（1）<br>`test_same_proof_recovers_orphan_without_losing_old_bytes`（1）<br>`test_select_new_and_explicit_new_do_not_overwrite_same_second_output`（1）<br>`test_selection_can_finish_without_any_file_changes`（4）<br>`test_symlink_does_not_allow_recovery_outside_output`（1）<br>`test_unproven_interruption_is_shown_and_selection_does_not_modify_it`（1）<br>`test_unproven_orphan_is_preserved`（7）<br>`test_untracked_execution_source_change_disables_recovery`（3） |
| `tests.refactoring.test_fedavg_experiment_run` | `test_existing_half_artifact_is_not_overwritten`（2）<br>`test_loader_rejects_other_method_and_unsafe_reference`（3）<br>`test_run_artifact_roundtrip`（2）<br>`test_stale_summary_and_inconsistent_predictions_are_rejected_before_write`（1） |
| `tests.refactoring.test_fedavg_experiment_sweep` | `test_print_plan_is_read_only_and_old_name_is_rejected`（1）<br>`test_real_sweep_saves_resumes_and_refuses_conflict`（2） |
| `tests.refactoring.test_fedavg_local_updates_per_aggregation` | `test_sweep_command_saves_runs_under_the_configuration_directory`（1） |
| `tests.refactoring.test_feddrift_experiment_run_artifacts` | `test_existing_output_is_preserved`（2）<br>`test_loader_rejects_unreadable_table_or_summary`（5）<br>`test_loader_rejects_wrong_version_method_or_path`（10）<br>`test_save_failure_cleans_its_files`（5）<br>`test_save_load_roundtrip_and_nonfinite_summary`（2）<br>`test_save_rejects_invalid_name_without_writing`（10）<br>`test_save_rejects_invalid_result_or_directory`（3） |
| `tests.refactoring.test_feddrift_experiment_sweep` | `test_actual_sweep_matches_single_run_and_resumes_without_execution`（2）<br>`test_cli_rejects_fedsda_only_explicit_axis_and_handles_missing_mnist`（1）<br>`test_cli_rejects_invalid_axis_before_outputs`（5）<br>`test_cli_writes_threshold_columns_records_and_resumes`（1）<br>`test_conflicting_feddrift_artifacts_are_not_overwritten`（5）<br>`test_feddrift_cli_reports_execution_failure`（1）<br>`test_real_cli_help_and_spawn_execution`（1） |
| `tests.refactoring.test_fedsda_configuration_variants` | `test_configuration_variant_matches_real_legacy_whole_run`（80） |
| `tests.refactoring.test_fedsda_experiment_run_artifacts` | `test_failed_save_leaves_no_partial_files`（1）<br>`test_golden_condition_change_response_columns_match_legacy_saved_arrays`（3）<br>`test_golden_condition_experiment_run_matches_separately_executed_run`（3）<br>`test_golden_condition_model_diagnostic_columns_match_legacy_saved_arrays`（3）<br>`test_golden_condition_record_tables_rebuild_legacy_saved_arrays`（3）<br>`test_golden_condition_round_model_counts_match_legacy_saved_arrays`（3）<br>`test_golden_condition_saved_artifacts_match_real_legacy`（3）<br>`test_loading_accepts_previous_format_without_run_provenance`（1）<br>`test_loading_rejects_unsupported_or_damaged_summaries`（1）<br>`test_round_model_counts_match_real_legacy_whole_run`（4）<br>`test_saved_artifacts_round_trip`（1）<br>`test_saving_refuses_to_overwrite_and_writes_nothing`（1）<br>`test_saving_rejects_invalid_arguments`（1）<br>`test_saving_rejects_invalid_destinations`（10） |
| `tests.refactoring.test_fedsda_fixed_share_time_scale_axis` | `test_command_rejects_invalid_time_scales_before_running`（1）<br>`test_command_saves_runs_with_time_scale_column_and_record_format_version`（1） |
| `tests.refactoring.test_fedsda_pending_assignment_fifo_capacity_axis` | `test_command_rejects_invalid_capacities_before_running`（1）<br>`test_command_saves_runs_with_capacity_column_and_record_format_version`（1） |
| `tests.refactoring.test_fedsda_run_metric_derivation` | `test_derivation_does_not_change_participant_state_and_is_repeatable`（3）<br>`test_derivation_rejects_invalid_arguments`（3）<br>`test_golden_condition_comparison_computation_metrics_match_real_legacy_counts`（3）<br>`test_golden_condition_metrics_match_real_legacy_experiment_result`（3）<br>`test_golden_condition_trace_arrays_match_real_legacy_saved_arrays`（3）<br>`test_small_run_metrics_match_real_legacy_metric_computation`（16） |
| `tests.refactoring.test_fmnist_and_rotated_mnist_runs` | `test_command_accepts_new_datasets_without_creating_output`（1）<br>`test_command_rejects_missing_image_files_before_output`（2） |
| `tests.refactoring.test_fmnist_and_rotated_mnist_sample_generation` | `test_cache_does_not_mix_mnist_and_fashion`（1）<br>`test_fashion_directory_resolution`（1）<br>`test_fashion_labels_and_sampling_follow_mnist`（2）<br>`test_fashion_missing_file_is_rejected_without_download`（2）<br>`test_fashion_reuses_idx_validation`（1）<br>`test_generators_belong_only_to_their_dataset`（1）<br>`test_rotation_matches_official_numpy_operations`（2）<br>`test_shared_directory_for_mnist_and_fashion_is_rejected`（3） |
| `tests.refactoring.test_fresh_process_smoke` | `test_fresh_process_smoke_runs_without_legacy_or_test_modules`（1） |
| `tests.refactoring.test_important_behavior_regression_goldens` | `test_corrupt_golden_is_rejected_before_environment_skip`（9）<br>`test_duplicate_environment_is_rejected`（1）<br>`test_generation_preserves_existing_output_before_starting`（1）<br>`test_generation_rejects_unrecorded_source_before_workers`（3）<br>`test_generation_requires_two_matching_process_results`（3）<br>`test_unicode_dtype_is_a_valid_digest`（1）<br>`test_unknown_environment_has_no_match_but_reads_structure`（1） |
| `tests.refactoring.test_mnist_sample_generation` | `test_damaged_files_are_rejected_as_invalid_format`（8）<br>`test_data_directory_follows_environment_variable_then_repository_default`（1）<br>`test_differing_image_and_label_counts_are_rejected`（1）<br>`test_invalid_file_formats_are_rejected`（6）<br>`test_loaded_training_data_is_reused_per_resolved_directory`（1）<br>`test_loading_requires_keyword_path_argument`（1）<br>`test_missing_files_are_rejected_without_fetching`（3）<br>`test_small_files_are_read_as_written`（1） |
| `tests.refactoring.test_seed_argument_parsing` | `test_invalid_main_arguments_create_no_output`（3）<br>`test_seed_syntax_does_not_change_plan_identifiers_or_legacy_resume`（1） |
| `tests.refactoring.test_synthetic_sample_generation` | `test_expected_generator_types_cover_all_defined_datasets`（1） |
| `tests.test_build_feddrift_baseline` | `test_build_baseline_groups_canonical_results_by_dataset`（1）<br>`test_build_baseline_keeps_same_sweep_for_multiple_schedules`（1）<br>`test_build_baseline_prefers_explicit_sweep_parameter`（1）<br>`test_build_baseline_refuses_to_overwrite_existing_output`（1）<br>`test_extend_baseline_adds_new_seed_atomically_and_keeps_backup`（1）<br>`test_extend_baseline_adds_new_sweep_value`（1）<br>`test_extend_baseline_rejects_conflicting_duplicate`（1） |
| `tests.test_clustering_functional_diagnostics` | `test_clustering_functional_diagnostics_separates_merged_and_retained`（1） |
| `tests.test_experiment_artifacts` | `test_rebuild_artifacts_recovers_pareto_metrics_from_old_raw_and_log`（1）<br>`test_rebuild_artifacts_uses_embedded_exact_metrics`（1） |
| `tests.test_experiment_manifests` | `test_backfill_existing_csv`（1）<br>`test_backfill_infers_missing_schedule_and_total_data_from_raw`（1）<br>`test_manifest_session_records_lifecycle_and_outputs`（1）<br>`test_overlap_distinguishes_exact_and_different_provenance`（1）<br>`test_preview_overlaps_does_not_create_manifest`（1）<br>`test_provenance_changes_when_regression_golden_changes`（1）<br>`test_recursive_backfill_separates_variants_and_ignores_analysis_csv`（1）<br>`test_study_manifest_is_generated_from_utf8_definition`（1） |
| `tests.test_paper_datasets` | `test_idx_reader_parses_mnist_files`（1） |
| `tests.test_pareto_sweep` | `test_load_csv_accepts_canonical_feddrift_baseline_names`（1）<br>`test_load_csv_rejects_legacy_parameter_columns`（1）<br>`test_plot_pareto_can_use_overall_accuracy`（1）<br>`test_plot_pareto_draws_baseline_standard_deviation_band`（1）<br>`test_replot_filter_selects_interval_sweeps_and_plot_accepts_compute_x`（1） |
| `tests.test_proposed_regression` | `test_proposed_regression`（1） |
| `tests.test_recovery_analysis` | `test_plot_recovery_skips_file_when_no_series`（1） |
| `tests.test_shared_backbone` | `test_drift_recovery_activation_is_saved_without_changing_history_shape`（1）<br>`test_shared_backbone_experiment_reports_component_metrics`（1） |
| `tests.test_telemetry` | `test_experiment_records_compute_and_model_telemetry`（1） |

研究用の専用検査:

| test module | test関数（件数） |
|---|---|
| `tools.research_prediction_initial_mass.checks` | `test_failed_worker_retains_reservation_and_no_completion`（1）<br>`test_invalid_plan_does_not_create_outputs`（3）<br>`test_plan_is_read_only_and_rejects_existing_reservation`（1）<br>`test_worker_reserves_output_before_training_and_writes_provenance`（1） |
| `tools.research_provisional_candidate_expert.checks_loss_monitor_rebaseline` | `test_artifact_comparison_finds_expected_runs_by_name_under_saved_results`（1）<br>`test_artifact_comparison_ignores_code_commit_but_not_baseline_version`（1）<br>`test_artifact_comparison_includes_all_arrays_and_non_time_summary_items`（1）<br>`test_combined_variant_product_sweep_records_limit_and_rebaselines`（1）<br>`test_execute_run_with_variant_enables_policy_and_writes_diagnostics_only_when_completed`（2）<br>`test_metrics_reject_conflicting_diagnostics_for_the_same_run`（1）<br>`test_observe_only_product_sweep_matches_all_saved_items`（3）<br>`test_parallel_workers_apply_variant_and_write_diagnostics`（1）<br>`test_run_calls_product_sweep_per_variant_with_patch_and_restores`（1）<br>`test_run_rejects_existing_output_without_variant_record`（1）<br>`test_run_rejects_other_variant_record_before_calling_product`（1）<br>`test_run_sets_evaluating_client_limit_only_for_combined_variant`（2）<br>`test_variant_record_is_written_before_product_runs`（1） |

</details>

環境条件・先行する照合の実行が必要なためのスキップ（sandboxによる事前除外とは別）:

- `tests.refactoring.test_fedsda_experiment_run_artifacts::test_cross_evaluation_table_conditions_cover_required_cases`: goldenの3ケースと、小さい条件の全部を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_experiment_run_artifacts::test_prediction_diagnostic_column_conditions_cover_required_cases`: goldenの3ケースと、小さい条件の全部を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_experiment_run_artifacts::test_model_diagnostic_column_conditions_cover_required_cases`: goldenの3ケースと、小さい条件の全部を実行したときだけ確かめる
- `tests.refactoring.test_feddrift_linux_core_metrics::test_linux_full_run_core_metrics_and_parameters_match_fixed_legacy[sine2]`: 固定Linux CPU基準環境専用。実旧全体run対照は各環境で別に実行する。
- `tests.refactoring.test_feddrift_linux_core_metrics::test_linux_full_run_core_metrics_and_parameters_match_fixed_legacy[sea4]`: 固定Linux CPU基準環境専用。実旧全体run対照は各環境で別に実行する。
- `tests.refactoring.test_feddrift_linux_core_metrics::test_linux_full_run_core_metrics_and_parameters_match_fixed_legacy[mnist2]`: 固定Linux CPU基準環境専用。実旧全体run対照は各環境で別に実行する。
- `tests.refactoring.test_feddrift_linux_core_metrics::test_linux_full_run_core_metrics_and_parameters_match_fixed_legacy[sine2_drift]`: 固定Linux CPU基準環境専用。実旧全体run対照は各環境で別に実行する。
- `tests.refactoring.test_fedsda_run_metric_derivation::test_small_run_conditions_cover_required_cases`: 対照の全条件を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_run_metric_derivation::test_change_response_metric_correspondence_distinguishes_metric_pairs`: goldenと、対照の全条件を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_run_metric_derivation::test_model_diagnostic_metric_correspondence_distinguishes_metric_pairs`: goldenと、対照の全条件を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_run_metric_derivation::test_model_pair_metric_correspondence_distinguishes_metric_pairs`: goldenと、対照の全条件を実行したときだけ確かめる
- `tests.refactoring.test_fedsda_run_metric_derivation::test_prediction_diagnostic_metric_correspondence_distinguishes_metric_pairs`: goldenと、対照の全条件を実行したときだけ確かめる
- `tests.test_main_ablation_suite::test_main_ablation_suite_lists_individual_variants`: POSIX bashを利用できない環境
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
- `tests.refactoring.test_environment_proposed_regression::test_legacy_final_configuration_matches_environment_golden`: 固定のgoldenの環境では、既存の回帰test（tests/test_proposed_regression.py）が照合する
- `tests.test_server_sweep_wrapper::test_server_sweep_wrapper_resolves_runtime_options`: POSIX bashが利用できない環境
- `tests.test_server_sweep_wrapper::test_server_sweep_wrapper_rejects_owned_options`: POSIX bashが利用できない環境

未実施・残る確認と懸念:

- 通常の環境での、上記sandbox除外testを含む`tests/`全体と研究用の保存・並列workerの専用検査、既存のproposed回帰とfresh-processの一時保存先の検査、resumptionの単独再試行が残る。WSL/Linux固定環境の数値照合もこのWindowsの実行では行っていない。数値・乱数の照合と実行可能な全体の検査は通過したが、通常環境の全pytestまでPASSとは報告しない（`MANUAL_VERIFY_REQUIRED`）。
- 通常runから候補検証の入口を外した状態で、部品を直接構築するtestは残る。設定・状態・保存の削除はtask 3以降の担当範囲である。モデル集合だけによるAdaHedge再始動の全runの経路は、上記の条件探索でも得られていない。部品の検査を残し、全runの網羅の除外を明示した。
- この記録はUTF-8で書込み後に再読し、置換文字・疑問符の連続への文字化けがなく、日本語の本文が保存されていることを確認した。過去の停止・承認の記録は書き換えていない。

### task 3（2026-10-28、実装担当の途中から主担当が仕上げた）

- 実装担当（gpt-6.1-sol）が、HEAD `769adeab`から変更の大部分を入れたところで、Codexの利用の上限が近いため主担当が止めた（ユーザーの判断。実装担当の記録は無い）。
  止めた時点の作業ツリーの全pytestは172 failed / 12948 passed / 24 skipped。主担当が残りを直した。
- 消したもの（実装担当）: 候補検証の実行と状態の13モジュール（`runtime/`の`candidate_validation_adaptation_recording.py`、`candidate_validation_session_holder.py`、`held_candidate_validation_progress.py`、
  `incomplete_post_alarm_candidate_validation_finalization.py`、`post_alarm_candidate_validation_progress.py`・`_resolution.py`・`_sample_observation.py`・`_session_start.py`、`post_alarm_reference_model_fixation.py`、
  `methods/fedsda/candidate_model_selection/`の`candidate_model_training_and_acceptance_settings.py`、`candidate_validation_decision_record_store.py`、`incomplete_post_alarm_candidate_validation_decision_record.py`、
  `post_alarm_candidate_loss_collection.py`、`post_alarm_candidate_loss_evaluation.py`、`post_alarm_candidate_validation_decision_record.py`）、設定の型と項目・検証ありの方針の値・制約、実行の契約の締めの段、
  関係するtest（`test_candidate_validation_adaptation_recording.py`、`test_candidate_validation_decision_record_store.py`、`test_held_candidate_validation_progress.py`、`test_incomplete_post_alarm_candidate_validation_finalization.py`ほか）。
  共有のtestの補助は`tests/refactoring/alarm_interval_test_support.py`へ移した。
- goldenの除く項目（`REMOVED_CANDIDATE_VALIDATION_ITEM_PATHS`）: `run_settings/run_participant_settings/run_client_settings/candidate_model_training_and_acceptance_settings`、
  `run_settings/run_participant_settings/run_client_settings/scalar_settings/minimum_candidate_mean_loss_improvement`、`record_tables/execution_event.{stage_name,client_id,sample_index,round_index}`。
  実行の出来事の表は、消した締めの段の行が無くなる。保存済みのgoldenは列ごとのhashだけを持つので、その行だけを除けず、表の4列ごと除いた。この表のほかの差も照合で見えなくなる（限界）。
  予測・パラメータ・runの指標・記録のほかの表と、要求4.7の乱数の状態は、引き続き照合する。段の回数は`fresh_process_smoke.py`で確かめる（締めの段が無いこと）。
- 主担当が直したもの:
  - 消したclientの判定の記録への参照の残り（`test_fedsda_run_metric_derivation.py`の2か所、`test_fedsda_stream_protocol_run.py`の補助と1か所）。旧の形の`provisional_*`の配列は、旧の検証なしのrunと同じく空にした。
  - 版どうしを比べるtest（`test_baseline_6_differs…`〜`test_baseline_13_runs…`、`test_next_baseline_differs…`）が、比べる版のどちらかで外したcaseを流さないよう、`_cases_runnable_in_both`で絞った。
  - `assert_required_paths`の候補検証の経路の網羅（候補の採用・棄却の件数、基準の引下げ0件）を、外したcaseでは確かめないようにした（外したcaseの条件で補助のtestが全体runを流すため）。
    壊れたgoldenの拒否のtestの`missing_path`は、残すcase（`fedsda-sine2-adoption-without-validation`の統合の数）を壊す形に替えた。
  - 要求4.5(b): 合うモデルが無いcaseは、新ではその場で候補を採用し、旧の`immediate`と一時のIDの割当てが違う（新−100、旧−103）。比べられる旧の経路が無いので、
    `test_alarm_training_interval_preparation.py`の`no_model_fits`と、`test_too_short_alarm_earlier_sample_release.py`のその場の採用の応答の、旧との照合を外した（理由をコメントに書いた。この経路は版7〜14のgoldenが照合する）。
  - 消した制約のtest: `test_minimum_description_length_is_rejected_with_candidate_validation`を消し、共有部の固定との組合せのtestを「共有部の2つの方針がどちらも組み立てられる」に替えた。
  - 期待値: FedSDAのrunの識別子のhash（`fc28fa415515`→`f9bff53a0cdc`、2つのtest）、`fedsda-sine2-events`の条件の検証なしのrunでの登録されない採用の数（2→4）。いずれも候補検証を消したことによる想定の変化。
  - 引数の消し残し（`test_alarm_wait_for_minimum_change_interval.py`の`minimum_validation_loss_decrease=0.0`）、消した設定の値を含む誤りの値の一覧（`test_run_settings_validation.py`）、
    使われない変数・import、空白だけの行、空になった依存の契約の区切り。
  - task 4の範囲の先取りは、`test_fedsda_experiment_run_artifacts.py`の対応表の比較から、候補検証の結果の値を除いたところだけ（製品の対応表から消すのはtask 4）。
- 確認（通常の環境）: 全pytest 13036 passed / 8 skipped / 1 failed（失敗は上の`missing_path`で、直した後にそのtestを含む構造の検査11件、依存の境界2257件、設定の検査1201件を流し直して全部通過）。
  版1〜14の残すcaseの数値の照合と、乱数の状態の照合は、全pytestで通った。pyright 0 errors、ruff check・formatは通過。
  消した名前（設定の型・項目・方針の値、検証の実行・状態の型と関数）は、保存済みのgoldenのJSONと、goldenの照合の一覧を除き、`src/`・`tests/`・`tools/`から参照されない（`git grep`）。

### task 4（2026-10-28、主担当）

- ユーザーの指示（Codexの利用の上限が近い）で、task 4から主担当が行った。基点はHEAD `a1e88ed4`。
- 製品のコードから消したもの:
  - 適応の結果の値7つ（`AdaptationOutcome`と`TRAINING_MODEL_SWITCH_ADAPTATION_OUTCOMES`）: `alarm_during_candidate_validation`、`alarm_interval_candidate_validation_started`、
    `post_alarm_validation_candidate_adopted`・`_held_model_reused`・`_current_model_maintained`・`_candidate_rejected`・`_incomplete_candidate_rejected`。粗い区分への対応表（`ADAPTATION_ACTION_BY_ADAPTATION_OUTCOME`）からも消した。
  - 適応の指標3つ（`start_candidate_validation_count`、`reject_candidate_model_count`、`defer_during_candidate_validation_count`）、`CandidateValidationDecisionMetrics`（23項目）と`candidate_validation_decision_metrics`、
    判定の値の補助（`_DecisionValues`、`_mean_of_finite_values`）、runの指標の`candidate_validation_decision_count`。
  - 記録の表`candidate_validation_decision.*`（16列）。
- goldenの除く項目に足したもの: `record_tables/candidate_validation_decision.*`の16列、`run_metrics/candidate_validation_decision_count`、`run_metrics/change_response_metrics/candidate_validation_decision_metrics`、
  `run_metrics/change_response_metrics/adaptation_action_metrics/{start_candidate_validation_count,reject_candidate_model_count,defer_during_candidate_validation_count}`、
  網羅の`coverage/{candidate_accepted_count,candidate_rejected_count,decision_reasons}`（判定の記録の表から求めていた項目。網羅の計算からも消した）。
  候補検証の経路の網羅の確かめ（外した4 caseだけが対象で、もう確かめる場面が無い）と、その場の採用の経路での「判定が無い」の確かめは消した。
- testで直したもの:
  - 旧の指標・配列への対応（`test_fedsda_run_metric_derivation.py`、`test_fedsda_experiment_run_artifacts.py`）: 旧の候補の判定の指標と配列は、旧の検証なしのrunでは0か空なので、
    新の指標から写すのをやめ、0を写す（`_NO_CANDIDATE_VALIDATION_DECISION_METRICS`）か、旧の判定の配列が空であることを確かめる（`LEGACY_CANDIDATE_VALIDATION_DECISION_ARRAY_NAMES`）形にした。
  - 旧のactionへの対応（`test_fedsda_experiment_run_artifacts.py`、`test_alarm_response_completion.py`）から、候補検証の区分と結果の値を消し、task 3で入れた一時の除外を元に戻した。
  - 記録の保存の順のtest（`test_alarm_adaptation_recording.py`）で、消した結果の値の代わりに、学習帰属を切り替えず、適合の数にも数えない`alarm_change_interval_too_short`を使った。
  - 消した網羅の項目の数え方のtest（`test_incomplete_candidate_is_not_counted_as_rejection`）を消した。`fresh_process_smoke.py`の判定の数の確かめを消した。
- 研究用の道具: `tools/research_shadow_learner_diagnostic/adoption.py`は、自分の遅れた採用を、消した`post_alarm_validation_candidate_adopted`で記録していた。
  同じく学習帰属を切り替える採用の`alarm_interval_candidate_adopted_as_new_model`へ替えた（粗い区分は同じ「候補の採用」で、指標の数え方は変わらない。この道具の採用は`detector_name="research_shadow_score"`で区別する）。
  過去の結果の記録は変わらない。
- 確認（通常の環境）: 全pytest 13008 passed / 8 skipped / 1 failed（失敗は上の消したtest）。消した後に、goldenの構造の検査79件を流し直して全部通過。
  版1〜14の残すcaseの数値の照合と、乱数の状態の照合は、全pytestで通った。pyright 0 errors、ruff check・formatは通過。
- 残り（task 5）: `with_candidate_validation`の引数、`RECORDING_ORACLE_CASES`の候補検証の結果の値のcaseなど、testの補助に残る候補検証の名前の整理と、要求5.3の参照の検査。


### task 5（2026-10-28、実装担当）

- 基点は`refactor/architecture`、HEAD `15748ff6`。task 5だけを実装した。task 6、版15のgolden生成・配置、stage・commit・pushは行っていない。開始時からの未追跡`.tmp/`は保持し、今回の一時検証資料もそこへ置いた。
- 共通引継ぎ手順とkiro-implのmanual・`--review off`で進め、完了の判定はkiro-verify-completionで証拠と照合した。task単位の独立レビューは増やしていない。全task後の独立CLI実装レビューはtask 6で主担当が行う。

変更したものと判断:

- `with_candidate_validation`の引数と到達しない分岐を、`build_completed_recording_oracle`、`build_sample_processing_oracle`、`make_run_client_settings`、`build_run_client_oracle`、`build_processed_run_client`から除いた。
  `RECORDING_ORACLE_CASES`の候補採用は`alarm_interval_candidate_adopted_as_new_model`で直接表し、検証中の警報caseを除いた（元から通常の照合のparametrizeには入っていなかった）。
- `LEGACY_REASON_BY_DECISION_REASON`と、空の判定記録を走査していた配列の組立てを除いた。旧の`provisional_*`配列は空の配列を直接作り、旧の候補判定の指標へは0を直接対応させる。
  task 4の`_NO_CANDIDATE_VALIDATION_DECISION_METRICS`も除き、消した新の指標のfieldを持つ代用品を残さない。旧の名前は旧実装の数値・配列との照合の側にだけ残した。
- test用の`MINIMUM_CANDIDATE_MEAN_LOSS_IMPROVEMENT`は、用途どおり`MINIMUM_VALIDATION_LOSS_DECREASE`へ改めた（0.0001のまま）。
  候補検証の結果の値を挙げた到達不能の分岐・網羅の除外、fresh smokeの未使用定数、空の依存契約の区切りを整理した。
  残る依存契約testは実態どおり`test_adaptation_record_store_dependency_contract`へ改めた。
- 要求4.5(a): `test_buffer_response_rejects_owned_inputs_before_updates[session]`を削除した。削除済みの`active_validation_session`を注入し、未知の引数としてのTypeErrorだけで通っていたため。
  `test_run_settings_validation.py`の無効値の一覧から、消した方針名`two_segment_candidate_validation`も除いた（14パラメータ分）。他の無効値・型・更新前の拒否の検査は残した。
  全体runを流した後の参照検査で見つけた上記sessionのcaseは、この段の最小の修正として含め、同ファイル全55件を再検証した。
- 研究用の`run.py`から、廃止済みの検証モードの出力`session_starts`、`sequential_test_decisions`、`sequential_test_mean_sample_count`を除いた。
  専用検査の実際の製品対照runで、これらの項目が無いことも確かめる。既存の保存結果と解析側は変更していない。

版15（要求4.3・4.4）:

- `BEHAVIOR_BASELINE_VERSION`とhelperの`BASELINE_VERSION`を15にし、対応版へ15を足した。case一覧は版14の順を保ち、検証ありの4 caseを除いた18 case。
  `test_baseline_15_keeps_exactly_the_non_validation_cases_of_baseline_14`で件数・一覧・各caseの組立てた設定の版14との一致を確認した。
- `test_baseline_15_saved_values_match_baseline_14_except_removed_items[case_name]`（18件）は、Windows・Linuxの両goldenについて、要求4.2の完全な鍵の道の一覧で除いた後のsnapshot全体を版14と比較する。
  版15が無い間は明示的に失敗する。`test_committed_goldens_cover_both_reference_environments[15]`（1件）も同じ理由で失敗する。許容する失敗はこの19件だけ。
- `test_new_implementation_matches_important_behavior_golden[15-<case_name>]`（18件）は現在、対応する数値goldenが無い理由でskipする。既存の版1〜14の数値照合と保存済みJSONの構造検査は引き続き実行する。
  golden検出力のtestは版14を明示して読むようにし、構造検査用の壊すcaseは残る採用caseへ替えた。これらは版15のファイル不足に巻き込まれない。
- REDの実測: 版の定数を上げる前の新しい一覧testは`14 == 15`で1 failed。研究出力の削除前の専用testは、上記の出力項目が残ることで1 failed。変更後の対象検査は下記のとおり。
- 生成手順を読んでhelperのCLI `--help`と組立てを確認した。`--baseline-version`省略時は15、2独立processの一致・HEAD一致・src変更なし・既存出力の拒否は既存どおり。
  主担当はtask 5のsrc変更をcommitした後、文書の次のコマンドをそのまま使える（未使用の出力先を選ぶ）。今回は生成自体を実行していない。

```powershell
$goldenSourceCommit=git rev-parse HEAD
../../venv/Scripts/python.exe tests/refactoring/behavior_golden_cases.py --output ../../venv/refactoring-tests/behavior-candidate-windows.json --source-commit $goldenSourceCommit
```

文書（要求6.2・6.3）:

- `docs/overview/proposed-method.md`: 現在の構成・設定・標本処理順から候補検証を除き、版15と検証の正本を案内した。候補学習の早期終了用の検証は残る。
- `docs/experiments/experiment-sweep.md`: 現在の基準版と構成一覧を直し、版14までの結果の集計、データ・間隔・seedによる対応、FedSDAの末尾hashの変更、過去掃引へ再開・seed追加せず新しい保存先を使うことを書いた。
- `docs/experiments/refactoring-baseline.md`: 現在の版の説明と版15の行を足し、生成待ち・比較の対象と除く項目・生成コマンドの前提を記載した。版1〜14の行は書き換えていない。
- `docs/components/new-model-creation.md`・`model-reuse.md`: 今の実装の説明だけを直し、前向き検証などの旧実装の説明は旧側の資料として残した。
- 改善候補一覧のIMPROVE-023、IMPROVE-023の新しい日付つき結果、resumeの現在地を更新した。過去の日付つきの節と承認記録、既存の他specは変更していない。
- 集計の道具`accuracy_periods._run_key`と`research_feddrift_delta_comparison._run_key`を実行し、SINE2・SEA4・MNIST2・MNIST4の同じデータ・間隔100・seed7で、末尾hashだけが違う2つのrunが対応することを確認した（2関数×4条件）。
  消した項目を集計側が参照しないことも検索で確認した。保存済みの本規模結果全体の再集計は実施していない。

要求5.3の参照検査:

- `git grep -n -w -F -e <名前> ... -- src/ tests/ tools/ '*.py' ':!federated_drift_experiment/' ':!*.json'`を実行した。ルート入口scriptも対象。
  下の全名前を指定した（元のHEADから消したmoduleと、現行srcに残らない型・関数・method、設定・結果の値・記録16列・指標23項目ほかを列挙）。共有の`validation_sample_count`は早期終了でも使う別の名前なので削除対象に含めず、部分一致だけによる別名の一致を避けるため`-w`を用いた。
- 一致は`behavior_golden_cases.py`の旧方針・除く項目の定義25行と、`test_candidate_validation_removal_regression.py`の保存済み方針から外すcaseを求める検査3行だけ。
  前者は要求4.2・後者は要求4.6に必要なgolden除外定義の照合で、現在の設定や実行を参照していない。該当定義・照合を除いた検索は**0件、exit 1**（一致無し）。
  全結果は`.tmp/task5-reference-grep.txt`、最終の残りは`.tmp/task5-reference-filtered.txt`（空）、一覧は`.tmp/task5-reference-names.json`。

<details>
<summary>検査に使った名前の全一覧</summary>

```text
CANDIDATE_VALIDATION_CONFIGURATION_NAME
CandidateModelTrainingAndAcceptanceSettings
CandidateValidationDecisionMetrics
CandidateValidationDecisionRecordStore
CandidateValidationSessionHolder
FixedPostAlarmReferenceModels
HeldCandidateValidationAdvance
HeldIncompleteCandidateValidationFinalization
IncompletePostAlarmCandidateValidationDecisionRecord
IncompletePostAlarmCandidateValidationFinalization
LEGACY_REASON_BY_DECISION_REASON
PostAlarmCandidateLossCollection
PostAlarmCandidateLossCollectionState
PostAlarmCandidateLossEvaluation
PostAlarmCandidateValidationCompletion
PostAlarmCandidateValidationDecisionRecord
PostAlarmCandidateValidationProgress
PostAlarmCandidateValidationResolution
PostAlarmCandidateValidationSession
_CANDIDATE_POST_ALARM_VALIDATION_SAMPLE_COUNT
_DecisionValues
_MINIMUM_CANDIDATE_MEAN_LOSS_IMPROVEMENT
_evaluate_single_sample_bounded_loss
_held_validation_session
_mean_bounded_loss
_mean_of_finite_values
_select_validated_reusable_reference
_select_validation_assignment_sample_concept_ids
_validate_bounded_observed_loss
_validate_concept_ids
_validate_holder_and_record_store
_validate_incomplete_validation_finalization_inputs
_validate_loss_sequence
_validate_nonnegative_finite_number
_validate_post_alarm_candidate_validation_start_inputs
_validate_reference_loss_inputs
_validate_resolution_inputs
_validate_sample_index_not_before_proposal
_validate_session_training_tensor_pair
_validate_validation_progress_inputs
_validate_validation_progress_threshold
_validation_assignment_sample_concept_ids
accepted_decision_precision
accepted_decision_rate
accepted_mean_full_interval_loss_margin
accepted_mean_second_segment_loss_margin
active_validation_session
advance_held_candidate_validation
advance_post_alarm_candidate_validation
alarm_during_candidate_validation
alarm_interval_candidate_validation_started
alternative_reference_within_historical_loss_tolerance
alternative_reference_within_historical_loss_tolerance_decision_count
append_candidate_validation_decision_record
apply_alarm_response_to_validation_session_holder
apply_post_alarm_candidate_validation_resolution
both_segment_margins_failed
both_segment_margins_failed_decision_count
both_segment_margins_passed
candidate_model_acceptance_policy
candidate_model_training_and_acceptance_settings
candidate_post_alarm_validation_sample_count
candidate_validation_adaptation_recording
candidate_validation_decision.candidate_accepted
candidate_validation_decision.candidate_full_interval_mean_loss
candidate_validation_decision.candidate_second_segment_mean_loss
candidate_validation_decision.candidate_training_interval_sample_count
candidate_validation_decision.client_id
candidate_validation_decision.comparison_reference_model_id
candidate_validation_decision.decision_reason
candidate_validation_decision.detector_name
candidate_validation_decision.is_incomplete_at_run_end
candidate_validation_decision.matched_to_concept_change
candidate_validation_decision.proposal_sample_index
candidate_validation_decision.reference_full_interval_mean_loss
candidate_validation_decision.reference_historical_mean_loss
candidate_validation_decision.reference_second_segment_mean_loss
candidate_validation_decision.resolution_sample_index
candidate_validation_decision.validation_sample_count
candidate_validation_decision_count
candidate_validation_decision_metrics
candidate_validation_decision_record_store
candidate_validation_session_holder
count_rejected_decisions_with_reason
current_model_first_reuse_then_candidate_adoption_without_validation
current_model_first_reuse_then_two_segment_candidate_validation
current_reference_within_historical_loss_tolerance
current_reference_within_historical_loss_tolerance_decision_count
defer_during_candidate_validation_count
evaluate_candidate_using_post_alarm_losses
fedsda-with-candidate-validation
finalization_delay_sample_count
finalize_held_incomplete_candidate_validation
finalize_incomplete_candidate_validation
finalize_incomplete_post_alarm_candidate_validation
first_segment_margin_failed
first_segment_margin_failed_decision_count
fix_reference_models_at_alarm
full_validation_mean_loss_advantage
held_candidate_validation_progress
held_validation_session
hold_validation_assignment_sample_concept_ids
hold_validation_session
incomplete_at_run_end_decision_count
incomplete_candidate_validation_finalization
incomplete_post_alarm_candidate_validation_decision_record
incomplete_post_alarm_candidate_validation_finalization
matched_accepted_decision_count
matched_decision_count
matched_mean_full_interval_loss_margin
matched_mean_second_segment_loss_margin
matched_rejected_decision_count
mean_candidate_training_interval_sample_count
mean_reference_loss_excess_over_historical_mean
mean_resolution_delay_sample_count
mean_validation_sample_count
minimum_candidate_mean_loss_improvement
observe_losses_after_label_observation
observe_post_alarm_candidate_validation_sample
post_alarm_candidate_loss_collection
post_alarm_candidate_loss_evaluation
post_alarm_candidate_validation_decision_record
post_alarm_candidate_validation_progress
post_alarm_candidate_validation_resolution
post_alarm_candidate_validation_sample_observation
post_alarm_candidate_validation_session_start
post_alarm_reference_model_fixation
post_alarm_validation_candidate_adopted
post_alarm_validation_candidate_rejected
post_alarm_validation_current_model_maintained
post_alarm_validation_held_model_reused
post_alarm_validation_incomplete_candidate_rejected
ready_for_acceptance_evaluation
record_completed_candidate_validation
record_incomplete_candidate_validation_finalization
reference_mean_loss_difference_from_history
reject_candidate_model_count
rejected_mean_full_interval_loss_margin
rejected_mean_second_segment_loss_margin
release_validation_assignment_sample_concept_ids
release_validation_session
research_candidate_validation_waiting
run_candidate_validation_waiting_diagnostic
second_segment_margin_failed
second_segment_margin_failed_decision_count
second_segment_mean_loss_advantage
select_available_reference_within_historical_loss_tolerance
select_post_alarm_reference_historical_mean_loss
snapshot_candidate_validation_decision_records
start_candidate_validation_count
start_post_alarm_candidate_validation_session
started_validation_session
unmatched_mean_full_interval_loss_margin
unmatched_mean_second_segment_loss_margin
validation_assignment_sample_concept_ids
validation_completion_delay_sample_count
validation_session_holder
with_candidate_validation
```

</details>

<details>
<summary>変更したファイルの全一覧（29ファイル）</summary>

- `.kiro/specs/remove-post-alarm-candidate-validation/tasks.md`
- `.kiro/steering/resume.md`
- `docs/components/model-reuse.md`
- `docs/components/new-model-creation.md`
- `docs/experiments/experiment-sweep.md`
- `docs/experiments/refactoring-baseline.md`
- `docs/overview/proposed-method.md`
- `docs/research/improvement-candidates/README.md`
- `docs/research/improvement-candidates/improve-023-remove-settings-left-from-candidate-validation.md`
- `src/federated_learning_experiments/core/run_provenance.py`
- `tests/refactoring/behavior_golden_cases.py`
- `tests/refactoring/fresh_process_smoke.py`
- `tests/refactoring/test_alarm_adaptation_recording.py`
- `tests/refactoring/test_alarm_buffer_response.py`
- `tests/refactoring/test_alarm_change_interval_resolution.py`
- `tests/refactoring/test_alarm_occurrence_handling.py`
- `tests/refactoring/test_candidate_validation_removal_regression.py`
- `tests/refactoring/test_fedsda_configuration_variants.py`
- `tests/refactoring/test_fedsda_minimum_description_length_alarm_interval_assignment.py`
- `tests/refactoring/test_fedsda_run_client.py`
- `tests/refactoring/test_fedsda_run_metric_derivation.py`
- `tests/refactoring/test_fedsda_stream_protocol_run.py`
- `tests/refactoring/test_fedsda_warm_started_learner_and_cross_validated_initial_statistics.py`
- `tests/refactoring/test_important_behavior_regression_goldens.py`
- `tests/refactoring/test_observed_sample_processing.py`
- `tests/refactoring/test_run_settings_validation.py`
- `tests/refactoring/test_single_run_dependency_boundaries.py`
- `tools/research_provisional_candidate_expert/checks_loss_monitor_rebaseline.py`
- `tools/research_provisional_candidate_expert/run.py`

</details>

検証の実行条件と結果:

- Pythonは`../../venv/Scripts/python.exe`、`PYTHONPATH="src;."`、`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`、既存のMNIST固定入力。srcの乱数源とfixtureは変えていない。
- 事前の`tempfile.mkdtemp()`で作った場所のファイル書込みが`PermissionError`になることを確認した。権限・sandboxを変更せず、以前のtaskと同じ一時pluginで一時領域fixtureと内部TemporaryDirectoryの2関数を理由つきskipにした。
  pluginの読込みだけrunnerの`sys.path`へ`.tmp`を足した（PYTHONPATHは指定どおり）。全体は、ファイルの所属を保つ7群とgoldenのcase別6群、同時8 process。slowは除外せず全収集対象を実行した。
- 初回は全13040件、12562 passed / 459 skipped / 19 failed。後で上記の死んだsession入力case1件を消し、変更したfile単位で上書き照合した最終の全対象は**13039件、12561 passed / 459 skipped / 19 failed、exit 1**。
  失敗は上記の版15待ちだけ。開始時相当13016件＋新規19件＋対応版追加19件−無効値の14件−sessionの1件＝13039件。
- JUnit: 初回`.tmp/task5-final-files-{0..6}.xml`と`.tmp/task5-final-golden-{0..5}.xml`。
  最終のfile単位の再検証は`.tmp/task5-late-recheck.xml`（指標と削除の検査、32 passed / 36 skipped / 18 expected failed）と`.tmp/task5-buffer-recheck.xml`（55 passed）。
  初回の該当file分を再検証の全件で置き換えた集約は`.tmp/task5-final-merged.xml`、集計は`.tmp/task5-results.json`。重複実行を通過件数へ二重加算していない。
- 版1〜14の残すcaseの数値goldenは**170 passed**。要求4.6の版別除外一覧は14 passed。要求4.7のPython Random・NumPy RandomState・torch CPUのrun終端の全状態は**14 passed**。
  `tests/test_regression.py::test_regression`もpassed。保存済みのgolden JSONと乱数fixture、旧実装、`docs/reference/options.md`に今回の差分がないこと、固定旧`748c3aa`から旧package・2golden・2回帰testに差分がないことを`git diff --exit-code`で確認した。
- skip459件は、sandbox415件（fixture413＋内部保存2）、版15の数値golden未生成18件、Linux専用・POSIX bash不在・先行testの未実行で網羅前提未成立など26件。
  下の表へ実行不能の関数名と理由を記す。完全なパラメータ付き名はJUnit。

研究用の道具ごとの専用検証（調査の報告の5の表）:

| 道具 | 検証と結果 |
|---|---|
| 候補検証の待機 | 入口と専用checksはtask 2で削除済み。参照0件。残るモードなし |
| `research_prediction_initial_mass/checks.py` | 12 passed / 6 skipped / 0 failed |
| `research_provisional_candidate_expert/checks_{loss_monitor_rebaseline,class_conditioned_prediction,mixing_past_posteriors}.py` | 56 passed / 17 skipped / 0 failed |
| `research_component_ablation/checks.py` | 0 passed / 26 skipped / 0 failed |
| task 4で採用の記録の値を替えた`research_shadow_learner_diagnostic/checks*.py`（追加確認） | 26 passed / 0 skipped / 0 failed |
| S3寄与（checks*.pyが無いため既存の専用`tests/refactoring/test_research_s3_contribution.py`） | 8 passed / 0 skipped / 0 failed |
| 出力層学習（checks*.pyが無いため既存の専用`tests/refactoring/test_research_output_layer_learner.py`） | 9 passed / 0 skipped / 0 failed |

- 道具のXMLとlogは`.tmp/task5-tools-<道具名>.{xml,log}`。保存・並列workerを検証するskipは未検証で、通過とは扱わない。
  checks.pyの同名module衝突で最初の一括収集が2 errorsになったため、修正前の収集結果は採用せず、道具ごとに別processで再実行した。
- Pyright: 変更20ファイルを指定して実行し、2393 errors（開始時HEADの同じ20ファイルは2426 errors）。0 errorsとは報告しない。
  同じ型検査設定で基点のsourceを別の一時配置から検査し、file・rule・診断対象のsource行で比較すると新しい診断箇所は0。
  文面が変わった4診断は、無効な型を意図して渡すtestのliteral集合1件と、研究CLIの同じround/intの式3件の推論unionの差。変更した製品のsourceファイル自体の診断は0。
  設定と全診断は`.tmp/task5-pyright-{current,base}-comparison.json`、比較は`.tmp/task5-pyright-location-summary.json`。sandboxでvenvの自動探索が働かないため、固定site-packagesとtest用のimport先をextraPathsへ明示した。
- Ruff: プロジェクト設定の対象18ファイルはcheck・formatともexit 0。toolsの2ファイルは既存設定で除外されるので、追加でisolatedのE4/E7/E9/F検査も実施した。
  追加検査は開始時・変更後とも、`run.py:26`の入口bootstrap後のimportによる既存E402の1件だけ（新規0）。このtaskで既存のbootstrapの書き方は変えていない。
- `git diff --check`、変更Python20ファイルのASTの読込み、変更MarkdownのUTF-8再読と置換文字・疑問符の連続の検査は通過した。
  `spec_checks.py names`は変更sourceを基点と照合し未登録なし。`identity`全体は未コミット・版15待ちの19失敗・proposed回帰のsandbox skipのためNG。
  identityの固定旧の対象には研究用`tools/`全体も含まれるため、その欄もNG（開始時から研究道具の差分がある。固定した旧package・golden・回帰testそのものは上記の直接照合で差分0）。
- 対象の作業ツリーのsource hash（tracked Pythonと旧2golden、584パス、LF）は`bd7879e7f5000f3e5f82db34a50df0982ffd207029d74dbf320f801f82a86190`。HEADのhashと未コミット状態を混同していない。

<details>
<summary>通常環境で確認する関数名と理由（各関数の全パラメータが対象）</summary>

| test module | 関数と件数 | 理由 |
|---|---|---|
| `tests.refactoring.test_behavior_fix_comparison_scripts` | `test_analysis_reports_absolute_values_and_differences_per_dataset_only`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_behavior_fix_comparison_scripts` | `test_analysis_tables_are_unchanged_from_the_previous_script`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_behavior_fix_comparison_scripts` | `test_analysis_warns_right_after_the_title_when_baseline_versions_are_mixed`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_behavior_fix_comparison_scripts` | `test_analysis_without_input_tables_returns_one_and_writes_nothing`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_behavior_fix_comparison_scripts` | `test_analysis_writes_a_file_with_provenance_in_the_input_directory`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_behavior_fix_comparison_scripts` | `test_comparison_record_holds_code_revision_and_behavior_baseline`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_behavior_fix_comparison_scripts` | `test_comparison_record_is_not_overwritten_within_the_same_second`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_behavior_fix_comparison_scripts` | `test_comparison_requires_purpose_and_places_results_under_its_study`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_behavior_fix_comparison_scripts` | `test_comparison_stops_with_a_warning_when_existing_runs_have_another_baseline`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_behavior_fix_comparison_scripts` | `test_short_alarm_axis_rejects_a_registration_level`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_cross_evaluation_download_deduplication` | `test_independent_run_saves_corrected_communication_volume`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_environment_proposed_regression` | `test_damaged_golden_files_are_reported_with_their_file_name`（4） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_environment_proposed_regression` | `test_golden_collection_may_be_empty_and_detects_misplaced_files`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_environment_proposed_regression` | `test_golden_is_selected_only_by_exactly_matching_environment`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_environment_proposed_regression` | `test_goldens_with_the_same_environment_are_rejected`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_environment_proposed_regression` | `test_legacy_final_configuration_matches_environment_golden`（1） | 固定のgoldenの環境では、既存の回帰test（tests/test_proposed_regression.py）が照合する |
| `tests.refactoring.test_environment_proposed_regression` | `test_missing_golden_reason_tells_the_environment_and_how_to_add_one`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_environment_proposed_regression` | `test_writing_a_golden_follows_the_environment_and_refuses_unreviewed_overwrites`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_environment_proposed_regression` | `test_writing_a_golden_is_always_refused_in_the_fixed_golden_environment`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_code_revision_counts_only_tracked_files_as_uncommitted_changes`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_code_revision_describes_commit_or_reason`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_command_creates_no_readme_when_existing_runs_have_another_baseline`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_command_keeps_an_existing_readme_and_records_each_purpose`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_command_places_results_under_the_study_with_a_readme`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_command_rejects_invalid_plans_before_running`（5） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_command_rejects_names_that_are_not_plain_directory_names`（6） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_command_rejects_study_with_output_directory`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_command_requires_a_single_line_purpose_to_run`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_command_returns_nonzero_when_a_run_fails`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_command_runs_sweep_and_writes_table_and_record`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_command_stops_before_running_when_mnist_files_are_missing`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_command_stops_with_a_warning_when_existing_runs_have_another_baseline`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_command_writes_record_even_when_a_run_summary_is_unreadable`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_completed_runs_are_skipped_and_left_unchanged`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_conflicting_existing_files_are_reported_and_not_overwritten`（4） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_existing_runs_with_another_baseline_stop_the_sweep_before_anything_runs`（6） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_existing_runs_with_damaged_provenance_are_conflicting_not_stopping`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_existing_runs_with_the_same_baseline_are_skipped_even_with_another_commit`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_experiment_readme_is_written_once_and_validates_arguments`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_failed_run_is_reported_and_other_runs_continue`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_parallel_sweep_cancels_waiting_runs_when_interrupted`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_parallel_sweep_saves_the_same_content_as_sequential_sweep`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_print_plan_lists_runs_and_creates_nothing`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_repository_script_runs_parallel_sweep_in_separate_processes`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_repository_script_runs_without_installation_or_environment_variables`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_run_summary_table_holds_one_row_per_saved_run`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_run_summary_table_holds_run_provenance_columns`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_saved_summary_holds_the_given_run_provenance`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_sequential_sweep_saves_the_same_content_as_direct_run`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_sweep_execution_rejects_invalid_arguments`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep` | `test_sweep_record_holds_plan_command_revision_and_outcomes`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep_resumption` | `test_active_output_refuses_command_before_readme_or_records_change`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep_resumption` | `test_cli_recovery_counts_and_preserves_saved_npz`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep_resumption` | `test_completed_artifacts_are_not_changed_and_training_interrupt_resumes`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep_resumption` | `test_different_or_broken_plan_is_not_offered`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep_resumption` | `test_different_requested_run_set_does_not_resume_existing`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep_resumption` | `test_invalid_recovery_proof_types_and_unknown_current_code_are_refused`（4） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep_resumption` | `test_output_lock_is_cross_process_and_released`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep_resumption` | `test_partial_old_readme_can_be_selected_with_new_metadata_and_workers`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep_resumption` | `test_plan_identity_ignores_order_but_not_settings`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep_resumption` | `test_process_termination_releases_output_lock`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep_resumption` | `test_same_proof_recovers_orphan_without_losing_old_bytes`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep_resumption` | `test_select_new_and_explicit_new_do_not_overwrite_same_second_output`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep_resumption` | `test_selection_can_finish_without_any_file_changes`（4） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep_resumption` | `test_symlink_does_not_allow_recovery_outside_output`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep_resumption` | `test_unproven_interruption_is_shown_and_selection_does_not_modify_it`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep_resumption` | `test_unproven_orphan_is_preserved`（7） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_experiment_sweep_resumption` | `test_untracked_execution_source_change_disables_recovery`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedavg_experiment_run` | `test_existing_half_artifact_is_not_overwritten`（2） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedavg_experiment_run` | `test_loader_rejects_other_method_and_unsafe_reference`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedavg_experiment_run` | `test_run_artifact_roundtrip`（2） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedavg_experiment_run` | `test_stale_summary_and_inconsistent_predictions_are_rejected_before_write`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedavg_experiment_sweep` | `test_print_plan_is_read_only_and_old_name_is_rejected`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedavg_experiment_sweep` | `test_real_sweep_saves_resumes_and_refuses_conflict`（2） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedavg_local_updates_per_aggregation` | `test_sweep_command_saves_runs_under_the_configuration_directory`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_feddrift_experiment_run_artifacts` | `test_existing_output_is_preserved`（2） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_feddrift_experiment_run_artifacts` | `test_loader_rejects_unreadable_table_or_summary`（5） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_feddrift_experiment_run_artifacts` | `test_loader_rejects_wrong_version_method_or_path`（10） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_feddrift_experiment_run_artifacts` | `test_save_failure_cleans_its_files`（5） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_feddrift_experiment_run_artifacts` | `test_save_load_roundtrip_and_nonfinite_summary`（2） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_feddrift_experiment_run_artifacts` | `test_save_rejects_invalid_name_without_writing`（10） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_feddrift_experiment_run_artifacts` | `test_save_rejects_invalid_result_or_directory`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_feddrift_experiment_sweep` | `test_actual_sweep_matches_single_run_and_resumes_without_execution`（2） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_feddrift_experiment_sweep` | `test_cli_rejects_fedsda_only_explicit_axis_and_handles_missing_mnist`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_feddrift_experiment_sweep` | `test_cli_rejects_invalid_axis_before_outputs`（5） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_feddrift_experiment_sweep` | `test_cli_writes_threshold_columns_records_and_resumes`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_feddrift_experiment_sweep` | `test_conflicting_feddrift_artifacts_are_not_overwritten`（5） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_feddrift_experiment_sweep` | `test_feddrift_cli_reports_execution_failure`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_feddrift_experiment_sweep` | `test_real_cli_help_and_spawn_execution`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_feddrift_linux_core_metrics` | `test_linux_full_run_core_metrics_and_parameters_match_fixed_legacy`（4） | 固定Linux CPU基準環境専用。実旧全体run対照は各環境で別に実行する。 |
| `tests.refactoring.test_fedsda_configuration_variants` | `test_adahedge_prediction_weights_take_their_own_path`（1） | 構成の、小さい条件の全部を実行したときだけ確かめる |
| `tests.refactoring.test_fedsda_configuration_variants` | `test_adwin_detection_takes_its_own_path`（2） | 構成の、小さい条件の全部を実行したときだけ確かめる |
| `tests.refactoring.test_fedsda_configuration_variants` | `test_configuration_variant_matches_real_legacy_whole_run`（80） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_configuration_variants` | `test_independent_models_take_their_own_path`（1） | 構成の、小さい条件の全部を実行したときだけ確かめる |
| `tests.refactoring.test_fedsda_configuration_variants` | `test_loss_distance_clustering_takes_its_own_path`（1） | 構成の、小さい条件の全部を実行したときだけ確かめる |
| `tests.refactoring.test_fedsda_configuration_variants` | `test_overall_loss_esr_detection_takes_its_own_path`（1） | 構成の、小さい条件の全部を実行したときだけ確かめる |
| `tests.refactoring.test_fedsda_configuration_variants` | `test_single_model_prediction_takes_its_own_path`（1） | 構成の、小さい条件の全部を実行したときだけ確かめる |
| `tests.refactoring.test_fedsda_configuration_variants` | `test_without_candidate_validation_takes_its_own_path`（1） | 構成の、小さい条件の全部を実行したときだけ確かめる |
| `tests.refactoring.test_fedsda_configuration_variants` | `test_without_concept_specific_adapter_takes_its_own_path`（1） | 構成の、小さい条件の全部を実行したときだけ確かめる |
| `tests.refactoring.test_fedsda_configuration_variants` | `test_without_prediction_weight_recalibration_takes_its_own_path`（1） | 構成の、小さい条件の全部を実行したときだけ確かめる |
| `tests.refactoring.test_fedsda_experiment_run_artifacts` | `test_cross_evaluation_table_conditions_cover_required_cases`（1） | goldenの3ケースと、小さい条件の全部を実行したときだけ確かめる |
| `tests.refactoring.test_fedsda_experiment_run_artifacts` | `test_failed_save_leaves_no_partial_files`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_experiment_run_artifacts` | `test_golden_condition_change_response_columns_match_legacy_saved_arrays`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_experiment_run_artifacts` | `test_golden_condition_experiment_run_matches_separately_executed_run`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_experiment_run_artifacts` | `test_golden_condition_model_diagnostic_columns_match_legacy_saved_arrays`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_experiment_run_artifacts` | `test_golden_condition_record_tables_rebuild_legacy_saved_arrays`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_experiment_run_artifacts` | `test_golden_condition_round_model_counts_match_legacy_saved_arrays`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_experiment_run_artifacts` | `test_golden_condition_saved_artifacts_match_real_legacy`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_experiment_run_artifacts` | `test_loading_accepts_previous_format_without_run_provenance`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_experiment_run_artifacts` | `test_loading_rejects_unsupported_or_damaged_summaries`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_experiment_run_artifacts` | `test_model_diagnostic_column_conditions_cover_required_cases`（1） | goldenの3ケースと、小さい条件の全部を実行したときだけ確かめる |
| `tests.refactoring.test_fedsda_experiment_run_artifacts` | `test_prediction_diagnostic_column_conditions_cover_required_cases`（1） | goldenの3ケースと、小さい条件の全部を実行したときだけ確かめる |
| `tests.refactoring.test_fedsda_experiment_run_artifacts` | `test_round_model_counts_match_real_legacy_whole_run`（4） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_experiment_run_artifacts` | `test_saved_artifacts_round_trip`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_experiment_run_artifacts` | `test_saving_refuses_to_overwrite_and_writes_nothing`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_experiment_run_artifacts` | `test_saving_rejects_invalid_arguments`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_experiment_run_artifacts` | `test_saving_rejects_invalid_destinations`（10） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_fixed_share_time_scale_axis` | `test_command_rejects_invalid_time_scales_before_running`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_fixed_share_time_scale_axis` | `test_command_saves_runs_with_time_scale_column_and_record_format_version`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_pending_assignment_fifo_capacity_axis` | `test_command_rejects_invalid_capacities_before_running`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_pending_assignment_fifo_capacity_axis` | `test_command_saves_runs_with_capacity_column_and_record_format_version`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_run_metric_derivation` | `test_change_response_metric_correspondence_distinguishes_metric_pairs`（1） | goldenと、対照の全条件を実行したときだけ確かめる |
| `tests.refactoring.test_fedsda_run_metric_derivation` | `test_derivation_does_not_change_participant_state_and_is_repeatable`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_run_metric_derivation` | `test_derivation_rejects_invalid_arguments`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_run_metric_derivation` | `test_golden_condition_comparison_computation_metrics_match_real_legacy_counts`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_run_metric_derivation` | `test_golden_condition_metrics_match_real_legacy_experiment_result`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_run_metric_derivation` | `test_golden_condition_trace_arrays_match_real_legacy_saved_arrays`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fedsda_run_metric_derivation` | `test_model_diagnostic_metric_correspondence_distinguishes_metric_pairs`（1） | goldenと、対照の全条件を実行したときだけ確かめる |
| `tests.refactoring.test_fedsda_run_metric_derivation` | `test_model_pair_metric_correspondence_distinguishes_metric_pairs`（1） | goldenと、対照の全条件を実行したときだけ確かめる |
| `tests.refactoring.test_fedsda_run_metric_derivation` | `test_prediction_diagnostic_metric_correspondence_distinguishes_metric_pairs`（1） | goldenと、対照の全条件を実行したときだけ確かめる |
| `tests.refactoring.test_fedsda_run_metric_derivation` | `test_small_run_conditions_cover_required_cases`（1） | 対照の全条件を実行したときだけ確かめる |
| `tests.refactoring.test_fedsda_run_metric_derivation` | `test_small_run_metrics_match_real_legacy_metric_computation`（16） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fmnist_and_rotated_mnist_runs` | `test_command_accepts_new_datasets_without_creating_output`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fmnist_and_rotated_mnist_runs` | `test_command_rejects_missing_image_files_before_output`（2） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fmnist_and_rotated_mnist_sample_generation` | `test_cache_does_not_mix_mnist_and_fashion`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fmnist_and_rotated_mnist_sample_generation` | `test_fashion_directory_resolution`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fmnist_and_rotated_mnist_sample_generation` | `test_fashion_labels_and_sampling_follow_mnist`（2） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fmnist_and_rotated_mnist_sample_generation` | `test_fashion_missing_file_is_rejected_without_download`（2） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fmnist_and_rotated_mnist_sample_generation` | `test_fashion_reuses_idx_validation`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fmnist_and_rotated_mnist_sample_generation` | `test_generators_belong_only_to_their_dataset`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fmnist_and_rotated_mnist_sample_generation` | `test_rotation_matches_official_numpy_operations`（2） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fmnist_and_rotated_mnist_sample_generation` | `test_shared_directory_for_mnist_and_fashion_is_rejected`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_fresh_process_smoke` | `test_fresh_process_smoke_runs_without_legacy_or_test_modules`（1） | sandbox: 内部TemporaryDirectoryへのNPZ保存がPermissionError（task 1で実測、task 5の事前確認でも同じ制約） |
| `tests.refactoring.test_important_behavior_regression_goldens` | `test_corrupt_golden_is_rejected_before_environment_skip`（9） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_important_behavior_regression_goldens` | `test_duplicate_environment_is_rejected`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_important_behavior_regression_goldens` | `test_generation_preserves_existing_output_before_starting`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_important_behavior_regression_goldens` | `test_generation_rejects_unrecorded_source_before_workers`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_important_behavior_regression_goldens` | `test_generation_requires_two_matching_process_results`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_important_behavior_regression_goldens` | `test_unicode_dtype_is_a_valid_digest`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_important_behavior_regression_goldens` | `test_unknown_environment_has_no_match_but_reads_structure`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_mnist_sample_generation` | `test_damaged_files_are_rejected_as_invalid_format`（8） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_mnist_sample_generation` | `test_data_directory_follows_environment_variable_then_repository_default`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_mnist_sample_generation` | `test_differing_image_and_label_counts_are_rejected`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_mnist_sample_generation` | `test_invalid_file_formats_are_rejected`（6） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_mnist_sample_generation` | `test_loaded_training_data_is_reused_per_resolved_directory`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_mnist_sample_generation` | `test_loading_requires_keyword_path_argument`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_mnist_sample_generation` | `test_missing_files_are_rejected_without_fetching`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_mnist_sample_generation` | `test_small_files_are_read_as_written`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_seed_argument_parsing` | `test_invalid_main_arguments_create_no_output`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_seed_argument_parsing` | `test_seed_syntax_does_not_change_plan_identifiers_or_legacy_resume`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.refactoring.test_synthetic_sample_generation` | `test_expected_generator_types_cover_all_defined_datasets`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_build_feddrift_baseline` | `test_build_baseline_groups_canonical_results_by_dataset`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_build_feddrift_baseline` | `test_build_baseline_keeps_same_sweep_for_multiple_schedules`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_build_feddrift_baseline` | `test_build_baseline_prefers_explicit_sweep_parameter`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_build_feddrift_baseline` | `test_build_baseline_refuses_to_overwrite_existing_output`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_build_feddrift_baseline` | `test_extend_baseline_adds_new_seed_atomically_and_keeps_backup`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_build_feddrift_baseline` | `test_extend_baseline_adds_new_sweep_value`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_build_feddrift_baseline` | `test_extend_baseline_rejects_conflicting_duplicate`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_clustering_functional_diagnostics` | `test_clustering_functional_diagnostics_separates_merged_and_retained`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_experiment_artifacts` | `test_rebuild_artifacts_recovers_pareto_metrics_from_old_raw_and_log`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_experiment_artifacts` | `test_rebuild_artifacts_uses_embedded_exact_metrics`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_experiment_manifests` | `test_backfill_existing_csv`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_experiment_manifests` | `test_backfill_infers_missing_schedule_and_total_data_from_raw`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_experiment_manifests` | `test_manifest_session_records_lifecycle_and_outputs`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_experiment_manifests` | `test_overlap_distinguishes_exact_and_different_provenance`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_experiment_manifests` | `test_preview_overlaps_does_not_create_manifest`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_experiment_manifests` | `test_provenance_changes_when_regression_golden_changes`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_experiment_manifests` | `test_recursive_backfill_separates_variants_and_ignores_analysis_csv`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_experiment_manifests` | `test_study_manifest_is_generated_from_utf8_definition`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_main_ablation_suite` | `test_main_ablation_suite_lists_individual_variants`（1） | POSIX bashを利用できない環境 |
| `tests.test_paper_datasets` | `test_idx_reader_parses_mnist_files`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_pareto_sweep` | `test_load_csv_accepts_canonical_feddrift_baseline_names`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_pareto_sweep` | `test_load_csv_rejects_legacy_parameter_columns`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_pareto_sweep` | `test_plot_pareto_can_use_overall_accuracy`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_pareto_sweep` | `test_plot_pareto_draws_baseline_standard_deviation_band`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_pareto_sweep` | `test_replot_filter_selects_interval_sweeps_and_plot_accepts_compute_x`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_proposed_regression` | `test_proposed_regression`（1） | sandbox: 内部TemporaryDirectoryへのNPZ保存がPermissionError（task 1で実測、task 5の事前確認でも同じ制約） |
| `tests.test_recovery_analysis` | `test_plot_recovery_skips_file_when_no_series`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_server_sweep_wrapper` | `test_server_sweep_wrapper_rejects_owned_options`（1） | POSIX bashが利用できない環境 |
| `tests.test_server_sweep_wrapper` | `test_server_sweep_wrapper_resolves_runtime_options`（1） | POSIX bashが利用できない環境 |
| `tests.test_shared_backbone` | `test_drift_recovery_activation_is_saved_without_changing_history_shape`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_shared_backbone` | `test_shared_backbone_experiment_reports_component_metrics`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tests.test_telemetry` | `test_experiment_records_compute_and_model_telemetry`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_component_ablation.checks` | `test_all_declared_configuration_defaults_accept_current_canonical_settings`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_component_ablation.checks` | `test_cli_complete_grid_in_fresh_process`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_component_ablation.checks` | `test_complete_grid_has_hand_calculated_paired_statistics`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_component_ablation.checks` | `test_duplicate_baseline_invalidates_key_but_other_time_scale_is_just_excluded`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_component_ablation.checks` | `test_equal_metrics_have_zero_difference_independent_of_file_order`（4） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_component_ablation.checks` | `test_incomparable_input_is_excluded`（8） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_component_ablation.checks` | `test_invalid_input_and_duplicate_grid_arguments_create_no_outputs`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_component_ablation.checks` | `test_missing_run_does_not_bias_paired_mean`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_component_ablation.checks` | `test_non_finite_environment_extension_is_rejected_without_crash`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_component_ablation.checks` | `test_provenance_difference_requires_acknowledgement_and_is_reported`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_component_ablation.checks` | `test_stdout_read_only_and_unique_file_output`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_component_ablation.checks` | `test_undefined_metric_is_never_zero_filled`（5） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_prediction_initial_mass.checks` | `test_failed_worker_retains_reservation_and_no_completion`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_prediction_initial_mass.checks` | `test_invalid_plan_does_not_create_outputs`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_prediction_initial_mass.checks` | `test_plan_is_read_only_and_rejects_existing_reservation`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_prediction_initial_mass.checks` | `test_worker_reserves_output_before_training_and_writes_provenance`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_provisional_candidate_expert.checks_loss_monitor_rebaseline` | `test_artifact_comparison_finds_expected_runs_by_name_under_saved_results`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_provisional_candidate_expert.checks_loss_monitor_rebaseline` | `test_artifact_comparison_ignores_code_commit_but_not_baseline_version`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_provisional_candidate_expert.checks_loss_monitor_rebaseline` | `test_artifact_comparison_includes_all_arrays_and_non_time_summary_items`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_provisional_candidate_expert.checks_loss_monitor_rebaseline` | `test_combined_variant_product_sweep_records_limit_and_rebaselines`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_provisional_candidate_expert.checks_loss_monitor_rebaseline` | `test_execute_run_with_variant_enables_policy_and_writes_diagnostics_only_when_completed`（2） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_provisional_candidate_expert.checks_loss_monitor_rebaseline` | `test_metrics_reject_conflicting_diagnostics_for_the_same_run`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_provisional_candidate_expert.checks_loss_monitor_rebaseline` | `test_observe_only_product_sweep_matches_all_saved_items`（3） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_provisional_candidate_expert.checks_loss_monitor_rebaseline` | `test_parallel_workers_apply_variant_and_write_diagnostics`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_provisional_candidate_expert.checks_loss_monitor_rebaseline` | `test_run_calls_product_sweep_per_variant_with_patch_and_restores`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_provisional_candidate_expert.checks_loss_monitor_rebaseline` | `test_run_rejects_existing_output_without_variant_record`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_provisional_candidate_expert.checks_loss_monitor_rebaseline` | `test_run_rejects_other_variant_record_before_calling_product`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_provisional_candidate_expert.checks_loss_monitor_rebaseline` | `test_run_sets_evaluating_client_limit_only_for_combined_variant`（2） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |
| `tools.research_provisional_candidate_expert.checks_loss_monitor_rebaseline` | `test_variant_record_is_written_before_product_runs`（1） | sandbox: tempfile.mkdtemp内への書込みがPermissionError（task 5の事前確認） |

</details>

完了の範囲・未実施:

- task 5の実装と、実行可能な全対象の数値・乱数・専用検査は確認した。共通完了条件のsandbox例外と、このtaskで明示的に許される版15のファイル待ちを適用してtask 5に印を付ける。
  kiro-verify-completionの判定は、task 5の変更そのものと数値・乱数はVERIFIED、通常環境の保存・並列処理・全体成功まで含む主張はMANUAL_VERIFY_REQUIRED。
- 未実施は、上記の一時領域が必要な検査、WSL/Linux環境の数値実行、版15のgolden生成・配置とその19件の通過、全task後の独立CLI実装レビュー。
  版14までの保存済み結果全体の再集計、既存の型診断やtoolsのbootstrapのスタイル変更は行っていない。
- 残る限界: 旧goldenのexecution_eventは列全体のdigestのため、要求4.2の一覧で4列を除いており、その表の他の差も検出できない（task 3の記録どおり）。予測・他の記録・パラメータ・指標・乱数の照合とfresh smokeの段の検査は残す。
  次は主担当が通常環境でtask 5を確かめ、srcをcommitした後、task 6で版15のgoldenと独立レビューへ進むことを推奨する。今回はtask 6へ進んでいない。

### task 6（2026-10-28、主担当）

- task 5を通常の環境で確かめてcommitした（`c13149ea`）。全pytest 12993 passed / 27 skipped / 19 failed（19件は版15のgoldenが未配置のことによるものだけ）。
- 版15のgolden: 生成元`c13149ea`で、Windows（Python 3.13.15）とWSL（Python 3.14.4）で生成し（各18 case、helperの2回の独立したprocessの一致の検査を通過）、`behavior_regression_goldens/v15/`へ置いた。
- WSLの照合で3件が失敗し、次のように扱った。
  - 要求4.7の乱数の状態の2 case（`fedsda-sine2-adoption-without-validation`、`fedsda-mnist4-warm-started-learner`）: task 1のfixtureはWindowsで採ったもの。
    消す前のコード（`6adb3ee7`の一時のworktreeに、採取のhelperだけを写した）をWSLで実行して14 caseを採り、今のコードのWSLの値と比べると14 caseすべてで完全一致した。
    Windowsのfixtureと違うのは上の2 caseだけ（消す前のコードでも違う）なので、環境の差と判断した。
    Linuxのfixture`golden/fedsda_without_candidate_validation_random_states_linux.json`を、同じ形（生成元`6adb3ee7`、版14）で足した。
    `RANDOM_STATE_FIXTURE_PATHS`と`find_random_state_fixture`で数値の環境が一致するfixtureを選び、無い環境ではskipする。両環境のfixtureがあることも確かめる。task 1のWindowsのfixtureは変えていない。
  - `test_baseline_11_really_contains_zero_sample_cross_evaluations`: 版14まではWSLでも通っていた。残すcaseで件数0のクロス評価があるのはWindowsの`fedsda-mnist4-warm-started-learner`（6件）だけで、
    Linuxでは外した候補検証ありのcaseだけにあった。Windowsでは必須のまま、他の環境で該当が無ければ理由つきでskipにした。
- 確認（最終）:
  - 通常の環境の全pytest 13030 passed / 9 skipped / 0 failed（上の2つのtestの直しの前に開始）。直した2ファイル（`test_candidate_validation_removal_regression.py`の全部と、上の版11のtest）を直した後に流し直して53 passed。
  - WSL: `test_important_behavior_regression_goldens.py`・`test_experiment_run_code_provenance.py`・`test_candidate_validation_removal_regression.py`で430 passed / 1 skipped（上の版11のtest）。
  - 研究用の道具の専用の検証（task 5でsandboxのためskipしたものを含む）: `research_prediction_initial_mass/checks.py` 18 passed、
    `research_provisional_candidate_expert/checks_{loss_monitor_rebaseline,class_conditioned_prediction,mixing_past_posteriors}.py` 54・6・13 passed、`research_component_ablation/checks.py` 26 passed（skip・失敗なし）。
  - pyright（プロジェクトの設定）0 errors、ruff check・format通過。
- 独立の実装レビュー（gpt-6-luna、medium、`git diff 6adb3ee7 c13149ea`）: FAIL。Blockerなし。
  - Major: 版15のgoldenが未配置で、版15と版14の比較ができていない → 配置し、両環境で比較が通った。
  - Minor: 外したcaseでは候補検証以外の旧値も比べない／全経路の同等性は版1〜14の照合と乱数の照合に依る → 確認の範囲の限界として残す（外したcaseは今のコードで実行できない。検証なしの全構成は、数値goldenと乱数の状態の完全一致で確かめた）。
  - 主担当の判断で、Majorの解消は配置と比較の通過で確かめ、再レビューはしていない。
- 残る限界: 旧goldenの`execution_event`は列全体のdigestのため4列を除いている（task 3の記録）。
