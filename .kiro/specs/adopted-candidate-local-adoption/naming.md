# 命名 revision 2

|名前|型・役割・更新する状態・区別|
|---|---|
|runtime/adopted_candidate_local_adoption.py|採用が決まった候補についての、クライアント内状態更新の組立。初期ローカル登録（その一部として呼ぶ）や正式ID確認、採否判定ではない|
|adopt_candidate_as_current_training_model|採番・登録・計数・標本追加・現在ID切替えを行いTrainingModelAssignmentChangeを返す。自身の状態なし。register_adopted_candidate_as_temporary_held_model（登録だけ）と区別|
|temporary_model_id_allocator|既存具体型。次の一時IDを読み、登録成功後に採番を確定|
|adopted_candidate_classifier / candidate_concept_specific_parameter_optimizer_state / initial_statistics_input_features / initial_statistics_observed_class_labels / upload_delay_round_count|初期ローカル登録APIと同じ名前・役割。そのまま渡す|
|candidate_trained_sample_count|exact int非負。採用前の候補の学習で使った延べ標本数。一時IDの学習計数へ加算。既存計数APIのtrained_sample_countへ渡す|
|candidate_parameter_update_step_count|exact int非負。採用前の候補のparameter更新回数。既存計数APIのparameter_update_step_countへ渡す|
|pending_assignment_training_samples|exact tuple[ObservedTrainingSample,...]。警報後の判定中に帰属を保留していた標本。採用により新モデルの学習標本になる。initial_statistics_*（初期統計を作る標本）とは別の列|
|held_model_training_state_registry / loss_statistics_store / current_training_model_assignment / pending_model_upload_state|初期ローカル登録APIと同じ具体型・役割。current_training_model_assignmentは登録後に本関数が一時IDへ切り替える|
|training_sample_store|既存具体型。一時IDの未使用を確認し、保留標本を追加|
|model_training_and_assignment_counts_store|既存具体型。一時IDの未使用を確認し、候補の学習量を加算。割当概念計数は変更しない|
|_validate_adoption_inputs|3ownerと現在IDownerの型、2計数、保留標本を検証。変更なし|
|_reject_temporary_model_id_already_in_use_for_adoption|標本store・計数store・現在IDで一時IDの使用を確認しValueError。変更なし。登録moduleの同種helper（一覧・統計・送信保留）と対象ownerが異なる|
|temporary_model_id|採番ownerから読んだ、この採用で使う一時ID|
|assignment_change|既存変更record。現在ID切替えの結果として返す|
|model_training_sample_collection / counts_snapshot|使用済みID確認で読む、標本storeのcollection/計数storeのsnapshot|
|state_owner / expected_owner_type / owner_name|型検査ループ（登録・登録確認と同じ）|

## testで使用する名前

|名前|役割|
|---|---|
|build_local_adoption_oracle|上流の登録oracleへ採番owner・標本store・計数store・保留標本を加え、実旧clientへ採用分岐に必要な属性と実ForwardValidationSessionを設定|
|adoption_arguments|新関数のkeyword引数dict|
|legacy_client / legacy_candidate_model / legacy_session|実旧client/旧候補/実旧ForwardValidationSession|
|finalize_adoption_in_legacy_client|実旧_finalize_forward_validationを採用条件で実行するtest helper。手順を再構成しない|
|assert_local_adoption_matches_legacy|一時ID・計数・標本列・現在ID・採番次値を対照し、登録部分は上流のassert_initial_registration_matches_legacyを使う|
|snapshot_adoption_state / assert_adoption_state_unchanged|拒否時不変の観測と確認。上流の登録状態snapshotに採番次値・標本・計数を加える|
|pending_sample_count / candidate_training_counts|保留標本数/候補の(標本数, 更新回数)のtest軸|
|invalid_case / invalid_owner_name / invalid_owner / IntSubclass|拒否入力|
|expected_call_order / actual_call_order / operation_name / original_operation|順序観測|
|source_text / expected_acceptance|AST注入契約|
|test_*|契約を記述するpytest関数|

共同学習接続の名前は上流registration/confirmation specと同じ役割で再利用する。追加が必要になったら実装前に本表へ戻してレビューする。

## Task1・Task2で追加した名前（revision 2）

productionの追加名は、検証ループの局所変数count_name / count_value（2つの計数の引数名と値）とtraining_sample（保留標本の一要素）だけ。

|名前|役割|
|---|---|
|adoption_module|monkeypatchで登録関数の呼出しを観測するための新moduleの別名|
|assert_training_samples_match_legacy|新標本storeのモデル順・件数・各標本の特徴/ラベルTensorのidentityを、実旧train_data_storeと対照|
|assert_adoption_rejected_without_any_change|新関数が指定例外で拒否し、採番次値を含む全状態が不変であることを確認|
|valid_adoption_arguments|不正値へ差し替える前の引数一式。不変確認で正しいownerを読むために使う|
|registration_arguments / shared_optimizer_owners|上流の登録testと同じ役割|
|counts_store / training_sample_store / loss_statistics_store / registry / candidate / candidate_optimizer_owner / active|既存ownerと候補の局所名（上流testと同じ役割）|
|model_index / sample_index / training_samples|既存モデルへ事前に置く標本の構築|
|legacy_client.local_model_changes|実旧_on_local_model_changeへ渡された(変更前ID, 変更後ID)を記録するtest用属性|
|previous_model_id / expected_temporary_model_id / previous_loss_statistics / previous_counts / current_counts / previous_training_samples / current_training_samples / previous_collection / collection|採用前後の観測値|
|first_assignment_change / second_assignment_change / second_adoption_arguments / second_shared_optimizer_owners / second_legacy_client / second_legacy_candidate_model / second_legacy_session|連続2回の採用testで、2回目の候補とsessionを別oracleから取り出すための名前|
|used_temporary_model_id|拒否testで事前に使用済みにする、採番ownerの次の一時ID|
|owner_name_by_case_prefix / invalid_count_by_case_suffix / case_prefix / argument_name / owner_type / owner_subclass|拒否caseの文字列から差し替える引数と不正値を引く対応|
|original_registration / record_then_register / record_then_update|呼出しを記録してから元の登録関数/owner APIへ委譲する順序観測|
|run_joint_update_in_both_implementations / legacy_training_batches / training_bindings / model_training_sample_collections / participating_training_batches / expected_joint_loss / actual_joint_loss|上流testと同じ役割。本testでは両実装とも各自の標本storeの全標本を固定batchにする|
|candidate_features / candidate_labels / local_training_settings / registered_candidate_optimizer / confirmation_change / temporary_model_id / random_states / numpy_state|上流testと同じ役割の局所名と、正式ID確認が返す変更record|
|adopted_candidate_local_adoption_cpu_smoke.py|共有venvのrefactoring-testsへ置く、旧importなしのfresh CPU smoke。Git管理外|
|test_adopted_candidate_local_adoption_exact_dependency_contract|依存境界testの注入契約|
