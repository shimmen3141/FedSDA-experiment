# 命名 revision 3

|名前|型・役割・更新する状態・区別|
|---|---|
|runtime/adopted_candidate_initial_local_registration.py|採否で採用された候補を、一時IDの保有モデルとしてクライアント内へ最初に登録する組立。正式ID確認（held_model_registration_confirmation）や送信ではない|
|register_adopted_candidate_as_temporary_held_model|共有反映・学習状態一覧・統計・送信保留へ登録しNoneを返す。自身の状態なし。確認のconfirm_held_model_registrationと対になる|
|temporary_model_id|exact int負。サーバ確認前のクライアント内ID。採番は呼出し側。registered_global_model_id（非負の正式ID）と区別|
|adopted_candidate_classifier|既存共有反映APIと同じ名前・役割。採否で採用済みの学習済みResidualAdapterClassifier。登録後は保有モデルの分類器になる|
|candidate_concept_specific_parameter_optimizer_state|既存共有反映APIと同じ名前。候補の概念固有parameter用ParameterOptimizerState。resetされ、登録後は保有モデルの管理器になる|
|initial_statistics_input_features|Tensor、初期損失統計を作る標本の特徴（batch×特徴数）。学習標本storeへは追加しない|
|initial_statistics_observed_class_labels|Tensor、同じ標本の観測ラベル。既存損失評価のobserved_class_labelsへ渡す|
|upload_delay_round_count|exact int 1以上、既存送信保留APIと同じ名前。送信可能までのラウンド境界回数|
|held_model_training_state_registry|既存具体型。保有確認・反映先選択に読み、候補を末尾へ登録|
|loss_statistics_store|既存具体型。一時IDの未登録を確認し、初期統計を設定|
|current_training_model_assignment|既存具体型。反映先選択のため現在IDを読むだけ。変更しない|
|pending_model_upload_state|既存具体型。対応IDの重複を確認し、snapshotと待機を登録|
|_validate_registration_inputs|一時ID・待機ラウンド数・4ownerの型/値を検証。変更なし|
|_reject_temporary_model_id_already_in_use|一覧・統計・送信保留の3箇所で一時IDの使用を確認しValueError。変更なし|
|_select_active_shared_feature_extractor|保有一覧と現在IDから反映先のSharedFeatureExtractorを返す。空はLookupError。値は変更しない|
|held_model_training_states|一覧snapshotのtuple。重複確認と反映先選択で一度だけ取得|
|held_model_training_state|一覧の一record（反復変数）|
|current_training_model_id|現在の学習帰属ID（読取り値）|
|pending_model_upload|既存の保留record、またはNone|
|active_shared_feature_extractor|既存共有反映APIと同じ名前。全保有モデルが参照する反映先|
|per_sample_bounded_losses|既存初期統計APIと同じ名前。候補の標本順の有界損失Tensor|
|initial_loss_statistics|ModelAndClassLossStatistics。登録する全体/クラス別の初期統計|
|parameter_snapshot|既存送信保留APIと同じ名前。候補の独立した全parameter値|
|state_owner / expected_owner_type / owner_name|型検査ループの具体owner/期待具体型/例外で示す名前（登録確認と同じ）|

## testで使用する名前

|名前|役割|
|---|---|
|build_initial_registration_oracle|新4ownerと保有モデル群・候補、同じ初期値の実旧client/候補を構築|
|registration_arguments|新関数のkeyword引数dict|
|legacy_client / legacy_candidate_model|固定旧登録を呼ぶ共有部構成clientと旧候補モデル|
|register_candidate_in_legacy_client|旧_register_trained_new_model(pending_ready=False)と旧FedSDA待機設定を同じ順で行うtest helper|
|assert_initial_registration_matches_legacy|共有部値/一覧順/出力/統計/保留/待機を対照|
|snapshot_registration_state|拒否時不変を比較する、owner構造・共有部値・候補接続先identity・optimizer stateのsnapshot|
|held_model_ids / current_model_is_held / existing_pending_model_id|一覧順/反映先選択/既存保留のtest軸|
|class_count / optimizer_variant / update_shared_features|上流joint-update testと同じ軸|
|statistics_sample_labels / statistics_sample_count|singleton・欠落クラスを含む統計標本の構成|
|expected_call_order / actual_call_order / operation_name / original_operation|順序観測用wrapper|
|invalid_temporary_model_id / invalid_upload_delay_round_count / invalid_owner_name / invalid_owner / IntSubclass / OwnerSubclass|拒否入力・派生型|
|source_text / expected_acceptance|AST注入契約|
|test_*|契約を記述するpytest関数、永続状態なし|

共同学習接続のbatch/optimizer/settings/Random状態の名前は上流joint-training/registry/confirmation specと同じ役割で再利用する。追加が必要になったら実装前に本表へ戻してレビューする。

## Task1で追加した名前（revision 2）

productionの追加名はない。型注釈用に既存の型HeldModelTrainingStateとSharedFeatureExtractorをimportする（design revision 2）。

|名前|役割|
|---|---|
|TEMPORARY_MODEL_ID|test定数-7。登録する一時ID|
|OWNER_NAMES|新関数が受け取る4ownerの引数名のtuple。owner別拒否のtest軸|
|registration_module|monkeypatchで呼出順を観測するための新moduleの別名|
|convert_legacy_parameter_name|旧state_dictのparameter名を新分類器の名前へ対応させるtest helper。値は変えない|
|held_models_share_feature_extractor|test軸。Falseでは保有モデルが別々の共有部を持ち、反映先選択（現在ID優先/先頭）を値で観測できる|
|shared_optimizer_owners|保有モデルと候補の共有部parameter用ParameterOptimizerStateの列。共有optimizerの蓄積state保持を旧backbone.optimizerと対照|
|legacy_models / classifiers / optimizer_owners|oracle構築中の旧モデル列/新分類器列/個別optimizer管理器列。末尾が候補|
|model_index / is_candidate / reuse_first_feature_extractor|構築ループの位置/候補かどうか/先頭の共有部を再利用するか|
|optimizer_settings / shared_optimizer_owner / shared_optimizer / stepped_shared_optimizer_ids|上流testと同じ設定値/共有部の管理器/そのoptimizer/共有optimizerを一度だけstepするためのid集合|
|arguments|旧登録helperへ渡すregistration_argumentsの別名引数|
|shared_optimizer_owner_by_extractor_id / first_shared_parameter|共有部の先頭parameterのidから管理器を引く対応/その先頭parameter|
|registered_parameter_storage_addresses|snapshotが登録モデルのparameterとstorageを共有しないことを確かめるdata_ptr集合|
|assert_registration_state_unchanged|snapshot_registration_stateの観測値と現在の全状態が同じことを確認|
|assert_rejected_without_any_change|新関数が指定例外で拒否し、全状態が不変であることを確認|
|select_valid_owners / valid_owners|registration_argumentsから4ownerを取り出すhelper/不正ownerへ差し替える前の正しいowner一式|
|previous_snapshot / previous_states / previous_optimizers / previous_candidate_optimizer / concept_parameter_snapshots / random_states|登録前の観測値。保持比較に使う|
|expected_active_feature_extractor / active_parameters / active / active_snapshot|期待する反映先とそのparameter参照・値|
|invalid_case|候補・管理器・特徴・ラベルの拒否入力の種類|
|record_then_call / record_then_update|呼出しを記録してから元の関数/owner APIへ委譲する順序観測wrapper|

## Task2で追加した名前（revision 3）

productionの追加名はない。

|名前|役割|
|---|---|
|assert_held_model_states_match_legacy|保有一覧の順序と各モデルの全値/grad/個別・共有optimizer state/共有参照構造/出力を実旧clientと対照するtest helper。assert_initial_registration_matches_legacyから統計・送信保留以外の部分を分離し、登録確認で保留が消えた後の学習でも使う|
|run_joint_update_in_both_implementations|test内closure。registryの現在bindingの順で実旧_train_heads_togetherと新共同更新を一回ずつ行い、lossと全状態を対照|
|legacy_training_batches|実旧共同学習へ渡す(モデルID, 特徴, ラベル)のlist。上流joint-update testと同じ役割|
|training_batch_tensors|保有2モデルと候補の固定学習batch（特徴, ラベル）の列。一覧順に対応|
|training_batch_index / sample_count / batch_features / batch_labels|固定batchの位置/標本数/その特徴/ラベル|
|training_bindings / training_binding|registryから取得した現在の学習binding列/その一要素|
|expected_joint_loss / actual_joint_loss|実旧/新の共同更新の標本数加重loss|
|candidate_features / candidate_labels|候補だけの登録前学習に使うbatch|
|local_training_settings|上流testと同じ共同学習設定|
|pending_parameter_snapshot|登録直後の送信保留値の複製。後続学習で保留値が変わらないことの比較用|
|registered_candidate_optimizer|登録後の学習で蓄積した候補の個別optimizer。正式ID確認後も同じobjectであることの比較用|
|assignment_change|既存登録確認が返す現在ID変更record|
|numpy_state|NumPy乱数状態の比較用の現在値|
|adopted_candidate_initial_local_registration_cpu_smoke.py|共有venvのrefactoring-testsへ置く、旧importなしのfresh CPU smoke。Git管理外|
