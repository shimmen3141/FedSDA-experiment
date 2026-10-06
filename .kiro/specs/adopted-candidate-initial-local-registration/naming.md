# 命名 revision 1

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
