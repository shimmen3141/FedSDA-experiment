# 命名: 保有モデル学習状態のID付替え

revision: 2

|名前|役割・型/単位・副作用・類似名との違い|
|---|---|
|reassign_held_model_training_state_id|一保有状態のID付替え→None。NN/optimizer登録・reset・サーバ対応表と区別|
|original_model_id / reassigned_model_id|exact builtin intの元/先ID、signed、単位なし。正式ID採番/負→非負制限を含めない|
|parameter_name|既存ID検査へ渡す拒否項目名。既定model_idで旧APIの拒否messageを維持|
|held_model_training_state / reassigned_held_model_training_state|元のfrozen record/変更先IDの新wrapper。NN/管理器参照は同じ|
|test_held_model_training_state_id_reassignment.py|本specの実旧順序/参照/拒否/接続検証|
|build_legacy_registration_client / build_registry_classifier_and_owner / register_training_state|前spec/上流のtest helperを同じ役割で借用。実旧confirmを使い、玩具oracleを作らない|
|test_training_state_id_reassignment_matches_legacy_registration|実旧負元IDの順序/NNidentityと先上書き|
|test_training_state_id_reassignment_accepts_signed_ids|汎用signed/同IDの期待順|
|test_training_state_id_reassignment_rejects_invalid_ids_without_mutation|両ID事前拒否と一覧/学習保持|
|test_training_state_id_reassignment_preserves_old_records_and_current_optimizer|古いwrapper/bindingと現在owner/resetの寿命|
|test_reassigned_training_state_continues_joint_updates_and_statistics|12条件の実旧共同更新前後と上位統計付替え/保留保持|
|registry / classifier / optimizer_owner / shared_optimizer / training_batches / legacy_client|既存test helperの新registry/NN/個別owner/共有optimizer/参加batch/実旧学習owner|
|legacy_registration_client / loss_statistics_store / pending_upload_state|実旧confirm用owner/新統計owner/新保留owner。学習ownerと区別|
|initial_model_ids / expected_model_ids / invalid_model_id / invalid_parameter_name|初期/期待順・拒否値/拒否項目|
|previous_state / previous_binding / previous_states / previous_bindings / reassigned_state / reassigned_binding|過去frozen記録/tuple/変更先の現在記録|
|parameter_snapshots / previous_optimizer / previous_optimizer_state / previous_random_states / previous_numeric_environment|値/grad・optimizer参照/蓄積state・3乱数/defaultsの保持証拠|
|class_count / optimizer_variant / update_shared_features / monkeypatch / step_index|class数/方式/共有更新フラグ/fixture/0始まり共同更新位置|
|concept_owners / participating_training_batches / training_bindings / training_batch / model_id|上位で保持するowner列/現在参加batch/binding/反復要素とID|
|expected_losses / actual_losses / initial_parameter_snapshot / current_loss_statistics / pending_model_upload|旧/新loss列/生成済み固定モデル値/現在統計/借用保留record|
|source_present / destination_present / source_state / destination_state|元先有無/付替え前の別実体record|
|legacy_training_batches / optimizer_settings / concept_parameters|実旧samplerのID付きbatch列（test側でIDだけ対応）/既存個別optimizer設定/adapter→headのparameter列|
|registered_states_by_model_id / expected_states / input_features / observed_class_labels|比較用元ID別record/期待一覧/既存batchの特徴とラベル|
|DerivedModelId|拒否用int派生型、test-only|

既存helper `build_joint_update_oracle_pair`/`run_legacy_joint_update`/`assert_joint_update_states_equal`/`assert_nested_state_equal`/parameter-grad保存比較を同じ役割で借用。
fixture・標準型・短いループ添字は既存慣例。公開API/永続状態を増やす場合は再レビューする。
