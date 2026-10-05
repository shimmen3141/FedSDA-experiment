# 命名案 revision 2

|種類|名前|役割・型・更新する状態|
|---|---|---|
|module|held_model_shared_feature_reconnection.py|保有モデル全体の共有元選択と再接続の外側操作|
|frozen借用記録|HeldModelOptimizerBinding|モデルIDと分類器/共有/個別optimizer ownerの対応、所有や生成はしない|
|fields|model_id / classifier / shared_parameter_optimizer_state / concept_specific_parameter_optimizer_state|int識別子/分類器/共有parameter owner/概念固有parameter owner|
|公開操作|reconnect_held_models_to_shared_feature_extractor|全対応検証後、共有元以外だけ接続/resetし現在対応のtupleを返す|
|引数|held_model_optimizer_bindings|入力順のexact tuple、各bindingの実体は借用|
|private検査|_validate_optimizer_parameter_binding / _validate_held_model_optimizer_bindings|owner現在optimizerと期待parameter列の対応、全記録とID/共有適合検査|
|検査引数|optimizer_state / expected_parameters|借用ParameterOptimizerState、期待するParameter tuple|
|局所|binding / source_binding / reconnected_bindings / nonnegative_model_ids / model_ids / classifier_ids / concept_parameter_ids / current_concept_parameter_ids|現在記録/共有元/結果記録列/非負候補/重複検査ID集合|
|局所|parameter_optimizer / optimizer_parameters / parameter_group / parameter / expected_parameter / parameter_index / parameters|現在optimizer/実parameter tuple/groups/個別参照と期待参照/位置/対象列|
|test module|test_held_model_shared_feature_reconnection.py|旧選択/参照reset/拒否/共同学習を対照|
|test helpers|build_held_model_optimizer_binding / snapshot_reconnection_input_states / assert_reconnection_input_states_unchanged|明示新NN/owner記録生成、参照/値grad/optimizer保存、拒否で保持確認|
|test局所|model_ids_in_order / expected_source_model_id / optimizer_variant / initially_shared / invalid_case / invalid_value / parameter_snapshots / optimizer_snapshots / previous_bindings / result_bindings / previous_optimizer / previous_state_dict / python_random_state / torch_random_state / events / failure / failed_model_id|入力順/期待共有元/設定/共有条件/拒否例/保存値/旧新記録/操作順と例外位置|
|既存test名再利用|class_count / classifier / shared_feature_extractor / shared_optimizer_state / optimizer_settings / training_batches / participating_training_batches / legacy_client / legacy_model / legacy_loss / new_loss / local_training_settings / training_batch / model_id / monkeypatch|既存model/optimizer/共同学習testと同じ役割|
|既存helper再利用|snapshot_parameter_values_and_gradients / assert_parameter_values_and_gradients_unchanged / build_joint_update_oracle_pair / run_legacy_joint_update / assert_joint_update_states_equal|前specの値grad保存と実旧学習のexact対照|
test関数はtest_<観測する契約>。pytest fixtureと既に承認済みの明確な局所名は同じ役割で再利用する。

revision2追加: `ModelIdSubclass`（exact int拒否の型派生）、`HeldModelOptimizerBindingSubclass`（exact record拒否の型派生）、`bindings_by_model_id`（返却記録のID対応）、`legacy_models`（実旧モデルID辞書）、`legacy_shared_optimizers`（旧共有optimizer参照列）、`concept_specific_optimizer_states`（概念固有owner列）、`previous_concept_optimizer_snapshots`（旧概念optimizer参照/state保存）、`source_model_id`（選択結果の共有元ID）、`binding_index`（記録列の位置）。ASTのcase引数source_text/expected_acceptance/source_module_pathは既存の役割で再利用する。
