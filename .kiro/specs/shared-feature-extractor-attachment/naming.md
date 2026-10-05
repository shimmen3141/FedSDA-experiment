# 命名案 revision 2

|種類|名前|役割/入出力/状態更新/違い|
|---|---|---|
|既存型|ResidualAdapterClassifier / SharedFeatureExtractor|概念固有分類器と共有特徴抽出NN|
|公開method|attach_shared_feature_extractor|適合する既存抽出部の参照へ再接続、None。NN値やoptimizerのresetは行わない|
|引数|shared_feature_extractor|exact SharedFeatureExtractor、元入力寸法/hidden構成に一致|
|既存状態|feature_extractor|成功後だけ交換する借用参照|
|test module|test_shared_feature_extractor_attachment.py|接続/不正拒否と実旧NN対照|
|test helper|build_attachment_classifier / snapshot_parameter_values_and_gradients / assert_parameter_values_and_gradients_unchanged|明示寸法の新分類器構築、参照/値/grad保存、拒否/接続で保持確認|
|test局所|classifier / shared_feature_extractor / previous_feature_extractor / previous_residual_adapter / previous_classification_layer / previous_output_activation / previous_parameters / previous_gradients / parameter / parameter_values / parameter_index / input_features / class_count / hidden_layer_widths / same_reference / invalid_case / invalid_value / python_random_state / torch_random_state|分類器/接続前後NN/概念参照と値grad/期待寸法/入力/RNG・拒否条件|
|test局所|training_batches / participating_training_batches / legacy_client / legacy_model / legacy_feature_extractor / shared_optimizer_state / concept_specific_optimizer_states / previous_parameter_optimizer / optimizer_settings / optimizer_variant / existing_shared_optimizer / update_shared_features / legacy_loss / new_loss / training_batch / optimizer_state / model_id|実旧/新NN接続、現在parameterへ対応するowner、既存共有optimizer有無と実学習比較|
|既存helper|build_joint_update_oracle_pair / run_legacy_joint_update / assert_joint_update_states_equal|旧実共同更新と全値grad/stateの対照|
|test型|SharedFeatureExtractorSubclass|exact型拒否を確認する型派生|
test関数はtest_<観測する契約>で命名する。pytest fixtureと既存helperの明確な局所名を再利用する。

revision2追加局所名: `parameters`（検査するParameter列）、`parameter_snapshot`（参照/値/gradの保存tuple）、`expected_prediction`（明示新抽出経路の出力）、`shared_feature_extractor_snapshot`（接続先状態の保存）。metaには数値実体がないため参照/shape/dtype/deviceだけ確認する。
