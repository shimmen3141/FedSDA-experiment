# 命名: モデル学習標本のID付替え

revision: 1

|名前|役割・型/単位・副作用・類似名との違い|
|---|---|
|reassign_model_training_samples_id|一モデルの学習標本列を別IDへ移す→None。対応表の列連結・標本追加と区別|
|original_model_id / reassigned_model_id|exact builtin intの元/先、signed、単位なし。採番/正式通知を含めない|
|parameter_name / model_id / training_samples|検査項目名/検査ID/popした内部list。値を計算しない|
|test_model_training_sample_id_reassignment.py|実旧pop/拒否/借用/抽出・学習継続|
|test_training_samples_id_reassignment_matches_legacy_registration|実旧負元IDの列/順/参照/衝突|
|test_training_samples_id_reassignment_accepts_signed_ids|汎用signedと同IDの期待順|
|test_training_samples_id_reassignment_rejects_invalid_ids_without_mutation|両ID拒否と一覧・Tensor保持|
|test_reassigned_training_samples_preserve_snapshots_and_support_append|過去snapshot/後続追加/重複借用参照|
|test_training_samples_id_reassignment_does_not_inspect_payloads|opaque/契約外Tensorも触らず移す|
|test_reassigned_training_samples_continue_sampling_and_joint_updates|12条件の旧抽出・共同更新への上位接続|
|build_training_sample_storage_oracle / append_legacy_training_samples / assert_training_sample_storage_matches_legacy / snapshot_training_sample_reference_ids|上流標本test helperを同じ役割で借用|
|build_legacy_registration_client|前specの実旧confirm用最小owner生成。学習標本/NNはtest側で明示接続|
|build_training_iteration_oracle_pair / run_legacy_training_iterations / assert_joint_update_states_equal / assert_nested_state_equal|上流の実NN/旧抽出・更新/全state比較helper|
|sample_store / legacy_client / legacy_registration_client / training_samples|新標本owner/旧学習owner/実旧登録oracle/借用標本列|
|initial_model_ids / expected_model_ids / source_sample_count / previous_snapshot / previous_reference_ids|初期/期待ID順・元標本数・過去構造/参照ID一覧|
|invalid_model_id / invalid_parameter_name / source_present / destination_present / payload_case|拒否値/項目/元先有無/payload条件|
|parameter_snapshots / input_features / observed_class_labels / previous_parameter_values / previous_numeric_environment / previous_random_states|Tensor参照・全値/特徴/ラベル・比較用clone/既定環境/3RNG|
|class_count / optimizer_variant / update_shared_features / monkeypatch / step_index|class数/方式/共有更新/fixture/0始まり更新位置|
|training_batches / shared_optimizer / ordered_training_samples / training_bindings / participating_training_batches|元batch/共有optimizer/初期標本一覧/上位ID対応済みbinding/現在optimizer参加batch|
|python_random_generator / expected_losses / actual_losses / expected_random_state / sampled_batch_history|明示Random/旧新loss列/旧終端RNG/旧抽出履歴|
|sampled_batches / assert_sampled_batches_equal|既存sampler結果と旧batch全Tensor比較helper。Randomの独立deepcopyで抽出をpreviewし、実更新のRandom消費を増やさない|
|training_sample_collection / training_sample / training_batch / training_binding|既存record/反復要素を同じ役割で使用|
|DerivedModelId|test-only拒否用int派生型|

標準型・fixture・短いループ添字は既存慣例。公開API/永続状態や役割の追加変更は再レビューする。
