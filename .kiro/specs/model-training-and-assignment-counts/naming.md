# 命名・役割

revision: 1。承認待ち。正式名の正本。

## 型とファイル

|名前|役割/所有|区別|
|---|---|---|
|model_training_and_assignment_counts.py|モデルに帰属する診断計数|回数計画や標本保持ではない|
|ModelTrainingAndAssignmentCountsStore|独立3辞書を所有|model実体/実行を所有しない|
|ModelTrainingAndAssignmentCountsSnapshot|3辞書の独立copyのrecord|frozen record内のdictは呼出し側で変更可|
|test_model_training_and_assignment_counts.py|旧対照と公開接続の証拠|旧APIは新productionへ追加しない|

## メソッド

|名前|入力→出力/役割|副作用|
|---|---|---|
|record_completed_model_training|ID/延べ標本数/個別step数→None。完了分の計数|2辞書加算|
|record_assigned_sample_concept|ID/真conceptかNone→None。1標本の診断帰属|concept辞書加算|
|get_model_assigned_sample_concept_counts|ID→concept dict copy|なし/欠落生成なし|
|snapshot_model_training_and_assignment_counts|なし→snapshot|なし|
|transfer_model_training_and_assignment_counts|元ID/受取ID→None|元除去、先へ加算。reassignの上書きとは異なる|
|remap_model_training_and_assignment_counts|一回対応表→None|3辞書再編/合計|
|_validate_integer|値/項目名/下限→None|型・下限検査のみ|

## field・状態・局所値

|名前|型/単位・意味|
|---|---|
|model_id / original_model_id / receiving_model_id / mapped_model_id|signed builtin int。記録対象/移管元/加算受取先/一回対応先|
|observed_concept_id|signed builtin intまたはNone。真概念、classラベルではない|
|trained_sample_count / parameter_update_step_count|非負builtin int増分。延べ標本件数/モデル個別更新step数|
|trained_sample_counts_by_model_id / _trained_sample_counts_by_model_id|モデル別延べ標本数のcopy/owner|
|parameter_update_step_counts_by_model_id / _parameter_update_step_counts_by_model_id|モデル別個別step数のcopy/owner|
|assigned_sample_counts_by_model_and_concept_id / _assigned_sample_counts_by_model_and_concept_id|モデル→concept→帰属件数のcopy/owner|
|model_id_mapping|dict[int,int]。一回サーバ対応|
|remapped_trained_sample_counts / remapped_parameter_update_step_counts / remapped_assigned_sample_counts|3独立一時dict、完成後交換|
|assigned_sample_counts / receiving_assigned_sample_counts|元/受取先のconcept別dict|
|assigned_sample_count|concept件数int|
|parameter_value / parameter_name / minimum_value|検査値/不正項目名/任意下限|

## テストの名前と役割

oracles: `build_model_counts_oracle`（新store/旧最小client構築）、`assert_model_counts_match_legacy`（独立key順と全件数対照）、`record_training_counts_in_both_implementations`（旧外部compute差分と新増分を対応）、`record_concept_counts_in_both_implementations`（実旧concept記録と新対応）。
`counts_store / legacy_client / legacy_registration_client / previous_snapshot / actual_snapshot / expected_snapshot`は新owner/旧学習/旧正式確認/独立snapshot比較。
`count_case / mapping_case / operation_name / operation_arguments / invalid_value / invalid_parameter_name / model_id_mapping / model_ids / concept_ids / increment`は条件/操作/拒否入力/対応表/順序/増分。
`class_count / optimizer_variant / update_shared_features / monkeypatch / training_batches / shared_optimizer / ordered_training_samples / training_bindings / python_random_generator / step_index`は実旧学習fixtureから借用した明示接続。
`actual_losses / expected_losses / expected_random_state / sampled_batch_history / previous_random_states / previous_numeric_environment`は全数値・RNG・環境の比較基準。
型拒否用: `ModelIdIntSubclass / ModelIdMappingDictSubclass`。
テスト名は`test_<観測契約>`、既存公開helper/引数名は元の承認済み役割で再利用する。
