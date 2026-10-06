# 命名・役割

revision: 1。承認待ち。正式名の正本は本書。

## ファイル・型

|名前|役割/所有|区別|
|---|---|---|
|evaluation/|評価の入力/計算部品|学習trainingや通信とは別|
|model_evaluation_sample_records.py|不変宣言|演算/検査なし|
|ObservedEvaluationSample|観測済み入力と正解labelのTensorを借用|ObservedTrainingSampleと用途を区別|
|ModelEvaluationSampleCollection|一model IDと評価標本tupleのsnapshot|mutable所有storeではない|
|model_evaluation_sample_store.py / ModelEvaluationSampleStore|モデル別評価列と固定容量/抽出件数を所有|学習列は所有しない|
|test_model_evaluation_sample_storage.py|旧保持/対応/拒否/損失接続の証拠|runtimeを実装しない|

## メソッド

|名前|入力→出力|役割/副作用|
|---|---|---|
|__init__|2固定件数→store|型/範囲検査、空dict生成|
|sample_and_append_model_evaluation_samples|ID/標本tuple/Random→None|追加前に抽出、容量超過は末尾保持|
|reassign_model_evaluation_samples_id|original/reassigned ID→None|列全体を先上書き、Random不要|
|remap_model_evaluation_sample_collections|対応表/Random→None|列連結、一回ID対応、超過列だけ抽出|
|snapshot_ordered_model_evaluation_samples|なし→collection tuple|モデル初出順、構造copy、payload借用|
|_validate_model_id|model_id/parameter_name→None|exact int検査、エラー項目名を表示|

## field・引数・内部変数

|名前|型/単位・役割|
|---|---|
|input_features / observed_class_labels|Tensor。評価入力と観測済み正解、shape検査は利用側|
|model_id / original_model_id / reassigned_model_id / mapped_model_id|builtin int。所属/変更元/先/一回対応結果|
|evaluation_samples|tupleまたは内部listの評価record列。用途は型で区別|
|maximum_stored_sample_count_per_model / _maximum_stored_sample_count_per_model|正int/標本件数、モデル別容量|
|added_batch_sample_count / _added_batch_sample_count|非負int/標本件数、1回追加の抽出上限。学習batch sizeではない|
|python_random_generator|exact Random。呼出しで借用する抽出乱数、共有乱数を変更しない|
|_evaluation_samples_by_model_id|dict[int,list[ObservedEvaluationSample]]、唯一の列構造owner|
|model_id_mapping|dict[int,int]、サーバ提供済み一回対応|
|remapped_evaluation_samples_by_model_id|再編の一時dict、完成後交換|
|sample_count_to_append / sampled_evaluation_samples|追加抽出件数/抽出順record list|
|evaluation_sample / parameter_name|検査対象record/不正入力名|

## テスト内の名前

旧oracles: `build_evaluation_storage_oracle`（store/旧client/record列生成）、`run_legacy_evaluation_operation`（共有Python乱数を一時対応してfinally復元）、`assert_evaluation_storage_matches_legacy`（ID/順序/全payload identity照合）、`snapshot_evaluation_reference_ids`（不正前後比較）。
`operation_name / operation_arguments / initial_random_state / expected_random_state / python_random_generator`は操作選択/引数/開始・期待終端/新側明示Random。
`legacy_client / sample_store / evaluation_samples / previous_snapshot / previous_reference_ids / previous_random_states / previous_numeric_environment`は旧対照・新owner・借用record列・旧snapshot/参照・3共有RNG/torch defaults。
`storage_case / mapping_case / input_case / invalid_value / invalid_parameter_name / sample_count / model_ids / model_id_mapping / capacity / append_sample_count / step_index`はケース条件/不正対象/件数/ID列/対応/容量/追加上限/操作順。
`class_count / classifier / legacy_classifier / input_features / observed_class_labels / actual_losses / expected_losses / parameter_snapshot / collection / sampled_records`は損失接続のクラス数/新旧NN/Tensor/損失/共有初期値/一覧/対象標本。
検査用派生型: `ModelIdIntSubclass / EvaluationSamplesTupleSubclass / ModelIdMappingDictSubclass / ObservedEvaluationSampleSubclass / RandomSubclass`。名前通りexact拒否の証拠のみ。
