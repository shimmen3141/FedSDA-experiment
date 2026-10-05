# 命名: 保有モデルの学習バッチ抽出
revision 1。入力の観測標本/モデル別population/出力batchと、保有集合/参加順を区別する。

| 名前 | 役割・型・状態/単位 |
|---|---|
| model_training_sample_records.py | 観測標本・model別標本列・抽出batchの三record宣言 |
| ObservedTrainingSample | frozen/kwonly。input_features[1,D]/observed_class_labels[1,1]の借用Tensor参照 |
| ModelTrainingSampleCollection | frozen/kwonly。model_idとtraining_samplesの順序付きtuple |
| SampledModelTrainingBatch | frozen/kwonly。抽出対象model_idとcat済み特徴/ラベル |
| input_features / observed_class_labels | 観測特徴/ラベル。recordにより単一行またはbatch行数 |
| model_id / training_samples | Python builtinintの識別子/同モデル向け観測population。負IDも可 |
| held_model_training_batch_sampling.py | 保有/件数filterと一回の復元なし抽出 |
| sample_training_batches_for_held_models | 借用Randomだけを進め、ID付きbatchtupleを返す |
| held_model_ids | exactfrozenset。保有集合であり、反復順を規定しない |
| ordered_model_training_samples | exacttuple。モデル別collectionの入力順が参加/乱数順 |
| batch_sample_count | exactint>0。各参加モデルの抽出標本数 |
| python_random_generator | 外側が所有するexactRandom。globalrandomと区別 |
| _validate_sampling_request | draw前全検査、参加予定collectiontupleを返す |
| _validate_model_id | bool以外のbuiltinint検査。ID符号を勝手に変えない |
| _validate_observed_training_sample | 一標本のshape/layout/finiteと列内Dを検査しDを返す |
| training_sample / training_sample_index | 一観測record/collection内0始まり位置 |
| model_training_samples / collection_index | 一model別collection/外側0始まり位置 |
| eligible_model_training_samples / sampled_model_training_batches | 検証後参加collection tuple/返却batch列 |
| sampled_training_samples | Random.sample結果の抽出順sample列 |
| input_feature_count / expected_input_feature_count | feature列数/同collectionの期待列数 |
| seen_model_ids / tensor_name / training_tensor | ID重複検査/入力field位置名/featureまたはlabel検査Tensor |
| run_legacy_training_batch_sampling | test-only。実旧抽出methodのoracle seam |
| build_sampling_oracle_inputs | test-only。同順population/保有ID/借用Randomと旧clientを準備 |
| assert_sampled_batches_equal | test-only。ID/順序/全features/labelsの完全比較 |
| capture_training_sample_inputs | test-only。tuple/record/Tensor参照・値/gradの独立snapshot |
| test_training_batch_sampling_matches_legacy | 条件別実旧出力/終端RNGの複数call完全照合 |
| test_training_batch_sampling_observes_draw_order | 参加順の一回draw/skip未drawを実wrap観測 |
| test_training_batch_sampling_rejects_before_random_draw | 入力違反でdrawゼロ/RNGと全入力不変 |
| test_training_batch_sampling_skips_ineligible_payloads | 未保有/不足の内容未検査と検査範囲 |
| test_training_batch_sampling_preserves_borrowed_inputs_and_environment | 入力/grad/非contiguous/ambient/RNG/出力storage |
| test_training_sample_records_are_frozen_and_explicit | 三recordのfrozen/kwonly/defaultなし |
| test_training_batch_sampling_connects_fifo_and_joint_update | 上位位置解決/ID対応をtest-onlyで接続し実旧学習へ比較 |
| input_contract_case / invalid_inputs / input_snapshot_before_sampling | test-only。不正条件/要求辞書/事前全入力snapshot |
| expected_random_state / actual_random_state / global_python_random_state | 比較用終端RNG/実終端RNG/復元用外側Pythonstate |
| global_torch_random_state / global_numpy_random_state / original_default_dtype | 環境保持/復元のsnapshot |
| sampling_record_field / parameter / model_module | dataclasses.Field/NN Parameter/NN Moduleを混同しない |
| legacy_client / legacy_batches / sampled_batches / sampled_batch | test-only旧client/旧出力/新出力列/一batch |
| observed_samples_by_index / released_sample_indices / pending_assignment_buffer | 上位test-only観測解決dict/FIFO解放位置/既存buffer |
| classifiers_by_model_id / optimizers_by_model_id / participating_training_batches | 上位test-onlyNN/optimizer対応dict/既存joint入力tuple |
| expected_joint_loss / actual_joint_loss / optimizer_settings / parameter_name | 旧/新loss数値/既存optimizer条件/NN Parameter名 |

小さな一時変数はこれらの役割群で明確な名前を使う。新公開名や責務追加はrevision再レビュー後に実装する。
