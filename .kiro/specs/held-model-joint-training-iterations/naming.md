# 命名 revision 2

正式実装名の正本。要求承認後に作成し、Lunaレビュー前は未承認。
借用入力の実体を生成・所有しない。単位は標本数と共同更新の反復回数を区別する。

| 名前 | 役割・型・状態・類似名との違い |
|---|---|
| held_model_training_binding.py | 保有モデルIDと学習実体の対応宣言だけを置く |
| HeldModelTrainingBinding | frozen/kw_onlyのmodel_id→classifier/個別optimizer対応。標本や更新結果は持たない |
| model_id | builtin int。表示順/配列位置ではなく対応ID |
| classifier | 借用ResidualAdapterClassifier。共有特徴抽出器を含む既存NN参照 |
| concept_specific_parameter_optimizer | 借用Optimizer。共有部optimizerと別、回を跨いでstate継続 |
| held_model_joint_training_iterations.py | 毎回新バッチを抽出する共同学習反復の接続 |
| perform_held_model_joint_training_iterations | 明示回数の抽出→ID対応→共同更新。成功更新前lossのtuple[float,...]を返す |
| requested_joint_update_iteration_count | builtin int>=0。要求した試行回数であり実成功数ではない。回数算出は上位 |
| held_model_training_bindings | tuple[HeldModelTrainingBinding,...]。ID lookup用。参加順はこの列で決めない |
| ordered_model_training_samples | tuple[ModelTrainingSampleCollection,...]。既存samplerと同名で参加/抽出順の基準 |
| batch_sample_count | 正の標本数。samplerへ渡す既存名、反復回数と別 |
| python_random_generator | 借用Random。sampleごとに進み、再生成/restoreしない |
| local_training_settings | 既存LocalTrainingSettingsを共同更新へ渡す |
| shared_feature_extractor | 借用SharedFeatureExtractor、共有forward用。classmethod/ID mapではない |
| shared_parameter_optimizer | 共有部Optimizerか空共有のNone。個別optimizerと区別 |
| update_shared_features | bool。共有更新/共有凍結を切替える既存の一回更新引数 |
| _index_held_model_training_bindings | positive回数の対応表構造/一意ID検査を行いdict[int,HeldModelTrainingBinding]を返す。NN詳細は検査しない |
| bindings_by_model_id | 上記検査済みlookup dict。NN/optimizerは同一参照、参加順には使わない |
| binding_index / binding | 対応表の入力位置/一記録。エラー表示とID重複検査に使用 |
| _joint_update_iteration_index | rangeの未使用位置。先頭_で未使用を明示し標本indexと混同しない |
| sampled_training_batches / sampled_training_batch | samplerから得た順序付きtuple/一記録。ID/特徴/ラベルを持つ |
| participating_training_batches | IDをNN/optimizerへ解決した既存ParticipatingModelTrainingBatch列。新しい順序へ並べ替えない |
| completed_joint_update_losses | list[float]。空回は含まず、実行した共同更新の更新前lossを保持 |
| joint_update_loss | 一回更新のfloat/None。空参加は呼出し前にskipし非空Noneは契約違反 |
| test_held_model_joint_training_iterations.py | 実旧反復・拒否・環境/境界の検証 |

## テスト内の担当局所名
build_training_iteration_oracle_pair: 既存同初期NN/optimizer helperへ標本列・逆順binding・実旧sampler観測を接続。
legacy_client / training_batches / shared_parameter_optimizer: 既存helperの借用状態を引き継ぐ。
sampled_batch_history: 各回の(model_id,features,labels)をcloneして抽出順を観測する列。
legacy_weighted_loss_history: 実旧loss hookの標本数加重loss列。反復ごとに集計する。
training_samples / model_training_samples / training_sample_index / model_index: 順序付き標本と入力位置。
iteration_count / batch_size / class_count / optimizer_variant / update_shared_features: parameterizedケース。
global_python_random_state / global_torch_random_state / initial_random_state / expected_random_state: 開始・復元・比較状態。
actual_losses / expected_losses / new_sampled_batch_history / original_sampling_function: 実値と旧oracle/spy参照。
invalid_iteration_count / invalid_bindings / binding_contract_case: 契約外入力ケース。
optimizer_state_before / sample_values_before / classifier_parameters_before / generator_state_before: 変更前snapshot。
sampling_calls / update_calls / original_update_function / failing_update: call順/失敗時保持のspy。
empty_shared_classifier / empty_shared_optimizer: 空共有を持つ新NNと個別optimizer。
record_field / training_binding: frozen/defaultなしの明示参照検査。
これ以外の公開名・永続状態名を追加する場合はrevisionを更新してレビューする。

## テスト関数と補助値
run_legacy_training_iterations: 実旧の_train_heads_togetherを指定回数で一回呼び、各回の実損失・抽出を返す。
test_joint_training_iterations_match_legacy: 実旧反復と新反復の損失/状態/RNG対照。
test_joint_training_iterations_reject_before_sampling: 回数/対応構造・IDの事前拒否。
test_joint_training_iterations_zero_count_ignores_training_inputs: 0回で他入力未参照。
test_joint_training_iterations_skip_ineligible_models: 未保有/不足/空で無操作、未参加payload保持。
test_joint_training_iterations_preserve_environment: 外側RNG/dtype/device/gradmode・標本/設定保持。
test_joint_training_iterations_preserve_completed_updates_on_failure: 後続失敗時の完了更新/RNG非rollback。
test_joint_training_iterations_with_empty_shared_features: 共有optimizer=Noneで正常個別学習。
test_held_model_training_binding_is_frozen_and_explicit: constructor/default/frozenと借用参照。
test_joint_training_iterations_reject_invalid_batch_before_draw: sampler検査の委譲とRNG保持。
total_batch_sample_count / participating_model_count: 一回の損失加重分母/参加モデル件数。
original_sampling_function / original_update_function: spy内で本物の既存公開関数を呼ぶ参照。
loss_hooks / loss_hook: 旧loss_fnの一時観測hook列/一登録。finallyでremoveする。
keyword_arguments: テスト用公開入口の引数dict。productionに汎用設定を導入しない。

## Task 1–2の実装前補完
revision2は以下のtest-local名だけを追加し、公開production名・役割は変えない。
training_iteration_module: spyで実sampler/共同更新を観測する新module参照。
sample_legacy_training_batches: 旧samplerを一回呼び、独立cat出力をcloneして履歴へ保存して返すtest-local helper。
sampling_request: 実sampler呼出しのkeyword引数。入力を作り替えるhookではない。
iteration_index / loss_offset: 旧損失履歴の回位置と参加件数に基づくslice開始位置。
legacy_training_batches: 実旧samplerの一回の(ID,特徴,ラベル)列。
model_id / input_features / observed_class_labels: 既存記録のfieldと同じ意味でtupleを展開する値。
binding / sampled_training_batch / record_field: 既存命名表の単一記録/出力/fieldをそのまま使う。
classifiers_by_model_id / optimizers_by_model_id: testで既存NN/optimizerをIDへ対応付けるdict。productionの所有者ではない。
sample_values_before / expected_optimizer_state / invalid_sample: 既存名に統一した変更前標本/失敗後期待optimizer/拒否すべき標本。
initial_parameters / previous_grad_mode / previous_default_dtype: fresh/ambient保持検証の開始値。
