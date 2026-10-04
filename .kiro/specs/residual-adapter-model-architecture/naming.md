# 命名: Residual Adapterモデル構造
revision: 2

## 1. ファイル・型
基点: src/federated_learning_experiments/learning/models/。
| ファイル / 型 | 役割 |
|---|---|
| shared_feature_extractor.py / SharedFeatureExtractor | 全概念で共有する特徴抽出 |
| nonlinear_residual_adapter.py / NonlinearResidualAdapter | 圧縮/ReLU/展開の非線形残差 |
| residual_adapter_classifier.py / ResidualAdapterClassifier | 共有特徴＋概念別adapter＋分類層 |
FeatureExtractorは入力から共有特徴、Adapterは共有特徴から同寸法の適応後特徴、Classifierは分類出力。手法名や学習処理とは区別する。

## 2. 公開引数・状態
| 名称 / 型 | 意味・単位・更新 |
|---|---|
| model_architecture_settings: ModelArchitectureSettings | 既存の構造名と要求rank。コピー検証、入力不変 |
| input_feature_count: int | 入力ベクトルの要素数、構築後固定 |
| hidden_layer_widths: tuple[int,...] | 幅の順序、空はidentity、固定 |
| output_feature_count: int | 抽出後の要素数、末尾幅または入力数、固定 |
| feature_count: int | adapterの入出力特徴数、固定 |
| requested_rank / effective_rank: int | 要求値と特徴数で制限した実効値、固定 |
| class_count: int | 分類対象のクラス数（出力列数とは区別）、固定 |
| shared_feature_extractor: SharedFeatureExtractor\|None | 注入された共有部、コピーせず参照 |
| hidden_layers: Sequential | extractorのLinear/ReLU列、学習時Parameter更新可 |
| feature_compression / feature_expansion: Linear | 残差の圧縮/展開、Parameter更新可 |
| activation: ReLU | adapter内の非線形性、状態更新なし |
| feature_extractor / residual_adapter | classifier所有部品、抽出部だけ共有可 |
| classification_layer: Linear | 適応後特徴から分類スコア、Parameter更新可 |
| output_activation: Sigmoid\|Identity | binary確率/多クラスlogits経路、更新なし |
| input_features / shared_features: Tensor | 入力/抽出後、CPU32 vector/batch、入力不変 |

## 3. 関数
| 名称 | 入出力・役割・状態 |
|---|---|
| validate_feature_dimensions | extractorの正整数/tuple検査。None、状態/RNG不変 |
| validate_structure | 注入されたextractorの宣言/層/Parameter検査。None、状態/RNG不変 |
| forward | 各nn.Moduleの標準演算入口。Tensor→Tensor、更新/RNGなし、grad保持 |
| extract_shared_features | classifierから共有部だけ適用。入力→共有特徴 |
| forward_from_shared_features | 共有特徴→adapter→分類出力。再抽出なし |
| _validate_classifier_inputs | 設定コピー/寸法/class/共有構造を全初期化前に検査 |
| _validate_shared_feature_extractor | classifierの注入型とvalidate_structureへの接続 |
| _validate_feature_tensor | 各単体部品のCPU32/strided/rank/幅検査。小さい局所helper |
| __init__ / __post_init__ | Python/PyTorch既知の初期化名。後者は既存設定のみ |
private helperは必要なときだけ導入し、共有汎用validatorやfactoryへ拡張しない。

## 4. 局所名
validated_model_architecture_settingsは再検証済みコピー。
hidden_layers/previous_feature_count/hidden_layer_widthは層構築、classification_output_countは実際の出力列数。
adapted_features/residual_features/classification_scoresは適応後特徴/加算する残差/activation前出力。
feature_extractor_parameter/expected_layer_count/layer_index/linear_layer/relu_layer/expected_feature_countは構造検査。
validation_errorは例外。各数の単位は要素/層/index、その他Tensorは無単位。入力を更新しない。
同意味の標準nn.Module名self、parameter、weight、bias、shape、dtype、deviceは使用可。

## 5. test-only
tests/refactoring/test_residual_adapter_model_architecture.py。
- build_legacy_residual_adapter_classifier: 実旧構築、optimizerだけno-op。
- map_residual_adapter_state_to_legacy_keys: test-only prefix変換（新readerではない）。
- assert_residual_adapter_state_matches_legacy: 全state tensor直接比較。
- capture_torch_random_state_after_model_construction: 共通開始状態から構築後CPU RNG取得。
- test_residual_adapter_classifier_matches_legacy: 構造/出力/初期値照合。
- test_residual_adapter_classifier_shares_only_feature_extractor: object/storage共有・独立。
- test_residual_adapter_model_rejects_invalid_inputs_without_consuming_randomness: 事前拒否。
- test_residual_adapter_model_preserves_default_tensor_environment: 既定型/device/grad/RNG保持。
Task2のtest名は開始前に追加レビューする。fixtureのseed/旧参照/比較対象など局所名は意味が同じ既存test慣例を再利用する。

## 6. Task2 test-only命名追加（revision2）
production名・役割は変更しない。
| test名 | 観測する契約 |
|---|---|
| test_residual_adapter_model_gradients_match_legacy | zero/nonzero展開で入力・全Parameter.gradを実旧へ照合 |
| test_residual_adapter_model_accepts_empty_batches | 三部品の0行shape、旧との一致 |
| test_residual_adapter_classifier_reuses_extracted_shared_features | 抽出済み経路は再抽出せず通常forwardと同じ |
| test_residual_adapter_classifier_connects_to_probability_calculations | 実forwardから既存確率/観測後meanlossへ接続 |
| test_residual_adapter_classifier_loads_selected_initial_parameters | 既存snapshot選択から独立model標準load_state_dictへ接続 |
既承認拒否/共有/環境testは条件別suffixを付けて再利用可。
parameter_snapshot_before_call、input_features_before_call、shared_features_before_callは値不変比較用detach clone。
legacy_input_featuresは旧勾配用の同値独立入力。parameter_gradients_by_name/legacy_parameter_gradients_by_nameは名前別.grad。
use_nonzero_expansion_weightsはtest-only boolで非zero展開case（production設定ではない）。
source_classifiers_by_model_idはsnapshot供給側NN、initialized_classifierは標準load先の独立NN。
constructor_arguments/invalid_value/case/original/actual/expected/classifier/repeated_classifier/legacy_classifier/feature_extractor/adapterは既存testと同じ役割で使用可。
その他は既存API引数名、model_id/parameter_name、共有環境global_*を同役割で使う。新helper/classは不要。
