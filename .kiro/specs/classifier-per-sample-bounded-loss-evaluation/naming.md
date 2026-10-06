# 命名: 準備済み分類器の標本別有界損失

revision: 1。承認状態はspec.jsonを参照する。

## ファイル・型
|名称|役割・関連名との違い|
|---|---|
|learning/prediction/classifier_bounded_loss_evaluation.py|一分類器を呼び、標本別有界損失を返す。予測済み確率の平均計算とは入力境界が異なる|
|tests/refactoring/test_classifier_bounded_loss_evaluation.py|実旧損失/登録統計対照、拒否/副作用/接続|
|ResidualAdapterClassifier / Tensor|既存型を再利用。新record/Protocol/設定型は導入しない|

## 関数
|名称|入力→出力・役割|状態更新|
|---|---|---|
|evaluate_classifier_per_sample_bounded_losses|classifierとinput_features、observed_class_labels→Tensor[N]。一forwardで観測済みラベルへの有界誤差|なし、勾配記録なし|
|_validate_classifier_bounded_loss_inputs|同じkeyword入力→None。モデル/parameter/非空batchをforward前に確認|なし|
|_validate_classifier_outputs_for_bounded_loss|classifier_outputs、batch_sample_count、class_count→None。forward結果の型/shape/環境/有限/二値範囲を確認|なし|

per_sampleは平均にしないこと、boundedは学習BCE/CEと異なる0〜1誤差、classifierは実モデルを呼ぶことを表す。predict_*には観測ラベルを渡さない方針との区別をevaluateで表す。

## 引数・局所変数
|名称|型・単位|意味・更新|
|---|---|---|
|classifier|ResidualAdapterClassifier|準備済み参照。clone/登録/resetなし|
|input_features|Tensor[N,F]|観測batchの特徴、CPU float32、入力不変|
|observed_class_labels|Tensor[N,1]|観測済み正解クラス、CPU float32、入力不変|
|class_count / batch_sample_count|int、クラス/標本数|モデルのクラス数/入力行数|
|classifier_parameter|Parameter|検査中の既存parameter参照、変更なし|
|classifier_outputs|Tensor[N,1]または[N,K]|二値確率または多クラスlogit。forward一回の結果|
|class_probabilities|Tensor[N,K]|多クラスsoftmax結果。二値branchでは生成しない|
|correct_class_probabilities|Tensor[N]|正解ラベル列をgatherした結果|
|expected_output_count|int、出力列数|二値1、多クラスclass_count|

標準Tensor属性・self・parameter_name・validation_errorは同じ意味で使用可。永続する可変状態は導入しない。

## test-only
- build_bounded_loss_oracle_pair: 既存test helperで実旧モデルと新classifierの同値parameterを準備。
- assert_initial_loss_statistics_match_legacy: 新損失→既存初期統計と実旧登録の全fieldを比較。
- test_classifier_bounded_losses_match_legacy: binary/multiclass、batch/singleton、非連続、training flagと既存grad保持。
- test_bounded_loss_evaluation_rejects_invalid_inputs_before_forward: 不正入力をforward計数0で拒否。
- test_bounded_loss_evaluation_rejects_invalid_classifier_parameters: dtype/device不正parameter拒否。
- test_bounded_loss_evaluation_rejects_invalid_outputs: 戻り値だけを置換するtest hookで出力違反を検出。
- test_bounded_loss_evaluation_preserves_environment_and_independent_results: default dtype/device、RNG、grad有効設定、入力/parameter/grad保持と結果独立。
- test_bounded_loss_evaluation_connects_prepared_model_to_initial_statistics: 学習→登録準備→評価→初期統計のtest-only接続。
- test_classifier_bounded_loss_evaluation_dependency_contract: exact許可/禁止symbol注入。

局所test名は同意味の既存慣例（class_count, sample_count, invalid_value, monkeypatch, result, legacy_client, training_batches, state_before, forward_calls）を利用し、条件別suffixを使用可。一時smokeは同じ公開APIのみを呼ぶ。
