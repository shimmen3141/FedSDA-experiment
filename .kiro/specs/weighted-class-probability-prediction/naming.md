# 命名正本

revision: 1。承認状態とLF SHA256はspec.json。旧routing/score_loss等はoracleだけ。永続状態・互換aliasなし。

## ファイルとpublic関数

| 名前 | 意味と入出力 |
|---|---|
| learning/prediction/class_probability_calculations.py | 分類確率と予測スコアのstateless計算。torch所在を明示 |
| convert_model_outputs_to_prediction_probabilities | rawモデル別出力とclass_count→モデル別確率。binary再sigmoidなし/multi softmax |
| normalize_model_prediction_weights | モデル別重み→昇順の正規化済copy。状態更新なし |
| combine_model_prediction_probabilities | モデル別確率・重み・class_count→混合Tensor。重み再正規化なし |
| predict_class_labels_from_prediction_scores | 有限混合スコアとclass_count→[N,1]float32クラス値。scoreは厳密確率域を要求しない |
| compute_model_mean_bounded_losses_after_label_observation | モデル別確率・観測ラベル・class_count→モデルID別float平均損失 |

## private補助・引数・局所名

| 名前 | 役割・型/単位 |
|---|---|
| _validate_class_count | builtin int≥2を検査、返却None |
| _validate_prediction_tensor | prediction_tensor/class_count/require_probabilitiesで形・配置・精度・有限値、必要なら確率制約を検査 |
| _validated_model_prediction_tensor_ids | model_tensors_by_model_id/class_count/require_probabilities→検証済昇順tuple[int,...]。共通N確認 |
| _validated_model_prediction_weights | prediction_weights_by_model_id→確率検証copy、補正しない |
| _validate_observed_class_labels | observed_class_labels/batch_sample_count/class_count→ラベル形・整数域・配置検査、None |
| class_count / batch_sample_count | クラス数 / 呼出しの標本数、個・sample |
| model_outputs_by_model_id | Mapping[int,Tensor]、binary確率/multi logit |
| prediction_probabilities_by_model_id | Mapping[int,Tensor]、binaryクラス1確率[N,1]/multi全確率[N,K] |
| prediction_weights_by_model_id / normalized_prediction_weights_by_model_id | 元重み / 正規化済snapshot、確率・無次元 |
| prediction_scores / combined_prediction_probabilities | クラス判定入力 / 混合結果、Tensor。微小丸めを補正しない |
| prediction_tensor / model_tensors_by_model_id | private検査の個別Tensor / 同じ表現のMapping |
| require_probabilities | 確率値域・多クラス行総和を検査するbool。有限スコアではFalse |
| observed_class_labels / predicted_class_labels | ラベル観測後入力 / 予測前返却、Tensorクラス番号 |
| observed_losses_by_model_id | モデル別平均有界損失、dict[int,float]、無次元 |
| model_ids / model_id / prediction_weight / total_prediction_weight | 昇順tuple / 個別ID / 重み値 / 検証または正規化総和 |

似た名の違い: model_outputsはprob/logitがクラス数で異なる。prediction_probabilitiesは混合前に確率化済み、prediction_scoresは分類だけに使う有限値。weightはモデル間、probabilityはクラスに関する値。

## 検証の名前

ファイル`test_class_probability_calculations.py`。関数は`test_class_probability_`+以下の役割句。

- normalization_matches_reference_without_input_mutation
- model_outputs_become_probabilities_before_combination
- combination_matches_reference_in_sorted_model_order
- class_predictions_preserve_threshold_and_tie_rules
- observed_model_losses_match_reference
- invalid_inputs_are_rejected_without_mutation
- outputs_are_independent_and_have_no_gradient_history
- fixed_share_prediction_and_observation_match_reference
- functions_require_explicit_keyword_arguments
- numeric_calls_preserve_caller_random_states

局所: 上記入力名、`reference_prediction_mixin`（旧static）、`reference_router`、`controller`、`prediction_combination_settings`、`reference_prediction_weights`、`reference_prediction_probabilities`、`reference_predicted_class_labels`、`reference_observed_losses`、`reference_prediction_scores`、`input_tensors_before_call`、`controller_state_before_call`、`invalid_input_name`、`invalid_input_value`、`prediction_operation`（検証対象関数）、`operation_name`、`expected_exception_type`、`observation_index`（0始まり）、`global_python_random_state`、`global_numpy_random_state`、`global_torch_random_state`、`exception_info`、`caught_exception`。reference接頭辞は旧oracleの値だけに使う。

既存`valid_run_settings_mapping`fixture、`capture_fixed_share_controller_state`・`assert_fixed_share_state_matches_reference`、AST checkerの名前を再利用する。新たな意味の名前は本書へ追記しLunaレビュー後に実装する。
