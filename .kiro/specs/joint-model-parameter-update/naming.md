# 命名: 一回の共同モデルパラメータ更新
revision 4。正式API名に旧aliasなし。型/参照は借用、更新は関数にだけ集約する。

| 名前 | 役割・型・単位・状態 |
|---|---|
| participating_model_training_batch.py | 一分類器の参加batch記録、選別や抽出を行わない |
| ParticipatingModelTrainingBatch | frozen/kw_onlyの分類器・optimizer・特徴・ラベル参照記録 |
| classifier | ResidualAdapterClassifier。共有部を参照し個別部を所有するNN |
| concept_specific_parameter_optimizer | adapterと分類層のParameter列だけを更新する標準optimizer |
| input_features | CPUfloat32[N,D]の観測入力。shared_featuresと区別 |
| observed_class_labels | CPUfloat32[N,1]。二値softtargetまたは多クラス整数値 |
| joint_model_parameter_update.py | 参加列全体に対する一回更新の責務 |
| perform_joint_model_parameter_update | float共同loss/空列Noneを返し、borrowed grad/value/stateを更新 |
| local_training_settings | 既存正式方式の条件。公開fieldコピーで検証 |
| shared_feature_extractor | 全参加者が同一参照を使う特徴抽出NN |
| shared_parameter_optimizer | 共有Parameterを更新するoptimizer。空共有のみNone |
| participating_training_batches | exacttupleの参加列。この順序が集約とstep順 |
| update_shared_features | exactbool。共有forwardgrad/共有stepだけの有効可否 |
| _validate_joint_update_inputs | zero_grad前の全検査。空列の詳細検査を省略 |
| _validate_training_tensor | CPU32/layout/shape/finiteの入力検査 |
| _validate_parameter_optimizer | 期待Parameter列とoptimizerのid/順序を検査 |
| _validate_trainable_parameters | Parameter型/device/layout/必要なrequires_gradを検査 |
| validated_local_training_settings | public fields再検証用の一時copy |
| shared_parameters / concept_specific_parameters | 共有/個別Parameter tuple。NNmoduleと区別 |
| parameter_optimizer / expected_parameters / optimizer_parameters | 検査対象optimizer/期待列/全group平坦化列 |
| training_batch / training_batch_index | 一参加記録と0始まり位置。モデルIDではない |
| sample_count / total_sample_count | 一batch/全体の標本数、正整数 |
| combined_input_features / combined_shared_features | 入力順catの入力/一回forwardの結果 |
| shared_features / feature_offset | 参加者ごとのsliceと開始行index |
| classifier_predictions / target_class_indices | 二値確率または多クラスlogits/CE用int64vector |
| model_loss / weighted_model_losses / joint_loss | mean Tensor/標本数倍Tensor列/全体加重meanTensor |
| seen_classifier_ids / seen_optimizer_ids / seen_parameter_ids | 一呼出の重複検査用object id集合。永続登録台帳ではない |
| require_trainable / tensor_name / expected_shape / parameter / parameter_index | 検査局所の学習可否/field名/shape/Parameterと位置 |
| validation_error | field再検証時の例外 |
| build_joint_update_oracle_pair | test-only同値新旧モデルと実optimizerの準備 |
| run_legacy_joint_update | 実旧method一回呼出のtest-only seam |
| assert_joint_update_states_equal | 新旧全Parameter/grad/optimizerstate直接比較 |
| legacy_client / legacy_models / new_classifiers | test-only旧client/旧model列/新NN列 |
| operation_events / forward_call_count | zero/step入力順/共有forward回数の観測 |
| shared_feature_parameters / concept_specific_parameter_sequences | test fixtureの共有tuple/参加別個別tuple |
| optimizer_settings / optimizer_variant / class_count / batch_sample_counts | test生成条件。標本数列は参加順 |
| snapshot_before_update / expected_joint_loss / actual_joint_loss | 拒否前deep状態/旧捕捉loss/新返却loss |
| assert_nested_state_equal | state_dictのTensor値/primitive/list/dictを比較 |
| parameter_name / parameter_value / old_parameter_name | test-onlystate名対応とtensor。moduleとは区別 |
| model_module_name / model_module | named_modulesの名前/NNmodule。Parameterではない |
| test_joint_model_parameter_update_matches_legacy | test-only。条件別の実旧共同更新・複数step全state/gradの比較 |
| test_joint_model_parameter_update_observes_operation_order | test-only。zero/forward/backward/stepの呼出順と回数の観測 |
| test_joint_model_parameter_update_rejects_invalid_inputs_before_mutation | 不正条件を更新前拒否し、全値/grad/state不変を検証 |
| test_joint_model_parameter_update_empty_participation_preserves_state | 空列Noneで既存grad/stateを触らないことを検証 |
| test_joint_model_parameter_update_without_shared_parameters | 空幅identityと共有optimizerなしの個別更新を独立検証 |
| test_joint_model_parameter_update_preserves_environment_and_borrowed_inputs | 入力参照/値・RNG・ambient contextを検証 |
| test_joint_model_parameter_update_accepts_soft_binary_labels_and_noncontiguous_batches | 二値soft targetと非連続メモリを実旧比較 |
| test_joint_model_parameter_update_accepts_ordered_multiple_optimizer_groups | param_groups平坦列の一致を受理して更新を検証 |
| test_joint_model_parameter_update_preserves_frozen_shared_optimizer_state | warm stateを持つ共有optimizerのstep省略とgradclearを検証 |
| test_participating_model_training_batch_is_frozen_and_explicit | frozen/kw_only/defaultなしの記録契約 |
| capture_joint_update_state | test-only。全Parameter値/gradとoptimizerstateの独立snapshot |
| input_contract_case / invalid_inputs / snapshot_after_update | test-only。不正条件分類/入力辞書/実行後snapshot |
| global_torch_random_state / global_numpy_random_state / global_python_random_state | test-only。外側RNG復元/照合用状態 |
| original_default_dtype / parameter_groups / shared_parameters_before_update | test-only。ambientdtype/group列/凍結前共有値 |
| training_tensor | _validate_training_tensorの検査対象Tensor。入力特徴・ラベルどちらにも使うためinput_featuresと区別 |
| training_batch_field | test-only。dataclasses.fieldsで得る記録のField。NN Parameterと混同しない |

小さい一時変数はこの役割群の明確な名前を用いる。公開名/新状態/責務を追加する場合は命名revisionを再レビューする。
