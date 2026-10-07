# 命名と役割 — revision3

## 配置と公開契約

| 名前 | 役割・型・単位・更新 |
| --- | --- |
| learning/training/candidate_epoch_training_settings.py / CandidateEpochTrainingSettings | 候補の反復学習だけのfrozen/kw_only設定。採否設定と分離 |
| candidate_training_strategy | str: fixed_epoch_training / validation_loss_early_stopping / skip_training。旧fixed/early_stopping/none対応。旧名aliasなし |
| maximum_epoch_count | int>=0、エポック上限。0も旧FedDrift経路のため保持 |
| maximum_batch_sample_count | int>=1、1batchの標本上限。平時のbatch設定とは独立 |
| validation_sample_fraction | 注釈float、builtin int/floatを検査（bool不可）、0より大きく1未満。検証標本への割合 |
| consecutive_non_improving_epoch_limit | int>=1、連続非改善で停止する件数。patienceの意味を明記 |
| minimum_validation_loss_decrease | 注釈float、有限builtin int/float>=0（bool不可）、改善と認める厳密な損失減少の閾値 |
| learning/training/candidate_epoch_training.py / train_candidate_classifier_epochs | 借用の候補・二つのoptimizer管理器・区間tensor・設定を受け、更新と学習量recordを返す |
| CandidateEpochTrainingResult | frozen/kw_only。学習終了時の量だけを返し候補/ownerを二重所有しない |
| completed_epoch_count | 実行したepoch数。最良epoch indexではない |
| candidate_trained_sample_count | 延べ学習標本数、反復による重複を含む。既存採用APIへ渡す同義名 |
| candidate_parameter_update_step_count | optimizer二つを同時更新したbatch回数。optimizer個数で倍にしない |
| validation_evaluated_sample_count | 延べ検証標本数、学習量から分離 |
| candidate_classifier | ResidualAdapterClassifier、更新する独立候補 |
| candidate_shared_parameter_optimizer_state | ParameterOptimizerState、候補自身の共有部に結合 |
| candidate_concept_specific_parameter_optimizer_state | ParameterOptimizerState、adapter→分類層順に結合 |
| input_features / observed_class_labels | CPU float32 [N,F]/[N,1]、借用区間を変更しない |
| candidate_epoch_training_settings | 上記設定を受ける引数。学習率はoptimizer設定へ一度だけ指定 |

## 内部処理と一時値（実装前にレビューする）

- `_validate_candidate_epoch_training_inputs`: 更新・乱数消費前の全入力契約検査。返却なし。
- `_train_candidate_dataset_epochs`: 一つのTensorDatasetとエポック数に対するDataLoader反復。固定方式と早期停止の1epochから再利用。
- `training_dataset`, `epoch_count`, `maximum_batch_sample_count`, `training_data_loader`: helperの標本集合/指定回数/batch上限/旧と同じDataLoader。
- `local_training_settings`, `participating_training_batch`, `batch_input_features`, `batch_observed_class_labels`: 既存単回共同更新に渡す固定設定・単一参加record・batch値。
- `shared_parameters`, `concept_specific_parameters`, `parameter_optimizer_state`, `expected_parameters`, `optimizer_parameters`, `parameter_optimizer`, `classifier_parameter`, `training_tensor`, `tensor_name`, `expected_shape`: 事前検査の対象と結合順の期待値。
- `sample_count`, `validation_sample_count`, `shuffled_sample_indices`, `validation_sample_indices`, `training_sample_indices`: 区間件数と一回の無作為分割、位置は区間内の0始まり。
- `validation_input_features`, `validation_observed_class_labels`, `validation_loss`, `best_validation_loss`, `best_candidate_parameter_snapshot`, `consecutive_non_improving_epoch_count`: 検証batchと比較用損失・最良値・非改善の連続数。
- `epoch_training_result`, `completed_epoch_count`, `candidate_trained_sample_count`, `candidate_parameter_update_step_count`, `validation_evaluated_sample_count`: helper結果と呼出し内の累積量。外部診断ownerを更新しない。

## テスト名

- `tests/refactoring/test_candidate_epoch_training.py`: 実旧メソッドへの数値/乱数/状態/学習量対照と拒否。
- `build_candidate_epoch_training_oracle`: 既存生成oracleを使い、新旧候補と旧clientを用意。
- `assert_candidate_epoch_training_matches_legacy`: 全値・grad・optimizer・学習量・RNGを照合。
- `test_candidate_epoch_training_matches_legacy`: 方式/class/optimizer/epoch/小区間/batch端数の正常対照。
- `test_candidate_epoch_training_restores_only_best_parameters`: 最良parameterのみ復元し、その後の更新も実旧に一致。
- `test_candidate_epoch_training_rejects_before_mutation`: 不正値拒否と全借用状態/RNG不変。
- `test_candidate_epoch_training_settings_contract`: 設定必須field/値域/frozen/kw_only。
- `test_candidate_epoch_training_dependency_contract`: exact import guardへの注入契約。
- テスト準備・比較の名前は既存生成oracleの名前を同義で再利用する。追加する名前はtask着手前に列挙しレビューする。

## Task1のテスト準備

`valid_settings_arguments`は全6項目の正常constructor引数dict、`invalid_settings_field_name`/`invalid_settings_field_value`は拒否条件の項目/値、`settings_argument_values`は一条件だけ変更した引数dict。既存dataclasses.fields、FrozenInstanceErrorとpytest.raisesで必須field/frozen/kw_onlyを確認する。`configuration_parameter_name`/`specified_parameter_value`は既存設定エラーの同義field。
