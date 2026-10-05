# 命名案 revision 3

|種類|名前|役割・入出力・更新・似た概念との差|
|---|---|---|
|module|parameter_optimizer_state.py|固定parameter列の現在optimizerと再生成を管理、factoryとは別|
|型|ParameterOptimizerState|optimizer学習stateの所有者。モデル/parameter本体を所有しない|
|constructor引数|parameters|exact tuple[Parameter,...]、参照/順序借用、空不可|
|constructor引数|optimizer_settings|既存AdamParameterOptimizerSettings/SgdParameterOptimizerSettings、固定frozen条件を借用|
|readonly property|parameter_optimizer|現在のOptimizerを借用で返す。mutable学習状態でsnapshotではない|
|公開method|reset_parameter_optimizer|同parameter/設定から新optimizerへ交換、None、NN値/grad保持|
|状態|_parameters / _optimizer_settings / _parameter_optimizer|借用固定tuple/設定、所有する現在optimizer|
|test module|test_parameter_optimizer_state.py|実旧reset対照と実NN接続|
|test helper|build_optimizer_state_oracle_pair|独立new/oldparameterと実旧resetnamespaceを構築|
|test helper|assert_optimizer_state_matches_legacy|groups/state/parameter/gradのexact照合|
|test一時|optimizer_state / shared_optimizer_state / concept_specific_optimizer_states|単独/共有/概念別の所有者を区別|
|test一時|previous_parameter_optimizer / previous_optimizer_state / previous_gradients / parameter_values_before_reset|旧借用参照/stateのコピー/grad参照/値コピー|
|test一時|input_parameters / legacy_parameters / legacy_model / optimizer_settings / parameter / parameter_index / update_index / invalid_value / reset_case / training_batches / participating_training_batches / shared_parameter_optimizer / legacy_client / classifier / model_id|既存test helperと同じ役割の局所名|
|test一時|python_random_state / torch_random_state / legacy_loss / new_loss / optimizer_variant / class_count / update_shared_features|RNG/旧新loss/方式/クラス数/共有更新可否|
|既存helper|build_joint_update_oracle_pair / run_legacy_joint_update / assert_joint_update_states_equal|実旧共同更新と全NN/grad/optimizerの照合、再実装しない|
test関数はtest_<観測する契約>と命名する。pytest monkeypatchなど既存の明確なfixture名は維持する。
reset後は古いParticipatingModelTrainingBatch/HeldModelTrainingBindingのoptimizer参照も古いまま。新recordへ現在参照を入れる判断は上位で明示する。

test補助名: `legacy_concept_parameters`（旧別headのparameter）、`legacy_optimizer`（旧現在optimizer）、`previous_optimizer_state_dict`（reset前stateのdeepcopy）、`optimizer_state_dict`（比較用state）、`gradient_value`（同勾配値）、`training_batch_index`（参加順）、`previous_training_batches`（reset前の借用記録）、`local_training_settings`（既存学習設定）、`parameter_optimizer_state`（testのmodule参照）、`reset_target`（共有/概念1/全体/なしの操作種別）、`expected_parameter_optimizer`（保持される参照）を使う。

revision2の追加: `OPTIMIZER_SETTINGS_CASES` はtest-onlyのfrozen設定6条件（Adam standard/AMSGrad、weight decay、lr0、SGD）を表すtuple。runtimeの設定registryとは区別する。

revision3の追加: `previous_binding`（reset前の実HeldModelTrainingBinding）、`previous_concept_optimizers`（各概念の旧optimizer参照tuple）、`previous_concept_state_dicts`（各旧stateのdeepcopy tuple）。旧借用参照がresetで交換されないことと別owner独立性を検証する局所名。
