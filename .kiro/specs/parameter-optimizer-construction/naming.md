# 命名: パラメータoptimizer生成
revision: 1

## ファイル・型
learning/training/parameter_optimizer_settings.pyは方式別条件、parameter_optimizer_construction.pyは標準optimizer生成。
AdamParameterOptimizerSettingsはAdam専用条件、SgdParameterOptimizerSettingsはSGD学習率だけ。LocalTrainingSettingsの更新/勾配統合方針とは区別する。
ParameterはPyTorchの既存型。独自Optimizer型/registryは作らない。

## 公開名と状態
| 名称・型 | 役割・単位・更新 |
|---|---|
| learning_rate: float | 非負有限更新係数、構築時固定、builtin intも可 |
| weight_decay: float | Adam L2係数、非負有限、固定 |
| adam_variant: str | standard/amsgrad、固定。SGDにはない |
| create_parameter_optimizer | 検証済み入力から新標準Optimizerを返す、入力更新なし |
| parameters: tuple[Parameter,...] | 上位が確定した順序、同参照維持 |
| optimizer_settings | 二つのexact設定型、コピー検証、入力不変 |
| _validate_optimizer_settings | 公開fieldコピー再検証を返す、入力/RNG不変 |
| _validate_optimizer_parameters | 全Parameter契約と重複の生成前検査、None、入力不変 |
| __post_init__ | dataclassの標準検査入口、設定を書き換えない |

## 局所名
validated_optimizer_settingsは再検証コピー。parameter_indexは列内位置、parameterはParameter実体、parameter_idsは重複検査用id集合。
configuration_parameter_nameは設定field名、specified_parameter_valueは型検査する元の数値、validation_errorは文脈付例外。
optimizer/legacy_optimizerは新/旧optimizer実体。actual/expected/originalは既存test同役割。
global_*は保存した共有環境、parameter_snapshot_before_callは値保存、gradient_snapshot_before_callはgrad保存。input_parametersはtest用列。
学習進行のstep_count/epochなど永続状態を新設しない。

## test-only
test_parameter_optimizer_construction.py:
- test_parameter_optimizer_matches_legacy: actual旧builderと条件/state/複数step照合。
- test_parameter_optimizer_rejects_invalid_inputs_without_mutation: 不正設定/Parameterを生成前拒否、入力/grad不変。
- test_parameter_optimizer_preserves_random_state_and_tensor_environment: CPU/Python/NumPy/default/grad保持。
- test_parameter_optimizer_uses_separate_model_parameter_groups: 新NNの共有/個別列が非重複、モデル無変更。
- test_parameter_optimizer_settings_are_immutable_and_explicit: frozen/keyword/defaultなし/SGD非Adamfield。
- test_parameter_optimizer_state_is_independent: 別生成のstate独立。
旧builderのtest-only呼出はhelperを新設せずtest内で行う。state比较は既存test慣例の局所名/再帰比較を必要なら承認後使用する。
