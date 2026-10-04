# 設計: パラメータoptimizerの設定と生成

## Overview
モデル構造からoptimizer条件と生成を分離する。標準PyTorch optimizerを返し、モデルへ属性を追加しない。
Adam専用条件を持つ型と学習率だけのSGD型を分け、既存の型付きfield検証を再利用する。

## Boundary Commitments
### This Spec Owns
設定検証、Parameter列の生成前検査、同参照/順序で新規optimizerを一つ生成すること。
### Out of Boundary
データ別lr解決、optimizer attach/共有管理/reset/復元、loss/grad計算/step実行、batch/epoch/共同更新/候補進行/新全体run。
### Allowed Dependencies
- parameter_optimizer_settings.py: dataclasses(dataclass,field)、core.settings_field_validationのvalidate_settings_field_valuesだけ。
- parameter_optimizer_construction.py: torch.optim(Optimizer,Adam,SGD)、torch(float32,strided)、torch.nn(Parameter)、同settings moduleのAdamParameterOptimizerSettings/SgdParameterOptimizerSettingsだけ。
exact module/public symbolをASTで固定する。一般stdlib許可より前に2moduleを判定し、__future__.annotationsだけ共通許可。
math/random/NumPy、models/data/runtime/methods/legacy、torch内部/optim内部や他optimizerをimportしない。
### Revalidation Triggers
設定fields・variant・Parameter契約・optimizer既定値・返却所有を変えると後続学習とstate比較を再検証する。

## Architecture
標準Adam/SGDと標準state_dictを採用する。独自Optimizer wrapper/state型/registry/protocolは不要。
runtime/上位が設定型を選び、モデルから順序確定済みParameter tupleを渡す。戻りoptimizerの所有者も上位。

## File Structure Plan
基点src/federated_learning_experiments/learning/training/:
- parameter_optimizer_settings.py: 二つのoptimizer条件の宣言/構築時検証。
- parameter_optimizer_construction.py: 全入力検査後に標準optimizerを生成。
新規tests/refactoring/test_parameter_optimizer_construction.py: 実旧生成/state/stepと拒否・接続。
変更tests/refactoring/test_single_run_dependency_boundaries.py: exact2module/symbolの許可と注入test。
対象specと.kiro/steering/roadmap.mdへ証拠。実証した旧問題だけdocs/research/implementation-findingsへ記録。
既存LocalTrainingSettings/モデル構造/旧production/golden/旧回帰testは変更しない。

## Components and Interfaces
### Settings
frozen=True,kw_only=True、defaultなしのdataclass。
```python
AdamParameterOptimizerSettings(*, learning_rate: float, weight_decay: float, adam_variant: str)
SgdParameterOptimizerSettings(*, learning_rate: float)
```
数値metadata minimum_allowed_value=0,minimum_value_is_inclusive=True。単位はlearning_rate更新係数、weight_decay L2係数。
adam_variantのallowed_parameter_values=(standard,amsgrad)。新しい正式選択名で旧bool/aliasは読まない。
各__post_init__は数値のexact builtin int/floatを先に確認してから既存validate_settings_field_valuesで非負/有限/variantを検証する。bool/NumPy scalar/subclassは拒否する。
必要な小さい型検査のみ局所で実装し、共有validatorの受理範囲を変えない。設定値をfloatへ丸めず、入力設定を変更しない。

### Builder
```python
create_parameter_optimizer(
    *, parameters: tuple[Parameter, ...],
    optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
) -> Optimizer
```
設定はexact二型のみ。公開fieldの新dataclassコピーでforged値も再検証し、TypeError/ValueErrorは項目を含むValueErrorへまとめる。
parametersはexacttuple非空、各要素exact torch.nn.Parameter、CPU float32 strided/notnested、id重複無し。入力値の有限性・grad値は生成契約へ追加しない。
requires_grad=False、grad=None、noncontiguous、scalar/0要素shapeは受理（CPU32 stridedのまま）。外部モデルからParameterの意味を推論しない。
全検査後Adam(parameters,lr=learning_rate,weight_decay=weight_decay,amsgrad=adam_variant=="amsgrad")またはSGD(parameters,lr=learning_rate)だけを呼ぶ。
betas/eps/momentum/foreach/fusedなどを独自に指定しない。基準環境の旧と同じconstructor defaultを使う。
上位の共有Parameter集合が空なら呼び出さない判断は後続。builderは空tupleを拒否する。
生成時のParameter順/参照/gradを保ち、標準optimizerを返す。各呼び出しは独立state、同Parameterを別呼出しへ渡すグローバル重複検査は行わない。

## Requirements Traceability
| 条件 | 設計/検証 |
|---|---|
| 1.1 | Adam設定と旧builder/defaults/variant |
| 1.2 | SGD型はlrのみ、旧momentum/decayなし |
| 1.3 | exact参照順、空独立state、Parameter/grad不変 |
| 2.1 | exact設定コピー/metadata/builtin型 |
| 2.2 | tuple/Parameter/device/dtype/layout/重複全検査 |
| 2.3 | RNG/defaultdtype/meta/grad保持 |
| 3.1 | 実旧builder全groups/複数step/state照合 |
| 3.2 | 新NNから抽出/個別集合をtest-only選択 |
| 3.3 | exact依存/fresh boot/fullgolden/証拠 |

## Error Handling・状態
不正入力は位置またはfield名を含むValueErrorで、optimizer constructor以前に拒否。外部Parameter/grad/設定に書込まない。
標準Optimizer内部stateの所有/更新は返却先とPyTorch。生成関数はstate load、zero_grad、step、backwardを呼ばない。
Python/NumPy/CPU RNG、defaultdtype/device/gradflagを変更しない。

## Testing Strategy
Task1は対象test先行missing-module RED→GREEN。実旧SharedBackboneMLP._build_component_optimizerを呼び、Adam標準/AMSGrad、decay有無、lr0、SGDのgroups/defaults/state空と参照順を比較する。
test-onlyで同値独立Parameterと明示勾配（grad=Noneも）を複数stepし、全値/全stateを直接比較する。旧SGDはconfigのAdam条件を無視することも確認。
Task2はtest-onlyの不正型/forged/tuple/Parameter/dtype/device/layout/重複/後段不正を事前拒否、入力gradとRNG保持、defaultfloat64/meta/gradcontext保持、frozen/defaultなし/独立stateを検証。
新ResidualAdapterClassifierの共有extractorとadapter→classifierの列を選んで二optimizerを生成し、参照集合の非重複/モデルにoptimizer追加なしを確認する。学習進行APIを追加しない。
Task3はexact依存禁止/許可注入実RED→GREEN、旧importなしCPUfresh生成/外部step smoke、全tests（旧11/最終3golden含む）、旧748c3aa無差分、UTF-8/hash/配置/境界を確認する。
環境はdocs/experiments/refactoring-baseline.md。全体runの移植完了とは区別する。
