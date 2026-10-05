# 設計: optimizer状態の所有と明示リセット

## Overview / Goals
ParameterOptimizerStateが一つの固定parameter列と設定に対して現在のoptimizerを所有する。
既存factoryを使い、モデルとoptimizerの寿命・reset時機を分離する。

## Boundary Commitments
### This Spec Owns
現在のoptimizer参照、既存factoryによる生成と成功後交換、parameter/固定設定の借用。
### Out of Boundary
NN/parameter生成・共有接続・parameter付け替え・ID管理・学習処理・reset時機・学習率変更・state保存移送・空parameter optimizer。
古いbinding/optimizerの自動更新、外側のfrozen/private回避、並行resetは対象外。
### Allowed Dependencies
__future__.annotations、torch.nn.Parameter/torch.optim.Optimizerの公開型、同階層parameter_optimizer_settingsの公開2型、parameter_optimizer_construction.create_parameter_optimizerだけ。
モデル・sampler・更新executor・グローバルconfig・旧package・random/NumPyには依存しない。
### Revalidation Triggers
reset交換条件・参照寿命・設定/parameter列変更時は共同更新/反復/将来clientの接続を再照合する。

## Architecture / Contracts
constructor(*,parameters:tuple[Parameter,...],optimizer_settings:AdamParameterOptimizerSettings|SgdParameterOptimizerSettings):
既存factoryで検証/生成したあと固定tuple/設定を借用し現在optimizerを保存する。
設定のfrozen回避・モデルのparameter差し替えは通常契約外。現Parameterの不正dtype等はfactory再検証で拒否する。
readonly property parameter_optimizer:Optimizer は現在の可変optimizerを借用で返す。
reset_parameter_optimizer()->None は固定tuple/設定でfactory成功後に現在optimizerを交換する。
factory拒否または生成例外時は現在参照未交換、元state/値/grad保持。
旧参照は同じParameterに結び付いたまま外側に残る。reset後は上位が新参照を再取得してbindingを作り直す。
reset自体は値/gradをクリアしない。学習側がzero_gradを行う既存契約を維持する。
parameter tupleは空不可、CPU float32/exactParameter/重複なし、設定は既存factory契約を継承。独自検査を複製しない。
共有なし構造では共有管理器を作らず上位がNoneを渡す（LEGACY-010を修正しない）。

## File Structure Plan
|パス|変更|責務|
|---|---|---|
|src/federated_learning_experiments/learning/training/parameter_optimizer_state.py|新規|現在optimizerとreset交換の管理|
|tests/refactoring/test_parameter_optimizer_state.py|新規|実旧reset対照/独立性/拒否保持/実NN接続|
|tests/refactoring/test_single_run_dependency_boundaries.py|変更|exact import guardと禁止/許可注入|
|.kiro/specs/parameter-optimizer-state/*|新規|承認/仕様/実測|
|.kiro/steering/roadmap.md|変更|現在地|

## Requirements Traceability / Testing Strategy
|要件|検証|
|---|---|
|1.1–1.3|実旧SharedBackboneMLP.reset_optimizerを最小namespaceで呼びgroups/state/ref順照合、factory型/値拒否|
|2.1|同勾配更新0/2回→reset→再更新、値/grad参照と旧state非継承/新identityを確認|
|2.2|共有と2概念別管理器を実NN共同更新へ接続、頭一つだけ/共有だけのresetを実旧attach/reset相当操作へ照合|
|2.3|学習済みParameterの不正dtypeやfactory生成例外で旧optimizer/state保持|
|2.4|reset後の実旧/新更新全値一致、旧optimizer借用参照/bindingが勝手に交換されない|
|3.1–3.2|RNG不変/独立reset/production無上位import、exactAST RED→GREEN/fresh新CPU smoke/全golden回帰と品質|
Task1はmissingmodule実RED→実装GREEN。Task2はtest-only、実旧数値処理を置換せず、既存実NN共同更新へ現在参照を渡す。
Adam standard/AMSGrad/SGD、class2/4、共有更新/凍結、resetなし/全体/共有だけ/概念一つだけを確認する。
全回帰と源hash/旧固定差分を記録し、旧goldenの確認と新部品同値を区別する。
Windows sandboxのPyrightは共有venv絶対pythonpath+require_escalatedで子Python起動を許可する。
