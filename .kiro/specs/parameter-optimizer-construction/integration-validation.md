# 統合検証: パラメータoptimizer生成

## 完成範囲
方式別Adam/SGDの固定条件と、明示Parameter tupleから標準optimizerを一つ生成する部品を完成した。
CPU float32 strided、同参照/順序、全検査後生成。モデルへattachしない。
lr解決/共有所有/reset/復元/loss/backward/step実行/epoch・batch/共同学習/候補進行/新全体runは後続。

## 9条件と証拠
| 条件 | 観測済み契約 |
|---|---|
| 1.1 | Adam standard/AMSGrad・decay/lr0、旧defaults/groups一致 |
| 1.2 | SGDはlrだけ、旧momentum/decay0を維持 |
| 1.3 | Parameter同参照/順序・grad不変、初期state空、生成state独立 |
| 2.1 | exact設定・forged・型/variant/finite/非負拒否 |
| 2.2 | tuple/空/後段Parameter/dtype/device/layout/重複のconstructor前拒否、特殊shape受理 |
| 2.3 | CPU/Python/NumPy RNG・float64/meta/grad/default保持 |
| 3.1 | actual旧builder・同grad3step後の全値/state厳密一致 |
| 3.2 | 新NN共有/個別列非重複、モデルstate/grad/属性不変 |
| 3.3 | exact依存/fresh boot/fullgolden/旧無差分 |

## 配置・境界
src/federated_learning_experiments/learning/training/parameter_optimizer_settings.pyとparameter_optimizer_construction.py。
設計どおり二設定型と一つの生成関数、学習進行やmodel dependencyは追加していない。
対象testはtests/refactoring/test_parameter_optimizer_construction.py、ASTはtest_single_run_dependency_boundaries.py。
productionはdataclasses/公開fieldvalidatorまたは標準torch optimizer/Parameter/専用設定のexact importのみ。

## 実測結果
- Task1 missing-module RED1error/2.13s/exit1→7 passed/3.45s/exit0。Luna APPROVED、rootfresh7 passed/2.85s。
- Task2初回36 passed/1 failedは極大intの過剰拒否期待。整数は有限でfloat変換可能性/上限は規定していないため期待を修正し、実旧constructor-only受理へ照合した。sourceは変更していない。
- Task2追加後37 passed/3.28s/exit0、Luna APPROVED。test-only局所名revision2もLuna PASS、名前修正後rootfresh37 passed/3.04s/exit0。
- AST禁止32/許可11先行RED16 failed/330 passed/3.55s/exit1→GREEN346 passed/3.41s/exit0。rootfresh346 passed/3.39s/exit0。
- 全tests: 3364 passed / 3 skipped / 1 warning / 176.97s / exit0。skipは既存Windows非対応wrapper、warningは既存qint8 fixture deepcopyのTypedStorage非推奨。
- CPUfreshprocess: PARAMETER_OPTIMIZER_CONSTRUCTION_SMOKE_PASS/exit0、旧package importなし。
- 旧production/両golden/旧回帰testは748c3aaから差分なし。数値基準/許容誤差を変更していない。
- 空共有特徴抽出部の旧optimizer失敗を実再現し[LEGACY-010](../../../docs/research/implementation-findings/legacy-010-empty-shared-feature-optimizer.md)へ記録。旧未修正/通常過去影響未確認、上位のoptimizer省略判断は後続。

## 環境と再現
共有venv Python3.13.15、torch2.12.1+cpu、NumPy2.4.6、pytest9.1.1。基準はdocs/experiments/refactoring-baseline.md。
```powershell
$env:TMP=(Resolve-Path ../../venv/refactoring-tests).Path
$env:TEMP=$env:TMP
$env:MPLCONFIGDIR=(Resolve-Path ../../venv/matplotlib-cache).Path
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:FDE_MNIST_DATA_DIR=(Resolve-Path ../../data/mnist).Path
../../venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/parameter-optimizer-construction-final-20261005a
```
fresh smokeは別processにPYTHONPATH=srcを設定して次を実行した。stepは上位/test-only側の呼出で、生成関数の責務ではない。
```python
import sys
import torch
from federated_learning_experiments.learning.training.parameter_optimizer_settings import AdamParameterOptimizerSettings,SgdParameterOptimizerSettings
from federated_learning_experiments.learning.training.parameter_optimizer_construction import create_parameter_optimizer
for settings in (AdamParameterOptimizerSettings(learning_rate=.01,weight_decay=.001,adam_variant="amsgrad"),SgdParameterOptimizerSettings(learning_rate=.01)):
 p=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float32,device="cpu"))
 p.grad=torch.ones_like(p)
 rng=torch.get_rng_state().clone()
 optimizer=create_parameter_optimizer(parameters=(p,),optimizer_settings=settings)
 assert optimizer.param_groups[0]["params"][0] is p
 assert not optimizer.state
 assert torch.equal(rng,torch.get_rng_state())
 before=p.detach().clone()
 optimizer.step()
 assert not torch.equal(before,p)
 assert torch.equal(rng,torch.get_rng_state())
assert not any(n.startswith("federated_drift_experiment") for n in sys.modules)
print("PARAMETER_OPTIMIZER_CONSTRUCTION_SMOKE_PASS")
```

## 最終判定
Luna Task3はAPPROVED、最終feature統合はGO。9/9条件・相互接続・参照/状態・配置/依存にgapやblockedなし。Luna自身のfresh Adam/SGD生成と外部step smokeも成功。
主担当はfresh全suite/346件/smoke、その後code不変更、hash/tasks/UTF-8/diffcheckと旧基準無差分を照合しVERIFIED。全3tasksとspecを完了した。
完成範囲は設定と生成であり、optimizer lifecycleや全体学習runは後続。
