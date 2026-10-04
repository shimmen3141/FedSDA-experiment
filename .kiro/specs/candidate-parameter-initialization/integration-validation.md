# 統合検証: 候補パラメータ初期化
## 完了範囲
専用初期化元設定と、入力snapshotから独立した初期parameter snapshotを選択・平均して返す部品。
モデル生成/適用/optimizer/学習/共有部attach/採否/登録/通信/FIFO/新FedSDA全体runは未移植。
CPU denseの対応dtypeをdesignで明示し、旧parameter名を解釈・分割せず完全stateを扱う。

## 旧参照と接続
実旧FedSDAClient._select_initialization_paramsと_average_model_paramsをunbound helperとして呼ぶ。
get_paramsはdeepcopy stub、modelconstructorやforwardを実行しない。旧choice対応はtest-only。
公開警報後評価のreference_full_interval_mean_lossを明示(ID,mean)へ変換し、再利用不適合な最小loss参照も初期化元へ使えることを確認する。
候補採否result全体や上位状態をproductionへ渡さない。正常値/演算順を維持し、非有限結果はclip/castせず拒否する。

## 条件対応
| 条件 | 証拠 |
|---|---|
| 1.1 | assigned方式の非最小current oracle |
| 1.2 | 評価済み最小/先着tie/未評価除外/空current fallback/負ID oracle |
| 1.3 | 登録順stack.mean/整数bool先頭copy/complex/空average None |
| 1.4 | 15dtype/全key/value/order/shape/device、逆key順/scalar/空形状/noncontiguous/丸めの実旧照合 |
| 2.1 | 不正設定/ID/loss/後段未選択snapshot/型/構造/非有限、入力不変、LEGACY009旧inf→新ValueError |
| 2.2 | 双方向copy/別呼出/storage/入力内key alias解消 |
| 2.3 | leaf/nonleaf grad detach、Python/NumPy/torch RNG/default dtype/device/grad flag保持 |
| 3.1 | 実旧unbound helper直接照合、公開評価meanからのtest-only接続 |
| 3.2 | exact2module AST/CPU fresh smoke/全golden/旧無変更/正本・未移植範囲・台帳 |

9/9条件を対応付けた。LEGACY009は旧未修正・通常/過去への影響未確認。新拒否を旧経路の修正済みとは扱わない。

## Task検証
Task1 missing-module RED（collection1error/3.56s/exit1）→18 passed/3.14s/exit0。LunaAPPROVED。
Task2 test-only40追加/RED非該当→58 passed/4.09s/exit0。LunaAPPROVED。
Task3 AST新14禁止+7許可を先行追加: RED8 failed/209 passed/0.55s/exit1。
exact2moduleのtorch/Tensor/専用設定型・core公開検査関数だけの許可後: 対象58+AST217=275 passed/4.55s/exit0。
qint8拒否fixtureのdeepcopyによるtorch TypedStorage非推奨警告1件。production警告ではない。
依存境界の追加で既存ASTの許可範囲は広げていない。

## 固定環境と全回帰
Python3.13.15/torch2.12.1+cpu/NumPy2.4.6/pytest9.1.1の共有venv。
TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、OMP/MKL_NUM_THREADS=1、FDE_MNIST_DATA_DIR=../../data/mnist。
```powershell
../../venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/candidate-parameter-initialization-final-20261004a
```
2026-10-04: exit0、3182 passed /3 skipped /1 warning /226.33s。
旧11/最終3goldenを含む。3skipは既存Windows server wrapper2件/ablation suite1件。
旧production/比較test/goldenは748c3aaとの差分なし、goldenや許容差を更新していない。
全suite開始後productionとtestコードは変更していない。文書末尾空行の除去と承認hash更新は意味を変えていない。

## 独立起動
fresh Pythonプロセスで三方式/同率/storage/入力不変/RNG不変を検証。
CANDIDATE_PARAMETER_INITIALIZATION_SMOKE_PASS /exit0、旧package importなし。
```python
from pathlib import Path
import sys
sys.path.insert(0,str(Path("src").resolve()))
import torch
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import CandidateParameterInitializationSettings
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization import select_candidate_initial_parameter_snapshot
snapshots={2:{"backbone.weight":torch.tensor([1.],device="cpu"),"counter":torch.tensor(7,device="cpu")},-1:{"backbone.weight":torch.tensor([3.],device="cpu"),"counter":torch.tensor(9,device="cpu")}}
state=torch.get_rng_state().clone()
for source,value in (("assigned_training_model",3.),("lowest_evaluated_mean_loss_model",1.),("equal_mean_of_available_models",2.)):
 result=select_candidate_initial_parameter_snapshot(settings=CandidateParameterInitializationSettings(candidate_parameter_initialization_source=source),available_parameter_snapshots_by_model_id=snapshots,current_training_model_id=-1,evaluated_mean_losses_by_model_id=((2,.2),(-1,.2)))
 assert result["backbone.weight"].item()==value
 assert not result["backbone.weight"].requires_grad
 result["backbone.weight"].fill_(99.)
 assert snapshots[2]["backbone.weight"].item()==1.
 assert snapshots[-1]["backbone.weight"].item()==3.
assert torch.equal(state,torch.get_rng_state())
assert not any(n.startswith("federated_drift_experiment") for n in sys.modules)
print("CANDIDATE_PARAMETER_INITIALIZATION_SMOKE_PASS")
```

## 最終判断
全回帰と独立起動は成功。Task3のLunaレビューと最終統合gateの判断をreview.md/spec.jsonへ記録してから完了とする。
