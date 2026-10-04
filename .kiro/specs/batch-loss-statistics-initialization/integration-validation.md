# 統合検証の証拠

## 完成範囲
外部計算済みCPUfloat32 batch loss[N]/labels[N,1]/class_countから、モデル全体/class初期統計を生成するpure関数。
全体n1M2.1/classn1M2zero、class昇順/欠落省略、旧torch reduction→Pythonfloat倍算の数値を維持。
singletonの未定義variance warningは生成しない。事前学習の逐次Welfordseed、モデルprepare/forward/学習/登録/ID/送信、merge/監視進行・新全体runは含めない。

## 環境と全回帰
docs/experiments/refactoring-baseline.mdの共有venv。Python3.13.15/torch2.12.1+cpu/NumPy2.4.6/pytest9.1.1。
TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、OMP/MKL_NUM_THREADS=1、FDE_MNIST_DATA_DIR=../../data/mnist。
```powershell
../../venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/batch-initial-statistics-final-20261004b
```
2026-10-04: exit0、2965 passed /3 skipped /124.63s。初回の全回帰は2964 passed /3 skipped /156.49s。依存検査の修正と1件追加後に上記最終状態で再実行した。
旧11/最終3golden・schema/機能を含む。3skipは既存Windows shell非対応server wrapper2件/ablation suite1件。
旧production/golden/比較testsに748c3aaから差分なし。golden/許容誤差更新なし。

## 対象と依存
対象60+AST164=224 passed /2.08s /exit0。
task1 REDは未実装module ModuleNotFoundError/exit1、旧正常register直接oracle24件でGREEN。
task2はtest-only RED非該当、33異常入力と独立性/共有状態/接続を追加し対象60件。
task3 RED7 failed /156 passed、exactmodule/型+torch許可後に通過。
直接importはtorch本体/公開Tensor型、bounded_loss_moments/BoundedLossMoments、model_and_class_loss_statistics/ModelAndClassLossStatisticsとstdlibだけを許可。
13禁止注入はtorch._C/旧/NumPy/runtime/methods/accumulate/private/別推定/store/子module/別module。7許可注入はtorch/公開Tensor/stdlibとabsolute/relative型参照。
Lunaのtask3レビューでtorch.*がprivateも通す検査漏れを確認し、torch本体/Tensorへ限定。rootも修正前の関数をメモリ上だけ再現してprivate逃げを確認した。production集計の演算やAPIは変更していない。

## 条件対応
| 条件 | 証拠 |
|---|---|
| 1.1 | 旧register全体count/mean/M2完全一致、torchbatch順とPython倍算 |
| 1.2 | 二値/多クラス、class順/欠落、class全fieldと非連続/入力順一致 |
| 1.3 | batchsingleton全体.1/class0、クラス内singleton0 |
| 2.1 | 33異常入力（type/dtype/device/layout/shape/empty/mismatch/finiteness/range/class）拒否、input値/versions不変 |
| 2.2 | frozen/result間独立、入力変更/結果field改変の非伝播 |
| 2.3 | gradmode/grad非更新、Python/NumPy/TorchRNG/defaultdtype/device維持 |
| 3.1 | 12正常batch×連続/非連続=24旧直接register全値oracle、singleton/boundary/丸め |
| 3.2 | seed→store→次class帰属更新の旧全値照合とoverallbaseline明示選択 |
| 3.3 | exactAST/fullgolden/CPUtorch独立smoke、roadmapとLEGACY007追跡 |

9/9条件の証拠あり。モデル登録・送信を含まない明示test接続、設計配置/依存と一致、blockedなし。
LEGACY007は旧空batchのNaN統計登録と新非空拒否を同入力で対照。正常client/過去成果への影響と旧修正は未確認・未実施。

## 独立起動
fresh Python processで以下を実行しexit0 /BATCH_INITIAL_STATISTICS_SMOKE_PASS。CPUtorchを使用、旧productionをimportせず初期集計/保存/次観測更新と空拒否を確認。
```python
import sys
sys.path.insert(0,'src')
import torch
from federated_learning_experiments.learning.loss_statistics.batch_loss_statistics_initialization import initialize_model_and_class_loss_statistics_from_batch
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatisticsStore
loss_statistics=initialize_model_and_class_loss_statistics_from_batch(per_sample_bounded_losses=torch.tensor([.25,.75],dtype=torch.float32,device='cpu'),observed_class_labels=torch.tensor([[1],[0]],dtype=torch.float32,device='cpu'),class_count=2)
assert loss_statistics.overall_loss_moments.observed_loss_count==2
assert loss_statistics.overall_loss_moments.mean_loss==.5
assert loss_statistics.overall_loss_moments.sum_squared_loss_deviations==.125
assert tuple(class_id for class_id,_ in loss_statistics.class_loss_moments_by_class_id)==(0,1)
seed=initialize_model_and_class_loss_statistics_from_batch(per_sample_bounded_losses=torch.tensor([.25],dtype=torch.float32,device='cpu'),observed_class_labels=torch.tensor([[1]],dtype=torch.float32,device='cpu'),class_count=2)
assert seed.overall_loss_moments.sum_squared_loss_deviations==.1
assert seed.class_loss_moments_by_class_id[0][1].sum_squared_loss_deviations==0
store=ModelAndClassLossStatisticsStore(initial_loss_statistics_by_model_id={-1:loss_statistics})
store.record_assigned_loss(model_id=-1,observed_loss=.5,observed_class_id=0)
assert store.get_model_loss_statistics(model_id=-1).overall_loss_moments.observed_loss_count==3
try:
    initialize_model_and_class_loss_statistics_from_batch(per_sample_bounded_losses=torch.empty(0,dtype=torch.float32,device='cpu'),observed_class_labels=torch.empty((0,1),dtype=torch.float32,device='cpu'),class_count=2)
except ValueError:
    pass
else:
    raise AssertionError('empty batch accepted')
assert not any(n=='federated_drift_experiment' or n.startswith('federated_drift_experiment.') for n in sys.modules)
print('BATCH_INITIAL_STATISTICS_SMOKE_PASS')
```
