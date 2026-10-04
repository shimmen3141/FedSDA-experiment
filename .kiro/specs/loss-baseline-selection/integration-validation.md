# 統合検証の証拠

## 完成範囲
一系列の任意集計から監視・警報区間再利用・警報後履歴の基準平均を選ぶ3pure関数。
モデル/class統計map、seed/merge、snapshot時機、参照ID選択・候補採否/進行・学習、新FedSDA全体runは後続。
旧n2平均0の再利用除外と履歴保持の非対称性を維持し、アルゴリズムは変更しない。

## 環境と全回帰
docs/experiments/refactoring-baseline.mdの共有venv。Python3.13.15/torch2.12.1+cpu/NumPy2.4.6/pytest9.1.1。
TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、OMP/MKL_NUM_THREADS=1、FDE_MNIST_DATA_DIR=../../data/mnist。
```powershell
../../venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/baseline-selection-final-20261004a
```
2026-10-04: exit0、2829 passed /3 skipped /259.61s。
旧11/最終3golden・schema/機能テストを含む。3skipは既存Windows shell非対応server wrapper2件/ablation suite1件。
旧production・golden・比較テストに748c3aaから差分なし。golden/許容誤差更新なし。

## 対象と依存
対象89＋AST129=218 passed/3.89s/exit0。
task1 REDは未実装package ModuleNotFoundError/exit1、旧3用途48casesをLuna確認後、n5追加で69、task2で89を確認。
task2はtest-onlyでRED非該当。task3は依存未許可で1failed/205passed、exactmodule/型許可追加で通過。
許可はbounded_loss_momentsとBoundedLossMomentsだけ、privatevalidator/別public推定/子module/旧/数値lib/上位を10禁止注入し、absolute/relativeの2許可を確認。

## 条件対応
| 条件 | 証拠 |
|---|---|
| 1.1,1.2,1.3 | 旧_e_detector_baseline直接oracle、欠落/n0/1/2/5とclip境界、publicMonitorへ明示基準入力 |
| 2.1,2.2,2.3 | 旧_resolve_driftの候補callback捕捉、旧_begin_forward_validationのsession履歴捕捉、n5・mean0と欠落、public参照選択への明示履歴 |
| 3.1 | exact型/forgedfield/overflow拒否と全入力非変更、共有Python/NumPy/TorchRNG/defaultdtype/device、keyword |
| 3.2 | 旧3用途69ケースと監視・参照選択接続 |
| 3.3 | exactAST/全golden/独立stdlib smoke/roadmap・正本 |

9/9条件の証拠あり。入力集計→用途別平均→monitor/history/reference選択のcross-task接続はtest-only。
設計配置・API・依存方向と一致、blockedなし。旧不具合の新たな修正は行わない。

## 独立起動
fresh Python processで以下を実行、exit0 /BASELINE_SMOKE_PASS。旧/NumPy/torchをimportせず起動した。
```python
import sys
sys.path.insert(0,'src')
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import BoundedLossMoments
from federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection import select_loss_monitoring_baseline_mean_loss, select_alarm_interval_reuse_baseline_mean_loss, select_post_alarm_reference_historical_mean_loss
assert select_loss_monitoring_baseline_mean_loss(loss_moments=None)==.01
loss_moments=BoundedLossMoments(observed_loss_count=1,mean_loss=.2,sum_squared_loss_deviations=.1)
assert select_loss_monitoring_baseline_mean_loss(loss_moments=loss_moments)==.2
assert select_post_alarm_reference_historical_mean_loss(loss_moments=loss_moments) is None
loss_moments=BoundedLossMoments(observed_loss_count=2,mean_loss=0,sum_squared_loss_deviations=0)
assert select_alarm_interval_reuse_baseline_mean_loss(loss_moments=loss_moments) is None
assert select_post_alarm_reference_historical_mean_loss(loss_moments=loss_moments)==0.
try:
    select_loss_monitoring_baseline_mean_loss(loss_moments={})
except TypeError:
    pass
else:
    raise AssertionError('invalid input accepted')
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
assert not any(n=='federated_drift_experiment' or n.startswith('federated_drift_experiment.') for n in sys.modules)
print('BASELINE_SMOKE_PASS')
```
