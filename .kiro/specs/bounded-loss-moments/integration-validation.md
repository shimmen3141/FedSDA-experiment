# 統合検証の証拠

## 完成範囲

一系列の不変な件数・平均・偏差平方和、旧演算順の1件追加、2件以上の平均/不偏標本分散だけを所有する。
旧1件seedのM2=.1、サーバ由来のM2=0を補正しない。
モデル/class所属・batch seed算出・監視clip/履歴採用・学習/登録・統計merge・候補進行・新FedSDA全体runは後続。

## 基準環境と全回帰

docs/experiments/refactoring-baseline.mdに従う共有venv。Python3.13.15、torch2.12.1+cpu、NumPy2.4.6、pytest9.1.1。
TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、OMP_NUM_THREADS=1、MKL_NUM_THREADS=1、FDE_MNIST_DATA_DIR=../../data/mnistを明示。

```powershell
../../venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/loss-moments-final-20261004a
```

2026-10-04: exit0、2728 passed /3 skipped /119.28s。
旧11/最終3goldenとschema/機能テストを含む。3 skipは既存Windows shell非対応のserver wrapper2件・ablation suite1件。
旧production・旧/最終goldenと比較テストに748c3aaから差分なし。環境差を理由にgolden/許容誤差を変更していない。

## 対象と依存境界

対象53件＋AST117件＝170 passed /1.94s /exit0。
task1 REDは未実装packageのModuleNotFoundError/exit1、GREEN44passed。
task2/3は既存契約のtest-only接続/境界検証のためRED非該当。
新数値部品はdataclasses/mathのみをimport。ASTの新許可例外は不要。
NumPy/torch/globalconfig/旧package/runtime/監視/候補評価/予測/同層別moduleを禁止注入、dataclasses/mathを許可注入する。

## 条件対応

| 条件 | 証拠 |
|---|---|
| 1.1, 1.2, 1.3 | frozen値/float正規化、空/非零1件seed/サーバM2=0、型/有限/値域/空整合/overflow拒否 |
| 2.1, 2.2, 2.3 | 旧_update_running_stats全更新値完全一致、入力保存/別結果、forged値と損失の拒否、追加件数overflow時不変 |
| 3.1, 3.2, 3.3 | Noneとn2真の0推定区別、旧_get_model_stats直接比較、保存seedM2と不偏除算、frozen推定 |
| 4.1 | 一定/交互/微小損失系列の旧oracle、モデル全体/class系列の帰属順直接照合 |
| 4.2 | 保存meanによる旧baselineとpublic監視、n<2履歴欠落/n2のmean0保持とpublic候補選択、RNG/defaultdtype/device/keyword-only |
| 4.3 | AST/golden/stdlib-only smoke、LEGACY-005/006・正本・roadmapと部分完成 |

12/12条件の証拠あり。値→更新→推定→監視/参照選択の接続を明示testで確認。
設計のファイル配置/API/stdlib境界と一致し、blocked taskなし。
旧説明の不整合と不正class入力後の部分更新は共通台帳に記録し、旧productionは変更しない。通常client/過去成果への影響は未確認。

## 独立起動

新しいPython processで以下を実行しexit0 /MOMENTS_SMOKE_PASS。旧/NumPy/torchをimportせず、seed保持・純粋更新・推定・拒否まで起動した。

```python
import sys
sys.path.insert(0,'src')
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import BoundedLossMoments, accumulate_bounded_loss_observation, estimate_loss_mean_and_sample_variance
loss_moments=BoundedLossMoments(observed_loss_count=0,mean_loss=0,sum_squared_loss_deviations=0)
result=accumulate_bounded_loss_observation(loss_moments=loss_moments,observed_loss=.25)
assert result.observed_loss_count==1 and estimate_loss_mean_and_sample_variance(loss_moments=result) is None
loss_moments=BoundedLossMoments(observed_loss_count=1,mean_loss=.25,sum_squared_loss_deviations=.1)
result=accumulate_bounded_loss_observation(loss_moments=loss_moments,observed_loss=.75)
estimate=estimate_loss_mean_and_sample_variance(loss_moments=result)
assert result.observed_loss_count==2 and estimate.mean_loss==.5 and estimate.sample_variance==.225
try:
    accumulate_bounded_loss_observation(loss_moments=loss_moments,observed_loss=None)
except TypeError:
    pass
else:
    raise AssertionError('invalid observation must reject')
assert loss_moments.observed_loss_count==1 and loss_moments.mean_loss==.25 and loss_moments.sum_squared_loss_deviations==.1
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
assert not any(name=='federated_drift_experiment' or name.startswith('federated_drift_experiment.') for name in sys.modules)
print('MOMENTS_SMOKE_PASS: seed, pure update, estimate, rejection; stdlib only')
```
