# 統合検証の証拠

## 完成範囲
一所有者のモデルID別全体・クラス別損失集計、明示seedの独立保持と一括置換、帰属損失の原子的更新、独立した順序付き参照。
モデル実体、クラス上限、batch seed算出、統計merge・ID変更・削除/全reset、学習・警報後進行・サーバ同期、新FedSDA全体runは後続。
全体とclassの件数/M2に新しい整合条件を課さず、旧非零1件seedとclass指定なしを保持する。

## 環境と全回帰
docs/experiments/refactoring-baseline.mdの共有venv。Python3.13.15/torch2.12.1+cpu/NumPy2.4.6/pytest9.1.1。
TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、OMP/MKL_NUM_THREADS=1、FDE_MNIST_DATA_DIR=../../data/mnist。
```powershell
../../venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/model-class-statistics-final-20261004a
```
2026-10-04: exit0、2885 passed /3 skipped /132.42s。
旧11/最終3golden・schema/機能テストを含む。3skipは既存Windows shell非対応server wrapper2件/ablation suite1件。
旧production・golden・比較テストに748c3aaから差分なし。golden/許容誤差更新なし。

## 対象と依存
対象41＋AST144=185 passed /2.00s /exit0。
task1 REDは未実装moduleのModuleNotFoundError/exit1、正常旧oracle6ケースとseed/upsert/snapshotの8testsから開始。
task2はtest-onlyでRED非該当。拒否・独立性・上位接続の追加で対象41件。
task3 REDは依存未許可で4 failed /140 passed、exact moduleと2symbol許可追加で通過。
標準機能とbounded_loss_moments/BoundedLossMoments/accumulate_bounded_loss_observationだけを許可。
旧/NumPy/torch/runtime/methods/private/別public関数/子module/別moduleを11禁止注入。stdlib、module import、absolute/relativeの2symbol参照を4許可注入した。
入力クラス検査と両更新の検査後に一回commitする。初期mapも全検査後に確定する。

## 条件対応
| 条件 | 証拠 |
|---|---|
| 1.1 | 明示初期seed、非零n1M2、class欠落・全体のみ、件数差の保持 |
| 1.2 | 既存全class置換、新ID追加、他モデル値とmodel順の維持 |
| 1.3 | 欠落None/登録0件、frozen、model/class順、独立model/all snapshot |
| 2.1 | 正負ID・未登録/seed各観測の旧全体Welford完全一致 |
| 2.2 | 同損失の指定class追加、未登録class、class順と旧値完全一致 |
| 2.3 | classなし全体のみ、別model/class不変、件数合計不一致許容 |
| 3.1 | ID/loss/forged seed/tupleshape/duplicate class拒否、全体/class countnextoverflowで全状態不変、emptyentryなし |
| 3.2 | 入力dict削除/seed改変、returnednested改変、別store非伝播 |
| 3.3 | Python/NumPy/Torch RNGとdefaultdtype/device非変更、keyword |
| 4.1 | 旧BaseClient._update_model_stats直接oracle6系列、各更新の全値/順 |
| 4.2 | overall平均.5/class平均.8から明示overall→baseline→monitor/history参照選択、store不変 |
| 4.3 | exactAST/fullgoldens/独立stdlib smoke/roadmap/LEGACY006台帳 |

12/12条件の証拠あり。全体統計→既存基準選択→監視/参照比較の接続はtest-only。
設計配置・API・依存方向と一致、blockedなし。LEGACY-006は旧partialと新atomicの同入力対照を追加した。旧修正済みとは扱わない。

## 独立起動
fresh Python processで以下を実行し、exit0 /MODEL_CLASS_STATISTICS_SMOKE_PASS。
旧/NumPy/torchをimportせず、seed・不正class拒否・通常更新・欠落・whole置換を確認。
```python
import sys
sys.path.insert(0,'src')
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import BoundedLossMoments
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatistics,ModelAndClassLossStatisticsStore
seed=ModelAndClassLossStatistics(overall_loss_moments=BoundedLossMoments(observed_loss_count=1,mean_loss=.25,sum_squared_loss_deviations=.1),class_loss_moments_by_class_id=((0,BoundedLossMoments(observed_loss_count=1,mean_loss=.25,sum_squared_loss_deviations=0)),))
store=ModelAndClassLossStatisticsStore(initial_loss_statistics_by_model_id={-100:seed})
state_before_call=store.get_state_snapshot()
try:
    store.record_assigned_loss(model_id=-100,observed_loss=.75,observed_class_id=object())
except TypeError:
    pass
else:
    raise AssertionError('invalid class accepted')
assert store.get_state_snapshot()==state_before_call
store.record_assigned_loss(model_id=-100,observed_loss=.75,observed_class_id=0)
result=store.get_model_loss_statistics(model_id=-100)
assert result.overall_loss_moments==BoundedLossMoments(observed_loss_count=2,mean_loss=.5,sum_squared_loss_deviations=.225)
assert result.class_loss_moments_by_class_id==((0,BoundedLossMoments(observed_loss_count=2,mean_loss=.5,sum_squared_loss_deviations=.125)),)
assert store.get_model_loss_statistics(model_id=99) is None
store.set_model_loss_statistics(model_id=-100,loss_statistics=ModelAndClassLossStatistics(overall_loss_moments=seed.overall_loss_moments))
assert store.get_state_snapshot()[0][0]==-100
assert store.get_model_loss_statistics(model_id=-100).class_loss_moments_by_class_id==()
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
assert not any(n=='federated_drift_experiment' or n.startswith('federated_drift_experiment.') for n in sys.modules)
print('MODEL_CLASS_STATISTICS_SMOKE_PASS')
```
