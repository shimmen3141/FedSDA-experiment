# 統合検証の証拠
## 完成範囲
一モデルの参加選別済みclient順の全体momentsから、統計観測件数を重みとする平均・合計件数・M2=0を返す純関数。空/全zeroはNone。
参加判定・モデルパラメータFedAvg・通信・ID/保存・クラス合成・分散pooling・候補進行・新FedSDA全体runは含めない。
上位はNoneなら既存whole serverrecordを保持、正nならclass=()でwhole置換する。今回はtest-onlyで接続する。

## 旧参照
BaseServer.update_global_modelsとSharedBackboneFedSDANoCachedServer.update_global_modelsを、モデル実体を持たないget_paramsのPythonfloat2keys stubから直接呼ぶ。
各10ケース、計20oracleで全fieldとwholeclassclear/既存保持をexact照合。空/zero/単一/非零seedM2/統計件数対training量/境界/丸め入力順/モデル有無・正training・stats有無を含む。
LEGACY008の2極大件数を実BaseServerで再現し、新ValueError拒否/入力不変と対照。旧productionは未修正、通常client/過去成果の影響は未確認。

## 環境・対象検証
共有venv: Python3.13.15/torch2.12.1+cpu/NumPy2.4.6/pytest9.1.1。
TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、OMP/MKL_NUM_THREADS=1、FDE_MNIST_DATA_DIR=../../data/mnist。
task1 REDは未実装module ModuleNotFoundError/exit1、GREEN20 passed。
task2はtest-only追加26/RED非該当、対象46 passed。
task3 AST許可前RED4 failed/192 passed/exit1。exactmodule/型許可後、対象46+AST196=242 passed/2.72s/exit0。
新12禁止注入は旧/torch/NumPy/config/runtime/methods/同module別関数2種/private/子module/別module/store。
新4許可はstdlib/exactmodule/absolute型/relative型。production参照はexactBoundedLossMomentsだけ。

## 条件対応
| 条件 | 実測証拠 |
|---|---|
| 1.1 | 両旧client順mean*n +=/n +=、training量との区別、丸め順の反転oracle |
| 1.2 | 正n/全n/mean/M2zero、入力非零M2を合成しない |
| 1.3 | 空/zero None、旧wholeclass付きrecord保持 |
| 2.1 | exacttuple/moment・後段forgedfield・zero不整合・極大算出範囲/overflow、拒否時入力非変更 |
| 2.2 | frozen/入力改変非伝播/別結果独立 |
| 2.3 | Python/NumPy/Torch RNG、grad/defaultdtype/device、keyword |
| 3.1 | 旧Base/shared両updateへ直接20ケース、外側参加gate |
| 3.2 | Nonewhole保持/正nclassclear→serverstore→ID補完→新store次観測を旧updateへ全値照合→3用途baseline |
| 3.3 | exactAST/fullgolden/stdlibfreshsmoke/roadmap/LEGACY008/承認記録 |

9/9条件を実測へ対応付ける。上位接続はtestで明示し、productionへ追加しない。
最終GPT-6 Lunaの統合判断はGO。主担当はkiro-verify-completionでVERIFIEDとし、全3tasksと本featureを完了した。

## 全回帰
```powershell
../../venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/server-loss-mean-final-20261004a
```
2026-10-04: exit0、3103 passed /3 skipped /156.65s。
旧11/最終3golden・schema/機能テストを含む。3skipは既存Windows非対応server wrapper2件/ablation suite1件。
全suite後production/testコードの変更なし。
旧production・golden・比較テストは748c3aaとの差分なし。golden/許容誤差を更新しない。

## 独立起動
fresh python -S - のstdinとして次を実行。SERVER_LOSS_MEAN_SMOKE_PASS/exit0、torch/NumPy/旧package import無し。
```python
from pathlib import Path
import sys
sys.path.insert(0,str(Path("src").resolve()))
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import BoundedLossMoments
from federated_learning_experiments.learning.loss_statistics.server_loss_mean_aggregation import aggregate_participating_client_loss_means
seed=BoundedLossMoments(observed_loss_count=1,mean_loss=.25,sum_squared_loss_deviations=.1)
other=BoundedLossMoments(observed_loss_count=3,mean_loss=.75,sum_squared_loss_deviations=.2)
result=aggregate_participating_client_loss_means(participating_client_loss_moments=(seed,other))
assert (result.observed_loss_count,result.mean_loss,result.sum_squared_loss_deviations)==(4,.625,0.)
assert aggregate_participating_client_loss_means(participating_client_loss_moments=()) is None
assert result is not seed and result is not other
assert "torch" not in sys.modules and "numpy" not in sys.modules
assert not any(n.startswith("federated_drift_experiment") for n in sys.modules)
print("SERVER_LOSS_MEAN_SMOKE_PASS")
```
