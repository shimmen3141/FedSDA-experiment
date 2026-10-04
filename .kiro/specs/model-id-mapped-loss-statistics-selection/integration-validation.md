# 統合検証の証拠
## 完成範囲
モデル全体・クラス別の順序付きsnapshotについて、一回のID対応、最大overall件数・先着同数のwhole record選択、サーバによる欠落/zero補完を行う純関数。
モデル操作・学習データ・予測重み・現在の帰属ID・登録/送信/サーバ集計・store全置換実行・警報後進行・新FedSDA全体runは含めない。
返却値のコピー独立性は新不変APIの契約。旧localdict参照共有を正常client不具合と断定しない。

## 旧参照と境界
旧clients/base.py:678–694をBaseClient.apply_server_mappingの空モデル/学習データstubから直接呼ぶ。
選択の全値/model順/class順を18正常ケースでexact照合。chain/cycleは一回、tie反転、zero server補完、positive local優先、serverID非対応と置換/追加順を含む。
旧実装の正常経路に新しい不具合は今回観測していない。推測のpending参照問題を台帳へ追加しない。既存LEGACY001–007の修正状態を変更しない。

## 条件対応
| 条件 | 実測証拠 |
|---|---|
| 1.1 | empty/missing/identity/negative/chain/cycle/unusedの旧oracle |
| 1.2 | 最大n/firsttie/入力順反転/classwholeと全値/order |
| 1.3 | 勝者置換でtarget位置保持・local初出順 |
| 2.1 | serverID再対応なし/missing/zero local補完 |
| 2.2 | local n2 vs server n100でlocal優先 |
| 2.3 | zero server/既存位置/末尾server順 |
| 3.1 | exact型/shape/ID/重複/後段統計/forged overall/class/未使用mapping、拒否時入力非変更 |
| 3.2 | frozen結果/深い独立/別呼出/共有入力record/RNG/default/keyword |
| 3.3 | snapshot→選択→新storeで元ID消失、次3観測を旧更新全field/class順へ照合、3用途baselineへ明示接続 |
| 3.4 | exact依存AST/全suitegolden/stdlibfreshsmoke/roadmap/LunaGO |

全10条件を検証へ対応付ける。上位接続はtest-onlyでありproduction coordinationを先取りしない。
最終GPT-6 Lunaの独立統合判定はGO。実装/テストは全suite時から不変、記録のUTF-8修復も再確認済み。主担当はkiro-verify-completionでVERIFIEDとし、全3タスク・本featureを完了した。

## 環境・対象検証
共有venv: Python3.13.15/torch2.12.1+cpu/NumPy2.4.6/pytest9.1.1。
TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、OMP/MKL_NUM_THREADS=1、FDE_MNIST_DATA_DIR=../../data/mnist。
task1 RED未実装module/exit1→18 passed。task2はtest-only追加42件でRED非該当、60 passed。
task3 ASTのexact型許可追加前: 4 failed/176 passed/exit1。許可追加後、対象60+AST180=240 passed/2.14s/exit0。
禁止注入12件（旧/torch/NumPy/config/runtime/methods/store/private/子module/別module/別publicsymbol）、許可4件（stdlib/exactmodule/absolute型/relative型）。
golden・旧production・比較テストは748c3aaとの差分なし。許容誤差を変更しない。

## 全回帰
```powershell
../../venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/mapped-statistics-final-20261004a
```
2026-10-04: exit0、3041 passed /3 skipped /142.93s。
旧11/最終3golden・schema/機能テストを含む。3skipは既存Windows shell非対応server wrapper2件/ablation suite1件。
golden/許容誤差/旧production・比較テストの更新なし。全体検証後production/testコードを変更していない。

## 独立起動
fresh Pythonのsite packagesを無効化（python -S -）し、次のstdinを実行。
MAPPED_STATISTICS_SMOKE_PASS/exit0。torch/NumPy/旧packageをimportしないことも確認。
```python
from pathlib import Path
import sys
sys.path.insert(0,str(Path("src").resolve()))
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import BoundedLossMoments
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import ModelAndClassLossStatistics
from federated_learning_experiments.learning.loss_statistics.model_id_mapped_loss_statistics_selection import select_loss_statistics_after_model_id_mapping
positive=ModelAndClassLossStatistics(overall_loss_moments=BoundedLossMoments(observed_loss_count=3,mean_loss=.25,sum_squared_loss_deviations=.1))
zero=ModelAndClassLossStatistics(overall_loss_moments=BoundedLossMoments(observed_loss_count=0,mean_loss=0.,sum_squared_loss_deviations=0.))
result=select_loss_statistics_after_model_id_mapping(local_model_loss_statistics=((-1,positive),(4,zero)),model_id_mapping={-1:2,2:7},server_model_loss_statistics=((4,positive),(9,zero)))
assert tuple(k for k,v in result)==(2,4,9)
assert result[0][1]==positive and result[0][1] is not positive
assert result[1][1]==positive
assert not any(m.startswith("federated_drift_experiment") for m in sys.modules)
assert "torch" not in sys.modules and "numpy" not in sys.modules
print("MAPPED_STATISTICS_SMOKE_PASS")
```
