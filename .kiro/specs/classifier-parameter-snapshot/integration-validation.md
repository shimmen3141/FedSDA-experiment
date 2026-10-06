# 実測: 分類器parameter snapshot

## Scope
一分類器の全parameter現在値の独立コピー。候補選択/平均/復元はtest-only接続で、登録/送信保留/新client/新全体runは未移植。要求/設計rev1、命名rev3、tasksが正本。

## Task1
- 未実装module先行RED: ModuleNotFoundError、1 collection error/2.09秒/exit1。
- GREEN: 16 passed/3.37秒/exit0。class2/4/10の実旧get_paramsと全native key順/shape/値/独立storageを照合。
- 隠れ層空/一層/二層、非連続parameter、相互変更と再取得、共有参照・既存grad・混合training flags保持、forward0、Python/NumPy/Torch乱数、default dtype/device/gradmodeとinference_modeを確認。
- 型/subclass、NaN/Inf/float64/meta/sparse拒否と不正parameterの値・grad参照/値保持。Ruff成功、対象format成功。
- 主担当の最終16 passed/3.62秒、Pyright0 errors/0 warnings。独立Luna実行も16passed・品質成功、Task1 APPROVED。

## 再実行環境
2026-10-06、既存Windows CPU基準、共有../../venv。OMP_NUM_THREADS=MKL_NUM_THREADS=1、MPLCONFIGDIR=../../venv/matplotlib-cache、TMP/TEMP=../../venv/refactoring-tests、FDE_MNIST_DATA_DIR=../../data/mnist。
版と基準はdocs/experiments/refactoring-baseline.md / environments/golden/windows-cpu。goldenは固定旧実装を実行し、新全体run完成を意味しない。
## Task2
- test-only接続なのでproduction RED N/A。初回は復元先state_dictのOrderedDictとsnapshotのplain dictをexact型比較するtestの誤りにより12 failed/16 passed。比較時だけdictへ明示変換し、productionは変更しない。
- class2/4×Adam標準/AMSGrad/SGD×共有更新/凍結の12条件、各3共同更新の全loss/parameter/grad/optimizer stateを実旧照合する。
- 両保有モデルのsnapshot→既存assigned initializer→新native load_state_dictを接続。全値/順序/予測一致、snapshot/選択結果/復元先の独立性、既存optimizer stateとgradient保持を確認。
- 28 passed/3.13秒/exit0、Ruff/format成功、実Luna独立28passed/Task2 APPROVED。有用なlocal renameと末尾空行除去を採用。
- 承認済みrename後も28 passed/7.04秒/exit0。

## Task3
- 26のexact依存注入（21拒否/5許可）の先行RED: 8 failed/18 passed/612 deselected/0.14秒/exit1。
- exact symbol解決とbare import拒否のguardを追加し、対象＋ASTは666 passed/4.80秒/exit0。
- fresh新CPU processでbinary/multiclass snapshot→既存initializer→native復元、全値/予測/storage独立と旧package非importを確認、exit0。
- Ruff成功、format111 files already formatted、Pyright0 errors/0 warnings、pip check成功、git diff --check成功。固定748c3aaとの旧production/両golden/旧回帰test差分は空。
- source211 Python/両goldenのSHA256: b4d869f6b1a83d661837ca8ebd3089e84ef19e9e43e098ac19df7aaae6a07131。Git追跡パスsort→UTF8相対path+NUL+LF正規化内容+NULで集計。要求rev1/設計rev1/命名rev3の承認hash一致。
- 全pytestは4399 passed/3 skipped/1既存warning/171.13秒。旧11/最終3goldenを含む。skipは既存Windows wrapper、warningは既存qint8 fixtureのTypedStorage非推奨。goldenの値・許容差変更なし。
- 全pytest exit0、JUnit classifier-snapshot-full.xmlは4402 tests/0 failures/0 errors/3 skipped。
- 新たな旧正常系不具合は今回の範囲では観測していない。
- 実Luna Task3 APPROVED。承認後の主担当completion gate: 対象＋AST666 passed/4.17秒/exit0、fresh CPU成功、diff-check/承認hash/JUnit確認。

## 全8要件trace
|要件|証拠|
|---|---|
|1.1|実旧class2/4/10全parameter/key順/shape/値、12学習後接続|
|1.2|storage独立・勾配なし・相互更新、initializer/復元先との独立|
|1.3|隠れ層空/一層/二層の再取得・過去結果保持|
|2.1|型/subclass・5parameter不正、既存値/grad参照/値保持|
|2.2|forward0、全parameter/grad・共有参照・flags・3乱数/default/gradmode/inference保持、12接続optimizer保持|
|3.1|実旧get_paramsのprefix明示対応と全値・順序・独立性、12実NN条件|
|3.2|26依存注入とfresh旧非import、productionの取得/copy限定|
|3.3|固定旧差分空、品質・全回帰/goldenの完了gate|

```text
python -m pytest tests/refactoring/test_classifier_parameter_snapshot.py tests/refactoring/test_single_run_dependency_boundaries.py -q -p no:cacheprovider
python -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/classifier-snapshot-full-20261006 --junitxml=../../venv/refactoring-tests/classifier-snapshot-full.xml
python ../../venv/refactoring-tests/classifier_parameter_snapshot_smoke.py
python -m ruff check .
python -m ruff format --check .
python -m pyright --pythonpath C:/Users/yshin/Research/FedSDA-experiment/venv/Scripts/python.exe
python -m pip check
```
