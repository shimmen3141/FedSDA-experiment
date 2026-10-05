# 統合検証: 共有特徴抽出部への再接続

## 範囲と正本
適合する共有特徴抽出部の借用参照への交換だけを分類器へ追加する。概念固有部と値/gradを保持する。
optimizer生成・reset・現在参照での新batch作成は上位の責務。旧共有parameterを持つ管理器は接続先へ流用しない。
spec.jsonが承認/進捗、naming revision3が正式名、review.mdが採否の正本。
全3tasks完了、Luna独立APPROVEDと最終feature統合GOを主担当が採用。7/7要件にblockerなし。
共有元選択・同期parameterロード・候補登録・モデル一覧の所有・新client/全体runは後続。

## 実測
- Task 1: 未実装methodのAttributeErrorで21 failed / 3.09秒 / exit1。実装後21 passed / 1.59秒 / exit0。Lunaも独立実行してAPPROVED。
- class2/4×hidden3構成×同/別参照の12正常条件。概念固有NN参照/class数/双方parameter値/grad参照と値/RNG不変、forward新経路を確認。
- 型/派生型/None/寸法/層/shape/float64/metaの9拒否条件。接続変更前に検証し、元接続/双方値gradを保持。metaは数値実体がないため参照/shape/dtype/deviceで照合。
- Task 2: 対象33 passed / 3.93秒 / exit0。LunaAPPROVEDと命名改善後にも33 passed / 3.96秒 / exit0。test-onlyなのでREDはN/A。
- class2/4×Adam standard/AMSGrad/SGD×接続先共有optimizer既存/欠落=12条件。概念側stateを一回学習で蓄積後、新接続→接続先共有owner選択→個別reset→新batchで実旧attach_backboneへ照合。
- 共有更新/凍結/更新の3stepで全loss/NN値/grad/optimizer stateが実旧共同更新とexact一致。
- 旧共有optimizerのparameter列は旧抽出部、現在共有ownerは接続先parameter列であることをidentity確認。旧共有owner/旧抽出部の値grad/stateを全後続stepで保持。旧借用batchの概念optimizer参照/stateも保持。
- 接続先に既存optimizerがある場合は一回step後のstateを接続後も維持。旧attachによる個別resetに対し、新接続単体ではoptimizer不変を確認してから上位で明示resetする。
- Task 3: 既存AST込み540 passed / 5.27秒 / exit0。import変更なし、既存guardで上位/optimizer非依存を検証（架空AST REDなし）。
- fresh新CPU接続/共同更新smoke成功 / exit0。借用参照・概念固有部・RNG保持、CPU float32学習で値更新と旧package非importを確認。
- Ruff成功、format101 files already formatted、Pyright0 errors/0 warnings、pip check成功、git diff --check成功。
- 固定748c3aaとの旧production/両golden/旧回帰test差分は空。
- 現在検証source hash: d2643464a3c6850d1c8db2141fc2f8919eba6e4d841c3df50db1f46f196f3fd9。
  tracked Pythonと両golden201パスをsortし、UTF8相対path+NUL+LF正規化内容+NULでSHA256集計する。
- 全回帰4033 passed / 3 skipped / 1既存warning / 230.35秒 / exit0、旧11/最終3goldenを含む。値・許容差未変更。
  skipは既存Windows非対応wrapper、warningは既存qint8 fixtureのTypedStorage非推奨。Task 3はLuna APPROVED、別feature GOも確認。
- Lunaはfresh新CPU smokeを独立再実行し、sourcehash一致・全7要件・責務境界・全task同期を確認して最終GO。

## 環境とコマンド
2026-10-06、Windows CPU、共有../../venv: Python3.13.15/Torch2.12.1+cpu/NumPy2.4.6/pytest9.1.1。
基準はdocs/experiments/refactoring-baseline.mdとenvironments/golden/windows-cpu。
OMP_NUM_THREADS=MKL_NUM_THREADS=1、TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、FDE_MNIST_DATA_DIR=../../data/mnist。

```text
python -m pytest tests/refactoring/test_shared_feature_extractor_attachment.py tests/refactoring/test_single_run_dependency_boundaries.py -q -p no:cacheprovider
python -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/feature-attachment-final-20261006a --junitxml=../../venv/refactoring-tests/feature-attachment-final.xml
python -m ruff check . --output-format concise
python -m ruff format --check . --output-format concise
python -m pip check
```

fresh smoke: PYTHONPATH=srcでgit管理外../../venv/refactoring-tests/feature_attachment_smoke.pyを別processで実行。
Windows sandbox上のPyrightはrequire_escalatedと共有venvの絶対pythonpathで子Python起動を許可する。

```powershell
C:/Users/yshin/Research/FedSDA-experiment/venv/Scripts/python.exe -m pyright --pythonpath C:/Users/yshin/Research/FedSDA-experiment/venv/Scripts/python.exe
```

## 全7要件の証拠
|要件|実装/検証|
|---|---|
|1.1|適合する借用先と参照identity、NN生成なし・双方値grad保持|
|1.2|adapter/分類層/activation参照/class数保持、実旧全値grad照合|
|1.3|明示した接続先forwardと同出力、再接続後3stepの学習一致|
|1.4|9異常条件の事前拒否、元接続と双方状態保持|
|1.5|同参照の12正常matrix部分で接続と概念固有部保持|
|2.1|Python/Torch RNG保持、接続単体optimizer不変、旧借用batch不変、既存AST/fresh新CPU|
|2.2|接続先parameter対応ownerと明示個別reset/新batch、旧owner非流用/不変、12実旧NN条件|

## 旧所見と限界
今回新たな旧正常系不具合は確認していない。旧不具合/改善点の正本はdocs/research/implementation-findings/README.md。
既存LEGACY-010（空共有optimizer）を変更しない。隠れ層なしの再接続自体は対応するが、共同更新は既存の非空共有optimizer契約に従う。
通常構築されたモデルの宣言寸法/層構造を外側で改変すること、並行再接続は通常契約外。
旧全体goldenは旧コード固定値の確認、新旧同値性は部品対照で確認する。新全体run golden完成を主張しない。
