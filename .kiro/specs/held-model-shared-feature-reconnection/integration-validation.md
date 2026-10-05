# 統合検証: 保有モデル全体の共有再接続

## 範囲と正本
保有モデルの順序付き借用記録に対する共有元選択・事前対応検証・source以外の接続/個別reset・現在ownerの結果を移植する。
一覧の所有・ID採番・同期load・候補採用・予測/統計・学習実行・新client/全体runは後続。
spec.jsonは承認/進捗、naming revision2は正式名、review.mdは採否、本文は実測の正本。
全3tasks完了、設計revision2・命名revision2・Luna独立APPROVEDと最終feature統合GOを主担当が採用。11/11要件にblockerなし。

## 実測
- Task 1: 未実装moduleでModuleNotFoundError、1 collection error / 3.22秒 / exit1。実装後37 passed / 3.11秒 / exit0、Lunaも37 passedを独立実行してAPPROVED。
- 空/単独非負/単独負/混合ID/非昇順/全負/極大ID×初期共有有無2=14条件で実旧_share_model_backbonesの選択・shared/head optimizer stateへ対照。
- 20末尾不正条件（型/ID/重複参照/owner/parameter対応順/正常構築だが接続寸法違い/概念parameter共有/層/dtype/meta）を全操作前に拒否。元接続と全入力の値grad/optimizer参照/state、Python/Torch RNGを保持。
- sourceをskipし入力順にattach→resetする実呼出順、予期しないreset例外でprefix/現在接続保持とsuffix未処理、frozen借用記録を確認。
- Task 2: 49 passed / 3.73秒 / exit0、Lunaも49 passedを独立実行してAPPROVED。test-onlyなのでRED N/A、数値production変更なし。
- class2/4×Adam standard/AMSGrad/SGD×初期共有有無2=12実NN条件。shared/個別stateをstepで蓄積後、逆順入力(-7,4)でも共有元4を選択し、返却列から新HeldModelTrainingBinding/参加batchを作る。
- 共有更新/凍結/更新の3stepで実旧の全loss/NN値/grad/shared/head optimizer stateにexact一致。
- source owner維持、non-source旧共有owner非流用/不変（初期非共有の場合）、旧借用個別optimizer参照/state保持を明示確認。初期共有済みの場合はsource ownerを共有し、個別だけをresetする。
- Task 3: 新exact AST契約17禁止/7許可、先行RED9 failed / 15 passed / 507 deselected / 0.09秒 / exit1。guard追加後、対象+AST580 passed / 4.55秒 / exit0。
- fresh新CPU全保有再接続→共同更新smoke成功 / exit0。source skip/全shared owner対応/個別reset/旧optimizer state保持/RNG不変/値更新/旧package非importを確認。
- Ruff成功、format103 files already formatted、Pyright0 errors/0 warnings、pip check成功、git diff --check成功。
- 固定748c3aaとの旧production/両golden/旧回帰test差分は空。
- 現在検証source hash: e855121fe782df5b69ad5997b84a0bb6fc47c5683e3d5424f9d6c7abd04920d6。
  tracked Pythonと両golden203パスをsortし、UTF8相対path+NUL+LF正規化内容+NULでSHA256集計する。
- 全回帰4106 passed / 3 skipped / 1既存warning / 161.90秒 / exit0、旧11/最終3goldenを含む。値・許容差未変更。
  skipは既存Windows非対応wrapper、warningは既存qint8 fixtureのTypedStorage非推奨。Task 3はLuna APPROVED、別feature GOも確認。
- Lunaは580対象/ASTとfresh新CPU smokeを独立実行し、全11要件・現在owner/旧借用状態の整合・全task同期・責務境界を確認して最終GO。

## 環境とコマンド
2026-10-06、Windows CPU、共有../../venv: Python3.13.15/Torch2.12.1+cpu/NumPy2.4.6/pytest9.1.1。
基準はdocs/experiments/refactoring-baseline.mdとenvironments/golden/windows-cpu。
OMP_NUM_THREADS=MKL_NUM_THREADS=1、TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、FDE_MNIST_DATA_DIR=../../data/mnist。

```text
python -m pytest tests/refactoring/test_held_model_shared_feature_reconnection.py tests/refactoring/test_single_run_dependency_boundaries.py -q -p no:cacheprovider
python -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/held-model-reconnection-final-20261006a --junitxml=../../venv/refactoring-tests/held-model-reconnection-final.xml
python -m ruff check . --output-format concise
python -m ruff format --check . --output-format concise
python -m pip check
```

fresh smoke: PYTHONPATH=srcでgit管理外../../venv/refactoring-tests/held_model_reconnection_smoke.pyを別processで実行。
Windows sandbox上のPyrightはrequire_escalatedと共有venvの絶対pythonpathで子Python起動を許可する。

```powershell
C:/Users/yshin/Research/FedSDA-experiment/venv/Scripts/python.exe -m pyright --pythonpath C:/Users/yshin/Research/FedSDA-experiment/venv/Scripts/python.exe
```

## 全11要件の証拠
|要件|実装/検証|
|---|---|
|1.1|空列無操作/空結果、実旧対照|
|1.2|混合/非昇順/極大IDで最小非負のsource参照を旧選択へ対照|
|1.3|全負で先着source、負の大小で順序を変えない|
|1.4|結果ID順/分類器と概念固有部参照/class数/双方値grad保持、実旧共同学習exact|
|2.1|source binding/shared/head optimizer identityとstate保持、skip呼出順|
|2.2|non-source attach→reset、既に共有済みでも新head optimizer/empty state、旧対照|
|2.3|source共有ownerを全返却bindingへ対応付け、旧shared owner不変/非流用|
|2.4|旧借用optimizer state保持、返却列→新学習binding/batch→実旧3step|
|3.1|全20拒否条件の末尾不正で全input状態保持、source適合検査を全副作用前に実行|
|3.2|reset例外注入で先行接続/resetと現在接続保持、suffix不変|
|3.3|値grad/RNG保持、exact依存guard/fresh新CPU/品質/全回帰gate|

## 旧所見と限界
今回新たな旧正常系不具合は観測していない。旧不具合/改善点はdocs/research/implementation-findings/README.mdで継続追跡。
LEGACY-010の空共有optimizerを変更しない。本操作は対応した非空parameterの共有ownerを入力として要求する。
返却列だけが現在のshared owner対応を表す。旧入力の分類器はlive参照だがshared owner欄は旧値なので、古い対応を学習へ流用しない。
通常構築済みの独立した概念モデルと公開owner契約を前提とし、private/frozen回避・概念層構造の外側改変・並行操作は通常契約外。
途中例外で全体を巻き戻す保証はない。旧全体goldenは旧固定値確認、新旧同値性は部品対照で確認する。新全体run golden完成を主張しない。
