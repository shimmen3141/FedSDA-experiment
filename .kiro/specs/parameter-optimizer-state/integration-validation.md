# 統合検証: optimizer状態所有と明示リセット

## 範囲と正本
全3tasks完了、Luna独立APPROVEDと最終feature統合GOを主担当が採用。9/9要件にblockerなし。
固定parameter/設定を借用し、現在optimizerと成功後reset交換だけ所有する。
モデル生成・共有接続・reset時機・parameter差し替え・学習・state移送・空共有optimizerは後続。
spec.jsonが承認/進捗、naming revision3が正式名、review.mdが採否の正本。

## 実測
- Task1: 新module未存在でImportError、1 collection error/3.35秒/exit1。実装後23 passed/4.56秒。
- 実旧SharedBackboneMLP.reset_optimizerへ6設定×更新0/2回を直接比較。groups/state/parameter値/grad一致、reset後3回同勾配（None混在）でもexact一致。
- 値/grad参照/parameter順保持、readonly現在参照、外側更新の反映、固定条件へ戻す再生成、旧optimizer state保持、繰り返しresetを確認。
- 不正params/settings・不正dtypeの再検証、生成例外で現optimizer/state/値/grad保持、Python/Torch RNG不変を確認。
- 対象Ruff/format成功、Pyright0 errors/0 warnings、diff-check成功。固定748c3aaとの旧production/golden/旧回帰test差分なし。

- Task2: 47 passed/5.23秒/exit0。class2/4×Adam standard/AMSGrad/SGD×reset4種の24条件で各3step、共有更新/凍結を混在させる。全loss/NN値/grad/optimizer stateが実旧共同更新とexact一致。
- 個別だけのresetは実旧attach_backbone、全体resetは実旧reset_optimizer、共有だけは実旧component builderで共有optimizerだけ再生成する部分へ照合。reset時機の移植とは区別する。
- 別owner/旧optimizerと旧HeldModelTrainingBinding/参加batchの参照が自動交換されないことを確認。reset後の学習は現在参照で新recordを作成するtest-only接続。既存数値production変更なし。
- Task3: exactAST14禁止/6許可の先行RED9 failed/11 passed/487 deselected/0.12秒。guard追加後、対象+AST554 passed/4.57秒/exit0。
- fresh新CPU smokeは生成→step→reset→step成功、旧参照state1/新state0→1・値/grad保持・RNG不変・旧非importを確認、exit0。
- Lunaは最終featureレビューで新CPU smokeを独立再実行し、現在sourcehash一致、全9要件と境界を確認してGO。
- Ruff成功、format100 files already formatted、Pyright0 errors/0 warnings、pip check成功、git diff --check成功。
- 全回帰4000 passed/3 skipped/1既存warning/173.13秒/exit0、旧11/最終3goldenを含む。値・許容差未変更。
  skipは既存Windows非対応wrapper、warningは既存qint8 fixtureのTypedStorage非推奨。
- 固定748c3aaと旧production/両golden/旧回帰test差分は空。現在検証source hash: 552ef70c243aeed2e2bf4672892d7680d5389659070d6ef08e188fc150124562。
  tracked Pythonと両golden200パスをsortし、UTF8相対path+NUL+LF正規化内容+NULでSHA256集計する。

## 環境とコマンド
2026-10-06、Windows CPU、共有../../venv: Python3.13.15/Torch2.12.1+cpu/NumPy2.4.6/pytest9.1.1。
基準はdocs/experiments/refactoring-baseline.mdとenvironments/golden/windows-cpu。
OMP_NUM_THREADS=MKL_NUM_THREADS=1、TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、FDE_MNIST_DATA_DIR=../../data/mnist。

```text
python -m pytest tests/refactoring/test_parameter_optimizer_state.py tests/refactoring/test_single_run_dependency_boundaries.py -q -p no:cacheprovider
python -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/optimizer-state-final-20261006a --junitxml=../../venv/refactoring-tests/optimizer-state-final.xml
python -m ruff check . --output-format concise
python -m ruff format --check . --output-format concise
python -m pip check
```
fresh smoke: PYTHONPATH=srcでgit管理外../../venv/refactoring-tests/optimizer_state_smoke.pyを別processで実行する。

### Windows Codex sandbox上のPyright
子Python起動の制約があるため、toolではsandbox_permissions=require_escalatedで下記完全commandを実行する。
作業directoryは.worktrees/refactoring。一般のローカル端末はsandbox外なので同じcommandを直接使う。
対象縮小や設定変更でimport未解決を回避しない。

```powershell
C:/Users/yshin/Research/FedSDA-experiment/venv/Scripts/python.exe -m pyright --pythonpath C:/Users/yshin/Research/FedSDA-experiment/venv/Scripts/python.exe
```

## 全9要件の証拠
|要件|実装/検証|
|---|---|
|1.1|既存factory生成/実旧reset12条件でgroups/state/refs順/値/grad照合|
|1.2|同じ現在参照/外部stepのstate反映/readonly property|
|1.3|不正params/settings拒否/値grad不変、factoryへ検査委譲|
|2.1|0/2学習後に新identity/empty state、同固定params設定/値grad維持|
|2.2|24実NN条件の共有/一概念/全体/なしで別owner参照とstate保持|
|2.3|生成例外/不正dtype再検証で現optimizer参照・state保持|
|2.4|旧binding/参加batch保持、現在参照で新recordを作り実旧と再学習exact一致|
|3.1|params借用/optimizer所有/新CPU生成resetでPython/Torch RNG不変|
|3.2|上位非依存/明示resetだけ/AST禁止許可/fresh新CPU/全回帰gate|

## 旧所見と限界
今回新たな旧正常系不具合を確認していない。既存LEGACY-010/空共有optimizerの拒否は変更しない。
reset後の旧bindingは旧optimizerのまま。上位が現在参照を取得し新bindingを作る。
parameter/settingsは借用。frozen/private回避や並行resetは通常契約外。
旧全体goldenは旧コード固定値の確認、新旧同値性は部品対照で確認する。新全体run golden完成を主張しない。
