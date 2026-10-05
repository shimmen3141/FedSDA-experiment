# 統合検証: モデル別学習標本の保持

## 範囲と正本
全3tasks完了、Luna独立APPROVEDと最終feature統合GOを主担当が採用。10/10要件にblockerなし。
保持/追加/snapshot/サーバID一回対応のみ。正式登録pop/上書き、統計、評価store、NN/optimizer、全体runは対象外。
spec.jsonが承認/進捗、naming revision1が正式名、review.mdが独立判定と採否の正本。

## 実測
- Task1: 新module未存在でModuleNotFoundError、1 collection error/3.23秒/exit1。実装後45 passed/1.97秒/exit0。
- 空/混合列×8対応表の16条件を実旧BaseClient追加/emptyextend/apply_server_mappingへ全標本参照と順序で照合。
- snapshotの構造分離とpayload借用、空列作成、型派生/bool拒否、後尾不正入力でstate保持、巨大ID、容量制限なし/重複保存を確認。
- Task1対象Ruff成功、Pyright0 errors/0 warnings、git diff --check成功。固定748c3aaとの旧production/両golden/旧回帰test差分なし。

- Task2: 94 passed/2.20秒/exit0。48条件（4対応表×batch1/3/7×seed2×保有集合2）で各3反復の実旧samplerと新snapshot→samplerを照合。全batch Tensor/モデル順/終端Random一致、保持器stateとglobal Random不変。
- 無効Tensorは保持時に参照だけ保存し、未保有/不足時skip、参加時にsamplerがRNG消費前に拒否する。接続test以外のproduction変更なし。
- Task2追加後のRuff unused loop指摘は不要なループ名を除去して解消。全Ruff/diff-check成功。
- Task3: exactAST14禁止/4許可注入、先行RED6 failed/12 passed/469 deselected/0.08秒。guard追加後対象+AST581 passed/2.94秒/exit0。
- fresh新package CPU smokeは保持→一回対応→抽出成功、旧package非import/CPU float32/モデル順/標本数を確認、exit0。
- Ruff成功、format98 files already formatted、Pyright0 errors/0 warnings、pip check成功、git diff --check成功。
- 全回帰3933 passed/3 skipped/1既存warning/143.24秒/exit0、既存11/最終3goldenを含む。
  3skipは既存Windows非対応wrapper、warningは既存qint8 fixtureのTypedStorage非推奨。値/許容差未変更。
- 固定旧748c3aaとの旧production/両golden/旧回帰test差分なし。検証source hash: e2f1c0ccb60c06c3bb81f98d016cfe3a867e9fa31404296981e0c543299a5a2c。
  tracked Pythonと両golden198パスをsortし、UTF8相対path+NUL+LF正規化内容+NULで集計。

## 環境と再実行
2026-10-06、Windows CPU、../../venv、Python3.13.15/Torch2.12.1+cpu/NumPy2.4.6/pytest9.1.1。
基準はdocs/experiments/refactoring-baseline.mdとenvironments/golden/windows-cpu。
OMP_NUM_THREADS=MKL_NUM_THREADS=1、TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、FDE_MNIST_DATA_DIR=../../data/mnist。

```text
python -m pytest tests/refactoring/test_model_training_sample_storage.py tests/refactoring/test_single_run_dependency_boundaries.py -q -p no:cacheprovider
python -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/sample-storage-final-20261006a --junitxml=../../venv/refactoring-tests/sample-storage-final.xml
python -m ruff check . --output-format concise
python -m ruff format --check . --output-format concise
python -m pyright --pythonpath <共有venv python.exe絶対path>
python -m pip check
```
fresh smoke: PYTHONPATH=srcでgit管理外../../venv/refactoring-tests/sample_store_smoke.pyを別process実行。

### Windows上のPyright再現条件
Codexのsandbox内ではPyrightが型情報収集用の子Pythonを起動できず、依存import未解決が出ることがある。
同じ共有venvの絶対pythonpathを指定し、Codex toolでは`sandbox_permissions=require_escalated`で実行する。
通常のローカル端末ではsandbox制約がないため、下記コマンド自体を使う。

```powershell
C:/Users/yshin/Research/FedSDA-experiment/venv/Scripts/python.exe -m pyright --pythonpath C:/Users/yshin/Research/FedSDA-experiment/venv/Scripts/python.exe
```
作業directoryは.worktrees/refactoring。共有venvにはNumPy/Torchを導入済み。対象縮小/型検査設定変更では回避しない。
主担当・Lunaともこの条件で0 errors/0 warningsを再確認し、Task3 APPROVED。

## 全10要件の証拠
|要件|実装/検証|
|---|---|
|1.1|空初期snapshot/二回読出しでstate不変|
|1.2|実旧absorbで交互追加/負ID/重複の全参照と順序一致|
|1.3|空extendと同じ空列作成/空読出しでキー非作成|
|1.4|snapshot後追加/対応でも元列保持、record identity/payload共有/frozen|
|1.5|全入力事前検証/後尾不正でstate保持/無効payloadはsamplerに委譲|
|2.1|実旧16条件/連鎖と循環は一回対応/未指定保持|
|2.2|宛先衝突/空列/巨大ID/重複60標本の順序と参照保持/無容量制限|
|2.3|exact dictと未使用を含むキー/値型拒否/全state保持|
|3.1|既存record形式から実samplerへ48条件×3反復、旧batch/RNG一致|
|3.2|exact依存guard/保持器の副作用なし/CPU fresh smoke/全回帰と品質gate|

## 旧所見と限界
新たな旧正常系の不具合は今回確認していない。正式登録の上書きは別責務として後続確認する。
payloadの完全不変性/deepcopyや、MemoryError・非同期更新のrollbackを保証しない。
旧全体goldenの確認と新旧部品対照を区別する。新全体run golden移植の完了とは主張しない。
