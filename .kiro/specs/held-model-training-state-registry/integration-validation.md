# 統合検証: 保有モデルの学習状態管理

## 範囲と正本
ID指定の保有NNと個別optimizer管理器の一覧、登録/置換/取得/現在bindingを移植する。
一覧構造はregistry、現在個別optimizerは既存owner、共有optimizerと学習実行は外側に置く。
候補採否/初期学習/標本/登録統計/parameter送信snapshot/ID対応/正式登録全体/新client・新全体runは含まない。
承認と進捗はspec.json/tasks.md、要求revision3、設計/命名revision1、採否はreview.md。

## 実測
- Task1: 新module未存在のimport RED、1 collection error/3.32秒/exit1。GREEN20 passed/2.74秒/exit0、Ruff/format成功。Luna独立20 passedとTask APPROVED。
- 正常ID・初出順・同ID置換・snapshot構造分離・readonly record・取得エラー・7不正ID・10不正対応・reset後の新binding/旧借用保持を検証。値/grad/state/RNGを保持。
- Task2: test-onlyでRED N/A。実旧BaseClient._register_trained_new_modelのmodels境界を、準備済み実体とidentity prepareで呼び、(-7,4,-7)の登録/置換順を比較する。
- class2/4×Adam standard/AMSGrad/SGD×共有更新/凍結の12条件。step0学習→実旧同共有attachによる個別reset/新owner reset→step1/2学習。
- registry順(-7,4)と標本store順(4,-7)を区別し、新bindingはIDで対応する。全loss/NN値/grad/共有・個別optimizer state/終端Random一致、古いbinding state保持。32 passed/3.24秒/exit0、Luna独立32 passed・Task APPROVED。
- Task3 AST: 16禁止/7許可の先行RED9 failed/14 passed/555 deselected/0.10秒/exit1。exact guard後、対象＋AST610 passed/4.06秒/exit0。
- Lunaのmodule import受理の指摘を採用。7追加拒否caseが修正前7 failed/0.10秒。fromの具体symbolだけの検査とbare import拒否へ修正し、対象＋AST617 passed/4.20秒/exit0。実装の数値処理は変更なし。
- fresh新CPU登録→reset→現在binding→3共同更新smoke成功/exit0。旧package非importを確認。
- Ruff成功、format107 files already formatted、Pyright0 errors/0 warnings、pip check成功、git diff --check成功。
- 固定748c3aaとの旧production/両golden/旧回帰test差分は空。golden値・許容差を変更していない。
- 修正後sourcehash: b9082c1726b904b40034daa21b2febb984de26418af34faa60be103c528fe463。
  tracked Pythonと両golden207パスをsortし、UTF8相対path+NUL+LF正規化内容+NULでSHA256集計。承認済み要求/設計/命名hashの一致を確認。
- 全回帰4222 passed/3 skipped/1既存warning/143.51秒/exit0。旧11/最終3goldenを含み値・許容差未変更。skipは既存Windows非対応wrapper、warningは既存qint8 fixtureのTypedStorage非推奨。独立Task3/featureレビューは次のゲート。
- guard修正後の最終全回帰4229 passed/3 skipped/1既存warning/361.15秒/exit0。修正前のsourcehashはa278ed7da2de717810982abb05ee34adff41d1f407ed13347a402fcf86720737。最終証拠はheld-state-final-b.xml。
- 最終件数はstdoutとJUnitの4232 tests/0 errors/0 failures/3 skippedで一致する。361.15秒はpytestのstdout表示、JUnit suite.timeは149.081秒で、時間表示差の原因は調査していない。Lunaは全suiteを再実行せずXMLを変更していない。時間を性能指標として使わない。
- Luna独立target617/fresh smoke/品質を確認し、module importの抜けの解消と命名文面の再承認を確認。Task3最終VERDICT: APPROVED。全checkbox完了後のfeature GOは別ゲート。
- 今回の範囲で旧正常経路の新しい不具合は観測していない。既存LEGACY-001〜010は判断/修正状態を変更しない。

## 環境と再実行
2026-10-06、Windows CPU、共有../../venv: Python3.13.15/Torch2.12.1+cpu/NumPy2.4.6/pytest9.1.1。
基準はdocs/experiments/refactoring-baseline.mdとenvironments/golden/windows-cpu。
OMP_NUM_THREADS=MKL_NUM_THREADS=1、TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、FDE_MNIST_DATA_DIR=../../data/mnist。

```text
python -m pytest tests/refactoring/test_held_model_training_state_registry.py tests/refactoring/test_single_run_dependency_boundaries.py -q -p no:cacheprovider
python -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/held-state-final-20261006b --junitxml=../../venv/refactoring-tests/held-state-final-b.xml
python ../../venv/refactoring-tests/held_model_training_state_registry_smoke.py
python -m ruff check .
python -m ruff format --check .
python -m pyright --pythonpath C:/Users/yshin/Research/FedSDA-experiment/venv/Scripts/python.exe
python -m pip check
```

全pytest/PyrightはWindows tmp ACL/子Python起動のためrequire_escalated。smokeはgit外の一時検証scriptであり、同じ公開API接続はtracked対照testで検証する。
旧goldenは旧packageを実行するため、新全体runのgolden完成を意味しない。

## 全11要件trace
|要件|証拠|
|---|---|
|1.1|空状態と空binding tuple|
|1.2|負/ゼロ/正/大整数の実体保持、実旧登録12条件|
|1.3|同ID別NN置換・位置保持・古い記録保持、実旧順序対照|
|1.4|readonlyrecord、tuple構造分離、取得で登録増加なし|
|1.5|7不正ID/10対応不正の事前拒否と既存一覧/学習状態保持|
|1.6|KeyError(-7)、型拒否と空一覧維持|
|2.1|複数状態の現在optimizer binding、旧NN/更新接続|
|2.2|state蓄積後reset→新旧binding寿命、12条件step1/2対照|
|2.3|単体値/grad/state/RNG保持、共有owner外側保持、実NN対照|
|3.1|実旧registerのmodel一覧順と12条件3stepのexact数値/状態/Random対照|
|3.2|exact AST、fresh新CPU、品質、固定旧差分/hash、最終全4229回帰成功|
