# 統合検証: 採用候補の共有特徴学習の反映

## 範囲と正本
上位で採用済みとされた候補の値を上位指定のactive共有部へ反映し、候補を接続して個別optimizerだけをresetする。
共有値反映はSharedFeatureExtractor公開method、順序はtraining公開操作。登録・ID・統計生成・候補採否・新client/全体runは後続。
承認/進捗はspec.json、正式名はnaming revision1、独立レビュー/採否はreview.md、実測は本文。
全3tasksはLuna APPROVED。全checkbox・spec・roadmap同期後、Lunaの独立最終feature DECISION: GOを主担当が採用。全9/9要件にblockerなし。

## 実測
- Task1: 未実装moduleのimport RED、1 collection error/3.60秒/exit1。GREEN25 passed/2.86秒/exit0、関連82 passed/4.42秒。Luna独立105 passedと品質/境界確認、Task APPROVED。
- 空hidden tuple/1層/2層×初期共有有無6、16入力不正、途中reset例外、双方直接構造拒否2。値・参照・grad/RNG・旧借用個別optimizer stateの保持を検証。
- Task2: test-onlyのためRED N/A。class2/4×Adam standard/AMSGrad/SGD×初期共有有無12条件で共有/個別state蓄積→実旧_prepare_model_for_registration→共有更新/凍結/更新3step。
- 全loss/NN値/grad/shared/head optimizer stateがexact一致。activeparameter/grad参照・共有optimizer/state維持、同一候補共有ownerの継続利用、別owner非再利用/不変、旧借用個別optimizer不変を確認。37 passed/3.31秒/exit0、Luna独立37 passed、Task APPROVED。
- Task3 AST: 18禁止/6許可の先行RED10 failed/14 passed/531 deselected/0.11秒/exit1。exact guard後、対象+AST592 passed/4.00秒/exit0。
- fresh新CPU候補反映→共同更新smoke成功/exit0。sharedparameter参照/候補値反映/個別reset/旧package非importを確認。
- Ruff成功、format105 files already formatted、Pyright0 errors/0 warnings、pip check成功、git diff --check成功。
- 固定748c3aaとの旧production/両golden/旧回帰test差分は空。golden値・許容差未変更。
- 現在sourcehash: 80459180777488ad3ef122e41210ebc84aa64f43be12dfbcf9bf0c950f1235dc。
  tracked Pythonと両golden205パスをsortし、UTF8相対path+NUL+LF正規化内容+NULでSHA256集計。
- 全回帰4167 passed/3 skipped/1既存warning/146.59秒/exit0。旧11/最終3goldenを含み値・許容差未変更。skipは既存Windows非対応wrapper、warningは既存qint8 fixtureのTypedStorage非推奨。
- この範囲で旧正常経路の新しい不具合は観測していない。既存LEGACY-001〜010の判断は変更しない。
- Luna最終GOでは全実測/JUnitとtask内容hash、9要件、所有状態・公開境界を確認し、fresh新CPU smokeを独立再実行して成功。対象範囲は本候補準備に限る。

## 環境と再実行
2026-10-06、Windows CPU、共有../../venv: Python3.13.15/Torch2.12.1+cpu/NumPy2.4.6/pytest9.1.1。
基準はdocs/experiments/refactoring-baseline.mdとenvironments/golden/windows-cpu。
OMP_NUM_THREADS=MKL_NUM_THREADS=1、TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、FDE_MNIST_DATA_DIR=../../data/mnist。

```text
python -m pytest tests/refactoring/test_adopted_candidate_shared_feature_integration.py tests/refactoring/test_single_run_dependency_boundaries.py -q -p no:cacheprovider
python -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/adopted-candidate-final-20261006a --junitxml=../../venv/refactoring-tests/adopted-candidate-final.xml
python ../../venv/refactoring-tests/adopted_candidate_integration_smoke.py
python -m ruff check .
python -m ruff format --check .
python -m pyright --pythonpath C:/Users/yshin/Research/FedSDA-experiment/venv/Scripts/python.exe
python -m pip check
```
Pyright/全pytestはWindows tmp ACL対策にrequire_escalatedを使用。全suiteの旧goldenは旧packageを実行するため、新全体run goldenの完成を意味しない。
smoke scriptはgit外の検証用一時ファイル。同じ公開API接続はtracked実NN対照testでも検証する。

## 全9要件trace
|要件|証拠|
|---|---|
|1.1|単体6条件と実旧prepare12条件、既存active参照モデルの値も一致|
|1.2|activeparameter/grad参照、共有optimizer/state保持、3共同step対照|
|1.3|source値grad/個別値grad/旧owner/RNG保持|
|2.1|実順序copy→attach→reset、実旧prepare対照|
|2.2|同じownerの現在optimizer交換と旧借用optimizer/state保持|
|2.3|初期共有済み条件でも個別reset|
|3.1|16不正条件/双方直接構造不正で全状態保持|
|3.2|注入reset失敗で値反映と接続保持・例外伝播|
|3.3|AST exact境界/fresh新CPU/固定旧差分、登録と学習を含まない公開API|
