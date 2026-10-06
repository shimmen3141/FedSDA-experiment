# 実測: 準備済み分類器の標本別有界損失

## 範囲
一分類器forwardと標本別有界損失。学習/準備/初期統計のtest-only接続と、登録全体/新client/新全体run完成は区別する。
正本はspec.json/tasks.md、要求revision2・設計revision2・命名revision1。
全3task完了後の別統合gateでLuna DECISION: GO。全11要件、境界・依存方向・接続・共有状態・ファイル計画・実測にblockerなし。主担当completion gateも本scopeでFEATURE_GO VERIFIED。

## Task1
- 未実装moduleのimport RED: ModuleNotFoundError、1 collection error/2.86秒/exit1。
- 初回GREEN75 passed/3.79秒/exit0。nested fixture生成時のprototype warningは予期したものなのでそのfixture構築だけで抑制した。
- 最終単体77 passed/3.28秒/exit0。class2/4/10×singleton/5件×連続/非連続×train/evalの24条件でadapter非ゼロの実旧per_sample_errorとexact一致。一forward/no_grad、値/既存grad/子別flags、逆順、独立結果を確認。
- 型/環境/shape/値/labelの31入力拒否、parameter/meta/nested/class_count拒否、12output違反を確認。極大有限parameterによる非有限logitは明示拒否し状態保持。多クラスの範囲制限しないlogitとinference_modeも確認。
- default dtype=float64/device=meta、gradmode True/False、inputの既存grad・Torch/Python/NumPy RNG保持を4条件で確認。
- Ruff成功、対象format成功、Pyright0 errors/0 warnings/exit0。未実装import RED以外に対象testの失敗なし。
- Lunaの検証context漏れ指摘を採用してscoped deviceへ修正。77 passed/3.41秒/exit0・品質成功。別fresh processで旧setterのstack0→1と新contextの0→0を実証。productionは変更なし。

## Task2
- test-only統合のためRED N/A。class2/4×Adam標準/AMSGrad/SGD×共有更新/凍結の12条件。
- 各3共同更新で全loss/NN値/gradient/共有・個別optimizer stateを実旧へexact照合した後、実旧prepareと新採用共有反映へ接続する。
- 準備後の新評価→既存batch初期統計と、実旧register(identity prepare)の全体/クラス別全fieldを、5件とsingletonでexact照合。評価/統計の前後で全parameter/gradient/optimizerの実体とstate/RNG保持。
- 対象89 passed/3.94秒/exit0、Ruff/format成功。production変更なし。

## Task3
- exact依存の27注入case（22拒否/5許可）を先行RED: 8 failed/19 passed/585 deselected/0.12秒/exit1。
- 個別symbolを解決し、bare importも拒否するguardを新moduleへ追加。対象＋AST701 passed/4.27秒/exit0。
- fresh新CPU processの実分類器→損失→既存初期統計はbinary/multiclassで成功/exit0、旧package非importを確認。
- 全Ruff成功、format109 files already formatted、Pyright0 errors/0 warnings、pip check成功、git diff --check成功。
- 固定748c3aaとの旧production/両golden/旧回帰test差分は空。最終全pytestは4345 passed/3 skipped/1既存warning/147.96秒/exit0。旧11/最終3goldenを含み、値・許容差を変更していない。
- JUnit classifier-loss-full.xmlは4348 tests/0 failures/0 errors/3 skipped。skipは既存Windows wrapper、warningは既存qint8 fixtureのTypedStorage非推奨。
- source209 Python/両goldenパスのhashは437a69bc1dae9b9cc3f8e94f804f8d531f5d1bcce8ebf92aa86b01fc19032637。Git追跡Pythonと両golden（新2ファイルを含む）をsortしUTF8相対path+NUL+LF正規化内容+NULでSHA256集計。要求rev2/設計rev2/命名rev1の承認hash一致を確認。
- 新しい旧正常経路の不具合は今回の範囲で観測していない。development findingは新testのcontext復元問題で、旧研究実装の修正を含まない。
- Luna独立701件/fresh新CPU/品質・境界、全回帰/JUnit点検後にTask3 VERDICT: APPROVED。全3tasks完了後、別feature統合レビューもDECISION: GO。

## 環境と再実行手順
2026-10-06、既存Windows CPU基準、共有../../venv（Python3.13.15/Torch2.12.1+cpu/NumPy2.4.6/pytest9.1.1）。基準と版はdocs/experiments/refactoring-baseline.md・environments/golden/windows-cpu。
OMP_NUM_THREADS=MKL_NUM_THREADS=1、MPLCONFIGDIR=../../venv/matplotlib-cache、TMP/TEMP=../../venv/refactoring-tests、FDE_MNIST_DATA_DIR=../../data/mnist。

```text
python -m pytest tests/refactoring/test_classifier_bounded_loss_evaluation.py -q -p no:cacheprovider
python -m pytest tests/refactoring/test_classifier_bounded_loss_evaluation.py tests/refactoring/test_single_run_dependency_boundaries.py -q -p no:cacheprovider
python -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/classifier-loss-full-20261006 --junitxml=../../venv/refactoring-tests/classifier-loss-full.xml
python ../../venv/refactoring-tests/classifier_bounded_loss_smoke.py
python -m ruff check .
python -m ruff format --check .
python -m pyright --pythonpath C:/Users/yshin/Research/FedSDA-experiment/venv/Scripts/python.exe
python -m pip check
```

全pytest/PyrightはWindows tmp ACL/子Python起動のため必要時にrequire_escalated。旧goldenは旧packageを実行し、新全体runのgolden完成を意味しない。

## 全11要件trace
|要件|証拠|
|---|---|
|1.1|class2実旧24条件中の8条件、12統合中の6条件|
|1.2|class4/10実旧16条件、class4の6統合条件、有限logit範囲無制約|
|1.3|全24条件のsingleton/逆順/非連続、全12統合の5件/singleton|
|1.4|正常の一forward/no_grad記録、先行検証のforward0|
|2.1|31入力違反と4parameter/環境/クラス/nested拒否|
|2.2|12output違反と実有限parameterからの非有限logit拒否|
|2.3|通常親/子flags、値/grad/入力/Random/default/context保持、12統合optimizer保持|
|2.4|shape[N]のCPU float32/grad_fnなし、結果変更が入力/parameterへ波及しない|
|3.1|実旧損失の全24条件と12統合、実旧register全統計field一致|
|3.2|22拒否/5許可exact依存注入、bare import拒否、production境界点検|
|3.3|全4345回帰/旧11最終3golden・固定旧差分空・品質/fresh新CPU/sourcehash|
