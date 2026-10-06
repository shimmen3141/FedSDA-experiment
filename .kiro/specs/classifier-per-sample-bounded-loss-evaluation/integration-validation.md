# 実測: 準備済み分類器の標本別有界損失

## 範囲
一分類器forwardと標本別有界損失。学習/準備/初期統計のtest-only接続と、登録全体/新client/新全体run完成は区別する。
正本はspec.json/tasks.md、要求revision2・設計revision2・命名revision1。

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

## 環境と再実行
2026-10-06、既存Windows CPU基準、共有../../venv（Python3.13.15/Torch2.12.1+cpu/NumPy2.4.6/pytest9.1.1）。基準と版はdocs/experiments/refactoring-baseline.md・environments/golden/windows-cpu。
OMP_NUM_THREADS=MKL_NUM_THREADS=1、MPLCONFIGDIR=../../venv/matplotlib-cache、TMP/TEMP=../../venv/refactoring-tests、FDE_MNIST_DATA_DIR=../../data/mnist。

```text
python -m pytest tests/refactoring/test_classifier_bounded_loss_evaluation.py -q -p no:cacheprovider
python -m ruff check .
python -m ruff format --check .
python -m pyright --pythonpath C:/Users/yshin/Research/FedSDA-experiment/venv/Scripts/python.exe
```

全pytest/PyrightはWindows tmp ACL/子Python起動のため必要時にrequire_escalated。旧goldenは旧packageを実行し、新全体runのgolden完成を意味しない。
