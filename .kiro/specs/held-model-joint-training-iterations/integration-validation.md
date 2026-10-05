# 統合検証: 保有モデルの共同学習反復

## 判定範囲
全3task完了、Lunaの最終統合判定GOを主担当が採用した。全12要件にblockerはない。
回数を外側から受け取り、同じ借用モデル・optimizer・Randomで抽出→ID対応→共同更新を繰り返す。
標本/optimizer所有、回数算出/pending/interval、候補進行/同期/診断counter・PCGrad、新全体runは範囲外。
承認・現在地の正本はspec.json、採否はreview.md。

## 環境と手順
2026-10-05、Windows CPU、共有../../venv: Python3.13.15、Torch2.12.1+cpu、NumPy2.4.6、pytest9.1.1。
固定環境の再構築はdocs/experiments/refactoring-baseline.mdとenvironments/golden/windows-cpuを参照。
OMP_NUM_THREADS=1、MKL_NUM_THREADS=1、FDE_MNIST_DATA_DIR=../../data/mnist。
TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache。

```text
python -m pytest tests/refactoring/test_held_model_joint_training_iterations.py -q -p no:cacheprovider
python -m pytest tests/refactoring/test_single_run_dependency_boundaries.py tests/refactoring/test_held_model_joint_training_iterations.py -q -p no:cacheprovider
python -m pytest tests -q -p no:cacheprovider --basetemp=../../venv/refactoring-tests/training-iterations-final-20261005b --junitxml=../../venv/refactoring-tests/training-iterations-final.xml
python -m ruff check . --output-format concise
python -m ruff format --check . --output-format concise
python -m pyright --pythonpath <共有venvのpython.exe絶対パス>
python -m pip check
```

## 実測
- Task1 missingmoduleの実RED: collection error/exit1。実装後72 passed/5.78秒、Luna独立72 passed/4.83秒。
  二値/4class×Adam/AMSGrad/SGD×共有更新/凍結×0/1/4回×batch1/3、population9/100、非昇順ID・逆順binding。
  実旧samplerと_train_heads_togetherの一回の複数反復呼出しで、全抽出Tensor/順序/loss/parameter/grad/optimizer/RNGをexact照合。数値本体のmockなし。
- Task2はtest-only。主担当94 passed/4.89秒、Luna独立94 passed/5.33秒。0未参照、事前拒否/skip、環境/標本/frozen参照保持、空共有の独立BCE/Adamstep、2回目拒否後の先行更新/2draw state非rollback。
  初回Luna指摘のtest device context漏れをscope方式へ修正。旧共同更新の操作順testも同時実行して171 passed/4.96秒。
- ASTの22禁止/14許可を先行し、RED16 failed/20 passed/403 deselected/0.13秒。
  exact公開symbol許可を一般stdlibの前へ配置し、対象94+AST439の533 passed/5.15秒。
- fresh CPU smokeはPYTHONPATH=srcの別processで実施しexit0。4loss/全個別Adamstep=4、CPU32、parameter/RNG更新、sys.modulesに旧packageなしを確認。
  一時scriptはgit管理外の共有venv/refactoring-tests/training_iterations_smoke.pyに保存。
- Ruff check成功、format93 files already formatted、Pyright0 errors/0 warnings、pip check整合、git diff --check成功。
- 最終全回帰: 3721 passed/3 skipped/1 warning/178.13秒/exit0。旧11/最終3goldenを含む。
  3skipは既存Windows非対応wrapper、1warningは既存qint8 fixture deepcopyのTypedStorage非推奨。
  最初の全回帰は修正前test device context漏れにより旧操作順2test失敗（3719 passed/3 skipped）を観測した。golden不一致ではない。Luna指摘を修正した最終実測は全件通過。
- git diff 748c3aa -- federated_drift_experiment tests/regression_golden.json tests/proposed_regression_golden.json tests/test_regression.py tests/test_proposed_regression.pyは空。旧production/golden/許容差は未変更。
- 検証ソースhash: bbac34bb295399188bc3a39f8c568109bd788abf573106561eabbfa9b2b3616c。
  tracked Pythonと両goldenの193パスをsortし、各UTF8相対パス+NUL+CRLFをLFへ正規化した内容+NULをSHA256集計。
- requirements/design/namingの承認LF hash一致とUTF8を確認。完成checkbox変更はtasksのcurrent_sha256_lfへ別記する。
- Luna Task3 APPROVED、独立対象+AST533 passed/7.97秒。独立fresh CPUの実4回Adam更新も成功し、旧package非importを確認。全12条件・タスク間契約・範囲を最終GOとして採用した。

## 12条件の証拠
| 条件 | 実装・検証 |
|---|---|
| 1.1 | 毎回実sampler→共同更新、72条件の全回抽出比較 |
| 1.2 | ID lookupだけにbinding使用、逆順bindingでも非昇順標本列を保持 |
| 1.3 | 3種empty/未保有/不足、4抽出試行・更新ゼロ・RNG不変 |
| 1.4 | 0回は全他引数Noneで()、sampler/更新未呼出 |
| 1.5 | 実旧loss hook加重集計と順序tuple exact |
| 2.1 | bool/負/小数/None/派生整数をdraw前拒否 |
| 2.2 | list/異型record/bool・派生ID/重複をdraw前拒否 |
| 2.3 | 同一NN/optimizer/Randomの終端stateと実旧複数反復一致 |
| 2.4 | 標本clone一致、frozen/defaultなし/borrowed、失敗時先行state非rollback |
| 2.5 | freeze/空共有・None optimizer、SGDのambient meta/float64とPython/NumPy/Torch RNG保持 |
| 3.1 | 実旧複数反復のbatch/loss/全parameter/grad/optimizer/RNG exact |
| 3.2 | exact AST・禁止許可注入・fresh CPU旧非import |

## 限界と後続
ambient metaで未初期化Adamを初めてstepするとTorch自身がmeta step scalarを作るため、meta初回Adamの正常動作までは保証しない。
CPU Adam/AMSGradの継続更新は実旧比較と空共有の標準BCE/stepで確認した。CPU32という既存学習契約は維持する。
通常経路の新しい旧不具合は観測していない。既存LEGACY findingsと旧productionの修正状態は変えない。
ローカルgoldenは旧実装の固定値確認、今回の新旧同値性は部品の実旧対照テストで確認する。新packageの全体run goldenは後続。
GitHubホストのgolden診断は別環境の観測であり、ローカル一致や新旧部品比較の代替にしない。
