# 実測: 警報後の候補検証標本の観測

2026-10-07、共有../../venvのWindows CPU基準環境。OMP/MKL各1、TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、FDE_MNIST_DATA_DIR=../../data/mnist。主担当はClaude Code、独立レビューはcodex exec -m gpt-6-lunaで起動したレビュー担当。

## Task1

- REDより前の確認: 実旧_snapshot_reference_modelsと実旧_observe_forward_validationを実clientで規定4件まで実行できた（結果はresearch.md）。
- RED: testを書いた後、実装ファイルが存在しない状態（runtime/に該当fileが0件）でpytestを実行し、ModuleNotFoundError、1 collection error/2.01秒、exit1。実装はその後に書いた。
- 実装後の初回実行は35 passed/12 failed/3.83秒。失敗12件は全て12条件testの最後の乱数不変assertで、数値の照合は全て通っていた。原因はtest側: 乱数状態を記録した後に参照モデルを生成しており、モデル生成の初期化がtorch乱数を消費した。実旧の参照複製（_snapshot_reference_models→_new_model）も同じくtorch乱数を消費する。記録位置を参照生成の後へ移して47 passed/3.56秒。productionは無変更。
- 検出力の確認: `../../venv/refactoring-tests/post_alarm_candidate_validation_sample_observation_red_evidence.py`で実装を一時的に差し替え、元へ戻して実装のLF sha256 `9b868be80e4f75b136df2fc4d6ed52a41ad7a15856732946aa76e2cad961f84b`が変わらないことを確認した。

|差し替えた実装|結果|
|---|---|
|何もせずFalseを返すstub|47 failed|
|収集へ追加しない|31 failed|
|常に未到達を返す|22 failed|
|常に到達を返す|4 failed|
|参照を候補の分類器で評価する|24 failed|
|候補を先頭の参照の分類器で評価する|24 failed|
|参照損失をモデル間で入れ替える|22 failed|
|複数行の標本を先頭行で受理する|2 failed|
|参照の評価より前に収集へ追加する|24 failed|
|dictの派生型の対応を受理する|1 failed|
|正しい実装|47 passed|

- 実旧対照4条件（class2/4×規定件数2/4）。各観測回の後に候補損失列・参照損失列（値・順序・ID順）・到達・最後の標本位置を実旧sessionと照合。候補と2つの参照が互いに異なる損失列を持つ入力であることを確認。分類器の学習mode（混在させた）・parameter/grad・torch/random/NumPy乱数の不変。
- 拒否24条件: 収集の型3、対応の型3、特徴5（2行・1次元はforwardより前に拒否されることを呼出し回数0で確認）、ラベル3、候補/参照の分類器2、標本位置4、参照IDの集合と型3、到達済み1。1件を正常観測した後の収集snapshotが各拒否で不変。呼出順1（候補→参照→収集1回）。
- AST RED: 注入契約追加直後10 failed/971 passed/1.01秒、exit1。exact4symbol guard追加後、対象＋AST 1028 passed/4.50秒。
- Ruff check成功/format 139files整形済み、Pyright基準venv明示0 errors/0 warnings。

## Task2

- 観測→評価→確定の照合6条件（class2/4×履歴平均なし/現行モデル/別モデル）は初回実行でGREEN。実旧は4件目の観測で確定まで進む。新は到達の戻り値を見て、収集snapshotの損失列を既存の評価関数へ渡し、既存の確定を適用（test-only接続）。旧判定と評価結果の一致、結果種別↔旧action、全owner状態を上流の確定testのassertで照合。履歴平均が現行モデルなら維持、別モデルなら再利用、なしなら棄却または採用（実損失による）。
- 12条件（class2/4×Adam標準/AMSGrad/SGD×共有部更新有無）: 共同更新→その時点の値で参照を固定→観測4件と確定（再利用）→共同更新2回。loss、全値/grad、optimizer state、学習計数、標本列、統計が一致。参照生成後の観測・評価・確定・学習で乱数が不変。
- fresh `../../venv/refactoring-tests/post_alarm_candidate_validation_sample_observation_cpu_smoke.py`成功、exit0。class2/4×履歴平均3通りで学習→収集開始→観測4件（到達はFalse,False,False,True、分類器不変）→評価→確定→学習。結果種別は棄却・維持・再利用を観測。旧importなし。
- 後続specへの引継ぎ: 実旧の参照複製はモデル生成でtorch乱数を消費する。候補sessionの開始を移植するときは、この乱数消費の位置と量を維持する必要がある。
- Task1/Task2はレビュー担当がAPPROVED、独立1028 passed/smoke/Ruff成功、指摘なし。
