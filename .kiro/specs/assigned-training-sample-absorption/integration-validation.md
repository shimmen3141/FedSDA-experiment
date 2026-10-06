# 実測: 帰属確定標本の吸収

2026-10-07、共有../../venvのWindows CPU基準環境。OMP/MKL各1、TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、FDE_MNIST_DATA_DIR=../../data/mnist。主担当はClaude Code、独立レビューはcodex exec経由のGPT-6 Luna。

## Task1

- 手順の逸脱: testファイルを書いた後、REDを実行する前に実装ファイルも書いた。実装が存在しない時点でのRED実行はない。その後に実装をworktree外へ一時退避して実行した結果はModuleNotFoundError、1 collection error/2.09秒、exit1。戻した後の初回実行は対象109 passed/3.93秒。testと実装は退避の前後で無変更。
- 代わりの証拠（testが振る舞いを拘束すること）: `../../venv/refactoring-tests/assigned_training_sample_absorption_red_evidence.py`で承認対象の実装を一時的に差し替え、対象testを実行し、元へ戻して実装のLF sha256 `dbd66914cf0443356c3e2cf488db40d2c0f36bbfb6edfc8cb3e71df6e105f0d6`が変わらないことを確認した。最終（対象127件）の結果:

|差し替えた実装|結果|
|---|---|
|何もしないstub|103 failed/24 passed|
|割当概念計数を省く|51 failed|
|統計へ観測クラスを渡さない|66 failed|
|損失評価の途中で標本store APIを呼ぶ|1 failed|
|複数行recordを平均で受理する|1 failed|
|全標本を一括追加する|7 failed|
|全損失評価の完了前に標本を追加する|8 failed|
|概念計数より先に統計を更新する|1 failed|
|正しい実装|127 passed|

- 一括追加の誤実装は、初回（対象109件）では呼出順test1件でしか検出されなかった。要求1.4（空列で標本列を作らない）を値で観測する条件が欠けていたため、吸収先が標本列を持たない条件を実旧対照testへ追加した（productionは無変更）。
- 実旧対照72条件（class2/4×吸収先4種: 現在のモデル/別の保有モデル/統計未登録/標本列なし×標本0/1/5件×概念IDあり/None混在/概念要素なし）。標本列の順序とTensor identity、割当概念計数、統計全field（クラス初出順を含む）を実旧_absorb_into_storeと照合。一覧record identity・全parameter/grad・全optimizer・学習計数・他モデルの標本と統計・現在ID・torch/random/NumPy乱数の不変。
- 拒否35条件: model_id4、標本列3、概念ID列7、未保有2（空列を含む）、複数行record/特徴/ラベルの不正7（列の先頭・途中・末尾）、4owner×3型12。各拒否で標本・計数・統計・parameter・optimizer・乱数の不変を確認。呼出順1（全損失評価→標本ごとに追加→概念→統計）。
- 旧の途中失敗時の部分更新を実旧で再現（1件目は全更新、不正な2件目は標本と概念だけ追加され統計は未更新、3件目は未処理）し、同じ入力で新が何も変更しないことを同じtestで確認。LEGACY-015へ記録。
- AST RED: 注入契約追加直後13 failed/896 passed/0.89秒、exit1。exact6symbol guard追加後、対象＋AST 1036 passed/4.95秒（条件追加後）。
- Ruff check成功/format 135files整形済み、Pyright基準venv明示0 errors/0 warnings。

## Task2

- 実旧_finalize_forward_validationの非採用3分岐×class2/4の6条件は初回実行でGREEN。候補損失0.9・参照損失0.2/0.3で非採用にし、履歴平均なし→棄却（create_rejected、現在のモデルへ吸収）、現在のモデルの履歴平均あり→現行維持（maintain）、別モデルの履歴平均あり→再利用（reuse、現在IDが4へ切替わりそのモデルへ吸収）。旧が決めた帰属先へ新の吸収を適用し、標本・割当概念計数・統計を照合。保有一覧・採番次値・parameter・送信保留が変わらないことを確認。再利用分岐の現在ID合わせは後続specの責務のためtest-only接続。
- 12条件（class2/4×Adam標準/AMSGrad/SGD×共有部更新有無）は初回実行でGREEN。共同更新（parameterが変わる）→吸収→吸収標本を含む共同更新2回を、両実装とも各自の標本storeの全標本を固定batchにして実旧と照合。loss、全値/grad、optimizer state、標本列、割当概念計数、統計、乱数が一致。
- fresh `../../venv/refactoring-tests/assigned_training_sample_absorption_cpu_smoke.py`成功、exit0。class2/4で学習→吸収（parameter不変、標本identity、概念計数、統計件数と平均）→吸収標本を含む学習。旧importなし。
- 実Luna Task2 APPROVED（1回目）、Task1 APPROVED（2回目）。経緯はreview.md。
