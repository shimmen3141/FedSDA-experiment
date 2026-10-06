# 実測: 一時モデルIDの採番

2026-10-07、共有../../venvのWindows CPU基準環境。OMP/MKL各1、TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、FDE_MNIST_DATA_DIR=../../data/mnist。主担当はClaude Code、独立レビューはcodex exec経由のGPT-6 Luna。

## Task1

- RED: 未実装moduleによるModuleNotFoundError、1 collection error/2.08秒、exit1。
- GREEN: 対象29 passed/3.88秒、exit0（初回実行で成功）。
- AST RED: 注入契約test追加直後10 failed/822 passed/0.79秒、exit1。generic規則が`__future__`の他import等を許可していた。exact guard（ast.Import全拒否、ImportFromは`__future__.annotations`だけ）追加後、対象＋AST 861 passed/4.63秒。
- 実旧対照16条件（client_id 0/1/7/10**40×採番回数0/1/2/5）。実旧`BaseClient(client_id, {0: object()}, verbose=False)`の実__init__と実_alloc_temp_idの戻り値を正解とし、各採番後の次ID読取り値も照合。読取り非消費1、owner独立1、client_id拒否8、注入契約24。
- 登録への接続2条件（class2/4）: 採番した2つのIDで既存の初期ローカル登録を連続実行し、一覧・統計・送信保留が各IDで更新されることを確認。採番moduleはruntimeへ依存しない。
- stdlib単独起動: 別プロセスで採番し、torch/NumPy/旧packageが未ロードであることを確認。
- Ruff check成功/format 131files整形済み、Pyright基準venv明示0 errors/0 warnings。
- 実Luna Task1 APPROVED、独立861 passed/Ruff成功、指摘なし。
