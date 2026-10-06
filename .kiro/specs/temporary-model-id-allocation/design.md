# 設計 revision 2

## 配置と責務

learning/training/temporary_model_id_allocation.pyへ、単調減少counterを一つ所有する`TemporaryModelIdAllocator`を置く。現在の学習帰属ID（current_training_model_assignment.py）や学習状態一覧と同じ学習層の状態ownerで、登録を組み立てるruntimeが借用する。runtimeや手法層へ依存しない。

## API

- `TemporaryModelIdAllocator(*, client_id:int)`: exact intでなければTypeError、負ならValueError。`_next_temporary_model_id = -100 - client_id`。
- `next_temporary_model_id`（property、int）: 次に採番する値。読んでも変化しない。
- `allocate_temporary_model_id() -> int`: 現在の次IDを返し、内部値を1減らす。引数なし、失敗しない。

client_idは保持しない（初期値の計算だけに使う）。採番済みIDの履歴・解放・再利用はない。

## 依存とファイル計画

実装はimport文を持たない。exact AST guardは、このmoduleの`ast.Import`ノードを全て拒否し、`ast.ImportFrom`は解決後の束縛symbolが`__future__.annotations`の場合だけ許可する。相対import（`from . import x`、`from .current_training_model_assignment import ...`）、別名import（`import random as r`、`from dataclasses import dataclass as D`）、`from __future__ import division`等の他のfuture import、`import *`、dataclasses・乱数・math・torch・NumPy・設定・旧実装・runtime・同じ層の他ownerを注入契約testで拒否する。stdlib単独起動はAST guardの代替ではなく、実行時にtorch等が読み込まれないことを別角度で確かめる補助検証とする。

|ファイル|役割|
|---|---|
|learning/training/temporary_model_id_allocation.py|一時IDの単調採番のみ|
|tests/refactoring/test_temporary_model_id_allocation.py|実旧採番対照・拒否・読取り非消費・登録への接続|
|tests/refactoring/test_single_run_dependency_boundaries.py|exact依存guardと注入契約|
|本spec正本・steering resume/roadmap|承認/証拠/再開|

## 検証

- Task1: RED module未実装→GREEN。実旧`BaseClient(client_id, {0: object()}, verbose=False)`の実__init__と実_alloc_temp_idを呼ぶ最小fixture（実行可能なことを確認済み、research.md）と同じclient_idで、採番列・読取り値を対照する。初期値・採番の式をtestへ複製して正解にしない（client_id 0/1/7/大int、採番回数0〜5）。読取りの非消費、client_id拒否、ownerごとの独立を確認。AST注入RED→exact guard GREEN。採番したIDを既存の初期ローカル登録へ渡し、2候補を連続登録して一覧/統計/送信保留が各IDで更新されることをtest-only接続で確認（採番値が登録APIの一時ID契約を満たすことの確認。採番moduleはruntimeへ依存せず、testだけがruntimeを呼ぶ）。stdlib単独起動（torch等をimportしないこと）を独立プロセスで確認。品質/型/独立Luna/主担当gate。
- Task2: 固定環境全pytest（旧11/最終3golden）、Ruff/Pyright/pip/diff、固定旧差分空。承認hash/source hash/JUnitを記録。全pytestの判定基準はsteering/agent-handoff.md。独立Luna/主担当gate後、別feature最終GO。
