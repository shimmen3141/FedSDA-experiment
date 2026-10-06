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

## Task2

- 検証対象実装commit: 785776b。要求revision1・設計revision2・命名revision3のLF hashは承認値と一致。tasksは承認時hash（revision2）を維持し、check後hashを別fieldへ記録。
- tracked Python＋2goldenの231パスをパス順、パスUTF8＋NUL＋内容CRLF→LF＋NULで連結したSHA256: `ede05cc17326b9ae6a5fb0505ef74188e5039304695b31dbef4d101c6e6b9073`（前specの229パスに新module/新testの2件を加えた数）。
- 全適用対象Ruff成功/format131files、Pyright基準venv明示0 errors/0 warnings、pip check成功。
- 固定旧基準748c3aaからHEADへの、federated_drift_experiment/・2golden・旧回帰test2本・tools/の差分は空。
- 全pytest（主担当実測）: 5731passed/3skipped/1既存warning、124.89秒、exit0。前spec完了時5678に今回の53（対象29＋AST契約24）を加えた件数と一致。旧11条件と最終3条件の固定goldenを含む。goldenは更新していない。
- JUnit: `../../venv/refactoring-tests/temporary-model-id-allocation-full.xml`。
- 全pytestの独立再現はsteering/agent-handoff.mdの基準（2026-10-07ユーザー決定）に従い必須としない。Luna側での全pytest再現は今回試みていない。新全体runを実行したとは扱わない。

## 要件trace

|要求|証拠|
|---|---|
|1.1|実旧対照16条件の初期値（実__init__後のnext_temp_idと一致）|
|1.2|実旧対照の採番列、負・一意・exact int、採番後の次ID|
|1.3|読取り非消費test、propertyへの代入拒否|
|2.1|client_id拒否8（負2、bool/float/str/None/派生型6）|
|3.1|importなしの実装、exact AST guardと注入契約24、stdlib単独起動、owner独立|
|3.2|実旧BaseClientの実__init__/実_alloc_temp_idとの採番列一致、登録への接続2条件|

採用時の接続（採番→登録→計数→標本追加→現在ID切替えと通知）・サーバ側ID対応・通信/new client/runは後続。新たな旧正常経路の不具合は観測していない。
