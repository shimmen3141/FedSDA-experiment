# 実測: 現在の学習帰属モデルID

2026-10-07。共有../../venvのWindows CPU固定環境。OMP/MKL各1、TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、FDE_MNIST_DATA_DIR=../../data/mnist。環境詳細はdocs/experiments/refactoring-baseline.md。

## Task1

- RED: 未実装moduleのModuleNotFoundError、1 collection error/1.93秒、exit1。sandbox実行に既存Windows temp/cache ACLのwarning2件あり。GREEN以降は既定の承認済みescalationで実行。
- GREEN: 対象72 passed/1.89秒、exit0。実旧ローカル切替16、実旧サーバmap28、初期拒否7、変更拒否7、map全項目拒否12、読取り専用/retained record1、入力map非保存/非変更1。
- 実旧methodを呼び、ローカルhookの前後ID・サーバeventの位置37/理由/前後ID・同値時無通知を比較。連鎖/循環/欠落/自己対応/負ID/大intを含む。
- Ruff対象成功、format2 files確認。取得済みrecordの後続変更不変を直接検証。
- Pyright: 無指定では基準venvを解決できず既存Torch/NumPy importから106 errors。`--pythonpath ../../venv/Scripts/python.exe`を明示して0 errors/0 warnings、exit0。解析環境のみを修正しcode変更なし。

## Task2

- AST RED: 新moduleを既存dataclass限定注入testへ追加、7 failed/737 passed/0.84秒、exit1。generic stdlibで許可される7入力をexact guardで拒否する前の失敗。
- GREEN: exact guard/resolver/module import拒否へ新module追加後、対象＋AST819 passed/2.99秒、exit0。
- 負ID-7→正式ID0/4/12の3条件で、現在ID変更は計数・旧fixture状態を変えないことを検証。その後に既存新計数storeを移管し、実旧confirmとID/計数を照合。後続サーバmapを両ownerへ一段適用し、再編結果と保持済みrecordの不変を確認。正式登録の呼出し順/transaction全体は未実装。
- Ruff対象成功/format2 files確認。snapshot期待値の一時名をID期待値から区別する改名を追加命名レビューへ戻す。
- 実Luna命名revision2 APPROVED後に改名。fresh `python -S ../../venv/refactoring-tests/current_training_model_assignment_stdlib_smoke.py`成功、torch/NumPy/旧importなしで設定→map→拒否→保持recordを確認。

- 実Luna独立819passed/2.65秒、stdlib-only/品質/diff成功、APPROVED。主担当改名後819passed/3.24秒、exit0。

## Task3

- Ruffをrootの`.`で探索するとWindowsの既存temp/cache ACLに触れ、lintに探索warning、formatにpanicを観測。対象の`src tests/refactoring`を明示することで同じ共有設定の全適用対象を検査し、lint成功・format125 files整形済み、各exit0を確認。対象外の旧コードや一時ディレクトリを品質対象へ追加していない。
- Pyright基準venv明示0 errors/0 warnings、pip check成功、diff check成功。旧production/2golden/2golden testの固定748c3aaからの差分は空。
- 要求revision3・設計revision1・命名revision2のLF hashは承認値と一致。tasksは承認時hashを保存し、check更新後hashも別fieldへ保持。
- 検証対象実装commitはa4959fb。tracked Python＋2goldenの225パスをパス順に、パスUTF8＋NUL＋内容CRLF→LF＋NULで連結したSHA256: `0489b9b6568ad15cea537ff394a6d576c3c22394448a52bc0bded059faa17ec1`。
- Task3でもfresh stdlib-only smokeを再実行してexit0。現在ID設定/対応/拒否/record保持に数値環境や旧importは不要。
- 全pytest: 5398 passed/3 skipped/1既存warning、234.74秒、exit0。旧11条件/最終Residual Adapter＋Switching3条件のgoldenを含む。既存Windows wrapper3skipとqint8 fixture TypedStorage warning。
- JUnit: `../../venv/refactoring-tests/current-assignment-full.xml`、5401 tests/0 failures/0 errors/3 skipped。
- 品質探索の観測/回避はdevelopment findingへ記録し、docs/research/code-quality.mdへ適用対象明示の手順を追加。Ruff内部やACLは変更していない。
- 実Luna Task3 APPROVED、独立対象＋AST819passed/7.61秒、stdlib-only/品質/diff/scan/JUnit照合成功。主担当も全suite/品質/hash/固定旧差分空を確認。

## 要件trace

|要求|証拠|
|---|---|
|1.1|明示signed初期ID、property setter拒否、初期型拒否|
|1.2|実旧ローカル16条件、同値None、frozen前後ID|
|2.1|実旧サーバ28条件、欠落/自己/循環/連鎖一段|
|2.2|map全項目拒否12条件、状態不変、map保存/変更なし|
|3.1|exact AST許可/禁止、ownerに他状態/callbackなし|
|3.2|実旧ローカルhook/server event対照、登録確認と計数移管3条件|
|3.3|後続更新後record保持、他owner非変更、取得済みsnapshot保持、stdlib単独|

正式登録の組立・候補session/client/server/new runは未実装。新たな旧正常経路の不具合は観測していない。
