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

## Task3（実施中）

- Ruffをrootの`.`で探索するとWindowsの既存temp/cache ACLに触れ、lintに探索warning、formatにpanicを観測。対象の`src tests/refactoring`を明示することで同じ共有設定の全適用対象を検査し、lint成功・format125 files整形済み、各exit0を確認。対象外の旧コードや一時ディレクトリを品質対象へ追加していない。
- Pyright基準venv明示0 errors/0 warnings、pip check成功、diff check成功。旧production/2golden/2golden testの固定748c3aaからの差分は空。
