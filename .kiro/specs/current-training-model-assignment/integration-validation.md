# 実測: 現在の学習帰属モデルID

2026-10-07。共有../../venvのWindows CPU固定環境。OMP/MKL各1、TMP/TEMP=../../venv/refactoring-tests、MPLCONFIGDIR=../../venv/matplotlib-cache、FDE_MNIST_DATA_DIR=../../data/mnist。環境詳細はdocs/experiments/refactoring-baseline.md。

## Task1

- RED: 未実装moduleのModuleNotFoundError、1 collection error/1.93秒、exit1。sandbox実行に既存Windows temp/cache ACLのwarning2件あり。GREEN以降は既定の承認済みescalationで実行。
- GREEN: 対象72 passed/1.89秒、exit0。実旧ローカル切替16、実旧サーバmap28、初期拒否7、変更拒否7、map全項目拒否12、読取り専用/retained record1、入力map非保存/非変更1。
- 実旧methodを呼び、ローカルhookの前後ID・サーバeventの位置37/理由/前後ID・同値時無通知を比較。連鎖/循環/欠落/自己対応/負ID/大intを含む。
- Ruff対象成功、format2 files確認。取得済みrecordの後続変更不変を直接検証。
- Pyright: 無指定では基準venvを解決できず既存Torch/NumPy importから106 errors。`--pythonpath ../../venv/Scripts/python.exe`を明示して0 errors/0 warnings、exit0。解析環境のみを修正しcode変更なし。

以後の検証は未実施。
