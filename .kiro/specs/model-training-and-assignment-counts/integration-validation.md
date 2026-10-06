# 実測: モデル別学習・割当件数

## 環境・範囲

2026-10-07、共有../../venvのWindows CPU基準。OMP_NUM_THREADS=MKL_NUM_THREADS=1、MPLCONFIGDIR=../../venv/matplotlib-cache、TMP/TEMP=../../venv/refactoring-tests、FDE_MNIST_DATA_DIR=../../data/mnist。
版/手順はdocs/experiments/refactoring-baseline.md / environments/golden/windows-cpu。
学習量・個別step・真concept診断件数の保持/加算移管/一回再編のみ。現在帰属ID・学習判断・全登録/新client/runは後続。

## Task1

- RED: 新module未実装のModuleNotFoundError、1 collection error/1.81秒、exit1。
- GREEN: 193 passed/1.84秒、exit0、Ruff/format成功。Pyright0 errors/0 warnings、exit0。
- 加算72条件（signed model/0・任意大int増分/None・signed concept/反復）、移管30条件（計数部分存在・先既存/新先・欠落元）、再編40条件（反復/連鎖/循環/多元先/0）、snapshot独立1、LEGACY011実再現/新拒否1、原子的入力拒否44、None/欠落時検査1、signed直接期待値4。
- 3辞書のkey順/概念内順と値を旧_attribute_model_training/_record/get_model_concept_counts/confirm/apply_server_mappingへ対照。旧同負ID通知の破損は実旧再現を残し、旧productionは変更しない。
- 初回Ruffでsnapshot変数をiterableの式とloopで兼用する警告を検出し、比較対象tupleを先に生成して修正。機能テストは初回189件から成功。
- 実Luna Task1 APPROVED、独立193passed/1.80秒、静的検査/scan成功、指摘なし。追加のTask2命名revision2もAPPROVED。
