# Research

## 今の実装と結果（2026-10-14、5ca896f）
- 掃引のコマンドの出力先の既定は、`results/<YYYYMMDD_HHMMSS>_<label>`（`--label`の既定は`sweep`）。比較scriptは、`--output-directory`が必須。
- 分析scriptは、Markdownを標準出力へ出す。ユーザーは、サーバで標準出力をファイルへ移し、`results/behavior-fix-comparison/behavior-fix-comparison.md`・`_2.md`・`_3.md`として残している（連番。中身に、どの実行の結果を集計したかが無い）。
- 新実装の比較の実行は、`results/behavior-fix-comparison/`（メインの作業ツリー）に、3つの記録が残る。
  - 2026-10-13 12:07〜17:13、commit c577e52、480 run（登録legacy・fixed×短い警報legacy・fixed、6データ、seed 0〜9、集約間隔50・200）。
  - 2026-10-14 04:35〜07:45、commit ad52310、240 run（登録waited-as-current×短い警報legacy・fixed）。
  - 2026-10-14 07:45〜09:04、commit ad52310、120 run（登録waited-as-current×短い警報fixed×警報を待つon）。
  - LEGACY-016の記録の「8通り」のうち、全部登録×002を直した側×待つ警報の120 runは、この出力先の記録に無い（別の出力先で実行したと思われる。ユーザーへ確認する）。
- `results/`は`.gitignore`の対象。READMEは、gitに残らない。台帳は`docs/`（gitの対象）。
- 固定旧の`tools/run_server_sweep.sh`と`run_pareto_sweep.py`の出力の規約（AGENTS.md）は、変えない。

## 判断
- READMEはコマンドが書き、台帳は人が書く（ユーザーが2026-10-14に了承）。
- 目的は必須（書き忘れを防ぐ）。研究の問いは、掃引では任意（今までの出力先の既定を保つ）。
