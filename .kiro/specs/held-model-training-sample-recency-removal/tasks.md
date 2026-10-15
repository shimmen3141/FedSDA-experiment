# Implementation Plan

- [x] 1. 外す（1つのまとまりで行い、終わった時点でtestが通る）
  - 前提: `git log --oneline 9420d4d..HEAD -- <戻す5つのmodule>`で、ほかのcommitの変更が無いことを確かめる。
  - 学習・設定の5つのmoduleを追加の前へ戻し、clientの設定の検査、設定の構築（source）、掃引の軸（計画、コマンド、表の列、記録の形式の版6と注釈）、goldenのhelperの登録、testの構築の引数（`fresh_process_smoke.py`を含む）、設定の検査のtestの表、記録・引数の一覧・容量の軸のtestの期待、FedSDAのrunの識別子のリテラル、依存境界の許可を合わせる。`test_held_model_training_sample_recency.py`を削除する。
  - 完了の条件: 3つの項目の名前、`fedsda_training_sample_selections`、`--fedsda-training-sample-selections`、列`training_sample_selection`が、src・testsで、保存済みのgoldenのJSON以外に残らない（`rg`で確かめる）。対象のtest（設定の検査、抜き出し、共同学習、掃引、容量の軸、依存境界、goldenの版1〜5）が通る。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2_

- [x] 2. 文書
  - 実験の入口の文書、研究バックログの7とまとめ、再開案内、完了specの一覧。
  - _Requirements: 3.1_

- [x] 3. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest（goldenの版1〜5を含む）・Ruff・Pyright・WSLの対象test。
  - _Requirements: 全部_

## 仕様レビュー2回目（PASS）のMinorの扱い（実装で行う）

- 前提の確認で、be9396b・01112c9・2d352cf以外のcommitが5つのmoduleを変えていたら、checkoutせず、抜き出し方の部分だけを手で外す。戻した後、`git diff 9420d4d -- <5つのmodule>`が空であることを確かめる。
- 依存境界の許可は、抜き出しのmoduleで使わなくなる名前の全部（`bisect`・`bisect.insort`・`math`・`math.floor`・`math.log`、`local_training_settings`のmodule名とクラス名）を外す。禁止の例（`from math import exp`）は、戻した後も禁止のままなので残す。
- 文書へ、台帳（`docs/experiments/experiment-log.md`のfedsda-training-recencyの行）の結論の参照を足す。
- FedSDAのrunの識別子のリテラルは、`make_run_identifier`で計算した値へ書き換える（推測しない）。

## Implementation Notes

- 前提の確認: `git log --oneline 9420d4d..HEAD -- <6つのmodule>`は、be9396b・01112c9・2d352cfだけだった（clientの設定の検査のmoduleは、be9396bだけ）。6つとも`git checkout 9420d4d`で戻し、`git diff 9420d4d -- <6つのmodule>`が空であることを確かめた。
- 設定を作る呼出しの全部から、3つのkeyword引数をASTで外した（37ファイル、153か所。各引数が1行以上を占める形だけを外し、それ以外は止める作り）。外した後、3つの名前ほかがsrc・testsに残らないことを`grep`で確かめた（保存済みのgoldenのJSONにも無い）。
- 依存境界のtestは、抜き出しのmoduleの許可と、禁止の例（`import math`）を、追加の前の形へ戻した。計画のmoduleの`describe_…`の許可も外した。
- FedSDAのrunの識別子のリテラルは、`make_run_identifier`で計算した`sine2_interval3_seed0_4e15de15e133`へ書き換えた（FedAvg・FedDriftの掃引のtestの2か所）。
- 実装レビュー（独立Haiku 5.5 medium、1回）: APPROVED（Minor 3）。1: 再開案内の更新日を上げた。2: このnotes。3: commitで対象のファイルを明示してstageする。
- 検証: 全pytest 14292 passed / 8 skipped、失敗0（削除したtestの分だけ減った。JUnit `../../venv/refactoring-tests/recency-removal-full.xml`）。Pyright 0件、Ruff成功。WSLで、goldenの版1〜5と、抜き出し・共同学習のtestの438件が通った。レビューの後の修正は文書だけ。
