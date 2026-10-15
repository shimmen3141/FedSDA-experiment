# Implementation Plan

- [x] 1. 軸
  - [x] 1.1 計画の項目と検査、変種の作り方（2つのFedSDAの軸の組）。計画の項目が掃引の記録に入るので、同じ変更で、掃引の記録の形式の版（5）と、既存のtestの期待（`test_experiment_sweep.py`の記録の版と、記録の計画の完全一致）を合わせる。testを先に書く（設計のValidation Designの1〜4の計画の部分）。
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.2_
  - [x] 1.2 コマンドの引数、一覧の表の列。testを先に書く（Validation Designの4のコマンドの部分と5・6）。
    - _Requirements: 1.2, 2.1, 2.2_

- [x] 2. 文書
  - 実験の入口の文書（軸、時間尺度も同じ値で替わること、目安と制約、組合せ、記録の形式の版）、研究バックログの8、再開案内、完了specの一覧。
  - _Requirements: 4.1_

- [x] 3. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest（goldenの版1〜5を含む）・Ruff・Pyright。
  - _Requirements: 全部、3.1_

## 仕様レビュー3回目（PASS）のMinorの扱い（実装で行う）

- 表の列は、経路の葉（`pending_assignment_buffer_capacity_samples`）を`.get`で読み、無い要約では空欄にする（行ごと外さない。既存の`training_sample_selection`の列と同じ扱い）。版1の要約の書出しにこの項目が在ることを、実装の前にgitの履歴で確かめ、結果をImplementation Notesへ書く。
- 要求2.2をtask 1.1に、設計のValidation Designの6をtask 1.2に対応させた。

## Implementation Notes

- 版1の要約にこの項目が在ること: 項目の名前`pending_assignment_buffer_capacity_samples`は2026-10-03（ed38fed）から変わっておらず、FedSDAのrunの要約を書くmodule（`runtime/fedsda_run_artifact_files.py`）は2026-10-11（030cdce）に作られ、run設定を`convert_settings_to_plain_mapping`で書く。要約の形式の版2で足したのは来歴だけ。testは、実物の版1の要約は作らず、版1と同じ形（来歴を持たない）の要約で列が埋まることを確かめる（この限界は、testのコメントにも書いた）。
- 表の列は、途中の項目（`training_data_assignment_settings`）も`.get`で読む。抜き出し方の軸の既存testが、この項目を持たない最小限の要約で表を作っていて、`[...]`で読むと、その行が読めない要約として外れたため（要求2.1の「行を外さない」に合わせた）。
- 抜き出し方の軸の処理は、FedSDAの変種を作る関数`_build_fedsda_settings_variants`へ移した（値と順は同じ。既存testが通る）。
- testを先に書き、実装の前に失敗することを確かめた（12件中10件が失敗。残る2件は、まだ無い項目を渡すと型の誤りになるため）。
- 実装レビュー（独立Haiku 5.5 medium、1回）: APPROVED（Minor 3）。1: このnotesを足した。2: 前の形式の要約のtestのコメントを直した。3: 要求3.1へ、コマンドの引数の一覧の期待の変更を足した。
- 検証: 全pytest 14358 passed / 8 skipped、失敗1件（コマンドの引数の一覧を完全に照合する`test_sweep_command_arguments_are_only_extended_by_the_artifact_organization`。新しい引数を期待へ足し、そのファイルを再実行して2 passed）。JUnit `../../venv/refactoring-tests/fifo-capacity-full.xml`。Ruff・Pyright 0件。レビューの後の修正は、文書とtestのコメントだけ。
