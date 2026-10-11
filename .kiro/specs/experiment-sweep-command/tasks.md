# Implementation Plan

逐次実行。testを先に書く。独立レビューは、全taskの後に1回受ける。全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [ ] 1. 計画
- [ ] 1.1 構成の名前と、掃引の計画、識別子
  - _Boundary: experiment_configurations、experiment_sweep_plan_
  - _Requirements: 1.1, 1.2, 1.3_

- [ ] 2. 実行
- [ ] 2.1 逐次・並列の実行、再開、失敗の扱い
  - _Depends: 1.1_
  - _Boundary: experiment_sweep_execution_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 3. 記録
- [ ] 3.1 一覧の表と、実行の記録
  - _Depends: 2.1_
  - _Boundary: experiment_sweep_records_
  - _Requirements: 3.1, 3.2_

- [ ] 4. コマンド
- [ ] 4.1 コマンドと、直下のscript、使い方の文書
  - _Depends: 3.1_
  - _Boundary: experiment_sweep_command、run_experiment_sweep.py_
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 5. 検証
- [ ] 5.1 独立レビューの指摘を反映し、全回帰を通す
  - 完了: 全pytest、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。WSLでの実行結果（並列の実行を含む）を、区別して記録してある。
  - _Depends: 4.1_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
