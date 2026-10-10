# Implementation Plan

逐次実行。testを先に書く。独立レビューは、全taskの後に1回受ける。全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [ ] 1. 指標
- [ ] 1.1 モデル対の予測の相補性の計算
  - 乱数の記録の列での、実旧の`pair_diagnostic_summary`との照合と、拒否のtestを先に書く。
  - _Boundary: cross_evaluation_metric_calculations_
  - _Requirements: 1.1, 1.2_

- [ ] 1.2 指標への追加
  - goldenの3ケースと、小さい条件での、実旧の値との照合を先に書く。
  - _Depends: 1.1_
  - _Boundary: FedsdaRunMetrics_
  - _Requirements: 2.1, 2.2, 2.3_

- [ ] 2. 記録の表
- [ ] 2.1 clientの評価ごとの表と、クラスごとの表
  - goldenの3ケースと、小さい条件の、実旧が保存した配列との照合を先に書く。
  - _Boundary: fedsda_run_record_tables_
  - _Requirements: 3.1, 3.2_

- [ ] 3. 記録と検証
- [ ] 3.1 移行の一覧を更新する
  - _Requirements: 4.1_

- [ ] 3.2 独立レビューの指摘を反映し、全回帰を通す
  - 完了: 全pytest、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。WSLでの実行結果を、区別して記録してある。
  - _Depends: 1.2, 2.1, 3.1_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
