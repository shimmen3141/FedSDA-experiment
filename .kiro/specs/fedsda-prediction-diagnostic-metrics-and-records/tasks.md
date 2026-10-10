# Implementation Plan

逐次実行。testを先に書く。独立レビューは、全taskの後に1回受ける。全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [ ] 1. 計算
- [ ] 1.1 区間ごとの正解の割合と、クラスの間のまとめ
  - 乱数の列での、実旧の関数・旧と同じ式との照合と、拒否のtestを先に書く。
  - _Boundary: prediction_diagnostic_metric_calculations_
  - _Requirements: 1.1, 1.2, 1.3_

- [ ] 2. 指標
- [ ] 2.1 予測の診断の指標の導出と、指標への追加
  - goldenの3ケースと、小さい条件での、実旧の値との、全部の項目の照合を先に書く。
  - _Depends: 1.1_
  - _Boundary: fedsda_prediction_diagnostic_metric_derivation、FedsdaRunMetrics_
  - _Requirements: 2.1, 2.2, 2.3_

- [ ] 3. 記録の表
- [ ] 3.1 標本ごとの予測の列と、clientごとの重みの更新の回数の表
  - goldenの3ケースと、小さい条件の、実旧が保存した配列との照合を先に書く。
  - _Boundary: fedsda_run_record_tables_
  - _Requirements: 3.1, 3.2, 3.3_

- [ ] 4. 記録と検証
- [ ] 4.1 移行の一覧を更新する
  - _Requirements: 4.1_

- [ ] 4.2 独立レビューの指摘を反映し、全回帰を通す
  - 完了: 全pytest、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。WSLでの実行結果を、区別して記録してある。
  - _Depends: 2.1, 3.1, 4.1_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
