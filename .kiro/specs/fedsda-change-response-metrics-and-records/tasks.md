# Implementation Plan

逐次実行。testを先に書く。独立レビューは、全taskの後に1回受ける。全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [ ] 1. 保存する表と列の名前
- [ ] 1.1 名前を直す（表`server_round`、`held_model_count_at_prediction`、`first_new_concept_sample_index`）
  - testの期待する名前を先に直し、失敗を確かめてから、sourceを直す。
  - 完了: 前のspecのtestが、新しい名前で成功する。
  - _Requirements: 1.1, 1.2, 1.3_

- [ ] 2. 指標
- [ ] 2.1 対応づけ・分類・変化点の誤差の計算
  - 乱数の列での、実旧の関数との照合と、拒否のtestを先に書く。
  - _Boundary: change_response_metric_calculations_
  - _Requirements: 2.1, 2.2, 2.3_

- [ ] 2.2 応答の指標の導出と、指標への追加
  - goldenの3ケースと、小さい条件での、実旧の値との、全部の項目の照合を先に書く。
  - _Depends: 2.1_
  - _Boundary: fedsda_change_response_metric_derivation、FedsdaRunMetrics_
  - _Requirements: 3.1, 3.2, 3.3_

- [ ] 3. 記録の表
- [ ] 3.1 警報の表、監視の値の表、適応と判定の列
  - goldenの3ケースの、実旧が保存した配列との照合を先に書く。
  - _Depends: 1.1_
  - _Boundary: fedsda_run_record_tables_
  - _Requirements: 4.1, 4.2_

- [ ] 4. 記録と検証
- [ ] 4.1 移行の一覧を更新する
  - _Requirements: 5.1_

- [ ] 4.2 独立レビューの指摘を反映し、全回帰を通す
  - 完了: 全pytest、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。WSLでの実行結果を、区別して記録してある。
  - _Depends: 2.2, 3.1, 4.1_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
