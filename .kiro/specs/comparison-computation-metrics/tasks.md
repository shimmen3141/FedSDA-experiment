# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。全pytestは、並列（`-n 8 --dist loadfile`）で実行する。

- [x] 1. 計数
- [x] 1.1 集約と統合が、パラメータの積和演算の数を、結果に含める
  - 既存の、実旧との対照へ、計数の照合（集約＝通信量の上りの値の数、統合・距離＝記録と層の形からの計算）を先に足す。
  - 完了: 集約と統合のtestが成功する。
  - _Boundary: ClientModelAggregation、ModelConsolidation_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 5.1, 5.2_

- [x] 1.2 clientが、標本ごとの保有モデル数を記録する
  - ownerのtestと、clientの全状態の新旧照合への照合を先に書く。
  - 完了: ownerのtest、clientのtest、依存境界のsuiteが成功する。
  - _Boundary: HeldModelCountRecordStore、FedsdaRunClient_
  - _Requirements: 2.1, 2.2, 2.3, 5.1_

- [x] 1.3 計測つきの全体runが、ラウンドごとのモデルの計算を返す
  - 合計の一致、同期の値が学習を含まないことのtestを先に書く。
  - 完了: 計測つきの全体runのtestが成功する。
  - _Depends: なし_
  - _Boundary: fedsda_measured_run_execution_
  - _Requirements: 3.1, 3.2, 3.4_

- [ ] 2. まとめと指標
- [ ] 2.1 計算量のまとめを計算する
  - 手計算との照合、NaN、拒否のtestを先に書く。
  - 完了: まとめのtestと、依存境界のsuiteが成功する。
  - _Boundary: computation_cost_summary_
  - _Requirements: 4.1, 4.2, 4.3_

- [ ] 2.2 指標の導出へ足し、goldenの条件で、実旧と照合する
  - 指標の導出のtestへ、サーバの計数・保有モデル数・ラウンドごとの値・まとめの照合を先に足す。共用scriptへ足す。
  - 完了: 指標の導出のtest（goldenの33指標の照合を含む）、共用script、依存境界のsuiteが成功する。
  - _Depends: 1.1, 1.2, 1.3, 2.1_
  - _Boundary: fedsda_run_metric_derivation_
  - _Requirements: 3.3, 4.4, 5.1, 5.3, 5.4_

- [ ] 3. 検証
- [ ] 3.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 2.2_
  - _Requirements: 1.4, 3.4, 5.4_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
