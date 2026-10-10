# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [x] 1. 計算の部品
- [x] 1.1 変更位置の抽出、対応づけ、精度、定常精度、検出の指標を計算する
  - 実旧の関数をoracleにしたtest（多数の入力と境界、拒否）を先に書く。
  - 完了: 部品のtestと、依存境界のsuiteが成功する。
  - _Boundary: run_metric_calculations_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 4.5, 5.2_

- [ ] 2. 全体runからの導出と照合
- [ ] 2.1 全体runの結果と参加者から指標を導出し、実旧・goldenと照合する
  - goldenの条件の照合（実旧の実行と保存結果、Windows用のgolden）、小さい条件の照合、状態を変えないこと、拒否のtestを先に書く。31の離散列は、testが、新の記録から作る。
  - 導出を実装する。共用scriptへ、導出を足す。
  - 完了: 導出のtest（Windowsでは、goldenの照合を含む）、依存境界のsuite、共用scriptが成功する。
  - _Depends: 1.1_
  - _Boundary: fedsda_run_metric_derivation_
  - _Requirements: 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4, 5.1, 5.2_

- [ ] 3. 検証
- [ ] 3.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 2.1_
  - _Requirements: 4.1, 4.3, 5.1, 5.2_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
