# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [x] 1. 判定記録の保持
- [x] 1.1 判定記録を起きた順に保持するownerを作る
  - ownerのtest（順、読取りの不変、2つの型、拒否）を先に書く。
  - 完了: ownerのtestと、依存境界のsuiteが成功する。
  - _Boundary: CandidateValidationDecisionRecordStore_
  - _Requirements: 1.4, 1.5, 3.4_

- [x] 1.2 clientが、候補検証の確定と終端の回収で、判定記録を保持へ足す
  - clientの全状態の新旧照合へ、実旧の候補の判定の一覧との照合を足す（先に書いて、失敗を確かめる）。組立てのtest。
  - ownerの束、組立て、2つの操作を変更する。全体runの対照へ、判定の種類の経路と、不干渉の一致を足す。共用scriptへ確認を足す。
  - 完了: clientのtest、全体runの対照の全条件、共用scriptが成功する。
  - _Depends: 1.1_
  - _Boundary: FedsdaRunClient、assemble_fedsda_run_client_
  - _Requirements: 1.1, 1.2, 1.3, 1.6, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3_

- [ ] 2. 検証
- [ ] 2.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 1.2_
  - _Requirements: 2.2, 3.1, 3.3, 3.4_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
