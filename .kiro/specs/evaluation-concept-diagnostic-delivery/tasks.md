# Implementation Plan

逐次実行（並行しない）。testを先に書く。独立レビューは、全taskの後に1回受ける（共通引継ぎ手順）。

- [x] 1. 契約と受渡し
- [x] 1.1 実行の枠が、真の概念を、診断専用の引数としてclientへ渡す
  - 実行の枠のtestを先に直す: 観測用のclientが真の概念を記録する、受渡しのtest（clientと位置の対応）、契約の引数のtest、概念列の拒否のtest。
  - 契約、区間の進行（検査と受渡し）、全体runの実行を変更する。区間の進行を直接呼ぶ、ほかのtestと共用scriptを、概念列を渡すように直す。
  - 完了: 実行の枠のtest、clientのtest、依存境界のsuite、共用scriptが成功する。
  - _Boundary: RunClientOperations、run_stream_protocol_intervals、execute_stream_protocol_run_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.2, 4.1, 4.3_

- [x] 2. 全体runの照合
- [x] 2.1 全体runの対照を、test専用の中継なしにし、判断への不干渉のtestを書く
  - 全体runの対照から、真の概念を渡す中継を外す。真の概念を渡さない中継で実行した全体runとの比較へ、不干渉のtestを書き換える。共用scriptの全体runの確認を、真の概念が渡っていることの確認にする。
  - 完了: 全体runの対照の全条件が、中継なしで、診断まで実旧と一致する。不干渉のtest、共用scriptが成功する。
  - _Depends: 1.1_
  - _Boundary: 全体runの対照test、共用のfresh process script_
  - _Requirements: 2.1, 3.1, 3.2, 4.2_

- [ ] 3. 検証
- [ ] 3.1 独立レビューの指摘を反映し、全回帰を通す
  - 独立レビュー（別session、検証コマンドだけを許可）を1回受け、指摘の採否を下の「Implementation Notes」へ書く。
  - 完了: 全pytest（旧11・最終3goldenを含む）、Ruff、Pyright、`pip check`、照合script（`names`・`identity`・`progress`）が成功し、件数とcommitを「Implementation Notes」へ記録してある。固定旧実装とgoldenの差分が空である。
  - _Depends: 2.1_
  - _Requirements: 3.1, 4.2, 4.3_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
