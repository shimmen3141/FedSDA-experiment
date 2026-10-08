# 実装タスク revision1

逐次実行。sourceは変更しない。

- [x] 1. 共用の検証3つを追加する
  - 共用のfresh process scriptとそれを別processで実行するtest、許可集合が広すぎないことの検査（見つかった使っていない許可を外す）、NEW-002の2件の修正を行う。
  - 追加するtestが、直す前の状態で失敗することを確かめる（許可集合の検査は、使っていない許可を外す前に失敗する）。
  - 完了: 対象test・依存境界suite・Ruffが成功し、独立レビュー承認。
  - _Boundary: testとtest用script_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2_

- [x] 2. 検出力を確認する
  - 設計5節の検出力の確認（scriptと許可集合の検査を一時的に壊して失敗を確かめ、元へ戻す）と、汎用の変異toolによる保持と進行のmoduleの確認を行い、結果と、等価と判断した変異の理由を証拠文書へ書く。
  - 完了: 証拠を記録し、独立レビュー承認。
  - _Depends: 1_
  - _Boundary: 検出力の証拠_
  - _Requirements: 1.2, 2.1, 2.2, 3.1, 3.2_

- [ ] 3. 固定基準の全回帰と証拠を確定する
  - commitして、Windowsの基準環境で全pytestとJUnit、旧11・最終3golden、Ruff・Pyright・pip check、`spec_checks.py identity`を実測し、integration-validation.mdへ要求の対応と未検証事項を記録する。NEW-002を修正済みへ改め、共通引継ぎ手順の「次のspecで作る」の記述を現在の状態へ改める。
  - 完了: 独立担当が照合して承認。小さいspec（sourceの変更なし）なので、同じ依頼でfeature最終の判定も受ける（`spec_checks.py progress`の後）。再開案内を更新する。
  - _Depends: 2_
  - _Boundary: 全回帰と統合証拠_
  - _Requirements: 4.1_
