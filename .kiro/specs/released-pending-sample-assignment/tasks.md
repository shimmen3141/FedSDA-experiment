# 実装タスク revision1

逐次実行。

- [ ] 1. 警報のない標本での帰属確定を実旧対照つきで実装する
  - 命名承認後にtestを先に追加し、sourceを実装する。依存の許可集合を登録する。共用のfresh process scriptへ、この接続を通る流れを足す。
  - 実旧の標本処理との対照、吸収が解放より前であること、拒否条件で全状態が不変であること、保留位置のownerの読取りの操作を確認する。
  - 完了: 対象test・依存境界suite・共用script・Ruff・Pyrightが成功し、独立レビュー承認。
  - _Boundary: 確定の関数、保留位置のownerの読取り、依存境界_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 3.1_

- [ ] 2. 検出力を確認する
  - 汎用の変異toolを、新しい関数と保留位置のownerの読取りの操作へ実行する。未検出はtestで補うか、等価と判断した理由を証拠文書へ書く。
  - 完了: 証拠を記録し、独立レビュー承認。
  - _Depends: 1_
  - _Boundary: 検出力の証拠_
  - _Requirements: 1.1, 1.4, 2.1, 2.2_

- [ ] 3. 固定基準の全回帰と証拠を確定する
  - commitして、Windowsの基準環境で全pytestとJUnit、旧11・最終3golden、Ruff・Pyright・pip check、`spec_checks.py identity`を実測し、integration-validation.mdへ要求の対応と未検証事項を記録する。
  - 完了: 独立担当が照合して承認。Task 1のレビューにBlocker・Majorが残らなければ、同じ依頼でfeature最終の判定も受ける（`spec_checks.py progress`の後）。再開案内を更新する。
  - _Depends: 2_
  - _Boundary: 全回帰と統合証拠_
  - _Requirements: 3.2_
