# 実装タスク revision2

逐次実行。

- [x] 1. 学習要求の記録と保有モデルの共同学習を実旧対照つきで実装する
  - 命名承認後にtestを先に追加し、sourceを実装する。依存の許可集合は、登録する前に依存境界のtestが新moduleで失敗することを確かめてから登録する。共用のfresh process scriptへ、この接続を通る流れを足す。
  - 実旧の学習stepとの対照（共同更新の合間の失敗を含む）、各回の共同学習→計数、全回の後に保留の消化の順、ownerの型の拒否で状態が変わらないこと、件数管理の読取りの操作を確認する。
  - 完了: 対象test・依存境界suite・共用script・Ruff・Pyrightが成功し、独立レビュー承認。
  - _Boundary: 学習要求の処理の関数2つ、件数管理の読取り、依存境界_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 4.1_

- [x] 2. 検出力を確認する
  - 汎用の変異toolを、新しい関数と件数管理の読取りの操作へ実行する。未検出はtestで補うか、等価と判断した理由を証拠文書へ書く。
  - 完了: 証拠を記録し、独立レビュー承認。
  - _Depends: 1_
  - _Boundary: 検出力の証拠_
  - _Requirements: 1.2, 1.5, 2.1, 2.3, 3.1_

- [ ] 3. 固定基準の全回帰と証拠を確定する
  - commitして、Windowsの基準環境で全pytestとJUnit、旧11・最終3golden、Ruff・Pyright・pip check、`spec_checks.py identity`を実測し、integration-validation.mdへ要求の対応と未検証事項を記録する。全pytestの件数が、前specの件数に今回足したtest数を加えた数と一致することを確かめる。
  - 完了: 独立担当が照合して承認。本specは2 moduleに触れるので、feature最終の判定は別sessionで受ける（`spec_checks.py progress`の後）。再開案内を更新する。
  - _Depends: 2_
  - _Boundary: 全回帰と統合証拠_
  - _Requirements: 4.2_
