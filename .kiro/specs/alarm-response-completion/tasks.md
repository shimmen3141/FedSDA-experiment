# 実装タスク revision2

逐次実行。既存ownerだけを使う1つの完了処理と、実旧oracleの再利用による検証。新しい状態ownerは作らない。runtimeの外部下書きは約180行、testは約800行（多くは既存の警報応答oracleと監視照合helperの接続）。各taskを独立レビューし、最後に別freshのfeature最終GOを受ける。

- [ ] 1. 完了recordと完了処理を実旧対照つきで実装する
  - 実装前に対象testを追加してREDを記録する（moduleなしの収集失敗）。依存境界は注入契約testのREDの後にexact 7 symbolのguardを両resolverへ登録する。
  - recordの受理集合（消費位置の各要素が0以上であることを含む）、検査→基準選択→reset→drainの順、不足時の保持を実装する。
  - 2/4class×3解決条件×区間長・最小件数4組と候補検証中の2回目の警報で、実旧のイベント・切替位置・再利用計数・戻り値・FIFO・reset後の検出器全状態・完了後の監視の継続を照合する。基準の5条件、全拒否条件の状態不変、再適用の拒否、呼出し順、recordの検査を確認する。
  - 完了: 対象testと依存境界suiteとRuffが成功し、独立レビュー承認。
  - _Boundary: 完了処理の実装と依存境界_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3_

- [ ] 2. 検出力と新実装単独の動作を確認する
  - test-only・証拠のtask。実sourceの代表変異（reset省略、基準を固定値に、基準を変更前モデルから選ぶ、drain省略、不足でもdrain、drainをresetより先に、変更前後IDの入替え、各対応検査の削除、型検査の緩和、余分な乱数消費）を各回finallyで元byteへ復元して実行し、意味のあるassertion失敗だけを検出と数える。未検出があればtestの条件を足す。
  - 旧実装とtest moduleをimportしないfresh CPU processで、不足・維持・再利用・候補検証開始・候補検証中の応答に続けて完了処理を実行し、監視の再開、保留位置の保持と消費、recordの値を確認する。
  - 完了: 変異の検出・byte復元・復元後の対象成功、fresh CPUの実測、guardと実importのexact一致を記録し、独立レビュー承認。
  - _Boundary: 検出力と独立動作の証拠_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 3.2, 3.3_

- [ ] 3. 固定基準の全回帰と証拠を確定する
  - 全pytestとJUnit、旧11・最終3golden、Ruff・Pyright・pip check、固定旧`748c3aa`からの空diff、承認hash・source hash・検証対象commitを実測する。
  - integration-validation.mdへ要求10項目の対応、各taskのレビュー、未検証事項（一覧owner・通知・session保持・新全体run）を記録する。
  - 完了: 主担当の全実測とJUnitを独立担当が照合して承認（独立の全pytest再実行は2026-10-07のユーザー決定により必須としない）。その後、別freshのfeature最終GOを受け、再開案内を更新する。
  - _Boundary: 全回帰・品質と統合証拠_
  - _Requirements: 3.3_
