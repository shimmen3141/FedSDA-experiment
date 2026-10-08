# 実装タスク revision2

逐次実行。既存のsource 1ファイル（約20行の追加）とtest 1ファイル（約170行の追加・変更）、依存境界testを変更する。

- [x] 1. 確定に伴う診断通知を実旧対照つきで実装する
  - 命名承認後にtestを先に変更してREDを記録し、sourceを変更する。注入契約testのREDの後に、exact集合へ2 symbolを登録する。
  - 確定4条件の通知（1回、帰属変更そのもの、記録と解除の後）と実旧のAdaHedgeとの一致、未到達・保持なしで診断が不変、ownerの型の拒否20条件（保持あり・保持なし）が上流の呼出しより前であること、既存の照合が変わらず成功することを確認する。拒否の文言（NEW-001）を直す。
  - 完了: 対象test・依存境界suite・Ruff・Pyrightが成功し、独立レビュー承認。
  - _Boundary: 進行の接続、依存境界_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 3.1_

- [ ] 2. 検出力と新実装単独の接続を確認する
  - 実sourceを1種ずつ変異させ（診断の型検査を上流の後へ移す・削除・isinstanceへ緩める・保持があるときだけ行う、通知の省略・二重実行・解除の前や記録の前への移動、帰属変更を渡さない、未到達でも通知する）、対応するtestが失敗することを確かめて元byteへ戻す。既存の変異（held-candidate-validation-progressの30種）のうち、今回の変更後のsourceへそのまま適用できるものも再実行する。未検出はtestで補う。
  - 旧実装とtest moduleをimportしないfresh CPU processで、`handle_alarm_occurrence`（候補検証の開始）→本関数（標本の観測と確定）を1つの保持・記録・診断のownerで実行し、確定の結果ごとの再始動の回数を確かめる。
  - 完了: 変異ごとの一覧・byte復元・復元後の対象成功、fresh CPUの実測を記録し、独立レビュー承認。
  - _Depends: 1_
  - _Boundary: 検出力と独立動作の証拠_
  - _Requirements: 1.1, 1.2, 2.1, 3.1_

- [ ] 3. 固定基準の全回帰と証拠を確定する
  - source/testをcommitして、Windowsの基準環境で全pytestとJUnit、旧11・最終3golden、Ruff・Pyright・pip check、`spec_checks.py identity`を実測し、integration-validation.mdへ要求7項目の対応と未検証事項を記録する。NEW-001の記録を修正済みへ改める。
  - 完了: 独立担当が照合して承認（全pytestの独立再実行は2026-10-07のユーザー決定により必須としない）。`spec_checks.py progress`の後、別sessionのfeature最終GOを受け、再開案内を更新する。
  - _Depends: 2_
  - _Boundary: 全回帰と統合証拠_
  - _Requirements: 3.2_
