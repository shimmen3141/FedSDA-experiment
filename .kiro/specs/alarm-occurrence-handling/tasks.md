# 実装タスク revision1

逐次実行。新しいownerはない。下書きはsource約200行（うちimportと引数の受渡しが大半で、処理は検査3つと5つの呼出し）、test約540行で、実旧対照は完了済みspecのoracleを再利用する。

- [x] 1. 警報1回ぶんの接続を実旧対照つきで実装する
  - 命名承認後にtestを先に追加してREDを記録し、sourceを実装する。注入契約testのREDの後に、新moduleのexact symbol（27）を両resolverへ登録する。
  - 引数の集合、2/4class×警報応答5種類の実旧対照（応答・完了・記録・保持・診断・乱数）、5段の順と受渡し、拒否条件（設計8節の10＋12）でどの段も呼ばれないこと、各段の失敗で後の段が呼ばれないこと、結果recordの形を確認する。
  - 完了: 対象test・依存境界suite・Ruff・Pyrightが成功し、独立レビュー承認。
  - _Boundary: 接続、依存境界_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 3.1_

- [ ] 2. 検出力と新実装単独の接続を確認する
  - 実sourceを1種ずつ変異させ（検査を応答の後へ移す、各検査の削除、exact型検査をisinstanceへ緩める、段の省略・順の入替え・二重実行、保持を読まずに進行中のsessionなしで応答する、提案位置を別の値にする、通知へ帰属変更を渡さない）、対応するtestが失敗することを確かめて元byteへ戻す。未検出はtestで補う。
  - 旧実装とtest moduleをimportしないfresh CPU processで、1つの保持・記録・診断のownerを使い、警報（候補検証の開始）→候補検証中の警報→標本の観測と確定→次の警報、の流れを本関数と既存の進行で実行する。
  - 完了: 変異ごとの一覧・byte復元・復元後の対象成功、fresh CPUの実測を記録し、独立レビュー承認。
  - _Depends: 1_
  - _Boundary: 検出力と独立動作の証拠_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.4, 3.1_

- [ ] 3. 固定基準の全回帰と証拠を確定する
  - source/testをcommitして、Windowsの基準環境で全pytestとJUnit、旧11・最終3golden、Ruff・Pyright・pip check、`spec_checks.py identity`を実測し、integration-validation.mdへ要求11項目の対応と未検証事項を記録する。基準環境が使えない間はWSLの結果を「WSLで成功」と記録し、このtaskを完了にしない。
  - 完了: 独立担当が照合して承認（全pytestの独立再実行は2026-10-07のユーザー決定により必須としない）。`spec_checks.py progress`の後、別sessionのfeature最終GOを受け、再開案内を更新する。
  - _Depends: 2_
  - _Boundary: 全回帰と統合証拠_
  - _Requirements: 3.2_
