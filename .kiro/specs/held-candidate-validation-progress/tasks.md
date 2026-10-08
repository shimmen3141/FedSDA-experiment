# 実装タスク revision2

逐次実行。新しいownerは保持の1つだけ。下書きはsource約220行（holder約35行、接続約185行）、test約480行で、実旧対照は完了済みspecのoracleを再利用する。

- [ ] 1. 保持のownerと3つの接続を実旧対照つきで実装する
  - 命名承認後にtestを先に追加してREDを記録し、sourceを実装する。注入契約testのREDの後に、2 moduleのexact symbol（1と20）を両resolverへ登録する。
  - holderの保持・解除・拒否、警報応答5種類の反映と拒否条件（設計8節）、到達時4条件と未到達、終端回収、保持が空の経路、ownerの型の拒否が上流の呼出しより前であること、記録→解除の順を確認する。
  - 完了: 対象test・依存境界suite・Ruff・Pyrightが成功し、独立レビュー承認。
  - _Boundary: 保持のowner、接続、依存境界_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 4.1_

- [ ] 2. 検出力と新実装単独の接続を確認する
  - 実sourceを1種ずつ変異させ（型検査を上流の呼出しの後へ移す、各検査と前提の削除、記録や解除の省略、記録と解除の順の入替え、開始以外の応答でも保持する、乱数消費）、対応するtestが失敗することを確かめて元byteへ戻す。未検出はtestで補う。
  - 旧実装とtest moduleをimportしないfresh CPU processで、1つのholderと1つの適応記録ownerを使い、警報応答→候補検証の開始→標本の観測→確定（採用・棄却・維持・再利用）と、開始→途中の警報→終端回収の流れを実行する。
  - 完了: 変異ごとの一覧・byte復元・復元後の対象成功、fresh CPUの実測を記録し、独立レビュー承認。
  - _Depends: 1_
  - _Boundary: 検出力と独立動作の証拠_
  - _Requirements: 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 4.1_

- [ ] 3. 固定基準の全回帰と証拠を確定する
  - source/testをcommitして、Windowsの基準環境で全pytestとJUnit、旧11・最終3golden、Ruff・Pyright・pip check、`spec_checks.py identity`を実測し、integration-validation.mdへ要求12項目の対応と未検証事項を記録する。基準環境が使えない間はWSLの結果を「WSLで成功」と記録し、このtaskを完了にしない。
  - 完了: 独立担当が照合して承認（全pytestの独立再実行は2026-10-07のユーザー決定により必須としない）。別sessionのfeature最終GOの後、再開案内を更新する。
  - _Depends: 2_
  - _Boundary: 全回帰と統合証拠_
  - _Requirements: 4.2_
