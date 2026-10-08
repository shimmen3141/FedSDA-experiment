# 実装タスク revision2

逐次実行。既存の記録ownerの拡張と、2つの記録関数。新しい状態ownerは作らない。下書きはsource約230行（うち新module約130行）、新test約480行で、実旧対照は完了済みspecのoracleを再利用する。

- [x] 1. 記録ownerの拡張と候補検証の記録を実旧対照つきで実装する
  - 命名承認後にtest（新testと、既存testのfield名・結果集合の更新）を先に追加してREDを記録し、sourceを実装する。注入契約testのREDの後に、新moduleのexact 9 symbolとevaluationの`typing.get_args`を両resolverへ登録する。
  - 到達時16条件・終端8条件の実旧対照、全10結果の切替位置・件数・ID整合の規則、設計4節の各検査を破る拒否条件（更新前の拒否）、乱数と上流状態の不変を確認する。既存の警報適応記録testが全件成功すること。
  - 完了: 対象test2件・依存境界suite・Ruff・Pyrightが成功し、独立レビュー承認。
  - _Boundary: evaluationの記録ownerとruntime変換、依存境界_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 3.1_

- [ ] 2. 検出力と新実装単独の接続を確認する
  - 実sourceを1種ずつ変異させ（検査を更新の後へ移す、各検査の削除、対応表・切替結果・件数の規則の破壊、field値の取り違え、乱数消費）、対応するtestが失敗することを確かめて元byteへ戻す。未検出はtestで補う。
  - 旧実装とtest moduleをimportしないfresh CPU processで、新上流の到達時の確定（採用・再利用・維持・棄却）と終端回収を実行し、同じstoreへ記録して値・切替位置・件数を確かめる。
  - 完了: 変異ごとの一覧・byte復元・復元後の対象成功、fresh CPUの実測を記録し、独立レビュー承認。
  - _Depends: 1_
  - _Boundary: 検出力と独立動作の証拠_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 3.1_

- [ ] 3. 固定基準の全回帰と証拠を確定する
  - source/testをcommitして全pytestとJUnit、旧11・最終3golden、Ruff・Pyright・pip check、`spec_checks.py identity`（承認hash・固定旧差分・source hash）を実測し、integration-validation.mdへ要求10項目の対応と未検証事項を記録する。
  - 完了: 独立担当が照合して承認（全pytestの独立再実行は2026-10-07のユーザー決定により必須としない）。別sessionのfeature最終GOの後、再開案内を更新する。
  - _Depends: 2_
  - _Boundary: 全回帰と統合証拠_
  - _Requirements: 3.2_
