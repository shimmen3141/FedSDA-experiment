# 実装タスク revision1

逐次3task。独立task graph sanity reviewはPASS。1は単一ownerと依存guardの明示的な統合taskで、数値操作の旧計算は約100行、実旧対照と拒否/copyのtestは約200行を目安とする。新しい通知や診断集計は追加しない。

- [x] 1. 診断証拠ownerを実旧対照と依存境界へ統合する
  - 命名承認後に対象testを先に追加してREDを記録し、その後ownerを実装する。注入依存契約testのRED後に、両AST resolverへ許可するstdlib依存3件をexactで登録する。
  - 初期状態、旧の取得/更新/再始動列、同率と有限/無限学習率、未更新の集合変更、単一/負ID、損失制限、集合変更を伴う直接update、拒否前不変、input/output copy、独立owner、3種の乱数状態全体を確認する。
  - 完了: 対象test・依存境界suite・Ruff・Pyrightが成功し、対象taskの独立レビューPASS。
  - _Boundary: 診断証拠owner・behavior tests・exact依存guardの統合_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 4.1_

- [x] 2. 単一ownerの検出力と独立動作を確認する
  - GREEN後、型/値/総和/集合検査の削除、検査を同期後へ移す、昇順/式/計数/再始動/copyの破壊を実sourceへ1種ずつ入れ、対応testで失敗を確認して元byteへ戻す。変異ごとの一覧と検出結果を保存し、復元後の対象suiteを成功させる。
  - fresh processで旧実装とtest moduleをimportせず、取得→更新→集合変更→再始動→継続を実行して診断証拠を確認する。
  - 完了: 変異の検出とbyte復元、対象suiteの再成功、fresh processの成功を実測し、対象taskの独立レビューPASS。
  - _Depends: 1_
  - _Boundary: 診断証拠の検出力と新単独動作の証拠_
  - _Requirements: 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 4.1_

- [ ] 3. Windows固定基準の全回帰と統合証拠を確定する
  - source/testをcommit後、Windows基準で全pytest/JUnit、旧11・最終3golden、Ruff・Pyright・pip check・spec_checks identityを実測する。要求と測定済み検証結果の対応、対象commit/hash、未検証範囲を統合証拠として保存する。
  - SACで止まったら保護設定・venv・goldenを変更せずWSLで続け、結果を区別する。Windows全回帰が残る間はこのtaskと最終GOを完了にしない。
  - 完了: 独立担当の照合と対象taskレビューPASS、別sessionのfeature最終GO。最終レビュー前は再開案内を現状へ更新してprogressを実行し、GO記録後もprogressを実行する。全pytestの独立再実行は既存ユーザー決定で必須としない。
  - _Depends: 2_
  - _Boundary: 固定環境の全回帰と統合証拠_
  - _Requirements: 4.2_
