# 実装タスク revision1

逐次順に依存する。全task承認後の別fresh feature GOは実装taskとは別のゲート。

- [x] 1. 不変の判定記録と再分析値を実装する
  - 旧の実判定recordとの数値対応をtest先行RED後に移植し、metadataと既存評価を保持する。
  - 14fields/4導出値の対応、履歴なしNone/旧NaN、frozen/kw_only、入力評価不変を確認する。
  - _Boundary: 不変判定情報_
  - _Requirements: 2.1_

- [ ] 2. 1標本の進行と到達時確定を組み立てる
  - test先行RED後に非active/未到達/到達の返却と観測→評価→記録→既存適用を実装する。
  - 2/4class×4分岐×保留0/3の16条件で実旧の記録/最終状態/RNGを照合し、各事前拒否と範囲限定も確認する。
  - _Boundary: 通常検証の進行runtime_
  - _Requirements: 1.1, 1.2, 1.3, 2.2, 2.3, 3.1, 3.2_

- [ ] 3. 実学習と観測と確定を接続する
  - test-onlyの2/4class×3optimizerの6条件で開始→観測→確定→共同更新を実旧に照合する。
  - 奇数件数/metadata None/履歴なし/空保留/確定時参照可用性と現在IDを確認し、代表変異を検出して元byteへ戻す。
  - _Boundary: 開始と進行と確定の接続_
  - _Requirements: 1.3, 2.1, 2.2, 3.2_

- [ ] 4. 依存境界と新CPUの進行を検証する
  - 2moduleのexact AST注入RED→guard/両resolver登録→GREENを記録する。
  - fresh新CPUで旧importなしの非active/未到達/到達と後続学習を確認する。
  - _Boundary: exact依存境界_
  - _Requirements: 2.3, 3.3_

- [ ] 5. 基準環境の全回帰と検証証拠を記録する
  - 明示的統合検証task。実装commitで全pytest/JUnit・旧11/最終3golden・Ruff/Pyright/pipを実測する。
  - 固定旧差分、承認/source hash、tested commit、各taskレビューと保証範囲を記録する。
  - _Boundary: 基準環境の全回帰_
  - _Requirements: 3.3_
