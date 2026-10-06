# 実装タスク revision2

- [x] 1. 独立した候補と新規学習状態を生成する
  - 実旧生成oracleの実行確認済み。単体testを先に作りREDを記録してから組立を実装する。
  - 値・参照独立性・乱数一致・初期optimizer・生成前拒否を実旧と照合し、誤実装差し替えの検出を確認する。
  - 共有部parameterが空の分類器の拒否と、候補生成前後のRNG不変を確認する。隠れ層なしの候補を生成してから拒否しない。
  - _Boundary: 候補生成のruntime組立、対象単体test_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 3.1_

- [x] 2. 選択済み初期値と既存学習更新へ接続する
  - test-onlyで初期snapshot選択→生成→単一候補の共同更新を接続する。
  - 二値/4クラス×標準Adam/AMSGrad/SGDの6条件、3batchの全loss/parameter/grad/optimizer state/RNGを実旧updateへ照合する。
  - 旧importなしのfresh新CPU起動を確認し、epoch/session未完成の範囲を記録する。
  - _Boundary: 対象接続test、fresh CPU検証_
  - _Depends: 1_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 3.3_

- [ ] 3. 依存境界と基準環境の回帰を検証する
  - AST注入契約のREDを確認してexact import guardを実装しGREENを確認する。
  - 実装commitに対する全pytest/JUnit、Ruff・Pyright・pip・旧固定差分・承認hashを確認する。
  - integration-validationへ実測と未完成範囲を記録する。別feature最終GOはtask承認の後に依頼する。
  - _Boundary: AST依存境界、統合検証と証拠_
  - _Depends: 2_
  - _Requirements: 3.2, 3.3_
