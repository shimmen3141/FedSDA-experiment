# 実装タスク revision1

逐次順に依存する。全task承認後の別fresh feature GOは実装taskとは別のゲート。

- [x] 1. 候補検証sessionの開始を組み立てる
  - 事前検査と開始recordをtest先行RED後に実装し、生成→学習→参照固定→空収集を各1回の順で呼ぶ。
  - 実旧開始oracleで2class×3optimizer×3方式×epoch0/3の36条件と最終構成6条件、全候補/grad/optimizer/参照順序/履歴平均/計数/RNG/metadataを照合する。
  - _Boundary: 開始runtime_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 3.1_

- [x] 2. 事前拒否と借用状態の不変性を検証する
  - 各preflight条件の拒否時RNG/保有parameter/grad/optimizer/統計/帰属/借用tensor不変、binding固定を確認する。
  - stub/順序入替/学習省略/参照live借用を差替え検出し、元byte/hashへ復元する。
  - _Boundary: 開始runtime_
  - _Requirements: 1.3, 3.1_

- [x] 3. 後続の候補検証観測へ接続する
  - test-onlyで開始→提案次位置から規定件数まで観測を実旧損失に照合する。
  - 候補の次batch更新でも固定参照と保有モデルが不変であることを確認する。
  - _Boundary: 開始runtimeと既存観測の接続_
  - _Requirements: 2.2_

- [x] 4. 依存境界と新CPUの接続を検証する
  - exact AST注入RED→guardとresolver登録→GREEN、許可/禁止symbolsを確認する。
  - fresh新CPUで開始→観測を実行し、旧importなしを確認する。
  - _Boundary: exact依存境界_
  - _Requirements: 3.2, 3.3_

- [ ] 5. 基準環境で全回帰と品質を統合検証する
  - 明示的統合検証task。実装commitで全pytest/JUnit・旧11/最終3golden・Ruff/Pyright/pipを実測する。
  - 固定旧差分・承認/source hashと検証対象commitを照合し、証拠を記録する。
  - _Boundary: 基準環境の全回帰_
  - _Requirements: 3.3_
