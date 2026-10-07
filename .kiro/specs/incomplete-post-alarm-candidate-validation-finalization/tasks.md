# 実装タスク revision1

逐次順で依存する。環境・上流開始/観測/吸収/共同学習・旧oracleは既存。全task承認後の別fresh feature GOは実装taskとは別のゲート。

- [ ] 1. 件数不足を表す不変判定記録を実装する
  - 旧の実不足判定との対応をtest先行RED後に移植し、位置・検出器・学習区間件数・実観測件数を保持する。
  - 比較未成立と棄却を型で表し、旧のNaN/理由とのtest内対応、位置差、frozen/kw_onlyが照合できる。
  - _Boundary: 不足の不変判定情報_
  - _Requirements: 2.1_

- [ ] 2. 件数不足sessionの終端回収を組み立てる
  - test先行RED後に非active/不足/要求到達拒否と現行IDへの既存吸収を実装する。
  - 非activeは他入力にアクセスせずNone、状態変更とRNG消費なしを確認する。
  - 2/4class×観測0/1/3×保留0/3の実旧対照、処理件数0/位置前/後、metadataなし、現行ID変更を確認する。
  - 拒否matrixは後半不正標本でも先行標本を含め全owner/collection/model/optimizer/RNG不変、成功出力は旧判定/eventへ対応し不変となる。
  - _Boundary: 件数不足の終端回収runtime_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 3.1, 3.2_

- [ ] 3. 実学習と未到達観測と終端回収を接続する
  - 明示的test-only統合task。2/4class×3optimizerで実NN開始→未到達観測→回収→2回共同更新を実旧と照合する。
  - 空保留/metadataなしも確認し、損失・標本・計数・統計・全parameter/grad/optimizer/RNGが一致する。
  - stub/開始時ID/位置/余分なRNG/二重吸収等の代表変異を検出し、元source byteを復元してGREENを再確認する。
  - _Boundary: 開始と観測と終端回収の接続_
  - _Requirements: 1.2, 2.1, 2.2, 3.2_

- [ ] 4. 依存境界と新CPUの終端回収を検証する
  - 2module exact AST注入のRED→guard/両resolver登録→GREENを記録する。
  - fresh新CPUで旧/test importなしの実開始→未到達観測→回収→後続学習と非activeを確認する。
  - _Boundary: exact依存境界_
  - _Requirements: 3.3_

- [ ] 5. 基準環境の全回帰と検証証拠を記録する
  - 明示的統合検証task。実装commitの全pytest/JUnit・旧11/最終3golden・Ruff/Pyright/pipを実測する。
  - 固定旧差分、承認/source hash、tested commit、各taskの独立レビューと保証範囲を照合できる証拠を残す。
  - _Boundary: 基準環境の全回帰_
  - _Requirements: 3.3_
