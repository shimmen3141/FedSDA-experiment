# タスク: 一回の共同モデルパラメータ更新
同一source/testを用いるため順次、(P)なし。モデル/設定/optimizer生成は実装済み。
taskごと独立Luna review→主担当fresh検証→commit。環境はdocs/experiments/refactoring-baseline.md。

- [x] 1. 確定済み参加バッチから一回共同更新を実装する
  - 対象testを先行しmissing-module実RED→指定production二つのGREENを確認する。
  - 実旧共同経路を固定batchで呼び、単一/複数・不均等標本数・二値/多クラス・Adam標準/AMSGrad/SGD・共有有効/凍結の複数step全値/grad/stateを照合する。
  - 共有forward一回、zero/step順、標本数加重loss返却を観測する。診断や抽出を新実装へ持ち込まない。
  - 完了は対象pytest成功、Luna APPROVED、主担当fresh検証。
  - _Requirements: 1.1, 1.3, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 5.1_
  - _Boundary: Joint update（指定二productionと対象test）_

- [x] 2. 更新前拒否・空列・空共有と借用状態を検証する
  - 対象testで型/forged/shape/finite/labels/後段不正/optimizer参照逆順・重複/grad無効を拒否し、zero前の値/grad/state不変を確認する。
  - 共有Parameterが存在するのに共有optimizer=Noneの入力を更新有効/無効の両方でzero前拒否し、既存値/grad/state不変を確認する。空共有へoptimizerを渡す逆ケースも拒否する。
  - 空列no-op、softtarget、非contiguous、空共有での個別学習、共有凍結での既存state保持、借用参照/RNG/ambientdtype・device・gradcontextを検証する。
  - 完了は対象pytest成功、Luna APPROVED、主担当fresh検証。test-only追加にfakeREDを要求しない。発見した実装欠落の修正は実RED→GREEN。
  - _Requirements: 1.2, 1.3, 2.1, 2.2, 2.3, 2.4, 3.4, 4.1, 4.2, 4.3, 5.2_
  - _Boundary: Test-only contract integration（対象test、必要な局所修正は同production）_

- [ ] 3. 依存境界・全回帰・完成証拠を揃える
  - exact二module/public symbolの禁止/許可testを先行し実RED→guard更新GREENを確認する。
  - 旧importなしfreshCPU共同更新、対象＋AST、全tests旧11/最終3golden、旧748c3aa無差分を確認する。
  - integration-validation/roadmapへ全要件・配置・未移植境界を記録する。既存LEGACY-010の正常拡張と旧未修正を区別し、通常旧不具合を未実証で増やさない。
  - 完了はLuna taskAPPROVED/最終featureGO、主担当fresh検証、hash/UTF8/metadataの整合。
  - _Requirements: 5.1, 5.2, 5.3_
  - _Boundary: AST/evidence（依存test・対象spec・roadmap）_

## Implementation Notes
旧loss/backward/stepは実物で比較し、torch.equalを緩和しない。旧名対応はtest-only。
日本語はapply_patch。torch.set_default_deviceを使わずcontext、dtypeはfinally復元。
