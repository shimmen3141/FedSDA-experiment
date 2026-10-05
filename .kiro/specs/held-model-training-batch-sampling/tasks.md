# タスク: 保有モデルの学習バッチ抽出
source/testを共有するため順次、(P)なし。既存Random/FIFO/NN/optimizer/共同更新は利用可能。
taskごとLuna独立レビュー→主担当current-state検証→指定commit。環境はdocs/experiments/refactoring-baseline.md。

- [x] 1. 参加選別と復元なし抽出を実装する
  - missingmodule対象testを先行し実RED→指定二productionを実装してGREENにする。
  - 実旧抽出へ小/大population、非昇順/負ID、未保有/不足/空混在、B1/B=N、同参照別位置、複数call全出力と終端RNGを完全照合する。
  - 参加者だけ一回drawを観測し、入力順/抽出順/skip時RNG不変を確認する。
  - 完了は対象pytest成功、Luna APPROVED、主担当fresh/current-state検証。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 3.1_
  - _Boundary: Training batch sampling（指定二productionと対象test）_

- [x] 2. 拒否と借用状態の契約を検証する
  - 後段/未抽出位置の不正Tensor/type/shape/finite/layout/device/dims、外側ID重複/bool/count/generatorをdraw前拒否し、Random.sample未呼出/借用RNGと全入力不変を確認する。
  - 未保有の不正payload、不足列の不正sample内容はskipし、保有列の不正コレクション型は拒否する検査範囲を確認する。
  - 三recordのfrozen/kwonly/defaultなし、非contiguous/requiresgrad/既存grad/出力独立storage/外側PythonNP/Torch RNG/dtype/device/gradcontextを検証する。
  - 完了は対象pytest/Luna APPROVED/主担当fresh検証。test-only追加にfakeREDを要求しない、欠落修正は実RED→GREEN。
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  - _Boundary: Sampler contract tests（対象test、必要な局所修正だけ同production）_

- [ ] 3. FIFOから共同更新までの上位接続を検証する
  - FIFO解放位置→上位観測解決/model別列→sampler→ID対応NN/optimizer→共同更新をtest-onlyで明示接続する。
  - 実旧抽出＋実旧共同学習へ複数step全値/grad/optimizerstate/終端RNGを完全照合する。学習本体はmockに置換しない。
  - 完了は対象pytest成功、Luna APPROVED、主担当fresh/current-state検証。sourceへFIFO/モデル依存を追加しない。
  - _Requirements: 3.1, 3.2_
  - _Boundary: Test-only cross-boundary integration（対象testのみ）_

- [ ] 4. 依存境界と全回帰・完成証拠を揃える
  - exact二module/publicsymbols禁止/許可注入testを先行し実RED→guardGREENを確認する。
  - freshCPU抽出→共同更新smoke/nolegacyimport、対象＋AST、全tests旧11/最終3golden、baseline748c3aa無差分を確認する。
  - integration-validation/roadmapへ12要件/配置/未移植境界/採否/実測を記録し、Luna taskAPPROVED/最終featureGOとmetadata整合を完了条件にする。
  - _Requirements: 3.1, 3.2, 3.3_
  - _Boundary: AST/evidence（依存test・対象spec・roadmap）_

## Implementation Notes
日本語はapply_patch。Torchdefaultdeviceはcontext、dtypeはfinally復元。旧globalRandomをfinally復元しgeneratorをcopy/reseedするproductionを作らない。
旧concept metadata/NN対応はtest-only。抽出位置の重複禁止と同一sample参照の重複禁止を混同しない。
