# タスク: パラメータoptimizer生成
共有source/testを使うため全taskは順次、(P)なし。設定field基盤/新NN/共有venvは実装済み。
環境はdocs/experiments/refactoring-baseline.md。taskごと独立Luna review→主担当fresh検証→commit。

- [x] 1. 方式別設定と標準optimizer生成を実装する
  - missing-module対象testを先に書いて実REDを確認し、指定2productionを実装してGREENにする。
  - Adam standard/AMSGrad・decay有無・lr0とSGDを実旧builderへ照合。全groups/参照順/初期stateと外部固定grad複数stepの全値/stateが一致することを確認する。
  - 完了は指定source/testのみ、対象pytest成功、Luna APPROVEDと主担当fresh検証。
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 3.1_
  - _Boundary: Optimizer configuration/construction（指定2sourceと対象test）_
  - _Depends: なし（field validator実装済み）_

- [x] 2. 拒否・独立性・環境とモデル接続を検証する
  - 対象testだけでforged/未知型・後段不正/空/重複/dtype/device/layout/boolを生成前拒否し、値/grad/設定とRNG不変を確認する。
  - grad=None/requires_gradFalse/noncontiguous/scalar/0要素shape、frozen/defaultなし、別optimizer state独立、ambientfloat64/meta/grad/PythonNP保持を検証する。
  - 新NNから共有extractor/adapter→classifier列をtest-only選択して二optimizer生成、集合非重複/モデルにoptimizer追加なしを確認する。
  - 完了は対象pytestとLuna APPROVED/主担当fresh検証。test-only追加にfakeREDを要求しない。
  - _Requirements: 1.3, 2.1, 2.2, 2.3, 3.2_
  - _Boundary: Test-only integration（対象testのみ）_
  - _Depends: 1_

- [ ] 3. 依存境界・全回帰と完成証拠を揃える
  - exact2module/symbol禁止/許可注入testを先行追加して実RED→allowlist更新GREENを確認する。
  - fresh CPU生成/外部step smoke（旧importなし）、対象＋AST、全tests旧11/最終3golden、旧748c3aa無差分を確認する。
  - integration-validationとroadmapへ9条件/配置/未移植境界を記録。実証した空共有optimizer問題は既存findings台帳に記録する。
  - 完了はLuna taskreview/最終featureGOと主担当fresh検証/metadata更新。
  - _Requirements: 3.1, 3.2, 3.3_
  - _Boundary: AST/evidence（依存test・対象spec・roadmap・実証findings）_
  - _Depends: 1, 2_

## Implementation Notes
日本語はapply_patchで書く。torch.set_default_deviceを使わずmeta devicecontext、dtypeはfinally復元。
旧builderの呼出/config変更はtest-only monkeypatch。productionにlegacyimport/モデル依存/学習進行を追加しない。
