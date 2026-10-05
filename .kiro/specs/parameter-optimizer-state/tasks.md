# 実装タスク

- [x] 1. 固定parameter列のoptimizer所有と明示リセットを検証する
  - 新module未存在の実RED後、現在optimizerの取得・同条件再生成・成功後交換を実装する。
  - 実旧リセットへ初期/学習後/reset後のgroups/state/parameter/gradを照合する。
  - readonly参照・外側のstate更新反映・繰り返しreset・拒否入力/生成例外で旧参照保持・RNG不変を確認する。
  - 完成は対象test GREENとLuna実装APPROVEDで確認する。
  - _Boundary: optimizer状態管理器と実旧対照test_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.3, 2.4, 3.1, 3.2_

- [x] 2. 個別リセットから実共同更新への接続を照合する
  - test-onlyで共有部と二概念固有部の管理器を作り、現在参照から新batch/bindingを作る。
  - 24条件（class2/4×Adam standard/AMSGrad/SGD×reset4種）の実旧共同更新と新共同更新へ、共有更新/凍結を混ぜた3回の学習を照合する。
  - 共有だけ/頭一つだけ/全体/なしのresetを区別し、別owner state保持と旧借用記録非更新を確認する。
  - 完成は全NN/grad/optimizer/lossの実旧exact一致、数値部品production無変更、Luna実装APPROVEDで確認する。
  - _Boundary: 上位外側のtest-only実NN接続_
  - _Depends: 1_
  - _Requirements: 1.2, 2.1, 2.2, 2.4, 3.1, 3.2_

- [x] 3. 依存境界と全回帰・統合証拠を確認する
  - exact禁止/許可import注入の先行RED後、AST guardを追加する。
  - fresh新CPU生成/reset/更新smoke、全pytest（旧11/最終3golden）、Ruff/format/Pyright/pip、旧固定差分/源hashを確認する。
  - 全9要件の実測・旧所見・限界を保存し、全task/roadmap同期後にLuna最終feature GOを確認する。
  - 完成はTask APPROVED・feature GOと現在の検証hash一致で確認する。
  - _Boundary: 依存test/対象spec/roadmap_
  - _Depends: 1, 2_
  - _Requirements: 3.1, 3.2_

## Implementation Notes
番号付きmanual実装、独立Lunaレビューrequiredで逐次実行する。
旧configの変更はtestのmonkeypatchで復元し、実旧reset/builder/数値共同更新は置換しない。
resetはgradをclearしない。次の共同学習が既存どおりzero_gradする。
既存bindingは自動更新されない。上位で現在参照を取得し新記録を作る。
frozen/private回避、共有接続、空共有optimizerの方針変更は対象外。
日本語文書はapply_patchで保存する。Windows sandboxでのPyrightは共有venv絶対pythonpathとrequire_escalatedを使う。
