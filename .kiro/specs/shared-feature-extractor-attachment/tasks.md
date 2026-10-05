# 実装タスク

- [x] 1. 共有特徴抽出部の検証と参照交換を実装する
  - 未実装操作の実RED後、適合検証・成功後交換・概念固有部保持を実装する。
  - class2/4・隠れ層3構成・同/別参照の12条件でforward新経路と双方値/grad/参照、RNGを確認する。
  - 型/寸法/構造/dtype/device異常拒否で接続と双方の状態保持を検証する。
  - 完成は対象test GREENとLuna実装APPROVEDで確認する。
  - _Boundary: 分類器接続操作/対象test_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1_

- [ ] 2. 接続先optimizerを選択して実旧学習へ照合する
  - test-onlyで新接続→接続先parameterの共有owner選択→個別reset→新batch作成を実旧attachへ比較する。
  - class2/4×optimizer3×共有optimizer既存/欠落の12条件、既存共有state保持と3回共同学習の全loss/NN値/grad/state一致を確認する。
  - 新旧extractorのparameter列と共有managerの対応を確認し、旧managerは変更・流用せず、接続先のparameter列を持つmanagerを使って更新する。
  - 接続単体で個別optimizerが変更されないことと外側resetを区別する。
  - 完成は実旧exact照合test GREEN、数値production無変更、Luna実装APPROVEDで確認する。
  - _Boundary: 上位外側test-only接続_
  - _Depends: 1_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2_

- [ ] 3. 全回帰と統合証拠を揃える
  - 新importなしの既存モデル依存guard、fresh新CPU接続/共同学習smoke、全pytest旧11/最終3goldenを確認する。
  - Ruff/format/Pyright/pip・旧固定差分/源hash・全7条件の実測を記録する。
  - 全taskとroadmapを同期し、Task APPROVEDとfeature GO/現在hash一致で完成を確認する。
  - _Boundary: 対象spec/roadmap/既存依存検査_
  - _Depends: 1, 2_
  - _Requirements: 1.4, 2.1, 2.2_

## Implementation Notes
番号manual実装、独立Lunaレビューrequired。既存guardに新importは加えず、架空AST REDを作らない。
Torch RNGはtest外へ漏らさず、meta Tensorでは数値実体がないのでshape/device/dtype/参照のみ照合する。
旧configはmonkeypatchで復元。共有optimizerは新接続先parameterのownerを使い、旧共有ownerを流用しない。
Windows Pyrightは共有venv絶対pythonpath+require_escalatedで実行する。

