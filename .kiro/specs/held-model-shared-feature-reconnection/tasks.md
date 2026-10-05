# 実装タスク

- [x] 1. 保有モデルの共有元選択と再接続を実装する
  - 新操作未実装の実RED後、全入力の対応検証・共有元選択・入力順接続/個別reset・現在の対応結果を実装する。
  - 空/単独/混合/全負/非昇順/極大ID、先行state保持、既に共有済みを実旧操作へ照合する。
  - 型/ID/owner対応/寸法/構造/CPU契約の末尾不正でも全状態保持、途中例外のprefix/現在接続/suffix保持を確認する。
  - 完成は対象test GREEN・Ruff/format・Luna実装APPROVEDで確認する。
  - _Boundary: 保有モデル全体の共有再接続と対象test_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3_

- [ ] 2. 現在の対応記録から実旧共同学習へ照合する
  - test-onlyで返却列から新学習binding/参加batchを作成する。
  - class2/4×optimizer3×初期共有有無2の12実NN条件で、個別/shared state蓄積後の実旧再接続と3回共同学習の全loss/NN値/grad/state一致を確認する。
  - source ownerの維持、non-source旧共有owner非流用/不変、旧借用optimizer/state保持と現在parameter対応を明示確認する。
  - 完成はexact対照test GREEN・数値production無変更・Luna実装APPROVEDで確認する。
  - _Boundary: 上位の学習準備test-only統合_
  - _Depends: 1_
  - _Requirements: 1.4, 2.1, 2.2, 2.3, 2.4, 3.3_

- [ ] 3. 依存境界と全回帰を検証する
  - 新module exact依存の許可/禁止ケースを先行RED後にguardへ登録し、上位/旧package/学習計算へ依存しないことを確認する。
  - fresh新CPU全保有再接続→共同学習smoke、全pytest旧11/最終3golden、Ruff/format/Pyright/pip、旧固定差分とsourcehashを記録する。
  - 全11要件trace・全task/roadmap同期・Luna Task APPROVEDと別feature GO/現在hash一致で完成を確認する。
  - _Boundary: 既存依存guard/対象spec/roadmap/検証_
  - _Depends: 1, 2_
  - _Requirements: 3.1, 3.3_

## Implementation Notes
番号manual実装、独立Lunaレビューrequired。新規src/先取りtestは命名承認後に作成する。
Torch RNGはfork_rngで復元し、metaは参照/shape/dtype/deviceだけ照合する。旧configはmonkeypatchで復元。
入力のshared ownerは対応するparameterのものを用意する。成功後の結果記録から現在optimizerを取得し、新学習記録を作成する。
通常拒否は副作用前、予期しない途中reset例外は旧順序のprefixと現在接続を保持する。全体rollbackを主張しない。
Pyrightは共有venv絶対pythonpathとrequire_escalated、全pytestもWindows tmp ACL対応でrequire_escalatedを使う。

