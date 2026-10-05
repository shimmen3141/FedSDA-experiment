# 実装タスク

- [x] 1. 共有値の反映と採用候補の準備を実装する
  - 新公開操作未実装のimport RED後、共有値コピーと候補copy→attach→個別resetを実装する。
  - 型/構造/寸法/CPU32/owner対応/独立性は全変更前に拒否し、値・参照・grad・旧optimizer/RNG保持を確認する。
  - 同一共有部、空共有層値反映、途中reset例外も確認する。
  - 完了は単体GREEN・Ruff/format・Luna Task APPROVEDで判断する。
  - _Boundary: モデル値反映と外側採用候補準備の明示統合_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3_

- [ ] 2. 実旧候補準備から共同学習へ照合する
  - test-onlyでclass2/4×optimizer3×初期共有2の12条件、先行step→実旧prepare→3共同stepを接続する。
  - 全loss/NN/grad/state一致、active共有optimizer学習state保持と旧借用個別optimizer保持を確認する。候補旧共有ownerがactiveと同一ならそのまま継続利用し、別ownerなら非再利用・不変を確認する。
  - 完了は対照GREEN・数値production無変更・Luna Task APPROVEDで判断する。
  - _Boundary: 上位の現在optimizerからの学習準備のtest-only統合_
  - _Depends: 1_
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3_

- [ ] 3. 依存境界と全回帰を検証する
  - 新moduleのAST禁止/許可ケースを先行RED→exact guard GREENにする。
  - fresh新CPUで値反映・候補接続・共同step、全pytest旧11/最終3golden、Ruff/format/Pyright/pip、旧固定差分/sourcehashを記録する。
  - 全9要件trace、検証記録、Luna Task APPROVEDで本taskを完了する。全task/spec/roadmap同期後のfeature GOは、全checkbox完了後の別ゲートとする。
  - _Boundary: 既存guard/対象spec/roadmap/統合検証_
  - _Depends: 1, 2_
  - _Requirements: 3.1, 3.2, 3.3_

## Implementation Notes
番号manual実装、独立Luna required。命名承認後にproduction/先取りtestを作成する。
Torch fork_rngを使用し、旧configはmonkeypatchで復元。Pyrightはshared venv絶対pythonpath＋require_escalated、全pytestはWindows tmp ACL対策require_escalated。
共有optimizer ownerは渡さず上位で保持、reset後の個別owner現在optimizerから学習batchを作る。
