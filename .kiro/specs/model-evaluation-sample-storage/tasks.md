# 実装task

要求/設計/命名とtask graphの承認後、依存順に実行する。

- [x] 1. 評価標本recordと保持ownerをTDDで実装する
  - 実旧追加/confirm/mappingを直接照合するtestをREDにし、固定条件と全入力拒否・snapshot/借用/payload非検査を確認する。
  - 2宣言と一storeを実装してGREEN。モデル・学習・評価forwardをownerへ持ち込まない。
  - 対象/品質/型/独立Luna実diffレビューと主担当gateで完了。
  - Requirements: 1.1,1.2,1.3,1.4,2.1,2.2,2.3,3.1,3.2
- [ ] 2. 依存境界と評価への接続を検証する
  - exact AST guardの許可/禁止testをRED→GREEN、package再export/child/private/global random/他ownerを拒否する。
  - class2/4で保持→単一付替え→ID再編→cat→損失評価の実旧NN対照、3共有RNG/defaults保持。production接続追加なし。
  - 対象＋AST/品質/独立Luna/主担当gateで完了。
  - Requirements: 3.3,3.4
- [ ] 3. 全回帰と新CPU独立起動を検証する
  - fresh新CPUで追加/付替え/再編と損失評価、旧非import。全pytest旧11/最終3goldenを固定環境で実行する。
  - Ruff/format/Pyright/pip、固定旧production/golden差分空、hash/JUnit/実測を記録。独立Luna承認と主担当gateで完了。
  - Requirements: 3.3,3.4

全task完了後、別feature最終Luna GO（全11条件/所有/依存/設計/ファイル計画）を確認し、再開案内・roadmapと承認状態を更新してcommit/pushする。
