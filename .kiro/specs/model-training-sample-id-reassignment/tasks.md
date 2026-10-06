# 実装task

要求/設計/命名revision1・graph PASS/task APPROVEDを正本に依存順に実行する。

- [x] 1. 学習標本storeの単一ID付替えをTDDで追加する
  - 実旧confirm順序/列上書き/空/欠落/同ID/借用、signed期待値、両ID拒否、過去snapshot/後続追加、payload非検査のtestを先にREDにする。
  - 既存storeへ一APIのみ追加してGREEN。対象/既存storage・samplerテスト/品質・独立Luna実diffレビューで完了。
  - Requirements: 1.1,1.2,1.3,1.4,2.1,2.2,2.3
- [x] 2. 抽出・共同更新へtest-only接続する
  - 12条件各3抽出/共同更新。最初の更新後に旧confirmと新標本store付替え、上位がbinding IDを更新。
  - batch/全loss/parameter/grad/両optimizer/Random終端と3共有RNG/defaultsをexact照合。production追加なし、RED N/A。
  - 対象/品質・独立Lunaレビューで完了。
  - Requirements: 2.4
- [ ] 3. 既存境界と固定goldenを統合検証する
  - 既存AST（変更/新RED N/A）・fresh新CPU標本store→sampler→学習接続、旧非importを確認。
  - 全pytest旧11/最終3golden、Ruff/format/Pyright/pip、固定旧差分空、承認hash/source hash/JUnitを記録。
  - 独立Luna task承認と主担当gateでcheck。
  - Requirements: 2.4

全task完了後、別feature最終Luna GO（8条件/契約/所有/依存/設計/ファイル計画）を得て、再開案内・roadmap更新とcommit/pushを行う。
