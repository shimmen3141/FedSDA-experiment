# 実装task

- [x] 1. 単一帰属ID ownerをTDDで実装する
  - ローカル切替・一段対応表・同値/欠落・signed ID・読取り専用・全入力拒否を実旧methodと直接期待値へ照合。取得済みrecordが後続ID変更で変わらないことも検証。RED→GREEN、Lunaレビューと主担当gate。
  - Requirements: 1.1,1.2,2.1,2.2,3.1,3.3
- [x] 2. 依存境界と登録確認の上位接続を検証する
  - exact AST許可/禁止注入RED→GREEN、stdlib単独fresh。実旧登録確認と新計数移管＋現在ID変更をtest-onlyで接続。理由別通知の区別を検証。Lunaレビューと主担当gate。
  - Requirements: 3.1,3.2,3.3
- [ ] 3. 固定環境の全回帰とfeature統合を確認する
  - 全pytest/旧11・最終3golden、Ruff/format/Pyright/pip/diff、固定旧とgolden差分空、承認hash/source hash/JUnit記録。taskレビューと主担当gateで完了。
  - Requirements: 3.2,3.3

全task完了後に別feature最終Luna GOを確認し、正本/再開案内/roadmapを更新してcommit/pushする。
