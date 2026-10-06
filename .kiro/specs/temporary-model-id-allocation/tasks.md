# 実装task revision 2

- [x] 1. 一時IDの採番ownerと依存境界をTDDで実装する
  - 実旧BaseClientの実__init__/実_alloc_temp_idを呼ぶ最小fixtureとの初期値・採番列の対照（式をtestへ複製しない）、読取り非消費、client_id拒否、owner独立。AST注入RED→exact guard GREEN。採番IDを既存の初期ローカル登録へ渡す連続登録のtest-only接続（採番値が登録APIの一時ID契約を満たす確認。採番moduleからruntimeへの依存は作らない）。stdlib単独起動。RED→GREEN、品質/型/独立Luna/主担当gate。
  - Requirements: 1.1,1.2,1.3,2.1,3.1,3.2
- [ ] 2. 固定環境の全回帰を確認する
  - 新moduleと依存guardの追加が既存testと固定旧実装の値へ影響しないことを同じcommitで示すため、worktree規約と前specまでの運用どおり全pytest/旧11・最終3goldenを実測する。品質/型/pip/diff、固定旧差分空、承認/source hash・JUnit/実測を記録。判定基準はsteering/agent-handoff.md。独立Luna/主担当gateで完了。
  - Requirements: 3.2

全task完了後、別feature最終Luna GOで正本・再開案内/roadmap更新、commit/pushする。
