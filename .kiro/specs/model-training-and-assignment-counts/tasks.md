# 実装task

要求/設計/命名とtask graphの承認後に依存順で実行。

- [ ] 1. モデル別学習・割当件数ownerをTDDで実装する
  - 実旧加算/読み取り/移管/再編の順序・拒否・snapshot独立をRED→GREENで検証。同ID旧破損と新拒否の対照を記録する。
  - ownerとsnapshotだけを実装し、対象/品質/型/実Lunaレビュー/主担当gateで完了。
  - Requirements: 1.1,1.2,1.3,2.1,2.2,2.3,3.1
- [ ] 2. exact依存と実学習への計数接続を検証する
  - AST許可/禁止注入をRED→GREEN。stdlib単独起動でtorch/旧非import。
  - 12条件各3共同学習、初回後に移管/上位ID対応、計数/全loss/NN値/grad/optimizer/RNG一致。接続はtest-only。
  - 対象＋AST/品質/独立Luna/主担当gateで完了。
  - Requirements: 3.2,3.3
- [ ] 3. 固定goldenと全回帰を検証する
  - fresh新CPUの学習→件数→移管を検証。全pytest旧11/最終3golden、Ruff/format/Pyright/pip、固定旧差分空を確認。
  - 承認hash/source hash/JUnit/実測と修正候補の追跡を記録し、独立Luna/主担当gateで完了。
  - Requirements: 3.2,3.3

全task完了後、別feature最終Luna GO（9条件/所有/依存/設計/ファイル計画）を確認し、再開案内/roadmap/承認状態を更新してcommit/pushする。
