# リファクタリングの再開案内

更新: 2026-10-07。これは短い案内であり、承認・進捗の正本は各specのspec.jsonとtasks.md。

Claude・Codexで交代する場合は[共通引継ぎ手順](agent-handoff.md)を参照する。
Claudeの入口は[CLAUDE.md](../../CLAUDE.md)。同じworktreeとspecを使い、GPT-6 Lunaを優先し、利用不能時はSonnetの独立レビューで承認する。

## 現在地

- 作業場所: `.worktrees/refactoring/`、ブランチ: `refactor/architecture`。
- 直近完了: [一時モデルIDの採番](../specs/temporary-model-id-allocation/README.md)（要求r1/設計r2/命名r3/tasks r2・全2task承認/完了、別feature最終GO）と、その前の[採用候補の初期ローカル登録](../specs/adopted-candidate-initial-local-registration/README.md)。どちらも主担当Claude Code、レビューはcodex exec経由のGPT-6 Luna。
- 直近の検証済み実装commit: `785776b`。全5731 passed/3 skipped/1既存warning（主担当実測、JUnit照合）、fresh新CPU/品質/旧11・最終3golden成功。採番ownerは実旧BaseClientの実初期化/実採番と対照し、採番IDでの連続登録までtest-only接続した。
- 全pytestの独立再現の基準は[共通引継ぎ手順](agent-handoff.md)の同名節（2026-10-07ユーザー決定）。Luna側sandboxでは全pytestを再現できないため、主担当実測＋JUnit照合で判定する。
- 実装途中のtaskはない。新client・新全体runの接続は未完了。部品の旧実装対照と、新全体runのgolden一致は別の完了条件。
- `federated_drift_experiment/`は固定旧実装との対照・既存golden実行用。`src/federated_learning_experiments/`は移植中の新実装。旧固定基準は`748c3aa`、旧名alias/互換読込みを追加しない。

## 次の候補（未仕様化・未承認）

1. 候補採用時の接続: 登録本体と一時IDの採番ownerは完了。次は旧fedsda.pyの前向き検証採用分岐から、採番→登録呼出し→候補学習量の計数加算（record_completed_model_training）→保留標本の学習標本storeへの追加（旧はextendのみで統計更新・概念計数なし）→現在の学習帰属ID切替え（変更recordを返し、旧_on_local_model_change相当の通知は上位）の組立を仕様化する。switch位置/episode記録/adaptation event、棄却・再利用・維持分岐（旧_absorb_into_storeは統計更新と概念計数を伴う）との境界、計算量診断の"initialization"記録（未移植）を明示する。確認APIはモデル保有を前提にし、欠落時snapshotからの再構築は呼出し側server/clientへ残す（旧分岐未移植）。現在ID同値設定はno-op、新計数同ID移管は拒否（LEGACY011、通常経路への影響未確認）。評価fallback/EVAL_MAX_SAMPLESはサーバ評価接続時、実送信は通信specで扱う。
2. 候補の生成・初期学習・前向き検証sessionの開始/終了を、既存の初期化・学習・採否部品へ接続する。
3. 警報後の帰属変更とclient進行、サーバ同期・ID対応へ順次接続する。

この順序は候補。再開時にコードと完了specを照合し、未移植の依存があれば先に仕様化する。既存の完了タスクを無条件に再実装しない。

新構成の機能名はproduct.md/configuration-foundation/naming.mdを参照。Residual Adapterの語は構造として維持し、部品はNonlinearResidualAdapter、構成はshared_backbone_residual_adapter。旧Switchingの新構成名はfixed_share_weighted_prediction。旧称は過去実験との対応説明で併記する。

## 最初に読む文書と手順

1. worktreeの[AGENTS.md](../../AGENTS.md)と[方針の正本](../../docs/research/refactoring-policy.md)。
2. [roadmap](roadmap.md)の「現在の状態」、[product](product.md)、[tech](tech.md)、[structure](structure.md)。
3. 対象specのREADME→spec.json/tasks.md→requirements/design/naming→review/integration-validation。未完了taskがあれば次のspecより先に扱う。
4. `git status --short`とブランチを確認し、別タスクの差分・未追跡資料を特定する。文書の案内と実際の状態が違う場合はGitとspecの正本を確認する。
5. cc-sddの要求→設計/命名→tasks→実装→統合検証へ進む。各段階はGPT-6 Luna（利用不能時はSonnet）の独立レビューと有用な指摘の反映で承認し、担当モデル・代替理由・採否・内容hashを記録する。命名承認前に新srcや先取りtestを作らない。

## 別タスクと記録の扱い

HTMLは元checkoutの`docs/overview/fedsda-processing-flow.html`にあるコミット保留資料。worktreeへ自動コピー・stageしない。他の保留2資料もAGENTS.mdに従う。
旧実装の不具合は[implementation-findings](../../docs/research/implementation-findings/README.md)、開発手順の問題は`development-findings/`に記録する。
中断時は対象specへ未完了task/検証/差分を記録し、この案内の現在地を更新する。別タスクの結果をリファクタリングの完了証拠に混ぜない。
