# リファクタリングの再開案内

更新: 2026-10-07。これは短い案内であり、承認・進捗の正本は各specのspec.jsonとtasks.md。

Claude・Codexで交代する場合は[共通引継ぎ手順](agent-handoff.md)を参照する。
Claudeの入口は[CLAUDE.md](../../CLAUDE.md)。同じworktreeとspecを使い、GPT-6 Lunaを優先し、利用不能時はSonnetの独立レビューで承認する。

## 現在地

- 作業場所: `.worktrees/refactoring/`、ブランチ: `refactor/architecture`。
- 直近完了: [保有モデルの正式登録確認](../specs/held-model-registration-confirmation/README.md)。要求/設計revision1・命名revision2・tasks revision1・全3taskをLuna承認/完了、別feature最終GO。
- 直近の検証済み実装commit: `03f24e6`。全5518 passed/3 skipped/1既存warning、対象＋AST864passed、fresh新CPU/品質/旧11・最終3golden成功。実測は同specのintegration-validation.md。runtimeに7ownerの確認組立を接続し、現在bindingによる後続学習の全数値/optimizer保持を実旧へ対照。
- レビュー待ち: [採用候補の初期ローカル登録](../specs/adopted-candidate-initial-local-registration/README.md)。主担当Claude Code、要求r2/設計r2/命名r3/tasks r3と全3taskをLuna承認、実装commit dd9f57d・記録09c399b。全5678 passed/3 skipped/1既存warning、対象＋AST928、fresh新CPU/品質/旧11・最終3golden成功（主担当実測）。別feature最終レビューはNO-GO: Luna側sandboxで全pytestを独立再現できないことだけが理由。解消手段はユーザー判断待ちで、同specのreview.md「レビュー待ち」を参照。completedではない。
- 実装途中のtaskはない。新client・新全体runの接続は未完了。部品の旧実装対照と、新全体runのgolden一致は別の完了条件。
- `federated_drift_experiment/`は固定旧実装との対照・既存golden実行用。`src/federated_learning_experiments/`は移植中の新実装。旧固定基準は`748c3aa`、旧名alias/互換読込みを追加しない。

## 次の候補（未仕様化・未承認）

0. 上のレビュー待ちの解消（最優先）。GO後、下の1のうち登録本体は同specで実装済みとなり、残りは候補session側の計数/標本追加/現在ID切替え・通知。
1. 採用済みモデルの初期ローカル登録: 各ownerと、保有済み一時modelの正式ID確認組立は完了。次は旧`clients/base.py::_register_trained_new_model`・各methodの呼出箇所から、共有部反映/再接続・学習状態登録・初期損失統計・独立snapshot・送信保留/待機の組立を仕様化する。計数/標本追加/現在ID切替えの呼出順と、候補採用・初期登録・確認の境界を明示する。確認APIはモデル保有を前提にし、欠落時snapshotからの再構築は呼出し側server/clientへ残す（旧分岐未移植）。現在ID同値設定はno-op、新計数同ID移管は拒否（LEGACY011、通常経路への影響未確認）。評価fallback/EVAL_MAX_SAMPLESはサーバ評価接続時、実送信は通信specで扱う。
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
