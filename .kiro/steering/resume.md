# リファクタリングの再開案内

更新: 2026-10-07。これは短い案内であり、承認・進捗の正本は各specのspec.jsonとtasks.md。

Claude・Codexで交代する場合は[共通引継ぎ手順](agent-handoff.md)を参照する。
Claudeの入口は[CLAUDE.md](../../CLAUDE.md)。同じworktreeとspecを使い、GPT-6 Lunaを優先し、利用不能時はSonnetの独立レビューで承認する。

## 現在地

- 作業場所: `.worktrees/refactoring/`、ブランチ: `refactor/architecture`。
- 直近完了: [採用候補のローカル採用](../specs/adopted-candidate-local-adoption/README.md)（要求r2/設計r2/命名r2/tasks r2・全3task承認/完了、別feature最終GO）。その前は[一時モデルIDの採番](../specs/temporary-model-id-allocation/README.md)と[採用候補の初期ローカル登録](../specs/adopted-candidate-initial-local-registration/README.md)。3件とも主担当Claude Code、レビューはcodex exec経由のGPT-6 Luna。
- 直近の検証済み実装commit: `af5c5b8`。全5876 passed/3 skipped/1既存warning（主担当実測、JUnit照合）、対象＋AST977、fresh新CPU/品質/旧11・最終3golden成功。runtimeの状態なし関数で、次の一時IDの読取り→登録→採番確定→候補学習量の計数→保留標本の追加→現在の学習帰属ID切替えを組み立て、変更recordを返す。実旧_finalize_forward_validationの採用分岐（forward_persistent）を実行して全状態を対照し、12条件で採用後学習と正式ID確認まで照合した。
- 全pytestの独立再現の基準は[共通引継ぎ手順](agent-handoff.md)の同名節（2026-10-07ユーザー決定）。Luna側sandboxでは全pytestを再現できないため、主担当実測＋JUnit照合で判定する。
- 旧実装の記録: 直近3specで見つけたLEGACY-012（登録途中失敗の部分更新と採番消費）・013（使用済み一時IDへの再登録の黙った置換）・014（採用時の保留標本が割当概念計数と損失統計へ反映されない。意図した仕様か未確認、oracle診断への影響未確認）をimplementation-findingsへ記録済み。旧挙動は移植で維持している。
- 実装途中のtaskはない。新client・新全体runの接続は未完了。部品の旧実装対照と、新全体runのgolden一致は別の完了条件。
- `federated_drift_experiment/`は固定旧実装との対照・既存golden実行用。`src/federated_learning_experiments/`は移植中の新実装。旧固定基準は`748c3aa`、旧名alias/互換読込みを追加しない。

## 次の候補（未仕様化・未承認）

1. 前向き検証の確定処理の残り: 採用時のモデル/データ状態更新は完了（変更recordを返す）。次は旧_finalize_forward_validationの他の部分を仕様化する。(a)棄却・再利用・維持分岐の保留標本吸収（旧_absorb_into_store: 標本追加＋標本ごとの損失評価と統計更新＋割当概念計数＋statistics計算量。採用分岐と異なる）、(b)再利用分岐の現在ID切替えとtournament参照の値反映/optimizer reset、(c)判定record・切替位置・検出エピソード・適応イベントの記録と、現在ID変更の通知（最終構成では予測重み再始動）。既存の採否判定（post-alarm-candidate-loss-evaluation）と損失収集（post-alarm-candidate-loss-collection）への接続もここで行う。実旧_finalize_forward_validationは最小属性の実clientで実行できる（adopted-candidate-local-adoptionのresearch.md/testのbuild_local_adoption_oracle）。確認APIはモデル保有を前提にし、欠落時snapshotからの再構築は呼出し側server/clientへ残す（旧分岐未移植）。計算量診断の"initialization"/"statistics"記録は未移植。現在ID同値設定はno-op、新計数同ID移管は拒否（LEGACY011、通常経路への影響未確認）。評価fallback/EVAL_MAX_SAMPLESはサーバ評価接続時、実送信は通信specで扱う。
2. 候補の生成・初期学習・前向き検証sessionの開始（旧_begin_forward_validation: 候補初期化、区間学習と学習量の記録、参照モデルのsnapshot、履歴平均）を、既存の初期化・学習・採否部品へ接続する。採用時に渡すcandidate_trained_sample_count/candidate_parameter_update_step_countはここで得る。
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
