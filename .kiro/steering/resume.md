# リファクタリングの再開案内

更新: 2026-10-06。これは短い案内であり、承認・進捗の正本は各specのspec.jsonとtasks.md。

## 現在地

- 作業場所: `.worktrees/refactoring/`、ブランチ: `refactor/architecture`。
- 直近完了: [保有モデルの学習状態管理](../specs/held-model-training-state-registry/README.md)。全3tasks完了、Luna最終GO。
- 直近の検証済み実装commit: `7eb04d5`。全4229 passed/3 skipped/1既存warning、旧11/最終3golden成功。実測は同specのintegration-validation.md。
- 実装途中のtaskはない。新client・新全体runの接続は未完了。部品の旧実装対照と、新全体runのgolden一致は別の完了条件。
- `federated_drift_experiment/`は固定旧実装との対照・既存golden実行用。`src/federated_learning_experiments/`は移植中の新実装。旧固定基準は`748c3aa`、旧名alias/互換読込みを追加しない。

## 次の候補（未仕様化・未承認）

1. 採用候補の正式なローカル登録: モデル一覧は上記registryで完了。旧`clients/base.py::_register_trained_new_model`を入口に、モデルの標本別損失評価→初期統計、parameter送信snapshot、標本/送信状態の不足依存と更新順を確認する。次は準備済みモデルの損失評価境界から検討し、一つのspecへ詰め込まない。
2. 候補の生成・初期学習・前向き検証sessionの開始/終了を、既存の初期化・学習・採否部品へ接続する。
3. 警報後の帰属変更とclient進行、サーバ同期・ID対応へ順次接続する。

この順序は候補。再開時にコードと完了specを照合し、未移植の依存があれば先に仕様化する。既存の完了タスクを無条件に再実装しない。

## 最初に読む文書と手順

1. worktreeの[AGENTS.md](../../AGENTS.md)と[方針の正本](../../docs/research/refactoring-policy.md)。
2. [roadmap](roadmap.md)の「現在の状態」、[product](product.md)、[tech](tech.md)、[structure](structure.md)。
3. 対象specのREADME→spec.json/tasks.md→requirements/design/naming→review/integration-validation。未完了taskがあれば次のspecより先に扱う。
4. `git status --short`とブランチを確認し、別タスクの差分・未追跡資料を特定する。文書の案内と実際の状態が違う場合はGitとspecの正本を確認する。
5. cc-sddの要求→設計/命名→tasks→実装→統合検証へ進む。各段階は実GPT-6 Lunaレビューと有用な指摘の反映で承認し、採否と内容hashを記録する。命名承認前に新srcや先取りtestを作らない。

## 別タスクと記録の扱い

HTMLは元checkoutの`docs/overview/fedsda-processing-flow.html`にあるコミット保留資料。worktreeへ自動コピー・stageしない。他の保留2資料もAGENTS.mdに従う。
旧実装の不具合は[implementation-findings](../../docs/research/implementation-findings/README.md)、開発手順の問題は`development-findings/`に記録する。
中断時は対象specへ未完了task/検証/差分を記録し、この案内の現在地を更新する。別タスクの結果をリファクタリングの完了証拠に混ぜない。
