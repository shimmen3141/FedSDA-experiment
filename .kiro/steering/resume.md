# リファクタリングの再開案内

更新: 2026-10-07。これは短い案内であり、承認・進捗の正本は各specのspec.jsonとtasks.md。

Claude・Codexで交代する場合は[共通引継ぎ手順](agent-handoff.md)を参照する。
Claudeの入口は[CLAUDE.md](../../CLAUDE.md)。同じworktreeとspecを使い、GPT-6 Lunaを優先し、利用不能時はSonnetの独立レビューで承認する。

## 現在地

- 作業場所: `.worktrees/refactoring/`、ブランチ: `refactor/architecture`。
- 直近完了: [帰属確定標本の吸収](../specs/assigned-training-sample-absorption/README.md)（要求r1/設計r1/命名r3/tasks r1・全3task承認/完了、別feature最終GO）。その前は[採用候補のローカル採用](../specs/adopted-candidate-local-adoption/README.md)、[一時モデルIDの採番](../specs/temporary-model-id-allocation/README.md)、[採用候補の初期ローカル登録](../specs/adopted-candidate-initial-local-registration/README.md)。4件とも主担当Claude Code、レビューはcodex exec経由のGPT-6 Luna。
- 直近の検証済み実装commit: `ff649a6`。全6038 passed/3 skipped/1既存warning（主担当実測、JUnit照合）、対象＋AST1036、fresh新CPU/品質/旧11・最終3golden成功。runtimeの状態なし関数で、全標本の検証と損失評価→標本順に学習標本追加・割当概念計数・損失統計更新を組み立て、実旧_absorb_into_storeと実旧確定処理の非採用3分岐（棄却/現行維持/再利用）へ照合した。
- レビュー運用の注意: (1)同specのTask1では、REDを実行する前に実装ファイルを書く手順逸脱があった。実装をstub/誤実装8種へ差し替えてtestの検出力を確かめる代替証拠でLuna承認を得た（integration-validation.md）。testを書いたら実装より前にREDを実行すること。差し替えscriptの形は`../../venv/refactoring-tests/assigned_training_sample_absorption_red_evidence.py`。(2)codex exec -m gpt-6-lunaで起動したレビュー担当は、自分のモデル名を内部から確認できないと回答し、それを理由に判定を保留/NO-GOにすることがある。主担当は起動時のmodel指定と実行ログを根拠に記録している。モデル自身による確認はできていない（ユーザーへ報告済みの未解消事項）。
- 全pytestの独立再現の基準は[共通引継ぎ手順](agent-handoff.md)の同名節（2026-10-07ユーザー決定）。Luna側sandboxでは全pytestを再現できないため、主担当実測＋JUnit照合で判定する。
- 旧実装の記録: LEGACY-015（標本吸収の途中失敗で先行標本の更新と不正標本が残る）も記録済み。直近3specで見つけたLEGACY-012（登録途中失敗の部分更新と採番消費）・013（使用済み一時IDへの再登録の黙った置換）・014（採用時の保留標本が割当概念計数と損失統計へ反映されない。意図した仕様か未確認、oracle診断への影響未確認）をimplementation-findingsへ記録済み。旧挙動は移植で維持している。
- 実装途中のtaskはない。新client・新全体runの接続は未完了。部品の旧実装対照と、新全体runのgolden一致は別の完了条件。
- `federated_drift_experiment/`は固定旧実装との対照・既存golden実行用。`src/federated_learning_experiments/`は移植中の新実装。旧固定基準は`748c3aa`、旧名alias/互換読込みを追加しない。

## 次の候補（未仕様化・未承認）

1. 前向き検証の確定処理の組立: 採用時の状態更新（adopt_candidate_as_current_training_model）と、非採用時の標本吸収（absorb_assigned_training_samples_into_held_model）は完了。次は旧_finalize_forward_validationの判断と分岐の組立を仕様化する。(a)収集済み損失から参照モデルの選択・再適合判定（旧select_forward_fitting_reference: 履歴平均と距離閾値、現行優先）・採否を決め、既存の採否判定（post-alarm-candidate-loss-evaluation）へ接続する。(b)結果に応じて採用/棄却/現行維持/別モデル再利用を選び、再利用では現在の学習帰属IDを切り替えてから吸収する。(c)判定record・切替位置・検出エピソード・適応イベントの記録と、現在ID変更の通知（最終構成では予測重み再始動）を上位へどう返すかを決める。tournament参照（shadow_tournament方針）の値反映/optimizer resetは最終構成（forward_persistent）では通らないため、移植要否を先に判断する。実旧_finalize_forward_validationは最小属性の実clientで全4分岐を実行できる（test_adopted_candidate_local_adoption.pyのbuild_local_adoption_oracle、test_assigned_training_sample_absorption.pyの非採用分岐test）。FIFOから1件ずつ確定する経路（旧fedsda.py 512-524行、統計→標本→概念の順のinline実装）は別に仕様化する。確認APIはモデル保有を前提にし、欠落時snapshotからの再構築は呼出し側server/clientへ残す（旧分岐未移植）。計算量診断の"initialization"/"statistics"記録は未移植。現在ID同値設定はno-op、新計数同ID移管は拒否（LEGACY011、通常経路への影響未確認）。評価fallback/EVAL_MAX_SAMPLESはサーバ評価接続時、実送信は通信specで扱う。
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
