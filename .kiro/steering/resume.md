# リファクタリングの再開案内

更新: 2026-10-08（警報応答の完了処理を完了）。これは案内であり、承認・進捗の正本は各specのspec.jsonとtasks.md。

**引継ぎ地点（2026-10-08）:** 主担当Claude Codeが[alarm-response-completion](../specs/alarm-response-completion/README.md)を完了した。要求r3・設計r5・命名r5・tasks r3、全3taskを独立承認（Task 1はHaiku 5.5、Task 2・3はLuna）、別fresh Luna feature最終GO。完了した警報応答に続けて、損失監視のresetと保留位置のdrain（不足ではdrainしない）を行い、記録用の不変recordを返す。全9254 passed、旧11/最終3golden・品質成功、33変異検出。直前は[alarm-buffer-response](../specs/alarm-buffer-response/README.md)。次は下の「次の候補」1。進行中の編集・レビューはない。

具体的な簡略化・効率化・局所的なアルゴリズム調整は[改善候補](../../docs/research/improvement-candidates/README.md)へ1候補1ファイルで記録する。広い研究アイデアは研究バックログ、不具合の疑いはimplementation-findings。同率現行優先IMPROVE-001（旧ALGO-001）は未検証/未採用で、旧保有順を維持する今回の移植に混ぜない。

Claude・Codexで交代する場合は[共通引継ぎ手順](agent-handoff.md)を参照する。
Claudeの入口は[CLAUDE.md](../../CLAUDE.md)。同じworktreeとspecを使う。2026-10-08の新方針は日常的・分量の多いレビューにLuna、やや複雑・難易度の高い実装レビューにHaiku 5.5を優先し、利用不能時は他方へ代替する。両方ともeffortは`high`を明示指定する。起動・判断の正本は共通引継ぎ手順。

## 現在地

- 過去の完了: [警報の変化区間の解決](../specs/alarm-change-interval-resolution/README.md)。要求r3・設計r3・命名r5・tasks r2、全5task独立レビュー承認、別fresh Luna feature最終GO、completed。runtimeの`resolve_alarm_change_interval`と不変record`AlarmChangeIntervalResolution`（結果種別は`alarm_interval_held_model_reused`/`alarm_interval_current_model_maintained`/`alarm_interval_candidate_validation_started`）。切出し済みの変化区間の標本（1件ずつ）を連結して既存の区間評価へ渡し、選択IDがあればそのモデルへ吸収して学習帰属を切り替え（同じIDなら維持）、なければ全保有モデルのsnapshot→既存の初期値選択→既存のsession開始を行う。実旧_resolve_driftを吸収・帰属切替・初期値選択・session開始を差し替えずに実行して最終状態を照合した。要求〜Task 2のレビューはCodexの利用上限中のため独立AgentのSonnetが代替し、Task 3以降はLuna。その前は[警報区間の保有モデル再利用評価](../specs/alarm-interval-model-reuse-assessment/README.md)（区間評価と再利用候補の選択。読取り専用）。
- 公開状態（2026-10-08）: GitHub復旧後の通常pushを1回実行し、未送信7commit（`5ddb9e3..db89dc4`）の送信に成功した。今後はユーザー指示に従いtaskごとにpushを1回だけ試す。失敗時は連続再試行や原因探索をせず、次taskのpush成功時に未送信commitも送る。
- 直近のpush状態: 全commitは`origin/refactor/architecture`へ送信済み（taskごとの通常pushはすべて成功）。
- 最新完了: [alarm-response-completion](../specs/alarm-response-completion/README.md)。runtimeの`complete_alarm_buffer_response`と不変record`AlarmResponseCompletion`（元の応答、警報位置、変更前後の学習帰属ID、推定変化点、episode ID、resetに使った基準平均、消費した保留位置。property `training_model_switch_sample_index`・`detection_episode_operation_required`）。詳細は同specのintegration-validation.md/spec.json/review.md。Task 1は独立レビューで4回差し戻され、検査をすべて更新の前に置く設計（recordを先に組み立てる、変更記録のexact型検査、未観測FIFOの拒否）へ改訂した。その前はalarm-buffer-response、alarm-training-interval-preparation。改善候補IMPROVE-004/005/006は未検証/未採用。研究アルゴリズムの変更は採用していない。
- 作業場所: `.worktrees/refactoring/`、ブランチ: `refactor/architecture`。元checkout（`main`、HEAD `748c3aa`、`src/`なし）と取り違えない。
- 作業状態: [候補検証session開始](../specs/post-alarm-candidate-validation-session-start/README.md)の全5tasksはLuna承認・完了。要求r2・設計r2・命名r4・tasks r1を維持し、別fresh GPT-6 Lunaのfeature最終GO、completed。候補生成とエポック学習もcompleted。承認・進捗は各specのspec.json/tasks.mdが正本。
- 直近の検証済みsource/test commit: `def37e4`。全pytest 9254 passed/3 skipped/2 warnings、155.63s（主担当実測、JUnit 9257件照合）、対象57＋AST2542、33変異検出、fresh新CPU 2/4class×5経路の10条件をTask 2・3のLunaも再現。Ruff 167files/Pyright/pip check成功、旧11・最終3golden成功。固定旧748c3aaから旧実装・golden・旧回帰test・toolsへのdiffは空。source hashは267パス、`b11a144ab4312c63f4dfd84492c5dc81c9ab11f7264caca9de5e32453fd9db2b`。以後は証拠/進捗文書だけを変更。
- 2026-10-07のClaude Code担当分（7spec、全てfeature最終GO・completed）。新しい順:
  1. [警報時点の参照モデルの固定](../specs/post-alarm-reference-model-fixation/README.md): 保有モデルと同じ値の独立した参照分類器と履歴平均損失。torch乱数の消費を実旧と一致させた。
  2. [警報後の候補検証標本の観測](../specs/post-alarm-candidate-validation-sample-observation/README.md): 標本1件の候補・参照の損失評価と損失収集への追加。
  3. [警報後の候補検証の確定](../specs/post-alarm-candidate-validation-resolution/README.md): 評価結果から採用/再利用/維持/棄却を選んで適用。
  4. [帰属確定標本の吸収](../specs/assigned-training-sample-absorption/README.md): 標本追加・割当概念計数・損失評価・統計更新。
  5. [採用候補のローカル採用](../specs/adopted-candidate-local-adoption/README.md): 採番→登録→計数→標本追加→現在の学習帰属ID切替え。
  6. [一時モデルIDの採番](../specs/temporary-model-id-allocation/README.md)。
  7. [採用候補の初期ローカル登録](../specs/adopted-candidate-initial-local-registration/README.md): 共有反映・初期統計・snapshot・一覧/統計/送信保留への登録。
- これで、警報後の「参照の固定→標本の観測→損失収集→採否評価→確定（採用/再利用/維持/棄却）→正式ID確認」までの部品が、実旧の対応するメソッドとの対照つきで揃った。新client・新全体runの接続は未完了。部品の旧実装対照と、新全体runのgolden一致は別の完了条件。
- `federated_drift_experiment/`は固定旧実装との対照・既存golden実行用。`src/federated_learning_experiments/`は移植中の新実装。旧名alias/互換読込みを追加しない。
- この期間に確立した進め方（実装前のRED、実装差し替えによる検出力の確認、実旧メソッドを使うoracle、検証コマンド、レビュー依頼の注意）は[共通引継ぎ手順](agent-handoff.md)の「2026-10-07に確立した運用」を参照する。

## ユーザー確認待ち・未解消の事項

- LEGACY-014: 旧の採用分岐は、新モデルへ移す保留標本を割当概念計数と損失統計へ反映しない（他の分岐と非対称）。2026-10-08ユーザー回答: LEGACYは後でまとめて判断するので、現状のまま置く。新実装は旧挙動を維持しtestで固定している。主担当からの再確認は不要。記録は[implementation-findings](../../docs/research/implementation-findings/README.md)。
- レビュー担当の同定: `codex exec -m gpt-6-luna`で起動したレビュー担当は、自分のモデル名を内部から確認できないと回答する。記録は起動時のmodel指定と実行ログのmodel行に依拠している。実際のルーティングの確認はユーザーへ依頼済み。
- 全pytestの独立再現: 2026-10-07のユーザー決定により、主担当の実測とJUnit照合で判定する（[共通引継ぎ手順](agent-handoff.md)の同名節）。レビュー担当のsandboxで再現できるようになった場合は、再現した事実を記録してよい。
- 移植しないと決めたもの: 参照も学習させる方針（旧`shadow_tournament`）。最終構成（`forward_persistent`）で通らず、goldenにも値がない。2026-10-07のユーザー判断で当面不要。
- 旧実装の記録（この期間の追加）: LEGACY-012（登録途中失敗の部分更新と採番消費）、013（使用済み一時IDへの再登録の黙った置換）、014（上記）、015（標本吸収の途中失敗で先行標本の更新と不正標本が残る）。012/013/015は契約外入力だけで、新実装は状態変更前に拒否する。

## 次の候補（未仕様化・未承認）

### 1. 適応記録・通知・候補検証sessionの保持（次に着手）

旧`FedSDAClient._resolve_drift`は、応答（`runtime/alarm_buffer_response.py::respond_to_alarm_with_buffered_samples`）と、その後の検出器reset・FIFO clear（`runtime/alarm_response_completion.py::complete_alarm_buffer_response`）まで部品になった。残りは、完了情報を受けて呼出側の状態を更新する部分。新実装にownerがまだないものが複数あり、候補検証の到達時（`PostAlarmCandidateValidationCompletion`）・未完了回収（`IncompletePostAlarmCandidateValidationFinalization`）・警報応答（`AlarmResponseCompletion`）の3つの完了情報を同じownerが受ける。1つのspecに収まらなければ分ける。要求・設計・命名・tasksはまだ作っていない。

- 適応イベントの一覧: 旧`AdaptationEvent`（position、detector、action、old/new model、estimated_change_point、episode_id）と`adaptation_events`。警報応答の結果5値と旧actionの対応はalarm-response-completion/design.md 3節。検出器名は呼出側が持つ。新しいevent名・型・結果値は命名レビューを通す。
- 切替位置の一覧（旧`local_switch_positions`）と再利用計数（旧`reuse_selection_counts`の`alternative_fit`/`current_fit`）。前者は各完了情報の`training_model_switch_sample_index`、後者は警報応答の結果値から決まる。旧の利用箇所は`federated_drift_experiment/metrics.py`と`experiment.py`。
- 検出episode（旧`DetectionEpisodeController`、`federated_drift_experiment/detection_episode.py`）: 最終構成の既定は無効（`FEDSDA_DETECTION_EPISODES_ENABLED=False`）。完了情報の`detection_episode_operation_required`が`mark_operation`に対応する。同一episode内の追加検出（旧`_resolve_episode_duplicate`）は未移植。移植の要否は最終構成とgoldenの使用有無を確認して決める。
- 学習帰属変更の通知（旧`_on_local_model_change`）: 最終構成ではAdaHedge系の`restart_for_concept`を呼ぶ。新`FixedSharePredictionWeightController`に何が必要かは未調査（旧のSwitching側routerが切替時に何をするかを先に確認する）。旧は切替の直後（吸収より前）に通知するが、通知先は標本・統計を読まない（alarm-change-interval-resolution/design.md）。
- 候補検証sessionの保持と解除: 応答の`active_validation_session`（候補検証中は同じ参照、開始時は新session、その他None）を誰が持ち、到達時・未完了回収で誰が外すか。
- 呼出側の保証として残っている事項（alarm-response-completion/research.md）: 応答を一度だけ完了させる、応答と完了の間に標本を観測しない。組立側のtestで確かめる。
- 許容損失増加量と最小変化区間件数の設定登録は組立側に残る。
- oracle: `tests/refactoring/test_alarm_response_completion.py`の`build_response_completion_oracle`と`run_legacy_alarm_with_real_completion`が、実旧`_resolve_drift`をイベント記録・検出器reset・FIFO clearを差し替えずに実行する（差し替えは推定区間長の供給だけ）。実旧の`adaptation_events`・`local_switch_positions`・`reuse_selection_counts`がそのまま照合に使える。`observe_monitored_losses_in_both_implementations`は新監視と実旧ClassESRへ同じ損失列を与える。
- 通常進行と終端回収は完了済み。呼出側session解除・一覧記録・通知は未実装。

### 2. その先

- 完了情報を呼出側へ接続し、session解除・判定/適応event一覧・切替位置・検出episode操作・学習帰属変更の通知（予測重みの再始動）を組み立てる。
- FIFOから1件ずつ帰属を確定する経路（旧`fedsda.py`の標本処理内。統計→標本→概念の順のinline実装で、吸収の部品とは更新順が違う）。
- 警報後の帰属変更とclient進行、サーバ同期・ID対応。

### 未移植として残している細目

- 正式ID確認は、モデルを保有している前提。欠落時に送信保留のsnapshotから再構築する旧分岐は未移植で、呼出し側のserver/clientで扱う。
- 計算量診断（旧`_record_model_compute`と`compute_counters`）は未移植。
- 現在IDへ同じIDを設定するとno-op、学習計数の同ID移管は拒否（LEGACY-011）。
- 評価fallbackと`EVAL_MAX_SAMPLES`はサーバ評価の接続時、実送信は通信のspecで扱う。

この順序は候補。再開時にコードと完了specを照合し、未移植の依存があれば先に仕様化する。既存の完了タスクを無条件に再実装しない。

新構成の機能名はproduct.md/configuration-foundation/naming.mdを参照。Residual Adapterの語は構造として維持し、部品はNonlinearResidualAdapter、構成はshared_backbone_residual_adapter。旧Switchingの新構成名はfixed_share_weighted_prediction。旧称は過去実験との対応説明で併記する。

## 最初に読む文書と手順

1. worktreeの[AGENTS.md](../../AGENTS.md)と[方針の正本](../../docs/research/refactoring-policy.md)。
2. [roadmap](roadmap.md)の「現在の状態」、[product](product.md)、[tech](tech.md)、[structure](structure.md)。
3. 対象specのREADME→spec.json/tasks.md→requirements/design/naming→review/integration-validation。未完了taskがあれば次のspecより先に扱う。
4. `git status --short`とブランチを確認し、別タスクの差分・未追跡資料を特定する。文書の案内と実際の状態が違う場合はGitとspecの正本を確認する。
5. cc-sddの要求→設計/命名→tasks→実装→統合検証へ進む。各段階は共通のLuna/Haiku 5.5選択方針に従った独立レビューと有用な指摘の反映で承認し、選択理由・モデル/effort・代替理由・採否・内容hashを記録する。命名承認前に新srcや先取りtestを作らない。

## 別タスクと記録の扱い

HTMLは元checkoutの`docs/overview/fedsda-processing-flow.html`にあるコミット保留資料。worktreeへ自動コピー・stageしない。他の保留2資料もAGENTS.mdに従う。
旧実装の不具合は[implementation-findings](../../docs/research/implementation-findings/README.md)、開発手順の問題は`development-findings/`に記録する。
中断時は対象specへ未完了task/検証/差分を記録し、この案内の現在地を更新する。別タスクの結果をリファクタリングの完了証拠に混ぜない。
