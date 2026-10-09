# リファクタリングの再開案内

更新: 2026-10-10（fedsda-run-client-assemblyの完了。主担当はClaude Code）。これは案内であり、承認・進捗の正本は各specのspec.jsonとtasks.md。完了specの検証commit・件数・hash・レビューの経緯は、各specのintegration-validation.mdとreview.mdを読む（ここへ写さない）。

具体的な簡略化・効率化・局所的なアルゴリズム調整は[改善候補](../../docs/research/improvement-candidates/README.md)へ1候補1ファイルで記録する。広い研究アイデアは研究バックログ、不具合の疑いはimplementation-findings。同率現行優先IMPROVE-001（旧ALGO-001）は未検証/未採用で、旧保有順を維持する今回の移植に混ぜない。

Claude・Codexで交代する場合は[共通引継ぎ手順](agent-handoff.md)を参照する。
Claudeの入口は[CLAUDE.md](../../CLAUDE.md)。同じworktreeとspecを使う。レビュー担当の選択・起動・回数の正本は共通引継ぎ手順。

## 現在地

- **現在地:** [fedsda-run-client-assembly](../specs/fedsda-run-client-assembly/tasks.md)（clientの組立て）まで完了。進行中の編集・未解消レビュー指摘はない。
- **手順（2026-10-10から）:** cc-sddのskillと標準の構成、下位taskごとの実装、最後に独立レビュー1回、変異テストは任意。observed-sample-predictionとfedsda-run-client-assemblyが、この手順で進めたspecなので、文書の形（spec.json、requirements.md、design.md、research.md、naming.md、tasks.mdと末尾の「Implementation Notes」）は、これを参考にしてよい。observed-sample-processingまでのspecの文書（README、review.md、証拠文書、局所名まで載せた命名表、revisionとhash）は、古い手順の形なので、雛形にしない。新しいspecは、cc-sddのskill（`.claude/skills/kiro-spec-*`、`kiro-impl`、`kiro-validate-impl`）と`.kiro/settings/templates/specs/`の雛形から作る。
- **次の一手:** 下の「次の候補」1のうち、**初期モデルの事前学習**を1つのspecにする（`/kiro-spec-init`から）。旧は`federated_drift_experiment/experiment.py`の`_pretrain_initial_model`（294〜334行。概念0の標本を作り、shuffleしてbatchごとに学習し、損失統計を求める）と、`_setup_server_and_clients`（337〜358行）のclientの生成。新は、事前学習の結果を、`assemble_fedsda_run_client`が受け取る形（分類器、概念固有部と共有部の`ParameterOptimizerState`、`ModelAndClassLossStatistics`）で作る。乱数の消費順（標本の生成、shuffle、モデルの初期化）を旧と合わせる必要があるので、数値の一致が繊細なspecである。新の実行の枠は、`prepare_run`へSINEの標本生成器と乱数源を渡す（`runtime/single_run_execution.py`）。旧の標本の作り方（`generate_data`）と新の生成器の対応、既存の部品（`learning/loss_statistics/batch_loss_statistics_initialization.py`、共同更新）で足りるかを、最初に確かめる。その次が、サーバ同期と`prepare_run`（全clientとサーバの準備）。モデル別の除外寄与の診断は、小さいspecとして、どの時点でも進められる。新しい接続は、共用の`tests/refactoring/fresh_process_smoke.py`へ流れを足す。
- 作業場所: `.worktrees/refactoring/`、ブランチ: `refactor/architecture`。元checkout（`main`、HEAD `748c3aa`、`src/`なし）と取り違えない。`federated_drift_experiment/`は固定旧実装との対照・既存golden実行用。`src/federated_learning_experiments/`は移植中の新実装。旧名alias/互換読込みを追加しない。
- **実行環境の注意:** Windowsの基準環境は、スマートアプリコントロールがtorchの読込み（`venv/Lib/site-packages/torch/_C.cp313-win_amd64.pyd`）を断続的にブロックする（10月3日・5日・8日に発生し、いずれも時間をおいて解消）。発生したら保護設定・venv・goldenを変えず、WSL Ubuntu（`wsl -d Ubuntu`→リポジトリ直下で`source .venv/bin/activate`→worktreeへ移動。Python 3.14.4）で作業を続け、結果を「WSLで成功」と区別して記録する。WSLでは既知の3件が失敗する（Python 3.14の構文解析の違いによる既存test 1件、golden回帰2件の環境差による不一致）。specの完了は、Windows基準での全回帰が済むまで記録しない。
- **レビュー担当の利用状況（2026-10-09）:** GPT-6 Luna（`codex exec`）は利用上限に達しており、CLIの表示では2026-10-14 12:54まで使えない。それまでは、共通引継ぎ手順の代替規則によりClaude Haiku 5.5（effort `medium`）がLunaの担当分も行う。起動コマンドは共通引継ぎ手順の「承認と独立レビュー」の起動例。Lunaが戻ったら通常の分担へ戻す。

## 決定事項

- **Linux用のgolden（2026-10-08ユーザー）:** 旧実装をLinuxで実行した結果のgoldenは、新実装の全体runを接続するspecに入る前に作る。既存のWindows用goldenと回帰testは変更せず、別ファイル・別testにする。作る前に、同じ環境で2回実行して結果が一致することを確かめる。それまでの部品specでは、WSLのgolden回帰の不一致（環境差。旧実装は固定旧から無変更）を理由にgoldenを変えない。
- **検出episode（2026-10-09、主担当の判断）:** 制御（旧`DetectionEpisodeController`）と同一episodeの追加検出の経路は当面移植しない。最終構成の既定で無効。記録は[UNPORTED-001](../../docs/research/implementation-findings/unported-001-detection-episode-control.md)。ユーザーが必要と判断すれば覆せる。
- **回復期だけの混合予測（2026-10-10、主担当の判断）:** 有効化方針`drift_recovery`と、そのときの警報側の通知・重みの再生は当面移植しない。最終構成は常時有効で、通知は何も変えない。記録は[UNPORTED-003](../../docs/research/implementation-findings/unported-003-drift-recovery-prediction-mixture-activation.md)。ユーザーが必要と判断すれば覆せる。
- **移植しないと決めたもの（2026-10-07ユーザー）:** 参照も学習させる候補検証方針（旧`shadow_tournament`）。記録は[UNPORTED-002](../../docs/research/implementation-findings/unported-002-reference-shadow-tournament.md)。
- **依存testの方式の一本化は行わない（2026-10-09ユーザー）:** 既存の登録と注入契約testは移さず、消さない。新しいmoduleにはsymbolごとの注入契約testを足さない。記録は[IMPROVE-009](../../docs/research/improvement-candidates/improve-009-unify-dependency-boundary-tests.md)。
- **手順（2026-10-09〜10ユーザー。正本は[共通引継ぎ手順](agent-handoff.md)。ここへ内容を写さない）:** 2026-10-10に、時間の実測をもとに軽くし、cc-sdd（kiro）のskillと標準の構成へ戻した。要点: kiroのskillで要求・設計・tasksを作り、下位taskを1つずつ実装し（下位taskごとのレビューはしない）、全taskの後に独立レビューを1回受ける。文書は標準の4つとnaming.md（公開する名前だけ）。review.md・証拠文書・READMEは作らない。変異テストは既定では実行しない。子エージェントは使わない。specは旧のひとまとまりの流れの大きさにする。fresh processは共用script。
- **学習の途中失敗のときの計数（2026-10-09ユーザー）:** 1回の共同更新の途中で例外が出た場合の学習量の計数は旧と違うが、現在の流れでは読まれないので、そのままにする。記録は[NEW-003](../../docs/research/implementation-findings/new-003-training-count-after-mid-update-failure.md)。
- 具体的な簡略化・効率化・局所的なアルゴリズム調整は[改善候補](../../docs/research/improvement-candidates/README.md)、不具合の疑い・移植しない判断・新実装で見つけた事項は[implementation-findings](../../docs/research/implementation-findings/README.md)へ、気づいた時点で1件1ファイルで記録する（2026-10-09ユーザー指示）。研究アルゴリズムの変更は採用していない。

## 完了specの一覧（新しい順。詳細は各spec）

clientの組立てと実行の枠の5操作（`FedsdaRunClient`）、標本1件の予測と学習側の処理（`process_observed_sample`）、警報後の「参照の固定→標本の観測→損失収集→採否評価→確定→正式ID確認」、警報時の「区間の準備→再利用評価→切替または候補検証の開始→完了処理→記録」、候補検証sessionの保持、診断証拠と通知までの部品が、実旧の対応するメソッドとの対照つきで揃っている。初期モデルの事前学習・サーバ側・新全体runの接続は未完了。部品の旧実装対照と、新全体runのgolden一致は別の完了条件。

- [fedsda-run-client-assembly](../specs/fedsda-run-client-assembly/tasks.md): 最終構成のFedSDAのclient 1つを、初期モデル（分類器と2つのoptimizerの状態）・初期の損失統計・設定の束から組み立てる`assemble_fedsda_run_client`と、実行の枠の契約の5操作（標本の処理、ラウンド境界での保留中の学習、登録できるモデルの有無、送信待ちの進行、終端での未完了の候補検証の回収）を持つ`FedsdaRunClient`。実`__init__`で作った実旧のclient（サーバなし）と、生成直後から終端まで照合した。既存の実行の枠（参加者の検査と区間の進行）で、サーバを何もしない代役にして動く。
- [observed-sample-prediction](../specs/observed-sample-prediction/tasks.md): 観測した標本1件の予測`predict_observed_sample_and_update_prediction_weights`（保有する全モデルの確率をFixed-Shareの重みで結合→標本ごとの記録→ラベル観測後のFixed-Shareの重みと診断証拠の更新）と、記録を持つ`SamplePredictionRecordStore`。`process_observed_sample`の最初の段として接続した。警報側の通知は、最終構成では何も変えないので移植していない（UNPORTED-003）。
- [observed-sample-processing](../specs/observed-sample-processing/README.md): 観測した標本1件に対する学習側の処理（候補検証の進行→損失の監視→保留→警報の処理、または帰属の確定と学習）をつなぐ`process_observed_sample`。保留中の標本そのものを持つ`PendingSampleObservationStore`と、警報の位置と監視の値を記録する`LossChangeAlarmRecordStore`を追加。
- [held-model-training-request-handling](../specs/held-model-training-request-handling/README.md): 学習要求を記録して間隔に達したら学習する`record_training_request_and_train_held_models_when_due`（旧`train_step`）と、保留中の要求を学習する`train_held_models_for_pending_training_requests`（旧`flush_pending_updates`）。共同更新の完了ごとにモデル別の学習量を計数へ足す。件数管理へ、間隔に達したかを読む操作を追加。呼ぶ位置は未接続。
- [released-pending-sample-assignment](../specs/released-pending-sample-assignment/README.md): 警報のない標本で、保留の容量を超えた最古の標本を既存の吸収で現在のモデルへ確定し、保留から解放する`assign_released_pending_samples_to_current_training_model`。保留位置のownerへ、容量を超える位置を解放せずに読む操作を追加。
- [shared-verification-infrastructure](../specs/shared-verification-infrastructure/README.md): Git管理下の共用fresh process scriptとそれを別processで実行するtest、依存の許可集合が広すぎないことの検査、汎用の変異toolで見つかったtestの穴2件の修正。sourceの変更なし。
- [held-candidate-validation-diagnostic-notification](../specs/held-candidate-validation-diagnostic-notification/README.md): 候補検証の進行（`advance_held_candidate_validation`）が、確定時に帰属変更を診断へ通知する。進行の関数の引数が増えたので、これより前のspecの個別fresh CPU script（Git管理外）は現在のsourceでは動かない。
- [alarm-occurrence-handling](../specs/alarm-occurrence-handling/README.md): 警報1回ぶんの処理（応答→完了処理→適応記録→session保持→診断通知）を`handle_alarm_occurrence`でつなぐ。
- [held-adahedge-diagnostic-notification](../specs/held-adahedge-diagnostic-notification/README.md)（主担当Codex）: globalと真の概念別oracleの診断証拠の保持と、帰属変更通知（globalだけ再始動）。
- [adahedge-diagnostic-evidence](../specs/adahedge-diagnostic-evidence/README.md)（主担当Codex）: 単一のAdaHedge診断証拠。
- [held-candidate-validation-progress](../specs/held-candidate-validation-progress/README.md): 候補検証sessionを1つ保持するownerと、警報応答の反映・標本ごとの進行・終端回収。
- [candidate-validation-adaptation-recording](../specs/candidate-validation-adaptation-recording/README.md): 候補検証の到達時の確定と未完了の終端回収の適応記録。
- [alarm-adaptation-recording](../specs/alarm-adaptation-recording/README.md)（主担当Codex）: 適応記録・切替位置・再利用件数のowner（`AdaptationRecordStore`）と、警報応答の記録。
- [alarm-response-completion](../specs/alarm-response-completion/README.md): 警報応答の後の、損失監視の再開と保留位置の消費。
- [alarm-buffer-response](../specs/alarm-buffer-response/README.md)、[alarm-training-interval-preparation](../specs/alarm-training-interval-preparation/README.md)、[alarm-change-interval-resolution](../specs/alarm-change-interval-resolution/README.md)、[alarm-interval-model-reuse-assessment](../specs/alarm-interval-model-reuse-assessment/README.md): 警報時の保留標本への応答（区間の準備、再利用評価、区間解決）。
- 2026-10-07の7 spec: [post-alarm-reference-model-fixation](../specs/post-alarm-reference-model-fixation/README.md)、[post-alarm-candidate-validation-sample-observation](../specs/post-alarm-candidate-validation-sample-observation/README.md)、[post-alarm-candidate-validation-resolution](../specs/post-alarm-candidate-validation-resolution/README.md)、[assigned-training-sample-absorption](../specs/assigned-training-sample-absorption/README.md)、[adopted-candidate-local-adoption](../specs/adopted-candidate-local-adoption/README.md)、[temporary-model-id-allocation](../specs/temporary-model-id-allocation/README.md)、[adopted-candidate-initial-local-registration](../specs/adopted-candidate-initial-local-registration/README.md)。
- それ以前（設定、単一runの実行順序、予測重み、損失監視、学習、候補の学習ほか）は[roadmap](roadmap.md)。

進め方（実装前のtest、実旧メソッドを使うoracle、検出力の確認、検証コマンド、レビュー依頼の注意）と、機械的な照合（`.kiro/settings/scripts/spec_checks.py`の`names`・`identity`・`progress`、`mutation_check.py`）は[共通引継ぎ手順](agent-handoff.md)。

## ユーザー確認待ち・未解消の事項

- LEGACY-014: 旧の採用分岐は、新モデルへ移す保留標本を割当概念計数と損失統計へ反映しない（他の分岐と非対称）。2026-10-08ユーザー回答: LEGACYは後でまとめて判断するので、現状のまま置く。新実装は旧挙動を維持しtestで固定している。主担当からの再確認は不要。記録は[implementation-findings](../../docs/research/implementation-findings/README.md)。
- レビュー担当の同定: `codex exec -m gpt-6-luna`で起動したレビュー担当は、自分のモデル名を内部から確認できないと回答する。記録は起動時のmodel指定と実行ログのmodel行に依拠している。実際のルーティングの確認はユーザーへ依頼済み。
- 全pytestの独立再現: 2026-10-07のユーザー決定により、主担当の実測とJUnit照合で判定する（[共通引継ぎ手順](agent-handoff.md)の同名節）。レビュー担当のsandboxで再現できるようになった場合は、再現した事実を記録してよい。
- 旧実装の記録（この期間の追加）: LEGACY-012（登録途中失敗の部分更新と採番消費）、013（使用済み一時IDへの再登録の黙った置換）、014（上記）、015（標本吸収の途中失敗で先行標本の更新と不正標本が残る）。012/013/015は契約外入力だけで、新実装は状態変更前に拒否する。

## 次の候補（未仕様化・未承認）

### 1. 事前学習・サーバ同期と、残る診断（未仕様化）

clientは、`assemble_fedsda_run_client`で組み立て、実行の枠の5操作で動かせるようになった。残るのは次のとおり。旧のひとまとまりの流れの単位でspecにし、境界を決めて下位taskへ分ける（数値の一致が繊細な部分が出てきたら、そこだけ別のspecへ切り出す）。

- モデル別の除外寄与の診断: 旧`federated_drift_experiment/diagnostics/routing.py`の`RoutingLeaveOneOutDiagnostics.observe`（予測の直後に、モデルを1つ除いた結合の損失の差を、モデル集合の世代・通信区間・モデルごとに集計する。予測と学習の判断へは戻らない）。入力（モデル別の確率、正規化後のFixed-Shareの重み、globalの診断重み）は`ObservedSamplePrediction`が持つので、`process_observed_sample`の予測の段の直後に呼出しを足せる。通信区間の長さ（集約間隔）が要る。最終3goldenはこの診断を比較しない。IMPROVE-007は診断の選択実行案で未検証/未採用。
- 予測の集計値: 旧の集計の計数（`routing_diagnostics`ほか）は、`SamplePredictionRecord`の列から導ける（対応はobserved-sample-predictionのresearch.md。testが全項目を実旧と照合している）。実効モデル数の合計は、逐次加算で求める（Python 3.13の組み込みの`sum`は浮動小数を補正つきで足すので、旧の標本ごとの加算と末尾の桁が違う）。集約後のFixed-Shareの重みの再較正（旧`fifo_replay`）は、サーバ同期のspecで扱う。
- 初期モデルの事前学習: 上の「次の一手」。
- サーバ同期と、全clientとサーバの準備（実行の枠の`prepare_run`を持つfactory）: モデルの回収・集約・配布、正式IDの確認（`held-model-registration-confirmation`は部品として移植済み）、クラスタリングと統合、集約後のFixed-Shareの重みの再較正（旧`fifo_replay`）、サーバ評価（評価fallbackと`EVAL_MAX_SAMPLES`）、通信量の記録。旧は、送信保留のモデルのIDを持たず、正式IDの確認のときに現行モデルのIDを一時IDとして使う（採用の後に現行モデルが別のモデルへ戻ると、送信保留のモデルと現行モデルが違う。新の送信保留はモデルIDを持つ）。この食い違いの扱いを、サーバ同期のspecで確かめる。
- 真の概念IDの配線: 実行の枠の契約は、clientへ真の概念を渡さない（clientの準備の後に真の概念の列を作る）。`FedsdaRunClient.process_observed_sample`は、任意の引数`evaluation_concept_id`で受け取れる。渡さなければ、概念別の診断と、モデル別の割当概念の計数が行われない。実行の枠から渡すかどうか（契約を変えるか）は、全体runのgoldenの比較項目を見て、全体runを接続するspecで決める。
- 設定の置き場所: clientの設定の束（`runtime/fedsda_run_client_settings.py`）は、機能別の設定型に置き場所のない値（許容する増加量、最小改善量、待ちラウンド数、最小件数、batchの件数、評価標本の件数、検出器の候補数の上限・賭け率・表示名）と、検証済みのrun設定の部分型に入っていない設定型（学習の間隔、候補の初期化、候補の学習、optimizer）をまとめた仮の形である。旧で1つの値を2箇所で使う3組は、同じ値であることを束が確かめる。完全なrun設定（保存表現、preset）を決めるspecで、正式な置き場所を決める。
- 標本ごとの結果種別の列（旧`history_drift_type`）と保存: 適応記録（位置と結果種別）から導けると見込むが、旧の数値との対応は確かめていない。
- 計算量と所要時間の記録（旧`_record_model_compute`・`compute_counters`・`phase_seconds`・`processing_times`）と、共有部の勾配の診断（旧`backbone_gradient_diagnostics`）は未移植。
- 記録のoracle: tests/refactoring/test_fedsda_run_client.py（`build_run_client_oracle`。実旧の事前学習と実`__init__`で実旧clientを作り、新clientを同じ初期モデルから組み立てる。`assert_run_client_matches_legacy`が全ownerを照合する）と、tests/refactoring/test_observed_sample_processing.py（`build_sample_processing_oracle`）を再利用する。
- 保留標本の並びの検査が、警報の処理・帰属の確定・標本1件の処理の3箇所にある。標本の型と形の検査も、標本1件の処理と予測の2箇所にある（IMPROVE-010）。
- 全体runを接続するspecの前に、Linux用のgoldenを作る（上の「決定」）。部品の旧対照と全体runのgolden一致を混同しない。

### 2. その先

- 判定記録の一覧（旧`provisional_model_decisions`）の保持と保存。
- client進行とサーバ同期・ID対応。

### 未移植として残している細目

- 検出episodeの制御（旧`DetectionEpisodeController`、`_resolve_episode_duplicate`）。最終構成で無効のため当面移植しない（上の「検出episodeの判断」）。
- 正式ID確認は、モデルを保有している前提。欠落時に送信保留のsnapshotから再構築する旧分岐は未移植で、呼出し側のserver/clientで扱う。
- 計算量診断（旧`_record_model_compute`と`compute_counters`）は未移植。
- 現在IDへ同じIDを設定するとno-op、学習計数の同ID移管は拒否（LEGACY-011）。
- 評価fallbackと`EVAL_MAX_SAMPLES`はサーバ評価の接続時、実送信は通信のspecで扱う。

この順序は候補。再開時にコードと完了specを照合し、未移植の依存があれば先に仕様化する。既存の完了タスクを無条件に再実装しない。

新構成の機能名はproduct.md/configuration-foundation/naming.mdを参照。Residual Adapterの語は構造として維持し、部品はNonlinearResidualAdapter、構成はshared_backbone_residual_adapter。旧Switchingの新構成名はfixed_share_weighted_prediction。旧称は過去実験との対応説明で併記する。

## 最初に読む文書と手順

1. worktreeの[AGENTS.md](../../AGENTS.md)と[方針の正本](../../docs/research/refactoring-policy.md)。
2. [roadmap](roadmap.md)の「現在の状態」、[product](product.md)、[tech](tech.md)、[structure](structure.md)。
3. 対象specのspec.json→tasks.md→requirements.md／design.md／naming.md（observed-sample-processingまでのspecは、README→spec.json/tasks.md→requirements/design/naming→review/integration-validation）。未完了があれば次のspecより先に扱う。
4. `git status --short`とブランチを確認し、別タスクの差分・未追跡資料を特定する。文書の案内と実際の状態が違う場合はGitとspecの正本を確認する。
5. kiroのskillで要求→設計（命名表）→tasks→下位taskごとの実装→独立レビュー1回→指摘の反映→全pytest、の順に進む。手順は共通引継ぎ手順に従う。

## 別タスクと記録の扱い

HTMLは元checkoutの`docs/overview/fedsda-processing-flow.html`にあるコミット保留資料。worktreeへ自動コピー・stageしない。他の保留2資料もAGENTS.mdに従う。
旧実装の不具合は[implementation-findings](../../docs/research/implementation-findings/README.md)、開発手順の問題は`development-findings/`に記録する。
中断時は対象specへ未完了task/検証/差分を記録し、この案内の現在地を更新する。別タスクの結果をリファクタリングの完了証拠に混ぜない。
