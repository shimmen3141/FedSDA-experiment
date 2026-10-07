# リファクタリングの再開案内

更新: 2026-10-08（警報区間の保有モデル再利用評価spec完了）。これは案内であり、承認・進捗の正本は各specのspec.jsonとtasks.md。

**引継ぎ地点（2026-10-08）:** 主担当Claude Code。[警報区間の保有モデル再利用評価](../specs/alarm-interval-model-reuse-assessment/README.md)は完了。続く[警報の変化区間の解決](../specs/alarm-change-interval-resolution/README.md)をTask 3のcommitまで進め、ユーザー指示で停止した。未コミット差分はない。次の一手は「現在地」の着手中の項を参照。

具体的な簡略化・効率化・局所的なアルゴリズム調整は[改善候補](../../docs/research/improvement-candidates/README.md)へ1候補1ファイルで記録する。広い研究アイデアは研究バックログ、不具合の疑いはimplementation-findings。同率現行優先IMPROVE-001（旧ALGO-001）は未検証/未採用で、旧保有順を維持する今回の移植に混ぜない。

Claude・Codexで交代する場合は[共通引継ぎ手順](agent-handoff.md)を参照する。
Claudeの入口は[CLAUDE.md](../../CLAUDE.md)。同じworktreeとspecを使い、GPT-6 Lunaを優先し、利用不能時はSonnetの独立レビューで承認する。

## 現在地

- 直近完了: [警報区間の保有モデル再利用評価](../specs/alarm-interval-model-reuse-assessment/README.md)。要求r2・設計r2・命名r4・tasks r1、全5task独立Luna承認、別fresh Luna feature最終GO、completed。methodsの純粋判定`assess_alarm_interval_model_reuse`と不変record`AlarmIntervalModelReuseAssessment`、runtimeの`evaluate_held_models_for_alarm_interval_reuse`。保有順に全モデルを区間評価し、履歴基準（2件以上・非零平均）を使えるモデルだけを評価列へ含め、区間平均−履歴平均が許容増加量以下のモデルを適合とし、平均最小・同率は保有順で選ぶ（現行優先ではない）。実旧_resolve_driftの評価済み候補列・適合列・選択と照合し、適合なしの場合は既存の初期値選択→session開始→観測までtest内で接続して実旧と照合した。その前は[終端の未完了候補検証の確定](../specs/incomplete-post-alarm-candidate-validation-finalization/README.md)。
- 公開状態（2026-10-08）: GitHub復旧後の通常pushを1回実行し、未送信7commit（`5ddb9e3..db89dc4`）の送信に成功した。今後はユーザー指示に従いtaskごとにpushを1回だけ試す。失敗時は連続再試行や原因探索をせず、次taskのpush成功時に未送信commitも送る。
- 直近のpush状態: 全commitは`origin/refactor/architecture`へ送信済み（taskごとの通常pushはすべて成功）。
- 着手中: [警報の変化区間の解決](../specs/alarm-change-interval-resolution/README.md)（主担当Claude Code）。2026-10-08、ユーザー指示でTask 3のcommit後に停止。要求r3・設計r3・命名r5・tasks r2は承認済み。Task 1（結果record）とTask 2（再利用・維持・候補検証開始の組立`resolve_alarm_change_interval`）は独立レビュー承認済み。Task 3（呼出し順・解決後の学習継続のtest）はcommit済みで独立レビュー未依頼。次の一手は、Task 3の独立レビュー→Task 4（fresh新CPUのsmoke作成と実行、guardとsourceのimportの最終照合）→Task 5（全pytestと証拠の記録）→別freshのfeature最終レビュー。このspecでは全pytestをまだ実行していない（対象241＋依存境界は成功）。レビューはLunaの利用上限（5:08まで）のため独立AgentのSonnetが代替した。再開時にLunaが使えるならLunaへ依頼する。経緯は同specのreview.md、進捗の正本はtasks.mdとspec.json。下の「次の候補」1の(b)(c)にあたり、(a)(d)は後続。
- 作業場所: `.worktrees/refactoring/`、ブランチ: `refactor/architecture`。元checkout（`main`、HEAD `748c3aa`、`src/`なし）と取り違えない。
- 作業状態: [候補検証session開始](../specs/post-alarm-candidate-validation-session-start/README.md)の全5tasksはLuna承認・完了。要求r2・設計r2・命名r4・tasks r1を維持し、別fresh GPT-6 Lunaのfeature最終GO、completed。候補生成とエポック学習もcompleted。承認・進捗は各specのspec.json/tasks.mdが正本。
- 直近の検証済み実装commit: `5785533`。全pytest 7924 passed/3 skipped/2 warnings（主担当実測、JUnit 7927件照合）、対象154＋AST1726、fresh新CPU 2/4class、Ruff 157files/Pyright/pip check成功、旧11・最終3golden成功、固定旧基準`748c3aa`から旧実装・golden・旧回帰test・tools/への差分は空。source hashは257パス。警告2件とskip 3件は前specと同じ既存のもの。証拠は同specのintegration-validation.md。
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

### 1. 警報処理の組立（次に着手）

旧`FedSDAClient._resolve_drift`（`federated_drift_experiment/clients/fedsda.py`）のうち、区間の評価と再利用候補の選択、初期値選択、session開始、進行、終端回収は部品として完了した。残りは、これらを順に呼ぶ組立と、その前後の状態更新。1つのspecに収まらなければ分ける。

- (a) 区間の切出しと前区間の処理: FIFOを推定変化点で前区間と変化区間へ分ける（`PendingTrainingAssignmentBuffer`のpartitionは実装済み）。前区間は評価標本の保存と現行IDへの吸収。変化区間が最小件数（旧`MIN_DRIFT_DATA`）未満なら何もしない。候補検証中の警報は、FIFO全体を現行IDへ吸収して終わる（旧`forward_validation_pending`）。
- (b) 再利用の適用: `evaluate_held_models_for_alarm_interval_reuse`の`selected_reuse_model_id`が現行と違えば学習帰属を切り替え（切替位置の記録、旧`alternative_fit`）、同じなら維持（旧`current_fit`）。どちらも変化区間を選択後の現行IDへ吸収する（既存`absorb_assigned_training_samples_into_held_model`）。
- (c) 適合なし: 評価情報の`baseline_supported_interval_mean_losses_by_model_id`を既存`select_candidate_initial_parameter_snapshot`の`evaluated_mean_losses_by_model_id`へ渡し、既存`start_post_alarm_candidate_validation_session`を呼ぶ。この接続はtest内では実旧と照合済み（`tests/refactoring/test_alarm_interval_model_reuse_assessment.py`の`test_alarm_interval_reuse_initialization_and_session_start`）。productionの組立は未実装。
- (d) 検出器のreset、FIFOのclear、適応イベントの記録（旧action: reuse/maintain/create_pending/insufficient_data/forward_validation_pending）。
- 許容損失増加量（旧`distance_threshold`）は、今回は明示引数で渡している。設定型への登録は組立側で扱う。
- oracle: 同testの`resolve_alarm_interval_in_legacy_client`が実旧_resolve_driftを実行する。現在は評価より後の副作用（吸収・イベント記録・検出器reset・帰属切替）を差し替えているので、組立のspecでは差し替えを外して最終状態を対照する。`build_alarm_interval_reuse_oracle`が同じ実NN・統計・区間の新owner群と実旧clientを作る。
- 通常進行と終端回収は実装済み（`runtime/post_alarm_candidate_validation_progress.py`、`runtime/incomplete_post_alarm_candidate_validation_finalization.py`）。呼出側のsession解除・一覧記録・通知は未実装。

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
5. cc-sddの要求→設計/命名→tasks→実装→統合検証へ進む。各段階はGPT-6 Luna（利用不能時はSonnet）の独立レビューと有用な指摘の反映で承認し、担当モデル・代替理由・採否・内容hashを記録する。命名承認前に新srcや先取りtestを作らない。

## 別タスクと記録の扱い

HTMLは元checkoutの`docs/overview/fedsda-processing-flow.html`にあるコミット保留資料。worktreeへ自動コピー・stageしない。他の保留2資料もAGENTS.mdに従う。
旧実装の不具合は[implementation-findings](../../docs/research/implementation-findings/README.md)、開発手順の問題は`development-findings/`に記録する。
中断時は対象specへ未完了task/検証/差分を記録し、この案内の現在地を更新する。別タスクの結果をリファクタリングの完了証拠に混ぜない。
