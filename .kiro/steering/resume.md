# リファクタリングの再開案内

更新: 2026-10-08（警報時の保留標本への応答を実装中）。これは案内であり、承認・進捗の正本は各specのspec.jsonとtasks.md。

**現在の作業（2026-10-08）:** [alarm-buffer-response](../specs/alarm-buffer-response/README.md)。要求r1・設計r2・命名r5・tasks r1を独立Luna承認。Task 1/2承認・実装済み、Task 3の実source変異検査中。公開runtimeはactive検証中の全件吸収、または前区間準備→最小件数判定→変化区間の公開解決を組み立てる。FIFOを変更せず、呼出側の消費判断と同じ/新しいsessionを返す。最終対象64＋AST2455成功。全回帰・fresh CPU・別feature最終GOはまだ未実施。ユーザー指示に従い本spec完了まで進める。

具体的な簡略化・効率化・局所的なアルゴリズム調整は[改善候補](../../docs/research/improvement-candidates/README.md)へ1候補1ファイルで記録する。広い研究アイデアは研究バックログ、不具合の疑いはimplementation-findings。同率現行優先IMPROVE-001（旧ALGO-001）は未検証/未採用で、旧保有順を維持する今回の移植に混ぜない。

Claude・Codexで交代する場合は[共通引継ぎ手順](agent-handoff.md)を参照する。
Claudeの入口は[CLAUDE.md](../../CLAUDE.md)。同じworktreeとspecを使い、GPT-6 Lunaを優先し、利用不能時はSonnetの独立レビューで承認する。

## 現在地

- 直近完了: [警報の変化区間の解決](../specs/alarm-change-interval-resolution/README.md)。要求r3・設計r3・命名r5・tasks r2、全5task独立レビュー承認、別fresh Luna feature最終GO、completed。runtimeの`resolve_alarm_change_interval`と不変record`AlarmChangeIntervalResolution`（結果種別は`alarm_interval_held_model_reused`/`alarm_interval_current_model_maintained`/`alarm_interval_candidate_validation_started`）。切出し済みの変化区間の標本（1件ずつ）を連結して既存の区間評価へ渡し、選択IDがあればそのモデルへ吸収して学習帰属を切り替え（同じIDなら維持）、なければ全保有モデルのsnapshot→既存の初期値選択→既存のsession開始を行う。実旧_resolve_driftを吸収・帰属切替・初期値選択・session開始を差し替えずに実行して最終状態を照合した。要求〜Task 2のレビューはCodexの利用上限中のため独立AgentのSonnetが代替し、Task 3以降はLuna。その前は[警報区間の保有モデル再利用評価](../specs/alarm-interval-model-reuse-assessment/README.md)（区間評価と再利用候補の選択。読取り専用）。
- 公開状態（2026-10-08）: GitHub復旧後の通常pushを1回実行し、未送信7commit（`5ddb9e3..db89dc4`）の送信に成功した。今後はユーザー指示に従いtaskごとにpushを1回だけ試す。失敗時は連続再試行や原因探索をせず、次taskのpush成功時に未送信commitも送る。
- 直近のpush状態: 全commitは`origin/refactor/architecture`へ送信済み（taskごとの通常pushはすべて成功）。
- 最新完了: [alarm-training-interval-preparation](../specs/alarm-training-interval-preparation/README.md)。詳細は同specのintegration-validation.md/spec.json/review.md。改善候補IMPROVE-004（履歴基準が使えないモデルの区間評価省略）とIMPROVE-005（前区間の事前検査と吸収での再評価を共用）は未検証/未採用。
- 作業場所: `.worktrees/refactoring/`、ブランチ: `refactor/architecture`。元checkout（`main`、HEAD `748c3aa`、`src/`なし）と取り違えない。
- 作業状態: [候補検証session開始](../specs/post-alarm-candidate-validation-session-start/README.md)の全5tasksはLuna承認・完了。要求r2・設計r2・命名r4・tasks r1を維持し、別fresh GPT-6 Lunaのfeature最終GO、completed。候補生成とエポック学習もcompleted。承認・進捗は各specのspec.json/tasks.mdが正本。
- 直近の検証済みsource/test commit: `d6cc32d`。全pytest 8811 passed/3 skipped/2 warnings、232.47s（主担当実測、JUnit8814件照合）、対象152＋AST2220、6変異検出、fresh新CPU 2/4class×正規/一時IDの4条件を独立Lunaも再現。Ruff163files/Pyright/pip check成功、旧11・最終3golden成功、固定旧基準`748c3aa`から旧実装・golden・旧回帰test・toolsへの差分は空。source hashは263パス、`ca7d146b8adec0f062ff3d3400d8d3d4399fd77f38920e4b9ccd9d90ebd345db`。以後は証拠/進捗文書だけを変更。証拠は最新specのintegration-validation.md。
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

### 1. 警報処理の残りの組立（次に着手）

旧`FedSDAClient._resolve_drift`（`federated_drift_experiment/clients/fedsda.py`）のうち、区間準備、区間評価、再利用・維持・候補検証開始の適用（`runtime/alarm_change_interval_resolution.py`の`resolve_alarm_change_interval`）、session開始・進行・終端回収は部品として完了した。次は、これらを呼び、active sessionの警報分岐・最小件数判定・後始末を担う制御の境界を決める。1つのspecに収まらなければ分ける。要求・設計・命名・tasksはまだ作っていない。

- (a) 完成した区間準備を利用: `runtime/alarm_training_interval_preparation.py::prepare_alarm_training_intervals`へFIFO順の`IndexedObservedTrainingSample`列を渡す。位置とTensor/概念IDの意味上の対応は供給側が保証する。返る前区間は保存/吸収済みで、変化区間だけを後続へ渡す。二重適用しない。FIFOはまだ保持されている。吸収後の履歴統計を区間解決に使う。制御側は各標本を再評価/再吸収する処理を複製しない。
- (b) 変化区間が最小件数（旧`MIN_DRIFT_DATA`）未満なら、前区間の処理だけ行って終わる（旧action insufficient_data）。このとき旧はFIFOをclearしない（LEGACY-002の記録を確認する）。
- (c) 候補検証中の警報は、FIFO全体を現行IDへ吸収して終わる（旧action forward_validation_pending）。
- (d) 検出器のreset、FIFOのclear、適応イベントの記録（旧action: reuse/maintain/create_pending/insufficient_data/forward_validation_pending）、切替位置と再利用計数の記録、学習帰属変更の通知（予測重みの再始動）、active sessionの保持。結果種別と旧action・戻り値の対応は同specのdesign.mdにある（再利用=reuse/1、維持=maintain/0、候補検証開始=create_pending/0）。
- 許容損失増加量（旧`distance_threshold`）は明示引数で渡している。設定型への登録は組立側で扱う。
- oracle: `tests/refactoring/test_alarm_change_interval_resolution.py`の`resolve_alarm_change_interval_in_legacy_client`が実旧_resolve_driftを実行する。現在は推定区間長を区間全体に固定し、イベント記録を記録用に、検出器resetを空に差し替えている。次のspecでは推定区間長と検出器を実物にして、前区間の処理を含む最終状態を対照する。`build_alarm_change_interval_resolution_oracle`が同じ実NN・統計・標本・計数・区間の新owner群と実旧clientを作る。
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
