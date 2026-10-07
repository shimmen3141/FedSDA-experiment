# リファクタリングの再開案内

更新: 2026-10-08（終端の未完了候補検証spec完了）。これは案内であり、承認・進捗の正本は各specのspec.jsonとtasks.md。

Claude・Codexで交代する場合は[共通引継ぎ手順](agent-handoff.md)を参照する。
Claudeの入口は[CLAUDE.md](../../CLAUDE.md)。同じworktreeとspecを使い、GPT-6 Lunaを優先し、利用不能時はSonnetの独立レビューで承認する。

## 現在地

- 直近完了: [終端の未完了候補検証の確定](../specs/incomplete-post-alarm-candidate-validation-finalization/README.md)。要求r4・設計r1・命名r2・tasks r1、全5task独立fresh Luna承認、別fresh Luna feature最終GO、completed。対象67条件＋AST1582条件、実NN7条件/6変異と元byte復元、fresh CPUの2/4classで非active/未到達観測/終端回収/共同更新を確認。
- 公開状態（2026-10-08）: GitHub復旧後の通常pushを1回実行し、未送信7commit（`5ddb9e3..db89dc4`）の送信に成功した。今後はユーザー指示に従いtaskごとにpushを1回だけ試す。失敗時は連続再試行や原因探索をせず、次taskのpush成功時に未送信commitも送る。
- 直近のpush状態: Task5までの`c6d5e33`は送信成功。別feature最終GOと再開案内の`d18802a`はGitHubのInternal Server Errorで1回のpushが失敗した。再試行・原因探索なし。この失敗記録もローカルcommitし、次taskの通常pushで未送信分をまとめて送る。再開時はGitのahead状態も確認する。
- 次の候補: 警報検出から候補検証session開始までの接続（推定変化点からの区間切出し、保留標本確保、初期parameter選択）。未仕様化・未承認。完了情報の呼出側への接続とsession解除・一覧記録・通知も後続。下記候補と公開部品の実態を確認し、次specの要求から始める。
- 作業場所: `.worktrees/refactoring/`、ブランチ: `refactor/architecture`。元checkout（`main`、HEAD `748c3aa`、`src/`なし）と取り違えない。
- 作業状態: [候補検証session開始](../specs/post-alarm-candidate-validation-session-start/README.md)の全5tasksはLuna承認・完了。要求r2・設計r2・命名r4・tasks r1を維持し、別fresh GPT-6 Lunaのfeature最終GO、completed。候補生成とエポック学習もcompleted。承認・進捗は各specのspec.json/tasks.mdが正本。
- 直近の検証済み実装commit: `237030b`。全pytest 7626 passed/3 skipped/2warnings（主担当実測、JUnit7629件照合）、Ruff154files/Pyright/pip check成功、旧11・最終3golden成功、固定旧基準`748c3aa`から旧実装・golden・旧回帰test・tools/への差分は空。警告は拒否test準備のnested Tensor prototypeと既存TypedStorage deprecated。skipはPOSIX bashのないWindows環境の既存3条件。Task5の独立Pyright再現は依存import解決で失敗したが、主担当基準環境の再実行は成功、両者を区別して記録。証拠は終端specのintegration-validation.md。
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

- LEGACY-014: 旧の採用分岐は、新モデルへ移す保留標本を割当概念計数と損失統計へ反映しない（他の分岐と非対称）。意図した仕様かをユーザーへ確認依頼済みで、回答はまだない。新実装は旧挙動を維持しtestで固定している。変更する場合は別specにする。記録は[implementation-findings](../../docs/research/implementation-findings/README.md)。
- レビュー担当の同定: `codex exec -m gpt-6-luna`で起動したレビュー担当は、自分のモデル名を内部から確認できないと回答する。記録は起動時のmodel指定と実行ログのmodel行に依拠している。実際のルーティングの確認はユーザーへ依頼済み。
- 全pytestの独立再現: 2026-10-07のユーザー決定により、主担当の実測とJUnit照合で判定する（[共通引継ぎ手順](agent-handoff.md)の同名節）。レビュー担当のsandboxで再現できるようになった場合は、再現した事実を記録してよい。
- 移植しないと決めたもの: 参照も学習させる方針（旧`shadow_tournament`）。最終構成（`forward_persistent`）で通らず、goldenにも値がない。2026-10-07のユーザー判断で当面不要。
- 旧実装の記録（この期間の追加）: LEGACY-012（登録途中失敗の部分更新と採番消費）、013（使用済み一時IDへの再登録の黙った置換）、014（上記）、015（標本吸収の途中失敗で先行標本の更新と不正標本が残る）。012/013/015は契約外入力だけで、新実装は状態変更前に拒否する。

## 次の候補（未仕様化・未承認）

### 1. 警報検出から候補検証session開始への接続（次に着手）

- 開始は実装済み: `runtime/post_alarm_candidate_validation_session_start.py`の`start_post_alarm_candidate_validation_session`。選択済み初期値・区間・保有状態から候補生成→学習→固定参照→空損失収集を実旧の順序/RNGで組み立て、`PostAlarmCandidateValidationSession`を返す。呼出し側がactive sessionを所有する。
- 開始の旧対照と接続は`tests/refactoring/test_post_alarm_candidate_validation_session_start.py`。正常54・拒否等112・観測接続6＝172条件。2/4class×3optimizer、最終30epoch設定、候補継続更新でも参照不変を照合済み。
- 候補生成・エポック学習・参照固定・標本観測・採否評価・確定・採用登録/帰属変更は既存public部品を再利用する。RunSettingsへの設定登録はclient組立時の後続作業。新client/全体runは未完成。

- 通常進行と不変判定record/完了情報は実装済み: `runtime/post_alarm_candidate_validation_progress.py`の`advance_post_alarm_candidate_validation`。未到達は同session、到達時は既存評価→判定情報→既存適用→完了情報。確定時の保有IDと現在帰属を使う。呼出側が返却後に解除・一覧記録・通知する。
- 終端回収も実装済み: `runtime/incomplete_post_alarm_candidate_validation_finalization.py`の`finalize_incomplete_post_alarm_candidate_validation`。非activeは他入力未読のNone、要求未満を棄却し現行IDへ既存吸収、位置=max(提案位置, processed_sample_count−1)、比較不成立は専用不変recordで表す。呼出側がsessionを解除し一覧へ記録する。要求到達は拒否し、帰属変更/切替位置/episode/通知を行わない。同session再投入の二重吸収防止は後続clientで検証する。
- 次の調査対象は旧警報処理から開始までの組立。検出情報と推定変化点から学習区間/保留標本を切り出し、候補parameterの初期値を選び、既存の公開開始APIへ渡す責務を定義する。新client全体を一度に実装せず、公開部品を再利用して責務・命名を独立レビュー後に移植する。

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
