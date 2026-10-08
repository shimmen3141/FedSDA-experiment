# リファクタリングの再開案内

更新: 2026-10-08（警報応答の適応記録を完了）。これは案内であり、承認・進捗の正本は各specのspec.jsonとtasks.md。

**引継ぎ地点（2026-10-08）:** 主担当Codexが[alarm-adaptation-recording](../specs/alarm-adaptation-recording/README.md)を完了。全3taskの独立承認と別fresh Luna feature最終GO。source/test commit 9b72182、全9363 passed/3 skipped/2 warnings、旧11/最終3golden・品質成功、22変異検出。進行中の編集・レビューはない。次は候補検証の到達時/終端の適応記録を要求から仕様化する。

具体的な簡略化・効率化・局所的なアルゴリズム調整は[改善候補](../../docs/research/improvement-candidates/README.md)へ1候補1ファイルで記録する。広い研究アイデアは研究バックログ、不具合の疑いはimplementation-findings。同率現行優先IMPROVE-001（旧ALGO-001）は未検証/未採用で、旧保有順を維持する今回の移植に混ぜない。

Claude・Codexで交代する場合は[共通引継ぎ手順](agent-handoff.md)を参照する。
Claudeの入口は[CLAUDE.md](../../CLAUDE.md)。同じworktreeとspecを使う。2026-10-08の新方針は日常的・分量の多いレビューにLuna、やや複雑・難易度の高い実装レビューにHaiku 5.5を優先し、利用不能時は他方へ代替する。effortはHaikuを`high`、Lunaを`medium`に明示指定する。起動・判断の正本は共通引継ぎ手順。

## 現在地

- **進行中（2026-10-08、主担当Claude Code、WSLで検証中）:** [candidate-validation-adaptation-recording](../specs/candidate-validation-adaptation-recording/README.md)のTask 1。Windowsの基準環境はtorchの読込みがスマートアプリコントロールでブロックされており、検証はWSL Ubuntu（Python 3.14）で進めている。WSLの結果はWindows基準の検証ではなく、Windowsでの最終回帰が済むまで完了ゲートを通過扱いにしない。経過と残る検証は同specのreview.md「WSLでの再開」。この項目より下の「次の一手」「最新完了」は、このspecの着手前の状態。
- 次の一手: 下の「次の候補」1。候補検証の到達時/終端の適応記録を先に分離する案。session保持と診断通知は境界を決めて別specへ。警報応答の記録を再実装しない。
- 作業場所: `.worktrees/refactoring/`、ブランチ: `refactor/architecture`。元checkout（`main`、HEAD `748c3aa`、`src/`なし）と取り違えない。
- 最新完了: [alarm-adaptation-recording](../specs/alarm-adaptation-recording/README.md)。要求r1・設計r3・命名r3・tasks r3、全3task/別fresh Luna最終GO。evaluationのAdaptationRecordStoreが履歴・切替位置・再利用/現行適合件数を所有し、runtimeのrecord_completed_alarm_responseが5種類の警報完了を不変AdaptationRecordへ変換する。検査は保存前、入力recordは再検査したcopyを保存する。
- 直近検証source/test commit 9b72182、全9363 passed/3 skipped/2 warnings（165.21s）、JUnit9366件、対象40＋AST2611、22変異検出、新CPU2/4class×5経路成功。source/golden271パスのLF hash14816d2b68990d224dc6c8ee4cc62e467e17e60974993f10f863f82cf8e06d1d。固定旧748c3aaから旧実装/golden/旧回帰/toolsのdiff空。Haiku Task1は静的、Luna Task2/3は対象/AST/fresh/Ruff再現、最終fresh Lunaは承認/hash/固定旧/JUnitの独立照合。全pytestは主担当のみ。最初のsandbox測定は29権限エラーで、同じテストを必要な権限で再実行して成功した。新全体runのgoldenは未検証。本spec完了までは以後の差分が証拠・進捗文書のみ。その後、検査scriptのhash読取りを一括化した（実験src/testsは変更なし、旧hash定義との一致を専用検証）。共通手順にレビュー/検証の重複を避ける条件を追記した。
- 直前完了: [alarm-response-completion](../specs/alarm-response-completion/README.md)。runtimeの`complete_alarm_buffer_response`と不変record`AlarmResponseCompletion`（元の応答、警報位置、変更前後の学習帰属ID、推定変化点、episode ID、resetに使った基準平均、消費した保留位置。property `training_model_switch_sample_index`・`detection_episode_operation_required`）。詳細は同specのintegration-validation.md/spec.json/review.md。Task 1は独立レビューで4回差し戻され、検査をすべて更新の前に置く設計（recordを先に組み立てる、変更記録のexact型検査、未観測FIFOの拒否）へ改訂した。その前はalarm-buffer-response、alarm-training-interval-preparation。改善候補IMPROVE-004/005/006は未検証/未採用。研究アルゴリズムの変更は採用していない。
- 直前specの検証済みsource/test commit: `def37e4`。全pytest 9254 passed/3 skipped/2 warnings、155.63s（主担当実測、JUnit 9257件照合）、対象57＋AST2542、33変異検出、fresh新CPU 2/4class×5経路の10条件をTask 2・3のLunaも再現。Ruff 167files/Pyright/pip check成功、旧11・最終3golden成功。固定旧748c3aaから旧実装・golden・旧回帰test・toolsへのdiffは空。source hashは267パス、`b11a144ab4312c63f4dfd84492c5dc81c9ab11f7264caca9de5e32453fd9db2b`。以後は証拠/進捗文書だけを変更。その後に追加した`.kiro/settings/scripts/spec_checks.py`はtrackedなPythonなので、以後のcommitのsource hashは268パスから数える。
- pushの扱い: taskごとに通常pushを1回だけ試す。失敗時は連続再試行や原因探索をせず、次taskのpush成功時に未送信commitも送る（2026-10-08ユーザー指示）。現在、全commitは`origin/refactor/architecture`へ送信済み。
- レビューに出す前の機械的な照合（命名表、承認hash・固定旧差分・source hash・JUnit）は`.kiro/settings/scripts/spec_checks.py`。レビュー依頼の確認項目（更新前の検査、要求との1文ずつの照合、変異ごとの一覧、Minorの基準）は[共通引継ぎ手順](agent-handoff.md)の「レビューに出す前と依頼文」。どちらも2026-10-08に追加した。
- それ以前の完了: [警報の変化区間の解決](../specs/alarm-change-interval-resolution/README.md)。要求r3・設計r3・命名r5・tasks r2、全5task独立レビュー承認、別fresh Luna feature最終GO、completed。runtimeの`resolve_alarm_change_interval`と不変record`AlarmChangeIntervalResolution`（結果種別は`alarm_interval_held_model_reused`/`alarm_interval_current_model_maintained`/`alarm_interval_candidate_validation_started`）。切出し済みの変化区間の標本（1件ずつ）を連結して既存の区間評価へ渡し、選択IDがあればそのモデルへ吸収して学習帰属を切り替え（同じIDなら維持）、なければ全保有モデルのsnapshot→既存の初期値選択→既存のsession開始を行う。実旧_resolve_driftを吸収・帰属切替・初期値選択・session開始を差し替えずに実行して最終状態を照合した。要求〜Task 2のレビューはCodexの利用上限中のため独立AgentのSonnetが代替し、Task 3以降はLuna。その前は[警報区間の保有モデル再利用評価](../specs/alarm-interval-model-reuse-assessment/README.md)（区間評価と再利用候補の選択。読取り専用）。
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
- この期間に確立した進め方（実装前のRED、実装差し替えによる検出力の確認、実旧メソッドを使うoracle、検証コマンド、レビュー依頼の注意）は[共通引継ぎ手順](agent-handoff.md)の「specの進め方」「検証コマンド」を参照する。

## ユーザー確認待ち・未解消の事項

- LEGACY-014: 旧の採用分岐は、新モデルへ移す保留標本を割当概念計数と損失統計へ反映しない（他の分岐と非対称）。2026-10-08ユーザー回答: LEGACYは後でまとめて判断するので、現状のまま置く。新実装は旧挙動を維持しtestで固定している。主担当からの再確認は不要。記録は[implementation-findings](../../docs/research/implementation-findings/README.md)。
- レビュー担当の同定: `codex exec -m gpt-6-luna`で起動したレビュー担当は、自分のモデル名を内部から確認できないと回答する。記録は起動時のmodel指定と実行ログのmodel行に依拠している。実際のルーティングの確認はユーザーへ依頼済み。
- 全pytestの独立再現: 2026-10-07のユーザー決定により、主担当の実測とJUnit照合で判定する（[共通引継ぎ手順](agent-handoff.md)の同名節）。レビュー担当のsandboxで再現できるようになった場合は、再現した事実を記録してよい。
- 移植しないと決めたもの: 参照も学習させる方針（旧`shadow_tournament`）。最終構成（`forward_persistent`）で通らず、goldenにも値がない。2026-10-07のユーザー判断で当面不要。
- 旧実装の記録（この期間の追加）: LEGACY-012（登録途中失敗の部分更新と採番消費）、013（使用済み一時IDへの再登録の黙った置換）、014（上記）、015（標本吸収の途中失敗で先行標本の更新と不正標本が残る）。012/013/015は契約外入力だけで、新実装は状態変更前に拒否する。

## 次の候補（未仕様化・未承認）

### 1. 候補検証の適応記録とsession保持、診断通知（次に着手）

警報完了の記録ownerはalarm-adaptation-recordingで分離した。次は候補検証の到達時（PostAlarmCandidateValidationCompletion）・未完了回収（IncompletePostAlarmCandidateValidationFinalization）の適応記録を同じownerへ接続する境界を要求から決める。現在のAdaptationOutcomeは警報応答5結果だけで、候補の採用等の結果値/切替・計数規則は追加仕様と命名レビューが必要。新しい要求・設計・命名・tasksは未作成。session保持と通知を一度に混ぜず、必要なら別specへ分ける。

- 記録のoracleはtests/refactoring/test_alarm_adaptation_recording.pyと直前specの実旧_resolve_drift。候補検証は対応する完了specの実旧oracleを使い、旧AdaptationEventの全field・切替位置・再利用件数を照合する。
- 予測と通知の確認は完了。最終Switching予測はFixed-Share側で、旧_on_local_model_changeのAdaHedge restartを直接参照しない。一方、NPZのrouting_concept_restart_counts、global gain、条件付きLOO診断はAdaHedge状態を使う。通知を全面的に省くことはできない。根拠はalarm-adaptation-recording/research.md。最終3goldenはこれら診断を比較しないので、その成功だけでは保存診断の同一性を証明しない。IMPROVE-007は診断の選択実行案で未検証/未採用。
- 候補検証sessionの保持と解除: 応答のactive_validation_session（検証中は同じ参照、開始時は新session、その他None）を誰が持ち、到達時・終端回収で誰が外すか。
- 検出episodeは最終構成の既定で無効。mark_operationと同一episodeの追加検出は移植の要否を最終構成から判断する。
- 呼出側で応答を一度だけ完了させ、応答と完了の間に標本を観測しない保証を接続testで確認する。検出器名等は上流更新の前に検査する。
- 許容損失増加量と最小変化区間件数の設定登録は組立側に残る。
- 新client・新全体run、候補完了/終端の記録、session解除、診断通知は未完了。部品の旧対照と全体runのgolden一致を混同しない。

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
