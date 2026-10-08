# リファクタリングの再開案内

更新: 2026-10-09（held-adahedge-diagnostic-notificationの全3task独立承認、別fresh sessionのfeature最終GO）。これは案内であり、承認・進捗の正本は各specのspec.jsonとtasks.md。

**引継ぎ地点（2026-10-08）:** 主担当Claude Codeが[held-candidate-validation-progress](../specs/held-candidate-validation-progress/README.md)を完了（全3taskの独立承認と別sessionのLuna最終GO）。その直前の[candidate-validation-adaptation-recording](../specs/candidate-validation-adaptation-recording/README.md)も完了（全3task承認、別sessionのLuna最終GOは3回目。1・2回目の指摘は進捗記録と再開案内の更新漏れ）。これら2 specの進行中の編集・レビューはない。2 specの検証commitは`733994b`、Windows基準で全9663 passed/3 skipped/2 warnings。引継ぎ時点では未コミット差分なし。引継ぎ後に下記のAdaHedge診断証拠specも完了した。

具体的な簡略化・効率化・局所的なアルゴリズム調整は[改善候補](../../docs/research/improvement-candidates/README.md)へ1候補1ファイルで記録する。広い研究アイデアは研究バックログ、不具合の疑いはimplementation-findings。同率現行優先IMPROVE-001（旧ALGO-001）は未検証/未採用で、旧保有順を維持する今回の移植に混ぜない。

Claude・Codexで交代する場合は[共通引継ぎ手順](agent-handoff.md)を参照する。
Claudeの入口は[CLAUDE.md](../../CLAUDE.md)。同じworktreeとspecを使う。2026-10-08の新方針は日常的・分量の多いレビューにLuna、やや複雑・難易度の高い実装レビューにHaiku 5.5を優先し、利用不能時は他方へ代替する。effortはHaikuを`high`、Lunaを`medium`に明示指定する。起動・判断の正本は共通引継ぎ手順。

## 現在地

- **現在地:** [held-adahedge-diagnostic-notification](../specs/held-adahedge-diagnostic-notification/README.md)完了。仕様r1・全3taskを独立承認し、別fresh sessionの内部Luna mediumがfeature最終GO。進行中の編集・未解消レビュー指摘はない。

- **実行環境の注意:** Windowsの基準環境は、スマートアプリコントロールがtorchの読込み（`venv/Lib/site-packages/torch/_C.cp313-win_amd64.pyd`）を断続的にブロックする（10月3日・5日・8日に発生し、いずれも時間をおいて解消）。発生したら保護設定・venv・goldenを変えず、WSL Ubuntu（`wsl -d Ubuntu`→リポジトリ直下で`source .venv/bin/activate`→worktreeへ移動。Python 3.14.4）で作業を続け、結果を「WSLで成功」と区別して記録する。WSLでは既知の3件が失敗する（Python 3.14の構文解析の違いによる既存test 1件、golden回帰2件の環境差による不一致）。Task 3と最終GOは、Windows基準での全回帰が済むまで完了にしない。
- **決定（2026-10-08ユーザー）:** Linux用のgolden（旧実装をLinuxで実行した結果）は、新実装の全体runを接続するspecに入る前に作る。既存のWindows用goldenと回帰testは変更せず、別ファイル・別testにする。作る前に、同じ環境で2回実行して結果が一致することを確かめる。それまでの部品specでは、WSLのgolden回帰の不一致（環境差。旧実装は固定旧から無変更）を理由にgoldenを変えない。
- 次の一手: 最終構成で既定無効の検出episodeの移植要否を判断し、標本ごとのclient進行を別specの要求から整理する。診断保持/通知部品を再実装せず、通知する時機・重複防止をclient側で決める。episode・client進行・同期を一つのspecへ混ぜない。
- 作業場所: `.worktrees/refactoring/`、ブランチ: `refactor/architecture`。元checkout（`main`、HEAD `748c3aa`、`src/`なし）と取り違えない。
- 最新完了: [held-adahedge-diagnostic-notification](../specs/held-adahedge-diagnostic-notification/README.md)。AdaHedgeDiagnosticEvidenceCollectionがglobalと真の概念別oracleのlive ownerを独立保持し、notify_diagnostics_of_training_assignment_changeが入力検査後にglobalだけを再始動する。検証commit24a5953、Windows全9811 passed/3 skipped/2既存warnings（199.13s）、JUnit9814件、旧11/最終3golden成功。対象24＋guard2881、変異23種の非等価21/21検出・等価2、fresh stdlib接続、Ruff180files/Pyright/pip check成功。source281パス8f94d784a3a1f0674c6ba76dae1ba114a4c96ca87480945e6041fb94ef99d2e2。Task3と最終別sessionは対象2905件/fresh/Ruffを独立実行、全pytestは主担当のみ。最終GO対象5cb1e61。以後は証拠・進捗文書だけ。client通知順序・重複防止・同期・episode・保存診断全体・新全体runは後続。
- その前の完了: [adahedge-diagnostic-evidence](../specs/adahedge-diagnostic-evidence/README.md)。単一のAdaHedgeDiagnosticEvidenceが診断用累積損失・gap・集合変更/概念再始動計数を所有し、重み取得・損失更新・明示再始動を提供。source/test commit d80c62a、Windows全9727 passed/3 skipped/2 warnings（176.44s）、JUnit9730件、旧11・最終3golden成功、Ruff177files/Pyright/pip check成功。実source変異24種は20非等価を検出、4等価は理由を記録。fresh新stdlibの5操作成功。source LF hash278パス cc175e8a0acbe8b2057c931384acdb553f5c5a4f8edc756682b097618e96bf28、固定旧748c3aa差分空。Task 2 Lunaは対象42+依存注入22、Task 3別Lunaは対象+依存2863/fresh/Ruff/pip/identityを再現。最終新session Lunaはidentity/progressを独立照合。全pytestは主担当のみ。Haiku外部CLIは非公開コード送信の承認不足で自動審査拒否、内部Luna mediumへ代替。以後の差分は証拠・進捗文書だけ。
- その前の完了: [held-candidate-validation-progress](../specs/held-candidate-validation-progress/README.md)。要求r1・設計r3・命名r2・tasks r2。runtimeの`CandidateValidationSessionHolder`（sessionを1つだけ保持）と、`apply_alarm_response_to_validation_session_holder`（候補検証を開始した応答だけが保持させる）、`advance_held_candidate_validation`（標本1件を観測させ、確定したら適応記録→保持の解除）、`finalize_held_incomplete_candidate_validation`（終端回収→記録→解除）。その前の[candidate-validation-adaptation-recording](../specs/candidate-validation-adaptation-recording/README.md)（要求r2・設計r4・命名r2・tasks r2）は、適応結果へ候補検証の5値を追加し、`record_completed_candidate_validation`と`record_incomplete_candidate_validation_finalization`が到達時の確定と未完了の終端回収を`AdaptationRecordStore`へ記録する。recordのfield名は`adaptation_sample_index`へ改めた。
- 上の2 specの検証: commit `733994b`、Windows基準で全9663 passed/3 skipped/2 warnings（327.73s）、JUnit 9666件、旧11・最終3golden成功、Ruff 175 files/Pyright/pip check成功。変異は51/51と30/30、fresh新CPUは各10条件。source hashは276パス、`f2348d1bede38ac4ebd4f1aa59468250821f35f7740a9831cf7926f21e73e69e`。固定旧748c3aaからのdiffは空。Task 2・3のLunaは対象test・依存境界・fresh CPU・Ruffを再現。全pytestは主担当のみ。以後の差分は証拠・進捗文書だけ。
- 次にheld_candidate_validation_progress.pyを変更するときに直すこと: 保持が空でないときの`ValueError`の文言（`response without an active validation requires an empty holder`）は、sessionを持つ開始の応答にも当たるので不正確（Haikuの任意の指摘。文言だけなので本specでは変更していない）。
- その前の完了: [alarm-adaptation-recording](../specs/alarm-adaptation-recording/README.md)（主担当Codex）。要求r1・設計r3・命名r3・tasks r3、全3task/別fresh Luna最終GO。evaluationのAdaptationRecordStoreが履歴・切替位置・再利用/現行適合件数を所有し、runtimeのrecord_completed_alarm_responseが5種類の警報完了を不変AdaptationRecordへ変換する。検査は保存前、入力recordは再検査したcopyを保存する。
- alarm-adaptation-recordingの検証: source/test commit 9b72182、全9363 passed/3 skipped/2 warnings（165.21s）、JUnit9366件、対象40＋AST2611、22変異検出、新CPU2/4class×5経路成功。source/golden271パスのLF hash14816d2b68990d224dc6c8ee4cc62e467e17e60974993f10f863f82cf8e06d1d。固定旧748c3aaから旧実装/golden/旧回帰/toolsのdiff空。Haiku Task1は静的、Luna Task2/3は対象/AST/fresh/Ruff再現、最終fresh Lunaは承認/hash/固定旧/JUnitの独立照合。全pytestは主担当のみ。最初のsandbox測定は29権限エラーで、同じテストを必要な権限で再実行して成功した。新全体runのgoldenは未検証。本spec完了までは以後の差分が証拠・進捗文書のみ。その後、検査scriptのhash読取りを一括化した（実験src/testsは変更なし、旧hash定義との一致を専用検証）。共通手順にレビュー/検証の重複を避ける条件を追記した。
- 直前完了: [alarm-response-completion](../specs/alarm-response-completion/README.md)。runtimeの`complete_alarm_buffer_response`と不変record`AlarmResponseCompletion`（元の応答、警報位置、変更前後の学習帰属ID、推定変化点、episode ID、resetに使った基準平均、消費した保留位置。property `training_model_switch_sample_index`・`detection_episode_operation_required`）。詳細は同specのintegration-validation.md/spec.json/review.md。Task 1は独立レビューで4回差し戻され、検査をすべて更新の前に置く設計（recordを先に組み立てる、変更記録のexact型検査、未観測FIFOの拒否）へ改訂した。その前はalarm-buffer-response、alarm-training-interval-preparation。改善候補IMPROVE-004/005/006は未検証/未採用。研究アルゴリズムの変更は採用していない。
- 直前specの検証済みsource/test commit: `def37e4`。全pytest 9254 passed/3 skipped/2 warnings、155.63s（主担当実測、JUnit 9257件照合）、対象57＋AST2542、33変異検出、fresh新CPU 2/4class×5経路の10条件をTask 2・3のLunaも再現。Ruff 167files/Pyright/pip check成功、旧11・最終3golden成功。固定旧748c3aaから旧実装・golden・旧回帰test・toolsへのdiffは空。source hashは267パス、`b11a144ab4312c63f4dfd84492c5dc81c9ab11f7264caca9de5e32453fd9db2b`。以後は証拠/進捗文書だけを変更。その後に追加した`.kiro/settings/scripts/spec_checks.py`はtrackedなPythonなので、以後のcommitのsource hashは268パスから数える。
- pushの扱い: taskごとに通常pushを1回だけ試す。失敗時は連続再試行や原因探索をせず、次taskのpush成功時に未送信commitも送る（2026-10-08ユーザー指示）。今回の変更と検証記録のpushは2026-10-08にユーザーが明示承認済み。Task 1の初回pushは自動審査に拒否され実行されなかったが、明示承認後のTask 2 pushでd80c62aまで送信済み。最終GO記録の送信状況はGit参照と終了報告で確認する。
- レビューに出す前の機械的な照合（命名表、承認hash・固定旧差分・source hash・JUnit、進捗記録）は`.kiro/settings/scripts/spec_checks.py`。進捗の照合`progress`は2026-10-08に追加した（最終レビューのNO-GOが記録の更新漏れで2回続いたため）。scriptはtrackedなPythonなので、追加後のcommitのsource hashは`733994b`の値から変わる。レビュー依頼の確認項目（更新前の検査、要求との1文ずつの照合、変異ごとの一覧、Minorの基準）は[共通引継ぎ手順](agent-handoff.md)の「レビューに出す前と依頼文」。どちらも2026-10-08に追加した。
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

### 1. 検出episodeの移植要否判断、標本ごとのclient進行（現specのGO後）

警報応答・候補検証の適応記録と、候補検証sessionの保持・解除までは部品が揃った。単一の診断証拠は完了。global/oracle保持・global再始動通知の部品は現specで実装済み、Task3承認とfeature最終GO待ち。残りは実clientの通知時機・重複防止、予測/保存診断全体runの接続、検出episode、標本ごとのclient進行。まず最終構成で既定無効のepisodeを別の任意機能として移植する要否を判断し、その後client進行を別specへ分ける。後続の要求・設計・命名・tasksは未作成。

- 通知と保存診断: 最終Switching予測はFixed-Share側で、旧_on_local_model_changeのAdaHedge restartを直接参照しない。一方、NPZのrouting_concept_restart_counts、global gain、条件付きLOO診断はAdaHedge状態を使うので、通知を全面的に省くことはできない。根拠はalarm-adaptation-recording/research.md。最終3goldenはこれらの診断を比較しないので、その成功だけでは保存診断の同一性を証明しない。IMPROVE-007は診断の選択実行案で未検証/未採用。
- 検出episodeは最終構成の既定で無効。mark_operationと同一episodeの追加検出は、移植の要否を最終構成から判断する。
- 標本ごとのclient進行で確認すること: 応答→完了処理→警報応答の記録→保持への反映の並べ方、応答を一度だけ完了させること、応答と完了の間に標本を観測しない保証、標本位置の連続性。検出器名等は上流更新の前に検査する。
- 既知の限界（held-candidate-validation-progressの設計4節）: 上流の進行は、標本位置が提案位置より後であることを検査しない。提案位置より前の位置で確定すると、上流の更新の後で記録が拒否される。位置の検査をどこへ置くかは、client進行のspecで判断する。
- 記録のoracle: tests/refactoring/test_alarm_adaptation_recording.py、test_candidate_validation_adaptation_recording.py、test_held_candidate_validation_progress.pyの実旧対照を再利用する。
- 許容損失増加量と最小変化区間件数の設定登録は組立側に残る。
- 新client・新全体runは未完了。部品の旧対照と全体runのgolden一致を混同しない。全体runを接続するspecの前に、Linux用のgoldenを作る（上の「決定」）。

### 2. その先

- 完了情報を呼出側へ接続し、session解除・判定/適応event一覧・切替位置・検出episode操作・学習帰属変更の通知：global診断証拠の再始動（最終Fixed-Share予測の通知ではない）を組み立てる。
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
