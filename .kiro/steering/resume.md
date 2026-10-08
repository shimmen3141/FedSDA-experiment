# リファクタリングの再開案内

更新: 2026-10-09（held-candidate-validation-diagnostic-notificationの完了と、手順の簡素化の反映。主担当はClaude Code）。これは案内であり、承認・進捗の正本は各specのspec.jsonとtasks.md。完了specの検証commit・件数・hash・レビューの経緯は、各specのintegration-validation.mdとreview.mdを読む（ここへ写さない）。

具体的な簡略化・効率化・局所的なアルゴリズム調整は[改善候補](../../docs/research/improvement-candidates/README.md)へ1候補1ファイルで記録する。広い研究アイデアは研究バックログ、不具合の疑いはimplementation-findings。同率現行優先IMPROVE-001（旧ALGO-001）は未検証/未採用で、旧保有順を維持する今回の移植に混ぜない。

Claude・Codexで交代する場合は[共通引継ぎ手順](agent-handoff.md)を参照する。
Claudeの入口は[CLAUDE.md](../../CLAUDE.md)。同じworktreeとspecを使う。2026-10-08の新方針は日常的・分量の多いレビューにLuna、やや複雑・難易度の高い実装レビューにHaiku 5.5を優先し、利用不能時は他方へ代替する。effortはHaiku・Lunaとも`medium`に明示指定する（Haikuは2026-10-09のユーザー指示で`high`から変更）。起動・判断の正本は共通引継ぎ手順。

## 現在地

- **現在地:** [held-candidate-validation-diagnostic-notification](../specs/held-candidate-validation-diagnostic-notification/README.md)（候補検証の確定に伴う診断通知）まで完了。進行中の編集・未解消レビュー指摘はない。
- **次の一手:** 下の「次の候補」1（警報のない標本での帰属確定と学習、標本1件の処理全体）を、境界を分けて要求から仕様化する。同期を同じspecへ混ぜない。次のspecで、共通引継ぎ手順の2026-10-09の変更のうち実装が要る2件（Git管理下の共用fresh process script、依存の許可集合のdata化と一致test）を行う。
- 作業場所: `.worktrees/refactoring/`、ブランチ: `refactor/architecture`。元checkout（`main`、HEAD `748c3aa`、`src/`なし）と取り違えない。`federated_drift_experiment/`は固定旧実装との対照・既存golden実行用。`src/federated_learning_experiments/`は移植中の新実装。旧名alias/互換読込みを追加しない。
- **実行環境の注意:** Windowsの基準環境は、スマートアプリコントロールがtorchの読込み（`venv/Lib/site-packages/torch/_C.cp313-win_amd64.pyd`）を断続的にブロックする（10月3日・5日・8日に発生し、いずれも時間をおいて解消）。発生したら保護設定・venv・goldenを変えず、WSL Ubuntu（`wsl -d Ubuntu`→リポジトリ直下で`source .venv/bin/activate`→worktreeへ移動。Python 3.14.4）で作業を続け、結果を「WSLで成功」と区別して記録する。WSLでは既知の3件が失敗する（Python 3.14の構文解析の違いによる既存test 1件、golden回帰2件の環境差による不一致）。Task 3と最終GOは、Windows基準での全回帰が済むまで完了にしない。
- pushの扱い: taskごとに通常pushを1回だけ試す。失敗時は連続再試行や原因探索をせず、次taskのpush成功時に未送信commitも送る（2026-10-08ユーザー指示）。

## 決定事項

- **Linux用のgolden（2026-10-08ユーザー）:** 旧実装をLinuxで実行した結果のgoldenは、新実装の全体runを接続するspecに入る前に作る。既存のWindows用goldenと回帰testは変更せず、別ファイル・別testにする。作る前に、同じ環境で2回実行して結果が一致することを確かめる。それまでの部品specでは、WSLのgolden回帰の不一致（環境差。旧実装は固定旧から無変更）を理由にgoldenを変えない。
- **検出episode（2026-10-09、主担当の判断）:** 制御（旧`DetectionEpisodeController`）と同一episodeの追加検出の経路は当面移植しない。最終構成の既定で無効。記録は[UNPORTED-001](../../docs/research/implementation-findings/unported-001-detection-episode-control.md)。ユーザーが必要と判断すれば覆せる。
- **移植しないと決めたもの（2026-10-07ユーザー）:** 参照も学習させる候補検証方針（旧`shadow_tournament`）。記録は[UNPORTED-002](../../docs/research/implementation-findings/unported-002-reference-shadow-tournament.md)。
- **手順の簡素化（2026-10-09ユーザー）:** 変異は汎用の`mutation_check.py`、命名表はsourceとtestのmodule直下の名前まで、設計・tasksへ条件数を書かない、小さいspecでは最終判定をTask 3のレビューと同じ依頼で受けてよい、resumeへ件数・hashを写さない、依存の注入契約testをsymbolごとに足さない、fresh processの確認は共用script。Haikuのeffortの既定は`medium`。正本は[共通引継ぎ手順](agent-handoff.md)。
- 具体的な簡略化・効率化・局所的なアルゴリズム調整は[改善候補](../../docs/research/improvement-candidates/README.md)、不具合の疑い・移植しない判断・新実装で見つけた事項は[implementation-findings](../../docs/research/implementation-findings/README.md)へ、気づいた時点で1件1ファイルで記録する（2026-10-09ユーザー指示）。研究アルゴリズムの変更は採用していない。

## 次に該当ファイルを変更するときに直すこと

- `tests/refactoring/test_held_candidate_validation_progress.py`: 汎用の変異toolで見つかったtestの穴2件（[NEW-002](../../docs/research/implementation-findings/new-002-held-validation-test-gaps-found-by-mutation-tool.md)。反映の関数の派生型の拒否、終端回収の記録→解除の順）。`test_owner_types_are_rejected_before_upstream_updates`のコメント「保持がなければ上流は何もしないが」を、mockへ差し替えた条件に合う説明へ改める。

## 完了specの一覧（新しい順。詳細は各spec）

警報後の「参照の固定→標本の観測→損失収集→採否評価→確定→正式ID確認」、警報時の「区間の準備→再利用評価→切替または候補検証の開始→完了処理→記録」、候補検証sessionの保持、診断証拠と通知までの部品が、実旧の対応するメソッドとの対照つきで揃っている。新client・新全体runの接続は未完了。部品の旧実装対照と、新全体runのgolden一致は別の完了条件。

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

### 1. 標本1件の処理の残り（未仕様化）

警報が起きた標本での処理は`handle_alarm_occurrence`で1回の呼出しになり、候補検証の進行（`advance_held_candidate_validation`）は確定時に診断へ通知するようになった。標本1件の処理全体（旧`process_one_step`）へ向けて残るのは次のとおり。一度に混ぜず、境界を決めてspecを分ける。

- 警報のない標本での帰属確定と学習: FIFOの容量を超えた分を1件ずつ現行モデルへ確定する経路（旧`fedsda.py` 511〜525行。統計→標本→概念の順のinline実装で、吸収の部品とは更新順が違う）と`train_step`。
- 標本1件の処理全体: 予測→候補検証の進行→検出→（警報なら`handle_alarm_occurrence`、なければ帰属確定と学習）の並べ方、標本位置の連続性、応答の前後に標本を観測しない保証。検出位置・推定変化点・検出器の候補開始位置の記録、保留中の学習更新の消化（`flush_pending_updates`）、標本ごとの履歴（旧`history_drift_type`）もここで扱う。
- 予測側の警報hook（旧`_on_drift_alarm`・`_on_drift_resolution`）は予測結合の接続で扱う。最終Switching予測はFixed-Share側で、AdaHedgeの再始動を直接参照しない。NPZの診断（routing_concept_restart_counts、global gain、条件付きLOO）はAdaHedge状態を使う。最終3goldenはこれらの診断を比較しないので、その成功だけでは保存診断の同一性を証明しない。IMPROVE-007は診断の選択実行案で未検証/未採用。
- 既知の限界（held-candidate-validation-progressの設計4節）: 候補検証の進行は、標本位置が提案位置より後であることを検査しない。`handle_alarm_occurrence`は提案位置を「保留位置の最終観測位置と一致する警報位置」に固定するので、以後の標本位置が保留位置の規則（最終観測位置の次だけを受け入れる）に従う限り問題にならない。その保証は標本1件の処理全体のspecで扱う。
- 記録のoracle: tests/refactoring/test_alarm_occurrence_handling.py（`build_alarm_occurrence_oracle`。実旧`_resolve_drift`と実旧の再始動hook）、test_held_candidate_validation_progress.pyの実旧対照を再利用する。
- 許容損失増加量と最小変化区間件数の設定登録は組立側に残る。
- 新client・新全体runは未完了。部品の旧対照と全体runのgolden一致を混同しない。全体runを接続するspecの前に、Linux用のgoldenを作る（上の「決定」）。

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
3. 対象specのREADME→spec.json/tasks.md→requirements/design/naming→review/integration-validation。未完了taskがあれば次のspecより先に扱う。
4. `git status --short`とブランチを確認し、別タスクの差分・未追跡資料を特定する。文書の案内と実際の状態が違う場合はGitとspecの正本を確認する。
5. cc-sddの要求→設計/命名→tasks→実装→統合検証へ進む。各段階は共通のLuna/Haiku 5.5選択方針に従った独立レビューと有用な指摘の反映で承認し、選択理由・モデル/effort・代替理由・採否・内容hashを記録する。命名承認前に新srcや先取りtestを作らない。

## 別タスクと記録の扱い

HTMLは元checkoutの`docs/overview/fedsda-processing-flow.html`にあるコミット保留資料。worktreeへ自動コピー・stageしない。他の保留2資料もAGENTS.mdに従う。
旧実装の不具合は[implementation-findings](../../docs/research/implementation-findings/README.md)、開発手順の問題は`development-findings/`に記録する。
中断時は対象specへ未完了task/検証/差分を記録し、この案内の現在地を更新する。別タスクの結果をリファクタリングの完了証拠に混ぜない。
