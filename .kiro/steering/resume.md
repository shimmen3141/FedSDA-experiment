# リファクタリングの再開案内

更新: 2026-10-07（Codexによる候補のエポック学習specの完了時）。これは案内であり、承認・進捗の正本は各specのspec.jsonとtasks.md。

Claude・Codexで交代する場合は[共通引継ぎ手順](agent-handoff.md)を参照する。
Claudeの入口は[CLAUDE.md](../../CLAUDE.md)。同じworktreeとspecを使い、GPT-6 Lunaを優先し、利用不能時はSonnetの独立レビューで承認する。

## 現在地

- 作業場所: `.worktrees/refactoring/`、ブランチ: `refactor/architecture`。元checkout（`main`、HEAD `748c3aa`、`src/`なし）と取り違えない。
- 作業状態: [候補のエポック学習](../specs/candidate-epoch-training/README.md)はcompleted。全5tasksが別Luna承認済み、要求revision2・設計revision1・命名revision7・tasks revision1。別fresh GPT-6 Lunaのfeature最終GOも取得。既存[候補生成](../specs/candidate-classifier-construction/README.md)もcompleted、次はsession開始の組立。承認・進捗は各specのspec.json/tasks.mdが正本。
- 直近の検証済み実装commit: `66ac055`。全pytest 6868 passed/3 skipped/2warnings（主担当実測、JUnit照合）、Ruff/Pyright/pip check成功、旧11・最終3golden成功、固定旧基準`748c3aa`から旧実装・golden・旧回帰test・tools/への差分は空。警告は拒否test準備のnested Tensor prototypeと既存TypedStorage deprecated。
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

### 1. 警報後の候補検証session開始の組立（次に着手）

旧`FedSDAClient._begin_forward_validation`（`federated_drift_experiment/clients/fedsda.py`）のうち、参照の固定・履歴平均、候補の生成と学習は移植した。次は(c)session開始の組立へ進む。各部品の完了状態は同specのspec.jsonで確認する。

- (a) 候補の生成は実装済み: `runtime/candidate_classifier_construction.py`の`create_independent_candidate_training_state`が、構造の参照分類器・選択済みsnapshot・optimizer設定から、独立した分類器と共有部/概念固有部の専用optimizer管理器を返す。旧`BaseClient._new_model`→`set_params`→`reset_optimizer`と全値・parameter順・乱数消費を照合済み。初期値の選択には既存`methods/fedsda/candidate_model_selection/candidate_parameter_initialization.py`の`select_candidate_initial_parameter_snapshot`を使う。候補の学習はこの生成部品を再実装せず利用する。
- (b) 警報区間での候補の学習: 旧`BaseClient._train_new_model`と、そこから呼ばれる`_train_new_model_early_stopping`・`_train_new_model_fixed`・`_update_new_model_epochs`・`new_model_initial_epochs`（`federated_drift_experiment/clients/base.py`。`_train_new_model`で検索する）。最終3goldenの設定は`tests/proposed_regression_golden.json`の`definition`にあり、`NEW_MODEL_TRAINING="early_stopping"`、`NEW_MODEL_EPOCHS=30`、`NEW_MODEL_VALIDATION_FRACTION=0.2`、`NEW_MODEL_EARLY_STOPPING_PATIENCE=3`、`NEW_MODEL_EARLY_STOPPING_MIN_DELTA=1e-4`、`NEW_MODEL_LR=0.01`、`CLIENT_BATCH_SIZE=32`。early stoppingは`torch.randperm`で学習/検証へ分け、`DataLoader(shuffle=True)`でbatchを作るので、torch乱数を消費する。最良parameterを保持して最後に復元する。学習量（延べ標本数と更新回数）は旧`compute_counters`の差分から得てsessionへ記録し、採用時に`candidate_trained_sample_count`/`candidate_parameter_update_step_count`として渡す。
- 新側の学習部品: `learning/training/candidate_epoch_training.py`の`train_candidate_classifier_epochs`と`CandidateEpochTrainingSettings`。固定エポック/検証損失早期停止/学習省略を実装し、正常値・小区間・0epoch・端数batch・RNG・全parameter/grad/optimizer・学習量を実旧へ照合した。最良parameterだけを戻しoptimizer/gradは最後の更新の状態を保持する。設定型は全6field必須。学習率は既存ParameterOptimizerSettingsから候補生成へ渡し、epoch設定には二重定義しない。最終構成30epoch・patience3・minimum decrease1e-4も生成から継続更新の接続testに含む。RunSettingsへの登録はsession組立時に必要項目を解決する。
- 新側で使える学習部品: `src/federated_learning_experiments/learning/training/`の`joint_model_parameter_update.py`（1回の共同更新。単一batchなら旧`ResidualAdapterMLP.update`と全値一致することを登録・採用のtestで確認済み）、`parameter_optimizer_state.py`、`parameter_optimizer_settings.py`、`local_training_settings.py`。
- (c) session開始の組立: 候補の生成→学習→参照の固定（`runtime/post_alarm_reference_model_fixation.py`）→損失収集の開始（`PostAlarmCandidateLossCollection`の生成）。乱数の消費順を実旧（候補の生成→学習→参照の生成）と一致させる。
- oracle: `tests/refactoring/test_post_alarm_reference_model_fixation.py`の`begin_forward_validation_in_legacy_client`が実旧のsession開始を実行している。現在は`_train_new_model`を無効化しているので、学習を有効にして候補の値・学習量・処理後の乱数状態を対照する。

### 2. 候補検証sessionの進行と記録

- 上位が観測の戻り値（規定件数への到達）を見て、既存の評価関数と確定を順に呼ぶ進行。
- 判定record（旧`ProvisionalModelDecision`）、切替位置、検出エピソード、適応イベントの記録。結果種別と変更記録から旧のaction・戻り値・切替位置の条件は導ける（対応表は`.kiro/specs/post-alarm-candidate-validation-resolution/design.md`）。
- 学習帰属変更の通知（最終構成では予測重みの再始動。旧`_on_local_model_change`）。
- 実験終端で未完了のsessionを棄却し、保留標本を現行モデルへ吸収する処理（旧`finalize_incomplete_forward_validation`）。

### 3. その先

- 警報の検出から候補検証sessionの開始までの接続（推定変化点からの区間切出し、保留標本の確保、初期parameterの選択）。
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
