# Implementation Plan

- [x] 1. 構成と挙動とtestを足す
  - 要求2.1〜2.3のtestを先に書き、REDを確かめてから、状態遷移の部品、設定の項目（割り当ての方式、学習の窓、DriftSurfの設定）、構成、掃引のオプション、runtimeの分岐（役割のモデルの登録と作り直し、clientの時刻の標本の保管と、学習の直前に窓の標本だけを学習の標本の集合へ入れる接続（design.mdの「窓と既存のstoreの接続」）、窓の標本数による集約の重み、clientの正解数の報告、サーバの状態遷移、昇格、予測の役割の配布、検出・作成・統合を行わない）、DriftSurfの記録を足す。Oracle・IFCA・FedAvg+Window／Adaptive-FedAvgで足した項目と分岐を再利用する。`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`と依存境界へ登録する。
  - 完了条件: 新しいtest、構成・掃引・識別子・依存境界のtest、FedDrift・Oracle・IFCA・FedAvg+Window・Adaptive-FedAvgのtest、基準版1〜16のgoldenの照合（Windows）が通る。Ruff・Pyright。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.2, 2.3, 2.4_
  - _Depends:_ FedAvg+Window／Adaptive-FedAvgのspecの実装（`f89a113b`）
- [x] 2. 文書と検証の記録
  - 要求3.1の文書（構成の一覧、`differences-from-feddrift.md`、`python -m tools.generate_option_docs`での設定参照の再生成）。関係するtestの結果をImplementation Notesに記録する。実装レビュー（独立CLI）と全pytestは主担当が行う。
  - _Requirements: 3.1_
  - _Depends:_ 1
- [ ] 3. 調整・本規模の実行と集計
  - commit・pushの後、主担当が内部確認をしてδの調整のコマンドを渡し、δを選んでから本規模のコマンドを渡す。結果を表1の手法と対で集計し、実験の台帳と論文のnotesに記録する。
  - _Requirements: 3.2_
  - _Depends:_ 2

## Implementation Notes


### Task 1・2の実装（2026-11-03、未コミット）

- 基点は`refactor/architecture`の`ecdcb08a`。Task 1の実装とTask 2の文書を追加した。commit・push・stash、本規模の調整・評価は実施していない。ユーザーの今回の分担に従い、独立CLIのレビューと全pytestは主担当が行う。checkboxは権限による未検証と主担当の確認を残して未チェックのままにした。
- TDD: 新しいtestを先に作り、`driftsurf_state`未実装のimportエラーでREDを確認。二値モデルの1列出力をそのままargmaxした誤りと、昇格の平均を和の比較へ置き換えた浮動小数点境界の差も、失敗する専用testを追加して修正した。二値の確率pは(1-p,p)へ対応付け、同値は0。平均は公式と同じ3要素（末尾の0を含む）の平均で比較する。
- 設定: `model_assignment_policy`の既定`detection_batch_assignment`、`training_sample_window`の既定`all_timesteps`を維持。新しい値は`driftsurf_global_role_assignment`と`role_timestep_window`。`driftsurf_settings`の既定None、`DriftSurfSettings.accuracy_drop_threshold`の既定0.10。反応3時刻・窓10時刻は固定。既定の設定は保存設定から省き、既存FedDriftの識別子`sine2_interval50_seed0_1fbf192c89f9`を確認した。
- 掃引: `ExperimentSweepPlan.driftsurf_accuracy_drop_thresholds`と`--driftsurf-accuracy-drop-thresholds`を追加。非正・非有限・重複と、構成を含まない指定を拒否。CLIからplanへの接続、非既定値による識別子の違い、明示した既定値との一致をtestした。OptionSpec・ActivationRule・ChoiceConstraint・parametersへ登録し、`python -m tools.generate_option_docs`で設定参照を再生成した。
- runtime: `feddrift_run_participant_factory.py`でpred/stabの2モデルを登録。`feddrift_client_stream_processing.py`はDriftSurfで検出バッチから学習storeへ追記せず、時刻の全標本を保留する。`driftsurf_timestep_preparation.py`で正解数を報告・合計し、`methods/feddrift/driftsurf_state.py`で状態遷移を行う。新IDとθ0・新optimizerによる役割の作り直し、旧IDの除去、昇格、予測役割の配布を接続した。昇格した時刻のpred/reacは同じ昇格IDへ対応付け、一度だけ学習する。
- 窓: `TimestepTrainingSampleStore`は直近10時刻だけを保管し、指定窓を時刻昇順に連結する。学習直前に`clear_model_training_samples`で全IDを空にしてから、学習する役割の窓だけを入れる。既存のround学習と集約を再利用し、集約の重みはこのstoreの窓の件数。検出・client別のモデル作成・クラスタリング・統合の経路を通らない。
- 記録: 保存形式を版6に上げ、DriftSurfだけ`driftsurf`表を追加。時刻、遷移前後の状態、役割の正解率と最高値、学習役割・予測役割、反応のcounter、進入・昇格、役割のIDと窓の時刻列を保存する。未計測の正解率はNaN、未登録ID/counterは−1、窓はカンマ区切り。サーバによる役割の登録は既存repositoryの`registering_client_id=None`で表す（この受理と不正値の拒否を共通testで確認）。既存構成の表・列は変更していない。
- 登録の漏れを防ぐ更新: `CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`、構成の一覧と実旧照合から除く集合、掃引の計画の記録の期待値、CLIの引数の期待値、IFCA/Windowの末尾位置の期待値、保存形式の版の期待値、typed settingsのパラメータの一覧、依存境界へ反映した。
- 文書: `docs/experiments/experiment-sweep.md`の構成とδの掃引、`differences-from-feddrift.md`の公式との差、生成した`docs/reference/options.md`、`naming.md`を更新した。公式との比較、初期値、局所学習、逐次評価、窓の標本数、model_keyの変更、δの選び方、二値スコアの対応を明記した。

#### 確認済みの証拠（最終集計は下に追記）

- 公式を複写した参照との照合は12ケース×100時刻（安定、最高精度からの低下、stabとの差、昇格しない終了、予測役割の交代、反応の反復、窓10超過、固定seedの乱数、δ0.02/0.10を含む）。状態・train_keys・model_key・各窓・acc_best・counter・作り直し・昇格を照合。平均の丸めの境界testも通過。
- 新規の部品・設定・掃引・CLIのtestは23 passed（全体run 3件を除く）。ログ: `.tmp/driftsurf-unit-latest.log`。
- 共通の登録・通信・設定保存は60 passed（`.tmp/driftsurf-shared.log`）。旧FedDriftと回帰は4 passed（`.tmp/driftsurf-legacy.log`）。
- 基準版1〜16のWindowsのgoldenの数値・記録の照合206件は全て通過。goldenと構成の2ファイル全体は391 passed / 10 skipped / 99 errors（`.tmp/driftsurf-goldens.xml`）。errorsは権限によるもの（構成80件、goldenの生成・破損入力等の補助test 19件）。skipは実旧と照合しない構成の指定によるもので、成功扱いにしていない。
- 一時ディレクトリの権限エラーは`WinError 5`。既定の`AppData/Local/Temp/pytest-of-yshin`と、worktree内に指定した`.tmp/driftsurf-*-temp`の双方で発生した。権限・ACL・sandboxの設定は変更していない。実行できなかったtestを成功扱いにしていない。
- 保存は、既に存在する`.tmp`への小さい2時刻runのJSON/NPZ保存・再読込みを別途実行し、版6と全配列の一致を確認した（`driftsurf-artifact-smoke.json/.npz`）。これは固定3 runの代わりではない。
- Ruff check / format checkは通過。Pyrightの通常の`--pythonpath ../../venv/Scripts/python.exe`だけではtorch/numpyを解決できなかった。同じ`pyproject.toml`をextendsし、依存先`C:/Users/yshin/Research/FedSDA-experiment/venv/Lib/site-packages`をextraPathsへ明示した一時設定`.tmp/driftsurf-pyright-config.json`で、設定の範囲全体を検査して0 errors / 0 warnings（`.tmp/driftsurf-quality-pyright.log`）。検査項目は無効化していない。命名検査は`spec_checks.py names ... --base ecdcb08a`で新しい公開名の未登録なし。

#### 変更ファイル

- source（新規）: `methods/feddrift/driftsurf_state.py`、`learning/training/timestep_training_sample_store.py`、`runtime/driftsurf_timestep_preparation.py`（いずれも`src/federated_learning_experiments/`配下）。
- source（変更）: 同パッケージの`runtime/feddrift_client_stream_processing.py`、`feddrift_run_client.py`、`feddrift_run_client_settings.py`、`feddrift_run_participant_factory.py`、`feddrift_run_server.py`、`feddrift_run_record_tables.py`、`feddrift_run_artifact_files.py`、`sweep/experiment_configurations.py`、`experiment_sweep_plan.py`、`experiment_sweep_command.py`、`methods/fedsda/model_registration/global_model_repository.py`。
- schema: `federated_drift_experiment/experiment_spec/options.py`、`parameters.py`。
- test（新規）: `tests/refactoring/test_feddrift_driftsurf_configuration.py`。
- test（変更）: `tests/refactoring/behavior_golden_cases.py`、`test_experiment_sweep.py`、`test_fedsda_configuration_variants.py`、`test_fedsda_default_waited_registration_and_alarm_wait.py`、`test_single_run_dependency_boundaries.py`、`test_feddrift_ifca_window_configuration.py`、`test_feddrift_window_and_adaptive_fedavg_configuration.py`、`test_feddrift_experiment_run_artifacts.py`、`test_global_model_repository_and_communication_volume.py`、`tests/test_parameter_schema.py`。
- 文書・状態: `docs/experiments/experiment-sweep.md`、`differences-from-feddrift.md`、`docs/reference/options.md`、本specの`naming.md`・`tasks.md`・`spec.json`。


#### 固定3 runの最終結果

全て間隔50、seed 0、10 client、各client 2500標本。条件を変えていない。

| データ | δ | 概念の変化の数 | 反応への進入 | 昇格 | 正解率の照合（役割×時刻） | 窓・重みの照合（client×モデル×時刻） | 初回学習前のθ0の照合（client×モデル） |
|---|---:|---:|---:|---:|---:|---:|---:|
| SINE2 | 0.10 | 24 | 9 | 3 | 89 | 970 | 150 |
| MNIST4 | 0.10 | 22 | 9 | 0 | 90 | 1000 | 140 |
| SINE2 | 0.02 | 24 | 13 | 7 | 86 | 930 | 200 |

- 3 runは全てPASS。進入は合計31回、昇格は合計10回で、合算の受入条件を満たす。最終のログとJUnitは`.tmp/driftsurf-audit-<dataset>-<delta>.log/.xml`（SINE2 δ0.10は134.70秒、MNIST4は787.80秒、SINE2 δ0.02は125.03秒）。
- 各runの全50時刻で、記録した正解率を公式の参照へ入れた状態・学習する役割・予測する役割・全窓・最高正解率・counterが一致した。正解率は、学習前の全clientの同一パラメータを確認し、全clientのその時刻の標本を連結した1回の推論による正解数÷件数と完全一致した（二値は明記したクラス別スコアへの対応）。
- 学習前のstoreは役割の窓に含む標本だけで、集約後の重みはその窓の標本数の全clientでの合計と一致した。標本の観測中に新しい標本を学習storeへ追記していないことも確認した。直近10時刻を保管する部品と、実runの最後の10時刻の標本の入力・ラベルを元のstreamと照合した。
- 各モデルの初回学習の直前にθ0を確認した。作り直したstabを次の時刻以降に初めて学習する場合も含む。各標本の予測IDは直前の時刻に決めたmodel_keyの役割のIDと一致し、最初は0。各時刻の大域モデルとclientの実体は同じID集合で3個以下、検出・クラスタリング・統合の記録は0件。


#### 関係test・品質検査の最終集計

- 関係testの元の実行（FedDrift系の全ファイル、依存境界、掃引、既定の構成、Option/Parameter schema）は3,354 passed / 4 skipped / 106 errors / 4 failed、3,468件。長時間実行の収集中に読み込まれていた、修正前のIFCAの一覧の期待値2件・生成文書の期待値1件・typed settingsのパラメータの期待値1件がfailedとなった。これらは現行のtestで個別に再実行して全てPASSを確認した（IFCAは11 passed / 4 deselected、schemaは24 passed。新規部品と合わせたschemaの実行は46 passed、その後追加したCLI接続のtestを含む新規部品は23 passed）。元のコマンドの終了コードは0ではなく、失敗を成功へ読み替えていない。最終の状態は再実行との対応で確認した。
- 現行のtest名ごとに重複を除き、再実行した最新の判定を採った合計は**3,839 passed / 14 skipped / 205 errors、未解消のfailure 0、計4,058件**。集計用の`.tmp/driftsurf_result_summary.py`、関係testの`.tmp/driftsurf-related.xml`、その他の上記JUnitを参照。
- 権限で実行できない205件の内訳: `test_fedsda_configuration_variants.py` 80、goldenの生成・破損入力等の補助test 19、`test_feddrift_experiment_run_artifacts.py` 37、`test_feddrift_experiment_sweep.py` 16、`test_experiment_sweep.py` 53。全てのerrorがPermissionErrorであることをJUnitで確認。pytest終了時の一時ディレクトリの掃除でも同じWinError 5が出た。14 skipは、実旧と照合しない構成10件とLinux専用の4件。
- FedDrift・Oracle・IFCA・FedAvg+Window・Adaptive-FedAvgの部品・全体runのtestは、上記の権限エラーと再確認済みの期待値の修正を除き通過。基準版1〜16のWindowsの数値golden照合206件は全て通過した。
- 最終のRuff checkは成功、format checkは392 files already formatted、Pyrightは同じpyproject範囲の検査で0 errors / 0 warnings（依存先を明示した一時設定を使用）。`pip check`もNo broken requirements。`git diff --check`は成功。全pytestと独立した実装レビューは未実施で、依頼どおり主担当へ引き継ぐ。
- 版6の小さい保存・読込みの照合は、値・dtype・shapeが全て一致した。これは一時ディレクトリを作る既存の保存test 37件の実行を代替するものではなく、それらは未検証のまま残す。
- 残る確認は、権限制約のない環境での上記205件と全pytest、主担当の独立した実装レビュー。固定3 runの条件を変えての再実験、goldenの更新、権限の設定変更はしていない。

### 主担当の確認（2026-11-03）
- 実装レビュー（GPT-6 Luna、effort medium、対象は`ecdcb08a`の上の未commitの差分）: PASS（Major以上なし）。
- 全pytest（`tests`、Windows、12並列、2442秒、JUnit `../../venv/refactoring-tests/driftsurf-full-pytest.xml`）: 13,334 passed、9 skipped、失敗0。実装担当の環境で権限のため実行できなかったtestを含む。
- 共通の`methods/fedsda/model_registration/global_model_repository.py`の変更は、`registering_client_id`に`None`（サーバが作るモデル）を許すだけで、整数の値の検査と記録は変わらないことを差分で確かめた。
- δの調整を、ユーザーの判断で0.04刻みの5値（600 run）にした（design.mdの「調整」）。
