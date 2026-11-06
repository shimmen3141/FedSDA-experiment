# 集約時判断の研究変種

仕様の正本は [計画書](../../docs/research/fedsda-aggregation-time-decisions-v16.md)。
3変種と `product-default` は、同じ研究CLIで指定する。
製品のsource・設定・golden・run識別子は変更しない。

## 差し込みと製品の関数境界

`aggregation_time_decisions.py` のcontext managerは、正常終了・例外終了のどちらでも差し込みを復元する。

| 包む関数 | 役割 |
| --- | --- |
| `FedsdaRunClient.process_observed_sample` | clientの識別と最初の集約より前の一様重み・βの写し |
| `fedsda_run_client.process_observed_sample` の参照 | 標本処理の引数を保持し、ストア・学習・FIFOから出た位置を記録 |
| `observed_sample_processing.predict_observed_sample_and_update_prediction_weights` | 既定の内部更新を済ませ、計算済みの同期側・局所側の確率を写しの重みで再結合 |
| `observed_sample_prediction.combine_model_prediction_probabilities` と `SynchronizedAndLocalPredictionMixture.mix_prediction_probabilities` | 上の呼出しの間だけ局所側のモデル別確率を取得。新たなNN評価は行わない |
| `SamplePredictionRecordStore.append_sample_prediction_record` | 上の呼出しの間だけ記録を保留し、再結合した実際の予測の記録を1回追加 |
| `OverallAndTrueClassLossMonitor.observe_loss_after_label_observation` | 最初の警報を保留し、以後は検出器を呼ばずに同じe値・更新件数0の観測結果を返す |
| `observed_sample_processing._validate_observed_sample_processing_inputs` | 待機中の検出器にだけ、入口の連続性検査の間、仮の直前の観測位置を提示する |
| `FedsdaRunClient.flush_pending_local_updates` | 登録可能の確認とサーバへの送信より前に、その時点のFIFOで選択・応答 |
| `observed_sample_processing.select_alarm_interval_assignment` | 選択位置・FIFOの位置・選択結果を記録 |
| `FedsdaRunClient.apply_global_model_distribution` | 配布のモデル集合を記録。「再較正しない」設定では戻った直後に重みを写す |
| `FedsdaRunClient.recalibrate_prediction_state_after_aggregation` | 再較正から戻った直後に重み・βを1回写す |
| `fedsda_run_server.synchronize_models_in_server_round` の参照 | 集約位置、登録結果、累積通信量を記録 |
| `alarm_interval_candidate_training.train_candidate_classifier_epochs` | 候補のepoch・更新回数・学習件数を記録 |
| `held_model_training_request_handling.perform_held_model_joint_training_iterations` | 学習の試行回数を記録 |
| `FedsdaRunClient.__init__` | 製品の学習量診断ストアだけを有効にする。責任度providerは付けない |

警報のその場でのMDL選択と `handle_alarm_occurrence` は、
`observed_sample_processing.process_observed_sample` の中で別々に呼ばれる。
後者だけを抑止しても先にMDLが実行されるため、監視結果の入口で抑止する。
集約直前には同じ `select_alarm_interval_assignment` と `handle_alarm_occurrence` を呼ぶ。

関数境界への対応:

- 製品の入口は監視位置の連続性を必須にする。待機中はこの検査だけを上記の方法で通す。
  実際の検出器の状態は変更せず、FIFOと予測記録の連続性の検査は残す。
- `handle_alarm_occurrence` は、警報位置がFIFOの最後の観測位置であることを要求する。
  応答・適応記録・MDL選択記録では集約区間の最後の位置を渡し、
  最初の警報位置は警報記録と診断の `alarms`・`pending_alarm`・`responses.alarm` に分けて残す。
- 製品では、1標本1回は学習の**試行**であり、batch件数が足りないストアでは更新を完了しない。
  既定の規則を保ち、診断の `training_attempts` と `joint_updates` を区別する。
- 最初の予測前や、空のFIFOで再較正した直後は、内部のFixed-Share辞書が未初期化の場合がある。
  その場合の写しは、製品が次の予測の入口で作るのと同じ、現在の保有集合上の一様重み。
- 5件未満のFIFOでは、既定の `wait_until_change_interval_reaches_minimum` に従い、
  選択・応答・検出器のresetを行わず、最初の警報を次の集約まで保持する。
  空の場合は、計画書の明記どおり、選択せず待機を消して検出器をresetする。

## 実際の経路で起きない境界

- 集約の間の保有モデルの削除はない。`HeldModelTrainingStateRegistry` の追加以外の操作は、
  `held_model_registration_confirmation.confirm_held_model_registration` のID付替えと、
  `global_model_distribution_application.apply_global_model_distribution` の置換。
  いずれも `server_round_synchronization.synchronize_models_in_server_round` の同期処理中である。
  新モデルの局所採用は追加であり、以前の保有モデルを削除しない。
  それでも、削除時の正規化は固定入力で検査する。
- 待機中の空のFIFOは通常の経路では起きない。各標本はFIFOへ追加され、容量超過の解放後も
  最新標本は残る。待機中は応答を行わず、保留の消費は集約直前の応答だけ。
  根拠は `observed_sample_processing.process_observed_sample` の追加・非警報分岐と
  `PendingTrainingAssignmentBuffer.release_sample_indices_exceeding_capacity`。
  空の場合のresetは固定入力で検査する。
- 計画書の間隔50・200・500では、待機する変種の集約直前に5件未満のFIFOは起きない。
  前回の集約で空になっても、その後に少なくとも50件追加され、容量30件が残る。
  間隔4の固定入力runで、5件未満なら待ち、次の境界で選ぶことを検査する。

## 診断

3変種を含むCLI実行では、`product-default` も含め `results.json` の各runの
`aggregation_time_diagnostics` に以下を保存する。
既存検査・既存変種との出力互換性を保つため、単独の `product-default` にはこの欄を追加しない。
Pythonから直接対照runを呼ぶ検査では、jobに `record_aggregation_time_diagnostics=True` を渡す。

既定ではrunの要約とイベントだけを保存する。検査用の
`--aggregation-time-sample-diagnostics`（既定: 無効）を指定したときだけ、
`samples`・`detector_waits`・`snapshots` の履歴を収集・保存する。
Pythonからはjobに `aggregation_time_sample_diagnostics=True` を渡す。
診断の `schema_version=2`、`sample_diagnostics_enabled` で収集方式を確認できる。
重みの現在の写しは、履歴の保存を無効にしても予測のために内部で保持する。

- `samples`（検査用）: client・標本位置・予測に使ったID別重みとβ・保留警報位置・現行モデル・
  FIFOからストアへ出た標本位置・各ストアの件数・モデル別累積学習件数と更新件数・
  局所学習の試行/完了件数・候補学習の累積回数。
  `internal_weights_after`・`internal_beta_after` に、その標本で更新した後の内部値も保存する。
- `snapshots`（検査用）: 初期/配布後/再較正後の重み・βと、その直前の標本位置。
- `alarms`、`detector_waits`（後者だけ検査用）: 最初の警報位置と検出器状態のSHA-256。
  待機中には実際の状態を比較し、変化していないことも確認する。
- `selections`、`responses`: 集約直前の選択・応答位置、元の警報位置、FIFOの位置、結果。
- `creations`、`synchronizations.registered`、`distributions`: 作成・登録・配布の時点とID。
- `candidate_training`: 候補の学習の内訳。MDL探索の2学習者の計算は製品の計算量指標に含む。
- 警報/選択/採用回数、選択の3種類の内訳、最終の保有モデル数、最終のモデル別学習量、
  上下通信量・転送モデル数・通知数、モデル計算量/MAC。
- 要約のキーは `alarm_count`、`selection_count`、`selection_outcomes`
  （`current_model`・`held_model`・`new_model`）、`new_model_adoption_count`、
  `final_held_model_counts`、`final_training_counts`（clientごと・モデル別の学習件数と更新件数）、
  `training_attempt_count_by_client`（局所学習の試行回数）、
  `detector_wait_sample_count_by_client`（検出器を止めた標本数。待機があったclientだけ）、
  `communication_volume`、`model_computation_counts`。
  候補学習の回数と内訳は警報に伴う `candidate_training` の各記録に残す。
- `training_mass_by_client_model_concept` と `training_mass_purity` は
  `fedsda-training-mass-diagnostics` と同じストア・関数を使う。
  後者は（clientの純度の等重み平均、全部を1モデルへ入れた場合の基準の平均、分母0で除外したclient数）。
  分母0を0%に置換しない。

実行時間は従来どおり各runの `seconds`。診断の収集・Python側での再結合の時間も含む。
`model_computation_counts` は製品のNN計算の指標であり、診断処理の所要時間とは区別する。

## 検査と本規模の実行

```powershell
python -m pytest tools/research_provisional_candidate_expert/checks_aggregation_time_decisions.py -q -p no:cacheprovider
```

`-m 'not slow'` で固定入力・短いrun・CLI検査を選べる。
標本単位の照合は検査側で詳細診断を明示的に有効にする。
追加の4変種の検査では引数を省いたrunも実行し、標本履歴が無いこと、
要約・予測結果・通信量・MACが詳細診断ありのrunと一致することを確認する。
`slow` は SINE2/MNIST4×間隔50/200×seed0×10 client×1000標本の既定・予測・割当の照合、
SINE2・間隔200の両方の判断、診断を無効にした製品経路との照合。
10データの同条件の保存済み結果との照合は、基準版16の研究結果JSONを
`FDE_AGGREGATION_TIME_REFERENCE_RESULTS` で指定して実行する。
未指定なら `test_product_default_matches_saved_baseline16` の20条件だけをskipする。
指定したファイルに同条件が無い、複数ある、主要な値が異なる場合は失敗にする。

本規模は800 run。workersは同じサーバで実行中の感度実験と合わせて割り当てる。
以下は4 workersを割り当てる例。出力先は未使用の名前を指定する。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m tools.research_provisional_candidate_expert.run \
  --output results/fedsda-aggregation-time-decisions-v16/20261107_full \
  --datasets sine2 circle2 sea2 sea4 mnist2 mnist4 fmnist2 fmnist4 rotmnist2 rotmnist4 \
  --intervals 50 500 --seeds 0 1 2 3 4 5 6 7 8 9 --sample-count 5000 --workers 4 \
  --variants product-default aggregation-time-prediction aggregation-time-assignment aggregation-time-decisions
```

開始前には研究CLIの `--print-plan` を内部確認に使う。再開は同じ引数に `--resume` を足す。
研究CLIは製品の掃引と異なり、manifestによる重複排除を行わない。
本規模は未実施であり、精度差の結論はまだ出していない。

## 手元での検査記録（2026-11-07）

- 小さいrun一式と初期の境界検査: **18 passed / 0 failed**。
  SINE2/MNIST4×間隔50/200の3変種、SINE2・間隔200の両判断、対照と製品経路の一致を含む。
- 最終コードの固定入力・短いrun・出力互換性とSINE2・間隔50の標本単位照合:
  **15 passed / 0 failed**。上の18件とは重複があるので合算しない。
  ここでは更新後の内部重み・βの照合も行った。
- 既存の `checks_candidate_start_diagnostics.py::test_small_run_exact_product_trajectory[sine2]`:
  修正後 **1 passed / 0 failed**。
  初回の既存検査の一括実行は、単独の対照に診断欄を常時追加したことによる互換性の失敗を検出して中断した。
  対照の診断を比較実行だけに限定し、既存検査自体は変更せず通した。
- 既存 `checks_*.py` の範囲を限定した再実行:
  **150 passed / 0 failed / 18 setup errors / 26 deselected**。
  全8ファイルを対象とし、`-m 'not slow'` と
  `-k 'not small_run and not small_runs and not parallel_workers and not gzip and not sweep_records'` を指定。
  既存の全検査が通過したという意味ではない。除外した長いrun・保存検査は未実施。
  18件はすべてpytestの `tmp_path` 準備時に、
  `AppData/Local/Temp/pytest-of-yshin` へのアクセスが拒否されたもの。
- `tests/test_regression.py`: **1 passed / 0 failed**。
- `tests/test_proposed_regression.py`: **0 passed / 1 failed**。
  製品値の不一致ではなく、`tempfile.TemporaryDirectory` 内へのNPZ保存・後片付けが
  `PermissionError: [WinError 5]` で止まった。sandbox外で再実行が必要。
- `test_fedsda_every_sample_change_interval_search_default.py` と
  `test_single_run_dependency_boundaries.py`: **2,291 passed / 0 failed**。
- `test_four_variants_share_one_cli_execution`: **setup error 1件**。
  pytestの一時ディレクトリへのアクセス拒否。
  pytest外でも同じ4変種の研究CLIを試したが、`ProcessPoolExecutor` の `mp.Pipe` 内の
  `_winapi.CreateFile` がアクセス拒否になった。sandbox外で再実行が必要。
  4変種の計画作成と、同一プロセスでの4 jobの連続実行は上の15件で通った。
- `test_product_default_matches_saved_baseline16`: **20 skipped**。
  同条件の保存済み基準版16の結果JSONが未指定。環境変数の指定後に実行する。
- Ruff（新しいPython 2ファイルのlintとformat）と `git diff --check`: 通過。
- 本規模の計画は **800 job、出力先未作成** を確認。本規模実験・全基準版goldenの一括実行は未実施。

既存検査でsetupできなかった名前（パラメータをまとめて記載）:

| ファイル | 検査名 | 件数 |
| --- | --- | ---: |
| `checks_change_interval_search_default.py` | `test_rebaseline_entry_passes_group_b_or_d_settings`（product-defaultと5 rebaseline変種） | 6 |
| `checks_loss_monitor_rebaseline.py` | `test_execute_run_with_variant_enables_policy_and_writes_diagnostics_only_when_completed`（completed/failed） | 2 |
| 同上 | `test_run_calls_product_sweep_per_variant_with_patch_and_restores` | 1 |
| 同上 | `test_run_sets_evaluating_client_limit_only_for_combined_variant`（combined/decrease） | 2 |
| 同上 | `test_run_rejects_existing_output_without_variant_record` | 1 |
| 同上 | `test_variant_record_is_written_before_product_runs` | 1 |
| 同上 | `test_metrics_reject_conflicting_diagnostics_for_the_same_run` | 1 |
| 同上 | `test_run_rejects_other_variant_record_before_calling_product` | 1 |
| 同上 | `test_artifact_comparison_includes_all_arrays_and_non_time_summary_items` | 1 |
| 同上 | `test_artifact_comparison_finds_expected_runs_by_name_under_saved_results` | 1 |
| 同上 | `test_artifact_comparison_ignores_code_commit_but_not_baseline_version` | 1 |

検査のJUnitと独立CLIの応答は、作業ツリーの `.tmp/aggregation-*` に残した。
主な記録は `aggregation-tests-all.xml`、`aggregation-final-focused.xml`、
`aggregation-existing-focused.xml`、`aggregation-existing-recheck.xml`、
`aggregation-product-boundaries.xml`、`aggregation-regression.xml`。

## 標本診断の出力制限の検証（2026-11-07、追加修正）

- `checks_aggregation_time_decisions.py` の既存44件は削除せず、4件を追加して48件を一括実行。
  JUnitは **27 passed / 0 failed / 20 skipped / 1 setup error**
  （`.tmp/aggregation-summary-tests.xml`）。SINE2/MNIST4×間隔50/200の照合、
  両判断の照合、診断の有無と製品経路の一致も通過。
  20件は `FDE_AGGREGATION_TIME_REFERENCE_RESULTS` 未指定による従来のskip。
  `test_four_variants_share_one_cli_execution` は `tmp_path` の準備中に
  `.tmp/aggregation-summary-pytest` の走査が `PermissionError: [WinError 5]` で失敗し、
  CLI本体は実行されなかった。さらに同じディレクトリの後片付けでもアクセス拒否になり、
  pytestプロセスは終了コード1。全体が正常終了した、全48件が通過したとは扱わない。
- 4変種それぞれで、詳細診断あり・なしを同じ固定入力で実行:
  **4 passed / 0 failed**（`.tmp/aggregation-summary-final-focused.xml`）。
  詳細なしで `samples`・`detector_waits`・`snapshots` が存在せず、
  要約と時間以外の主要なrun結果が詳細ありと一致する。
- 既存の `tests/test_regression.py` と
  `checks_candidate_start_diagnostics.py::test_small_run_exact_product_trajectory`
  （SINE2・MNIST4・ROTMNIST2）: **4 passed / 0 failed**。
  `.tmp/aggregation-summary-existing.xml` に記録。
- 本規模条件の1 runを最終コードで実測: `aggregation-time-decisions`、SINE2、
  間隔50、seed 0、10 client×5000標本、標本診断なし。
  CLIと同じ `json.dumps([result], indent=1)` をWindowsの `Path.write_text` で保存した
  `results.json` は **247,707 bytes（約241.9 KiB）**。
  `.tmp/aggregation-summary-size-final/results.json` に保存。
  警報70件・選択70件で、標本履歴3種類は存在しない。
  所要時間544.79秒、精度0.95988はこの出力サイズ確認runの値であり、
  本規模800 runの比較実験の結果ではない。
  並列用IPCが制限される環境なので、計画確認後に `_execute_job` を同一プロセスで呼び、
  CLIと同じ形式で保存した。CLIの並列実行を検証したという意味ではない。
- Ruffは対象のPython 2ファイルを `--isolated` でlint・format検査し、通過。
  通常のプロジェクト設定では `tools/` が除外されるので、通常設定での検査成功とは扱わない。
- 今回の追加修正の独立CLIレビューは未実施。
  日常的な出力制限の修正としてLuna・effort mediumを指定したが、
  app-server初期化のOS error 5でsessionを開始できなかった。
  代替のHaiku 5.5・effort mediumも `ECONNREFUSED` で失敗。
  session `7bc08708-44e4-4f88-a2b2-724a6bf6d721`、`is_error=true`、`modelUsage={}`。
  `.tmp/aggregation-summary-review-luna.log`・`aggregation-summary-review-haiku.json` に記録。
  前回の3変種の承認を今回の修正の承認へ流用せず、今回のレビュー待ちとして残す。

## 独立レビューの状態（2026-11-07）

未承認。自己レビューでは代替していない。

- 可変状態とNN数値の更新順序を含むため、Haiku 5.5・effort mediumを選択。
  独立CLIはAPI接続拒否（ECONNREFUSED）で失敗。session `5b242836-3bb4-4057-84ae-7a5f7508a9bc`、
  `is_error=true`、`modelUsage={}` で、モデルによるレビューは行われていない。
- 代替のGPT-6 Luna・effort mediumも、独立CLIのapp-server初期化がOS error 5（アクセス拒否）で失敗。
  レビューsessionは開始していない。sandbox解除・認証/ユーザー設定の変更は行っていない。
- 対象は基点 `7d8bb386e0ff2c28fcc432da9101a77f0a48376a` からの未コミット差分。
  commit・push・stashは行わず、差分を作業ツリーへ残す。
