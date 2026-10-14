# Design Document

## Overview
構成の一覧（`EXPERIMENT_CONFIGURATION_NAMES`）へ`fedavg-local-updates-per-aggregation`を足し、`build_experiment_run_settings`で、`fedavg`の既定のrun設定の`local_update_interval_per_client_samples`だけを、実行条件の集約間隔へ置き換える。FedAvgのclientは、τ>1のとき、保留の件数がτに達した標本の処理の終わりで、保留の件数×1標本あたりの更新回数だけまとめて更新する（既存の処理。`methods/fedavg/fedavg_client.py`の`process_observed_sample`と`flush_pending_local_updates`）。τ=集約間隔なら、roundの最後の標本の`process_observed_sample`の中で（予測の後、同期の前に）まとめて更新され、roundの終わりの`flush_pending_local_updates`（実行の枠が毎round、標本の処理の後・同期の前に呼ぶ）は、保留0で何もしない。roundの中の予測は、全部、そのroundの始めに配布されたモデルで行われる。更新は、累積の保持標本数（`training_samples`）がbatchの件数以上のときだけ行い、足りないときは保留を学習せずに0へ戻す（既存の挙動。変えない。集約間隔がbatchの件数より小さいと、最初の数roundの更新が捨てられる。掃引の例の50以上では起きない）。集約間隔で割り切れない末尾の標本は、実行の枠が処理しない（観測も保留もされない。goldenの`fedavg-no-complete-interval`が守る既存の挙動）。

## Boundary Commitments
### This Spec Owns
- `sweep/experiment_configurations.py`: 構成名の定数と一覧、run設定の組み立て。
- `tests/refactoring/behavior_golden_cases.py`: 基準版1〜5の事例の一覧を、構成の一覧から作らず、今の13構成の名前で固定する。golden以外のtestで守る構成の一覧（構成名→testのファイル）を持つ。
- `tests/refactoring/test_important_behavior_regression_goldens.py`: 事例の一覧の検査を、上の形へ合わせる。
- 構成の一覧の全部を回して`golden.build_case_settings`を呼ぶ2つのtest（`tests/refactoring/test_fedsda_default_waited_registration_and_alarm_wait.py`の`test_every_fedsda_configuration_uses_waited_registration_and_alarm_wait_by_default`、`tests/refactoring/test_too_short_alarm_earlier_sample_release.py`の`test_every_fedsda_configuration_defaults_to_releasing_earlier_samples`）: goldenの事例にない名前で`ValueError`になるので、golden側の13構成（`BASELINE_CONFIGURATION_CASE_NAMES`）で回す。どちらもFedSDAの構成だけを数えるので、件数の期待は変わらない。
- `tests/refactoring/test_fedsda_configuration_variants.py`の`test_configuration_names_build_run_settings_that_differ_only_in_the_changed_element`: 構成の一覧の完全なtupleへ新しい名前を足し、`CONFIGURATION_VARIANTS`（実旧と照合するFedSDAの構成）と比べる除外へ新しい名前を足す（旧実装に対応する構成が無いため）。
- 新しいtest: `tests/refactoring/test_fedavg_local_updates_per_aggregation.py`。
- 文書: `docs/experiments/experiment-sweep.md`、研究バックログの8、再開案内。
### Out of Boundary
FedAvgのclient・サーバ・保存、FedSDA・FedDrift、掃引の計画の軸、一覧の表の列、`BEHAVIOR_BASELINE_VERSION`。
### Allowed Dependencies
`experiment_configurations.py`は、既にimportしている`replace`・`FedavgRunSettings`・`build_default_fedavg_run_settings`・`ExperimentRunConditions`だけを使う。依存境界のtestの許可の一覧は、変える必要がない見込み（変わる場合は、使う名前だけを足す）。
### Revalidation Triggers
構成の一覧（goldenの事例、testの並列の組分け——`tests/conftest.py`は構成名の一覧の位置で組を決めるので、末尾に足せば既存の組は変わらない）。

## Architecture
### 構成
- 定数`FEDAVG_LOCAL_UPDATES_PER_AGGREGATION_CONFIGURATION_NAME = "fedavg-local-updates-per-aggregation"`。一覧の末尾（`"fedavg"`の後）に足す。
- `build_experiment_run_settings`: `"fedavg"`の分岐の後に、この構成の分岐を置く。`build_default_fedavg_run_settings`の結果に`replace`を2段（run設定→`run_participant_settings`）かけ、`local_update_interval_per_client_samples=experiment_run_conditions.server_aggregation_interval_per_client_samples`とする。設定の検査はもう一度行われる（τ≥1）。
- 実行・保存・一覧の表は、run設定の型（`FedavgRunSettings`）で振り分けているので変えない。要約の`method_name`は`fedavg`のまま。一覧の表（`run_summary.csv`）は構成のディレクトリごとに分かれるので、今の`fedavg`と同じ表には入らない。τの列は足さない（完全な値はrunの要約のrun設定にある）。

### golden
- 今、`_VERSION_1_CASE_NAMES`は`*EXPERIMENT_CONFIGURATION_NAMES`から始まるので、構成を足すと、基準版1〜5の事例が増え、保存済みのgolden（読込みの`load_golden`が、事例の集合を一覧と厳密に比べる）と合わなくなる。goldenを作る経路（`generate_golden`。既存のファイルは上書きせず、新しいファイルだけを作る）に、既存の版を更新する経路は無いので、一覧を固定すれば、読込み・比較・生成のどれとも矛盾しない。これを、今の13構成の名前を並べた定数`BASELINE_CONFIGURATION_CASE_NAMES`へ置き換える（版1〜5の事例の一覧は、値も順も今と同じ）。
- `CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`（構成名→専用のtestのファイル）を置く。task 1.1では空の辞書にし（構成の一覧は13個のままなので、下の(b)の集合の一致が通る）、task 1.2で、構成の一覧へ名前を足すのと同じ変更で、`"fedavg-local-updates-per-aggregation": "tests/refactoring/test_fedavg_local_updates_per_aggregation.py"`を入れる。
- `test_case_registry_preserves_all_registered_configurations`は、`golden.CASE_NAMES[:13] == EXPERIMENT_CONFIGURATION_NAMES`の代わりに、(a) `CASE_NAMES[:13]`が`BASELINE_CONFIGURATION_CASE_NAMES`と同じ、(b) 構成の一覧の集合が、golden側の13構成と`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`の鍵の和集合と等しく、両者が交わらない、(c) 一覧のtestのファイルが在る、を確かめる。今の一覧の先頭13個が`BASELINE_CONFIGURATION_CASE_NAMES`と同じ順であることも確かめる（既存の構成の位置を変えていない）。
- 新しい構成をgoldenの事例にしない理由: 新しい基準版を作るか、既存の版のgoldenへ事例を足す（WindowsとLinuxの両方で作り直す）必要があり、変えるのは既にgoldenで守られているFedAvgの処理の、τの値だけだから。代わりに、下の専用のtestで、τ>1の経路の振る舞いを確かめる。

## Validation Design
`tests/refactoring/test_fedavg_local_updates_per_aggregation.py`:
1. 設定: 集約間隔の2つの値（例: 3と50）で、新しい構成のrun設定を、保存の形（`convert_settings_to_plain_mapping`）で`fedavg`のものと比べ、違うのが`run_participant_settings.local_update_interval_per_client_samples`だけで、その値が集約間隔であること。
2. 振る舞い（小さい条件。sine2、client 2、集約間隔はbatchの件数32以上——例40——、標本は3 round分）: サーバの同期の後の`global_parameters`を記録し（同期の関数を包む。testの中だけ）、各clientの各roundの全標本の予測（記録の`predicted_class_id`）が、そのroundの始めの配布モデル（round 0は事前学習のモデル）で、その標本の入力を予測した結果と一致すること。各clientの更新の回数（`parameter_update_step_count`）が、roundの最後の標本の処理の時点で累積の保持標本数がbatchの件数以上だったroundの数×集約間隔×1標本あたりの更新回数であること。
3. 検出力: 同じ条件の`fedavg`（τ=1）では、2.の予測の一致が成り立たないこと（少なくとも1標本で違う）。
4. コマンド: `--configurations fedavg-local-updates-per-aggregation`の小さい掃引で、`<出力先>/fedavg-local-updates-per-aggregation/runs/`に保存され、`run_summary.csv`の構成名の列が新しい名前になること。

既存のtest: goldenの版1〜5、`test_important_behavior_regression_goldens.py`、構成の一覧を使う3つのtest（上の「This Spec Owns」）、`test_experiment_sweep.py`、依存境界、全pytest。

## 旧と違う点
旧の実験では、Oblivious（FedAvg）のτは1だけで実行していた。新しい構成は、旧実装に対応する構成を持たない（旧との照合はしない）。
