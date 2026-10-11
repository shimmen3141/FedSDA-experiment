# 命名表: experiment-sweep-command

sourceの公開する名前（package、module、class、公開の関数）と、コマンド・出力の名前を載せる。確認は、全taskの後の独立レビューで受ける。構成の名前の案（design.mdの表）は、ユーザーの確認を受ける。

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `sweep` | `src/federated_learning_experiments/` | 掃引（複数のrunの計画・実行・記録）とコマンド |
| `experiment_configurations` / `PROPOSED_FEDSDA_CONFIGURATION_NAME` / `EXPERIMENT_CONFIGURATION_NAMES` / `build_experiment_run_settings` | `sweep/` | 実験の構成の名前と、完全なrun設定の対応 |
| `experiment_sweep_plan` / `ExperimentSweepPlan` / `PlannedExperimentRun` / `build_planned_experiment_runs` / `make_run_identifier` | 同上 | 掃引の計画と、runの識別子 |
| `experiment_sweep_execution` / `ExperimentRunOutcome` / `ExistingRunState` / `find_existing_run_state` / `execute_planned_experiment_run` / `execute_experiment_sweep` | 同上 | 逐次・並列の実行と、再開 |
| `experiment_sweep_records` / `write_run_summary_table` / `describe_code_revision` / `write_sweep_record` | 同上 | 一覧の表と、実行の記録 |
| `experiment_sweep_command` / `main` | 同上 | コマンド |
| `run_experiment_sweep.py` | リポジトリ直下 | コマンドの入口 |

## 判断が必要な点

- 「sweep（掃引）」は、条件の組合せを、まとめて実行すること（旧の`run_pareto_sweep.py`・`run_server_sweep.sh`の語）。
- 「configuration（構成）」は、手法と、その選択肢の組（提案構成、ablationの各構成、比較手法）。実行条件（dataset、seed、規模、集約間隔）とは別。
- 「run identifier」は、1 runのファイルの名前になる文字列。
- 「sweep record」は、1回のコマンドの実行の記録（旧の「manifest」）。
