# Naming

sourceの公開名だけを対象とする。既存名の意味は維持する。

| 名前 | 役割・区別 |
| --- | --- |
| feddrift_default_run_settings | FedDriftの既定完全条件。実行器や参加者生成とは別。 |
| build_default_feddrift_run_settings | dataset・規模・seedから既定完全条件を作る。FedSDAのfinal configuration builderとは手法が異なる。 |
| experiment_configurations | 既存の構成選択。feddriftを追加。 |
| build_experiment_run_settings | 構成名から手法別の完全設定を返す既存入口。 |
| experiment_sweep_plan | 既存の掃引計画と識別子。 |
| ExperimentSweepPlan | 手法に適用される値だけを展開する計画。 |
| PlannedExperimentRun | 完全設定と構成・識別子の既存束。 |
| make_run_identifier | 構成と完全設定から既存規則の識別子を作る。 |
| build_planned_experiment_runs | 全条件を検証して順序を持つrunへ展開する。 |
| experiment_sweep_execution | 計画実行と再開の既存境界。 |
| find_existing_run_state | 計画の手法/条件に一致する既存成果物を確認。 |
| execute_planned_experiment_run | 計画の手法で単一runを実行・保存。 |
| execute_experiment_sweep | 既存の逐次/spawnによる掃引実行。 |
| experiment_sweep_command | 既存のCLI引数と実行接続。 |
| main | 既存CLIの終了コードを返す。 |
| experiment_sweep_records | 既存のCSV/実行記録。 |
| write_run_summary_table | 構成別の全指標とFedDrift条件を一覧化。 |
