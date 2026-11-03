# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `--feddrift-local-training-round-counts` | 掃引のCLI | `feddrift-window`と`feddrift-cfl-window`の1時刻のround数 $R$ の値の列。既定1 | `local_training_round_count`は1 runの設定の項目、こちらは掃引の値 |
| `local_training_round_count` | `FeddriftServerSettings`（既存） | 1時刻に学習・集約・配布を繰り返す回数 | FedSDAのroundや`upload_delay_round_count`とは異なる |
| `round_index` | 記録表`cfl_split_check`の列 | その時刻の中のroundの番号（0から） | `timestep_index`は時刻の番号 |
| `feddrift_local_training_round_counts` | `ExperimentSweepPlan` | 掃引の正の整数の列。`None`は既定の1だけ | 単一runの項目とは複数形で区別する |
| `feddrift_local_training_round_count` | OptionSpec・ParameterSpecのID | 新APIの`server_settings.local_training_round_count`に対応する宣言 | 既存のserverの項目を改名せず、旧設定にも追加しない |
| `FEDDRIFT_LOCAL_TRAINING_ROUND_COUNT_AXIS` | `experiment_spec/sweep.py` | 新APIのround数のSweepAxis宣言（既定の値は`(1,)`） | 旧SweepPlanの実行軸には混ぜず、新APIの型付き計画で展開する |
| `_replace_feddrift_training_round_count` | `experiment_sweep_plan.py` | 窓構成のserver設定のround数を差し替え、参加者の検査を再実行する | clientの設定を替える`_replace_client_settings`とは対象が違う |
| `_current_timestep_training_model_ids`（既存） | `FeddriftRunServer` | そのroundの直前の割り当てから学習・集約の対象集合を作る | 通知を数える操作とは分離する |
| `_current_training_round_index` | `FeddriftRunServer` | 現在実行中の時刻内のround番号 | `timestep_index`と区別し、CFLの判定の記録だけで使用する |
| `_cfl_record_round_indices` | `FeddriftRunServer` | `_cfl_records`の各判定行と同じ順のround番号 | 既存の公式照合用の判定tupleを変更せず、保存時に`round_index`列を加える |
| `split_cfl_training_round`（既存） | `runtime/cfl_training_round.py` | 学習後・集約前の判定、分割時の集約省略とθ0配布、割り当て変更・標本移動 | 分割の数値規則を所有する`CflClusterSplittingState.check`とは区別する |
| `reassign_model_training_samples_id`（既存） | `ModelTrainingSampleStore` | 元IDの標本列を新IDへ順序を保って移す | 累積の学習件数・step数・帰属診断の記録は移さない |
