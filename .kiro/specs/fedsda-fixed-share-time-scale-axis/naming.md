# 命名と役割

## 新しい公開名

| 公開名 | 場所 | 役割 |
|---|---|---|
| `--fedsda-fixed-share-time-scales` | 掃引のコマンドの引数 | FedSDAの構成の、Fixed-Shareの時間尺度T（混ぜる予測の重みの分け合いの割合1/Tも決める）を並べる軸 |
| `fedsda_fixed_share_time_scales` | `ExperimentSweepPlan`の項目 | 上の軸 |
| `fixed_share_time_scale` | 一覧の表の列 | FedSDAの行の、T |

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `--fedsda-pending-assignment-fifo-capacities`・`fedsda_pending_assignment_fifo_capacities` | 容量とTを同じ値に替える | 容量だけを替える |
| `FedsdaRunClientSettings`・`validate_experiment_run_settings` | 容量とTの同値を求める | 違う値も受理する |
| `build_planned_experiment_runs` | FedSDAの変種は容量の軸だけで作る | 容量の軸とTの軸の全部の組を作る |
| `SWEEP_RECORD_FORMAT_VERSION` | 6 | 7 |
