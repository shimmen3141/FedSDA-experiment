# 命名と役割

## 新しい公開名

| 公開名 | 場所 | 役割 |
|---|---|---|
| `--fedsda-pending-assignment-fifo-capacities` | 掃引のコマンドの引数 | FedSDAの構成の、帰属の保留（FIFO）の容量を並べる軸（ユーザーが選んだ名前）。Fixed-Shareの時間尺度も同じ値で替える |
| `fedsda_pending_assignment_fifo_capacities` | `ExperimentSweepPlan`の項目 | 上の軸 |
| `pending_assignment_fifo_capacity` | 一覧の表の列 | FedSDAの行の、帰属の保留の容量 |

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `build_planned_experiment_runs` | FedSDAの変種は、抜き出し方の軸だけで作る | 抜き出し方の軸と容量の軸の全部の組を作る |
| `SWEEP_RECORD_FORMAT_VERSION` | 4 | 5 |
