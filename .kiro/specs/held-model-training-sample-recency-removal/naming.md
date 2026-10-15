# 命名と役割

## 新しい公開名

無し。

## 外す公開名

| 公開名 | 場所 |
|---|---|
| `training_sample_selection_policy`・`most_recent_stored_sample_count`・`stored_sample_recency_decay_factor` | `LocalTrainingSettings`の項目 |
| `fedsda_training_sample_selections` | `ExperimentSweepPlan`の項目 |
| `parse_fedsda_training_sample_selection`・`describe_fedsda_training_sample_selection` | `sweep/experiment_sweep_plan.py` |
| `--fedsda-training-sample-selections` | 掃引のコマンドの引数 |
| `training_sample_selection` | 一覧の表の列 |

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `sample_training_batches_for_held_models` | keyword引数`local_training_settings`の選び方で抜き出す | 選び方の引数を持たず、全部の標本から一様に抜き出す（追加の前と同じ） |
| `SWEEP_RECORD_FORMAT_VERSION` | 5 | 6 |
