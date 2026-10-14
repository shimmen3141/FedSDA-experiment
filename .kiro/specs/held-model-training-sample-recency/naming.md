# 命名と役割

## 新しい公開名

| 公開名 | 場所 | 役割 |
|---|---|---|
| `training_sample_selection_policy` | `LocalTrainingSettings`の項目 | 保有モデルの学習batchの抜き出し方。`uniform_over_all_stored_samples`（全履歴から一様。既定）、`uniform_over_most_recent_stored_samples`（直近K件から一様）、`recency_decayed_stored_samples`（新しい標本ほど選ばれやすい指数減衰） |
| `most_recent_stored_sample_count` | 同上 | 直近K件のK |
| `stored_sample_recency_decay_factor` | 同上 | 指数減衰のα（0<α<1。末尾からa件前の標本の重みがαのa乗） |
| `fedsda_training_sample_selections` | `ExperimentSweepPlan`の項目 | FedSDAの構成について、抜き出し方を並べる掃引の軸（`uniform`・`recent:<K>`・`decay:<α>`） |
| `parse_fedsda_training_sample_selection` | `sweep/experiment_sweep_plan.py` | 軸の文字列を、設定の3つの値へ変換する |
| `--fedsda-training-sample-selections` | 掃引のコマンドの引数 | 上の軸 |
| `training_sample_selection` | 一覧の表の列 | FedSDAの行の、抜き出し方（軸と同じ書き方） |

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `sample_training_batches_for_held_models` | 全部の標本から一様に抜き出す | keyword引数`local_training_settings`（省略可）の選び方で抜き出す。省略か既定なら今と同じ |
| `perform_held_model_joint_training_iterations` | 設定を共同更新にだけ使う | 抜き出しにも渡す |
