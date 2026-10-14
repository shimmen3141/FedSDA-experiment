# 命名と役割

## 新しい公開名

| 公開名 | 場所 | 役割 |
|---|---|---|
| `latest_alarm_boundary_position` | `ModelTrainingSampleCollection`の項目 | そのモデルの保存の列の、最新の区切りの位置（区切りより後の標本は、警報の後に同じモデルで続けると決まった後に足されたもの）。無ければNone |
| `count_model_training_samples` | `ModelTrainingSampleStore`のmethod | そのモデルの列の標本の数（列が無ければ0） |
| `mark_latest_alarm_boundary` | `ModelTrainingSampleStore`のmethod | 指定の位置を、そのモデルの最新の区切りとして記録する |
| `uniform_after_latest_alarm_boundary` | 抜き出し方の値 | 区切りより後から一様（足りなければ末尾のbatchの件数から。区切りが無ければ全部から） |
| `recency_decayed_before_latest_alarm_boundary` | 抜き出し方の値 | 区切りより後の重み1、区切りより前の重みをαで減衰 |
| `alarm`・`alarm-decay:<α>` | 掃引の軸の書き方 | 上の2つ |

## 既存の公開名で、役割が変わるもの

| 公開名 | これまで | これから |
|---|---|---|
| `ModelTrainingSampleStore`の`reassign_model_training_samples_id`・`remap_model_training_sample_collections`・`snapshot_ordered_model_training_samples` | 標本の列だけを扱う | 区切りも移す・消す・運ぶ |
| `resolve_alarm_change_interval`・`apply_post_alarm_candidate_validation_resolution` | 標本を吸収する | 今のモデルで続けるとき、吸収の後に区切りを付ける |
| `sample_training_batches_for_held_models` | 全履歴・直近K件・指数減衰 | 区切りを使う2つも |
| `parse_fedsda_training_sample_selection`・`describe_fedsda_training_sample_selection` | 3つの書き方 | 5つの書き方 |
