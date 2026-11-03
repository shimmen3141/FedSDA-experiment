# 命名と役割

| 公開名 | 場所 | 役割 | 似た名前との違い |
|---|---|---|---|
| `--feddrift-local-training-round-counts` | 掃引のCLI | `feddrift-window`と`feddrift-cfl-window`の1時刻のround数 $R$ の値の列。既定1 | `local_training_round_count`は1 runの設定の項目、こちらは掃引の値 |
| `local_training_round_count` | `FeddriftServerSettings`（既存） | 1時刻に学習・集約・配布を繰り返す回数 | FedSDAのroundや`upload_delay_round_count`とは異なる |
| `round_index` | 記録表`cfl_split_check`の列 | その時刻の中のroundの番号（0から） | `timestep_index`は時刻の番号 |

（実装担当が、実際に使った設定の項目・内部の関数の名前を追記する）
