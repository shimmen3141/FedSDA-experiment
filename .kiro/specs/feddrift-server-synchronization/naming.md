# 公開名と役割

| 名前 | 役割・似た名前との差 |
| --- | --- |
| feddrift_model_distribution_application | FedDriftのclient ownerへの正式モデル反映。FedSDAの共有部再接続とは別 |
| apply_feddrift_model_distribution | 検査・準備済みの配布をclientの独立ownerへ反映 |
| FeddriftServerSettings | サーバ進行の設定。clientの検出/学習設定とは別 |
| feddrift_server_settings | 上の型とその値の検査 |
| loss_distance_model_clustering | FedDriftの損失距離と指定linkageの計算 |
| cluster_feddrift_model_ids | 4損失統計と件数条件からモデルを分割する |
| feddrift_run_server | 時刻準備・集約配布・R回進行の接続 |
| FeddriftRunServer | サーバ状態/記録のownerと進行入口 |
| prepare_timestep | 観測済み時刻の成熟モデル統合→新規登録 |
| aggregate_and_distribute_training_round | 学習済みモデルの1回のFedAvg→全配布。学習を重複実行しない |
| synchronize_timestep | 準備→設定R回のclient学習/集約/配布 |
| snapshot_model_weights | 次の統合へ使う過去重みのコピー |
| snapshot_isolation_until | 新モデルの隔離期限のコピー |
| python_random_generator | clientが借用する同じrunのRandom |
| get_held_model_ids | 評価の保持者判定に使うモデル/統計/学習/現在IDの集合 |
| apply_server_model_distribution | 組立済みclientから上の配布適用へ接続 |
