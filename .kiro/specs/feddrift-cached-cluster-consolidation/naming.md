# 公開名と役割

| 名前 | 役割・似た名前との差 |
| --- | --- |
| cached_model_cluster_consolidation | FedDriftの保存済みモデルを指定群で統合する数値計算。クラスタを決める判定やowner更新とは別 |
| FeddriftCachedClusterConsolidation | server側の統合計算結果。client自身の平均parameterは別に計算する |
| consolidate_feddrift_cached_model_clusters | 過去重みからserverのparameter/統計/隔離期限をまとめて計算する |
| average_feddrift_cached_cluster_parameters | 決定済みの群内重みを、呼出し側自身のsnapshotへ適用する。FedAvgやserver snapshot配布とは別 |
