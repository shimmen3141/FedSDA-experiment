# 公開名と役割

| 公開名 | 役割と似た名前との差 |
|---|---|
| FeddriftDetectionBatchProcessor.held_model_count | 予測直前の実保有モデル数。評価標本のID集合やglobalモデル数と区別する。 |
| FeddriftRunServer.snapshot_server_computation_counts | サーバで実行したパラメータMACの不変共通snapshot。モデルhook計数や通信量と区別する。 |
| FeddriftRunServer.client_consolidation_parameter_multiply_accumulate_count | serverの進行中にclient自身が行った統合MAC。サーバ統合やclientのLinear計算と区別する。 |

既存derive_feddrift_run_metricsは記録からの読取り導出として再利用。新module/classは追加しない。記録field/引数/私有名/test名は対象外。
