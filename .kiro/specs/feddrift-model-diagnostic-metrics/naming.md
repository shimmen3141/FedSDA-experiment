# 公開名と役割
| 名前 | 役割・既存名との違い |
|---|---|
| feddrift_model_diagnostic_records | FedDrift同期・終了時学習量の不変記録。判断やownerではない |
| FeddriftSynchronizedTimestepModelCounts | R回の配布完了後の時刻単位のモデル数。標本ごとのheld countとは別 |
| FeddriftClientHeldModelTrainingAmount | 終了時のclient保有モデル1個の学習量。全client集計ではない |
| snapshot_synchronized_timestep_model_counts | サーバ所有の成功同期記録の読取り |
| feddrift_model_diagnostic_metric_derivation | FedDrift ownerから適用可能なモデル診断を導出 |
| FeddriftModelDiagnosticMetrics | FedDriftのモデル数/学習量/クラスタ診断。FedSDA固有のruntime型ではない |
| derive_feddrift_model_diagnostic_metrics | 読取り専用の診断導出。モデル学習/判定を実行しない |
