# 公開名と役割

| 公開名 | 役割と似た名前との差 |
|---|---|
| runtime/feddrift_stream_run_execution.py | FedDriftの設定から全体観測・同期・終端までの実行接続。 |
| FeddriftStreamProtocolExecutionSettings | FedDrift固有の実行順を持つrunと概念スケジュールの設定。FedSDA逐次strategyとは別。 |
| execute_feddrift_stream_protocol_run | 準備から終端までの全体run入口。通常同期/終端単体とは別。 |
| FeddriftRunParticipantFactory.validate_run_conditions | 参加者を作らずrunと検出chunkの対応を検査する。 |
| runtime/feddrift_run_metric_derivation.py | FedDriftの記録から共通定義の基本指標を導く。 |
| FeddriftRunMetrics | FedDriftの現在移植済みのrun指標。FedSDA固有項目を持たない。 |
| derive_feddrift_run_metrics | 数値計算部へ記録を対応付ける読取り専用の導出入口。 |
