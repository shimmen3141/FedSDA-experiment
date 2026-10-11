# 命名表: feddrift-detection-batch-processing

| 名前 | 場所 | 役割・区別 |
|---|---|---|
| `feddrift_detection_batch_processing` | runtime | 数値判定後の状態ownerへの接続。数値判定moduleと区別 |
| `FeddriftDetectionBatchProcessor` | 同module | 一バッチの処理を接続し、前回値だけを所有。全clientやstreamのownerではない |
| `process_detection_batch` | 同class | 完成した検出バッチの評価・割当・保持を実行 |
| `previous_minimum_mean_bounded_loss` | 同classのproperty | 次のバッチと比較する前回値。read-only |
| `FeddriftDetectionBatchTransition` | 同module | 一バッチの分岐・位置・次の前回値。状態を変更する関数ではない |
| `queue_ready_model_upload` | 既存PendingModelUploadState | 待機せず即時送信可能な保留。既存遅延queueと区別 |
