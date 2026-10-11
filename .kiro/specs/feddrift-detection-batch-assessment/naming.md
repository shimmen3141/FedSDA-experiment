# 命名表: feddrift-detection-batch-assessment

| 名前 | 置き場所 | 役割・区別 |
|---|---|---|
| `detection_batch_assessment` | `methods/feddrift/` | 検出バッチの数値評価と判断。判定後の状態変更を含まない |
| `FeddriftDetectionBatchAssessment` | 同module | モデル別平均損失と選択・検出の不変な結果。clientや検出器の状態ownerではない |
| `assess_feddrift_detection_batch` | 同module | 保有順で全モデルを評価して、前回の最小平均損失と比較する。モデルの生成・切替・割当は行わない |
