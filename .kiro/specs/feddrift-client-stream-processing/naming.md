# Naming
公開source名だけを列挙する。引数/field/local/test/private名は対象外。最終独立レビューで確認する。

| 名前 | 役割・似た名前との違い |
| --- | --- |
| runtime/feddrift_client_stream_processing.py | clientの観測列・蓄積・位置を接続する。完成一バッチの処理とは別 |
| FeddriftSamplePrediction | 切替前の予測と発火時だけのtransitionを持つ不変記録。FedSDAの重み診断記録とは別 |
| FeddriftSamplePrediction.__post_init__ | 不変記録の型/対応を検査する |
| FeddriftClientStreamProcessor | 単一clientの観測数/バッファ/予測/完了結果owner。run participant全機能ではない |
| FeddriftClientStreamProcessor.processed_sample_count | 観測完了数を読む |
| FeddriftClientStreamProcessor.buffered_training_samples | 検出待ち列のtuple snapshot。Tensorは借用 |
| FeddriftClientStreamProcessor.sample_predictions | 予測記録のtuple snapshot |
| FeddriftClientStreamProcessor.completed_detection_batch_transitions | flushを含む完了結果のtuple snapshot |
| FeddriftClientStreamProcessor.process_observations | 時刻ブロックを順に進め、満杯検出の有無を返す。局所学習はしない |
| FeddriftClientStreamProcessor.flush_detection_buffer | 部分バッチを処理し結果を返す。予測列は変えない |
| FeddriftDetectionBatchProcessor.current_training_model_state | 同じ割当ownerの現在の保有stateを借用する |
| FeddriftDetectionBatchProcessor.validate_detection_batch_inputs | 既存のbatch検査を更新/forwardなしで呼ぶ。判定はしない |
