# 公開名と役割

| 公開名 | 役割・似た名前との違い |
|---|---|
| `runtime/feddrift_run_client_settings.py` | 一clientを組み立てて進める条件。全体runのR回進行を含まない |
| `FeddriftRunClientSettings` | 検出・学習・評価保持・新規optimizerの不変設定 |
| `runtime/feddrift_run_client.py` | 初期状態からのclient組立と操作の接続 |
| `FeddriftRunClientOwners` | 可変状態を持つ既存ownerへの固定参照束 |
| `FeddriftRunClient` | 組立済みclientの観測・round・終端操作。FedSDAの実行Protocolとは別 |
| `FeddriftRunClient.client_id` | clientの非負ID |
| `FeddriftRunClient.owners` | 借用owner参照束 |
| `FeddriftRunClient.detection_batch_processor` | 同じownerに作用するbatch処理への借用参照 |
| `FeddriftRunClient.stream_processor` | 観測位置・buffer・予測列の正本への借用参照 |
| `FeddriftRunClient.process_observed_block` | 観測値のblockを変換して処理し、満杯検出の有無を返す |
| `FeddriftRunClient.train_local_round` | 検出batch件数を倍率にした一roundの局所学習 |
| `FeddriftRunClient.flush_detection_buffer` | 残余検出の一度の処理。局所学習のflushではない |
| `FeddriftRunClient.has_model_ready_for_server_registration` | 送信ready値を変更せず読む |
| `assemble_feddrift_run_client` | 初期モデル0・optimizerを独立に写してownerと処理を接続する |

最後の独立レビューで役割・実装との一致を確認する。引数/field/private/test名は対象外。
