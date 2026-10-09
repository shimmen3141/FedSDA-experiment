# 命名表: server-model-registration-and-aggregation

sourceの公開する名前（module、class、公開の関数・メソッド）だけを載せる。確認は、全taskの後の独立レビューで受ける。

## ファイル・型

| 提案名 | 種類 | 役割・所有する状態 | 関連する名前との違い |
|---|---|---|---|
| `communication_volume_record_store` | module（evaluation） | 通信量の記録 | 他の記録のowner（適応、警報、予測）と同じ層 |
| `CommunicationVolumeSnapshot` | frozen dataclass | 上り・下りの、モデル転送数、メッセージ数、パラメータの値の数、バイト数 | — |
| `CommunicationVolumeRecordStore` | class | 通信量の8つの計数を持つ。判断をしない | `ModelTrainingAndAssignmentCountsStore`は、clientの中の学習量と割当の計数 |
| `global_model_repository` | module（methods/fedsda/model_registration） | サーバが持つグローバルモデルの状態 | `pending_model_upload`は、clientの中の送信保留 |
| `GlobalModelRegistrationRecord` | frozen dataclass | 正式IDが付いた事実（モデルID、ラウンド、client。初期モデルは、ラウンドとclientがNone） | `AdaptationRecord`は、clientの中の適応の記録 |
| `GlobalModelRepository` | class | グローバルモデルのパラメータ（モデルIDごとの完全なパラメータ）と損失統計、次の正式ID、登録の来歴 | `HeldModelTrainingStateRegistry`は、client 1つが保有するモデル（分類器とoptimizer）。こちらはサーバ側で、パラメータの値だけを持つ |
| `server_model_registration_and_aggregation` | module（runtime） | サーバの1ラウンドの前半（登録と集約）の関数 | `held_model_registration_confirmation`は、client 1つの中の正式IDの確認（本moduleが呼ぶ） |
| `RegisteredClientModel` | frozen dataclass | 登録1件の結果（client、正式ID、学習帰属の変更） | — |
| `ClientModelAggregation` | frozen dataclass | 集約1回の結果（対象のID、IDごとの参加した学習データの総件数） | — |

## 関数・メソッド

| 提案名 | 入力 | 出力 | 役割 | 状態更新・副作用 |
|---|---|---|---|---|
| `register_ready_client_models` | clientの列、グローバルモデルのowner、ラウンド | `RegisteredClientModel`のtuple | 送信できるモデルを持つclientへ、順に、正式IDを採番し、来歴を記録し、clientへ確認させる | グローバルモデルのowner（次のID、来歴）、各clientのowner |
| `aggregate_client_models_into_global_models` | clientの列、グローバルモデルのowner、通信量のowner | `ClientModelAggregation` | clientのモデルを、学習データの件数で重み付けて平均し、グローバルモデルと統計を置き換える | グローバルモデルのowner、通信量のowner。clientは読むだけ |
| `GlobalModelRepository.allocate_global_model_id` | なし | int | 次の正式IDを返して、1つ進める | 次のID |
| `GlobalModelRepository.record_model_registration` | モデルID、ラウンド、client | なし | 登録の来歴を記録する（同じIDは上書き） | 来歴 |
| `GlobalModelRepository.snapshot_model_registration_records` | なし | 来歴のtuple（ID昇順） | — | なし |
| `GlobalModelRepository.set_global_model_parameters`／`get_global_model_parameters` | モデルID（とパラメータ） | なし／パラメータの写し | モデルIDのパラメータを置く／読む | パラメータ／なし |
| `GlobalModelRepository.set_global_model_loss_statistics`／`get_global_model_loss_statistics` | モデルID（と統計） | なし／統計またはNone | モデルIDの損失統計を置く／読む | 統計／なし |
| `GlobalModelRepository.next_global_model_id`、`global_model_ids` | なし（property） | int、IDのtuple（設定した順） | — | なし |
| `CommunicationVolumeRecordStore.record_model_transfers` | 向き、モデル数 | なし | 論理的なモデル転送数を足す | 計数 |
| `CommunicationVolumeRecordStore.record_messages` | 向き、件数 | なし | 軽量メッセージ数を足す | 計数 |
| `CommunicationVolumeRecordStore.record_parameter_transfer` | 向き、パラメータ、転送回数 | なし | パラメータの値の数とバイト数を足す | 計数 |
| `CommunicationVolumeRecordStore.get_state_snapshot` | なし | `CommunicationVolumeSnapshot` | — | なし |

## 判断が必要な点

- 「global model」は、サーバが持つ、正式ID（非負）のモデル。clientの「held model」（保有モデル。一時IDを含む）と区別する。
- 向きは、旧の`"up"`・`"down"`ではなく、`"upload"`（client→サーバ）・`"download"`（サーバ→client）とする。
- 「ready」は、送信保留のモデルが、待ちを終えて送信できる状態（clientの`has_model_ready_for_server_registration`と同じ語）。
