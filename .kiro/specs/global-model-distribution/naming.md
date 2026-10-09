# 命名表: global-model-distribution

sourceの公開する名前（module、class、公開の関数・メソッド）だけを載せる。確認は、全taskの後の独立レビューで受ける。

## ファイル・型

| 提案名 | 種類 | 役割・所有する状態 | 関連する名前との違い |
|---|---|---|---|
| `shared_parameter_optimizer_state_holder` | module（learning/training） | 共有部のoptimizerの状態の保持者 | `parameter_optimizer_state`は、固定のパラメータ列のoptimizerを1つ持つ |
| `SharedParameterOptimizerStateHolder` | class | client 1つの、現在の共有部のoptimizerの状態への参照を1つ持ち、配布のときに置き換えられる | `CandidateValidationSessionHolder`は、候補検証のsessionの保持者（同じ「保持者」の形） |
| `global_model_distribution_application` | module（runtime） | client 1つが、配布を受け取る処理 | `global_model_distribution`は、サーバが全clientへ配る処理（本moduleの処理を、clientの操作を通して呼ぶ） |
| `GlobalModelDistributionApplication` | frozen dataclass | 受取り1回の結果（学習帰属の変更、受取りの後の保有モデルのID） | `ClientModelAggregation`は、集約1回の結果 |
| `global_model_distribution` | module（runtime） | サーバの配布 | `server_model_registration_and_aggregation`は、ラウンドの前半（登録と集約） |

## 関数・メソッド

| 提案名 | 入力 | 出力 | 役割 | 状態更新・副作用 |
|---|---|---|---|---|
| `apply_global_model_distribution` | ID対応、配布されたパラメータと統計、clientのowner、2つのoptimizerの設定、乱数生成器 | `GlobalModelDistributionApplication` | 統計・標本・計数をID対応で付け替え、非負のIDの保有モデルを配布された値で作り直し、共有部へつなぎ直し、現在の学習帰属を付け替える | clientのowner、torchの乱数、借りた乱数 |
| `distribute_global_models_to_clients` | clientの列、グローバルモデルのowner、通信量のowner、ID対応 | 受取りの結果のtuple | 下りの通信量を足し、全clientへ、全グローバルモデルを受け取らせる | 通信量のowner、各client |
| `FedsdaRunClient.apply_global_model_distribution` | ID対応、配布されたパラメータと統計 | `GlobalModelDistributionApplication` | 自分のownerと設定で、受取りを行う | 同上 |
| `SharedParameterOptimizerStateHolder.held_shared_parameter_optimizer_state`（property）、`replace_shared_parameter_optimizer_state` | なし／状態 | 状態／なし | 現在の共有部のoptimizerの状態を読む／置き換える | なし／保持者 |
| `HeldModelTrainingStateRegistry.replace_held_model_training_states` | 状態のtuple | なし | 保有モデルを、渡された順の状態で全部置き換える | registry |
| `ModelAndClassLossStatisticsStore.replace_model_loss_statistics` | （ID、統計）のtuple | なし | 損失統計を、渡された順で全部置き換える | 統計 |
| `split_shared_and_concept_specific_parameters`（`server_model_registration_and_aggregation`。非公開だった関数を公開にした） | 完全なパラメータ | （共有部、概念固有部） | 完全なパラメータを、名前の順を保って、共有部と概念固有部へ分ける。集約と配布の両方が使う | なし |
| `GlobalModelRepository.snapshot_global_model_loss_statistics` | なし | （ID、統計）のtuple | 統計を持つグローバルモデルの統計を、置いた順で返す | なし |

## 判断が必要な点

- 「distribution」はサーバが配ること、「application」はclientが受け取って自分の状態へ反映すること。旧の`broadcast_models`・`apply_server_mapping`に当たる。
- 束の新しいfield `rebuilt_model_parameter_optimizer_settings`は、配布で作り直す（rebuilt）モデルのoptimizerの設定。既存の`parameter_optimizer_settings`（候補と、つなぎ直しで作り直す概念固有部）と区別する。
- 適応記録の結果種別`server_consolidation_training_model_remapped`は、サーバの統合（consolidation）で、学習帰属のIDが付け替わった（remapped）こと。警報による切替（switch）と区別する。
