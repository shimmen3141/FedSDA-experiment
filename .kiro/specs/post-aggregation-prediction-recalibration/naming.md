# 命名表: post-aggregation-prediction-recalibration

sourceの公開する名前（module、class、公開の関数・メソッド）だけを載せる。確認は、全taskの後の独立レビューで受ける。

## ファイル・型

| 提案名 | 種類 | 役割・所有する状態 | 関連する名前との違い |
|---|---|---|---|
| `post_aggregation_prediction_recalibration` | module（runtime） | client 1つの、集約（と配布）の後の、予測の重みと診断証拠の再較正 | `global_model_distribution_application`は、配布されたモデルの受取り（予測の状態に触れない） |
| `PostAggregationPredictionRecalibration` | frozen dataclass | 再較正1回の結果（再生した標本数、列に含めたモデルID、再始動した真の概念ID） | `GlobalModelDistributionApplication`は、受取り1回の結果 |

## 関数・メソッド

| 提案名 | 入力 | 出力 | 役割 | 状態更新・副作用 |
|---|---|---|---|---|
| `recalibrate_prediction_state_after_aggregation` | 保有モデルのregistry、保留標本のowner、Fixed-Shareの重みのowner、診断証拠の集まり | `PostAggregationPredictionRecalibration` | 保留中の標本の損失の列を、現在の保有モデルで計算し、globalの診断証拠の再生、真の概念別の診断証拠の再始動、Fixed-Shareの重みの再生を行う | 診断証拠、Fixed-Shareの重みと計数 |
| `FedsdaRunClient.recalibrate_prediction_state_after_aggregation` | なし | 同上 | 自分のownerで、再較正を行う | 同上 |
| `AdaHedgeDiagnosticEvidence.restart_evidence_after_aggregation` | なし | なし | 証拠を消し、集約後の再始動と再較正の回数を1ずつ足す | 証拠、計数 |
| `AdaHedgeDiagnosticEvidence.replay_observed_losses_after_aggregation` | 損失の列 | なし | 空でない列について、再較正の回数と標本数を足し、証拠を消して、列の順に再構成する | 証拠、計数 |
| `AdaHedgeDiagnosticEvidence.aggregation_restart_count`・`aggregation_recalibration_count`・`aggregation_recalibration_sample_count`（property） | なし | int | 集約後の再始動の回数、再較正（再始動と再生）の回数、再生した標本数 | なし |

## 判断が必要な点

- 「recalibration」は、集約の後で、過去の比較（重みと証拠）を、配布の後のモデルに合わせて作り直すこと。既存の設定名`prediction_weight_recalibration_after_aggregation_policy`と、Fixed-Shareの`replay_observed_losses_after_aggregation`・`aggregation_recalibration_count`と、同じ語を使う。
- 「prediction state」は、予測の重み（Fixed-Share）と、予測の診断証拠（AdaHedge）を合わせたもの。重みだけではないので、`prediction_weights`ではなく`prediction_state`にする。
- 診断証拠の操作名は、既存の`restart_evidence_after_concept_operation`（概念操作による再始動）と対にする。再生の操作名は、Fixed-Shareの同じ役割の操作と同じ名前にする。
