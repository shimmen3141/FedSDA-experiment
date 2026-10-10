# 命名表: model-clustering-and-consolidation

sourceの公開する名前（module、class、公開の関数・メソッド）だけを載せる。確認は、全taskの後の独立レビューで受ける。

## ファイル・型

| 提案名 | 種類 | 役割・所有する状態 | 関連する名前との違い |
|---|---|---|---|
| `model_clustering_calculations` | module（methods/fedsda/consolidation） | クラスタリングの判定の、乱数もownerも使わない計算と、判定の基準の型 | `model_consolidation_settings`は、方式の選択（最終構成の1つずつ） |
| `ModelClusteringCriteria` | frozen dataclass | 判定の閾値・評価の件数の下限・信頼水準 | `ModelConsolidationSettings`は方式の名前を持つ。こちらは数値 |
| `model_clustering_record_store` | module（evaluation） | クラスタリングの診断の記録 | `cross_evaluation_record_store`は、クロス評価の記録 |
| `ModelPairClusteringObservation` | frozen dataclass | 1回のクラスタリングの、モデル対1つの観測（距離、判定の値、同じクラスタか、診断値） | `ModelClusteringObservation`は、モデル1つの観測 |
| `ModelClusteringObservation` | frozen dataclass | 1回のクラスタリングの、モデル1つの観測（最も近いモデル、代表、クラスタの大きさほか） | 上 |
| `ModelClusteringRecordStore` | class | 2種類の観測を、足された順に持つ | — |
| `model_clustering_and_consolidation` | module（runtime） | クロス評価の結果からの、クラスタリングと統合 | `model_cross_evaluation`は、その入力を作る |
| `ModelConsolidation` | frozen dataclass | クラスタリングと統合1回の結果（クラスタ、ID対応、吸収されたモデルID） | `ModelCrossEvaluation`は、クロス評価1回の結果 |
| `server_round_synchronization` | module（runtime） | サーバの1ラウンドの同期（登録からclientの再較正まで） | `server_model_registration_and_aggregation`は、その前半の2段 |
| `ServerRoundSynchronization` | frozen dataclass | 同期1回の、各段の結果 | — |

## 関数・メソッド

| 提案名 | 入力 | 出力 | 役割 | 状態更新・副作用 |
|---|---|---|---|---|
| `compute_binomial_proportion_lower_confidence_bound` | 成功数、標本数、信頼水準 | float | Wilsonの片側信頼下限 | なし |
| `compute_classwise_unique_correctness_decision_score` | 対の、全体とクラス別の数の組、クラスの標本数の下限、信頼水準 | float | クラス別の、片方だけが正解した割合の、同時信頼下限の最大（モデル対の判定の値） | なし |
| `cluster_model_ids_by_average_linkage` | モデルIDの列、対ごとの判定の値、閾値 | クラスタの列 | average linkageの逐次統合 | なし |
| `cluster_and_consolidate_global_models` | clientの列、クロス評価の結果、集約の件数、グローバルモデルのowner、診断の記録のowner、判定の基準、ラウンド | `ModelConsolidation` | 対の距離と判定の値を求めてクラスタリングし、記録を足し、クラスタが減るなら統合して、ID対応を返す | 診断の記録、グローバルモデルのowner |
| `synchronize_models_in_server_round` | clientの列、各owner、判定の基準、clientの上限、乱数生成器、ラウンド、クラスタリングが有効か | `ServerRoundSynchronization` | 登録→集約→（クロス評価→クラスタリングと統合）→配布→全clientの再較正 | 各owner、client、乱数 |
| `ModelClusteringRecordStore.append_clustering_observations`・`snapshot_pair_clustering_observations`・`snapshot_model_clustering_observations` | 観測の列／なし | なし／観測のtuple | 1回のクラスタリングの観測を足す／写しを返す | 記録／なし |
| `GlobalModelRepository.remove_global_model` | モデルID | なし | グローバルモデルのパラメータと損失統計を外す | グローバルモデルのowner |
| `FedsdaRunClient.get_model_assigned_sample_concept_counts` | モデルID | 概念ごとの件数 | モデルへ帰属させた標本の、真の概念ごとの件数（診断用） | なし |

## 判断が必要な点

- 「clustering」は、モデルを、判定の値でクラスタへ分けること。「consolidation」は、同じクラスタのモデルを、1つのモデルへまとめること（旧の`merge`）。既存の設定名`ModelConsolidationSettings`・`model_consolidation_policy`、適応記録の`server_consolidation_training_model_remapped`と同じ語。
- 「decision score」は、クラスタリングの判定に使う値（旧の`pair_decision_scores`）。「loss increase distance」は、互いの平均損失の増加（旧の`pair_distances`。最終構成では、判定に使わず、診断と、対を判定するかどうかの前提にだけ使う）。
- 「unique correctness」は、対の片方だけが正解すること（クロス評価の`ModelPairUniqueCorrectnessCounts`と同じ語）。
- 「synchronization」は、実行の枠の`synchronize_models`と同じ語（サーバとclientのモデルを同期させる、ラウンド境界の処理の全体）。
