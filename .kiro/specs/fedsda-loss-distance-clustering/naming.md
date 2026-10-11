# 命名表: fedsda-loss-distance-clustering

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `model_pair_comparison_strategy` | `ModelClusteringCriteria`のfield（統合の方式の宣言の、既存のfieldと同じ名前） | モデル対の比べ方（判定の値の作り方） |
| `MODEL_PAIR_COMPARISON_STRATEGIES` | `methods/fedsda/consolidation/model_clustering_calculations.py` | 判定の基準が許す、比べ方の値 |
| `mean_loss_increase` | 上の2つのfieldの値 | 互いの標本での平均損失の増加量の、大きいほうを、判定の値にする |
| `LOSS_DISTANCE_CLUSTERING_CONFIGURATION_NAME`（値`fedsda-loss-distance-clustering`） | `sweep/experiment_configurations.py` | 掃引のコマンドで選ぶ、構成の名前 |
