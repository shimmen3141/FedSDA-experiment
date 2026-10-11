# Design Document: fedsda-loss-distance-clustering

## 変更

- `ModelClusteringCriteria`（`methods/fedsda/consolidation/model_clustering_calculations.py`）へ、`model_pair_comparison_strategy`を足す。許す値は、`MODEL_PAIR_COMPARISON_STRATEGIES`の2つ: `classwise_unique_correctness_lower_confidence_bound`（提案構成）、`mean_loss_increase`。
- `cluster_and_consolidate_global_models`（`runtime/model_clustering_and_consolidation.py`）: `mean_loss_increase`のとき、損失の距離を、判定の値にする。
- `ModelConsolidationSettings.model_pair_comparison_strategy`（統合の方式の宣言）へ、同じ値を足す。`FedsdaRunSettings`が、判定の基準と宣言の、値の一致を確かめる。
- 最終構成のrun設定は、判定の基準へ、提案構成の値を置く。
- 掃引: 構成`fedsda-loss-distance-clustering`を足す（判定の基準と、宣言の、両方を替える）。

## 設定を2か所に持つ理由

サーバは、参加者の設定（`FedsdaRunParticipantSettings`）の、判定の基準だけを受け取る。統合の方式の宣言（`ModelConsolidationSettings`）は、run設定の全体にあり、条件の保存と、組合せの検証に使う。宣言をサーバまで渡すには、参加者の準備・サーバ・同期の関数の引数を変える必要があり、判定の基準へ足すほうが、変更が小さい。2か所の値がずれないように、run設定の生成時に確かめる。

## クロス評価は変えない

旧のFedSDAのサーバは、常に、モデル対の診断を集める（`servers/fedsda.py`）。`distance`でも、クロス評価で、全体の正誤の集計は作る。クラス別の正誤の集計は、提案構成の判定（`class_functional_confidence`）のときだけ作る。

新は、どちらの比べ方でも、クロス評価を変えず、クラス別の集計も作る。そのため、この構成では、記録の表`client_cross_evaluation_class`を、新だけが持つ（旧には、対応する記録が無い）。損失の統計、判定、指標、計算量の計数は、新旧で同じ（小さい条件の照合で一致）。構成の照合は、この表を比べていない。

## 検証

- `tests/refactoring/test_fedsda_configuration_variants.py`: 構成を足す（旧の側は、`FEDSDA_CLUSTERING_DECISION`を`distance`）。小さい条件の7つで、実旧の全体runと照合する。固有の経路のtestと、設定の不一致の拒否のtestを足す。
- 既存のtest: 判定の基準を作っている箇所へ、提案構成の値を足す。
