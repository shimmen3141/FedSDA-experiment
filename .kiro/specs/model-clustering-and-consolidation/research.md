# Research & Design Decisions

## Summary

- **Feature**: `model-clustering-and-consolidation`
- **Discovery Scope**: New Feature（判定の計算、クラスタリング、統合は新規。前後の段は移植済み）
- **Key Findings**:
  - 旧の最終構成のクラスタリングは、クロス評価の表と、対ごとの正誤の集計だけから決まる、乱数を使わない計算である。
  - 統合は、グローバルモデルのパラメータと統計を書き換え、代表でないモデルを消し、ID対応を配布へ渡す。配布と受取りは、ID対応つきで、実旧と照合済みである。
  - 前後の段がすべてそろうので、サーバの1ラウンドの同期を1つの関数にでき、旧の`run_round`そのもの（クラスタリングあり）と照合できる。
  - 実旧だけで32条件（2値・多クラス、8つの標本列、評価標本の追加の件数2種類、22ラウンド）を進めると、2モデルの統合、3モデルが1つへ集まる統合、統合しないクラスタリング、2回めのクラスタリングが、自然に起きた。

## Research Log

### 旧のクラスタリングと統合

- **Sources Consulted**: 旧`servers/fedsda.py`（`run_round` 49〜70行、`_cluster_and_consolidate` 89〜130行、`_merge_clusters`・`_weighted_average_params`・`_merge_stats` 335〜415行）、`servers/clustering.py`（`perform_hierarchical_clustering` 290〜400行、`_personalized_parameter_distance`、`_oracle_concept_labels`、`_clustering_cutoff`、`record_clustering_diagnostics`）、`clustering.py`（`FunctionalPairStats`、`binomial_proportion_lower_bound`、`mean_loss`、`cluster_models`、`_agglomerative_linkage`）、`model_lineage.py`（観測の型と`record_clustering`）、`servers/shared_backbone.py`（`run_round`、`broadcast_models`）、`experiment.py`（100〜112行）。
- **Findings**（要求の「調査で確かめたこと」のほか）:
  - 閾値は、サーバの`distance_threshold`（判定が`class_functional_confidence`のとき）。対照testでは0.1。
  - 対の走査の中の表示（`verbose`のときだけ、乱数を1回読む）は、`verbose`が偽なので、乱数を消費しない。
  - 対の正誤の集計は、クロス評価が直前に作ったもの（`_last_pair_functional_stats`）。キーは（小さいID、大きいID）。
  - パラメータの距離: 共有部（旧の名前で`backbone.`で始まるもの）を除くパラメータについて、倍精度で、差の2乗和÷（2乗和の和の半分）の平方根。分母が0なら、差が0のとき0、そうでなければ無限大。
  - 真の概念のラベル: モデルごとに、全clientの`get_model_concept_counts`を、clientの順に足し合わせ、最多の概念が1つだけなら、それ。
  - `record_clustering`の対の観測は、距離を持つ対だけ（対のIDの昇順）。判定の値は、なければ距離で代用する。パラメータの距離は、全部の対について求めてあるものを引く。
  - `_merge_clusters`は、クラスタの列（最小のIDの昇順）の順に処理する。代表のパラメータは、辞書の同じ位置で置き換わる（グローバルモデルの順は変わらない）。
  - `_weighted_average_params`: 重みは`max(件数, 0)`。メンバーの順に、最初の（重みが正の）メンバーの`値×重み`から始めて、`和＋値×重み`を足し、最後に合計で割る。パラメータは、共有部を含む完全なもの。
  - `_merge_stats`: `sum(...)`（組み込みの和）で、件数の合計と、`平均×件数`の和を求める。
  - 統計の`M2`は0.0、クラス別の統計は持たない（集約が作る統計と同じ形）。
- **Implications**: 判定の計算は、乱数もownerも使わない関数にする。クラスタリングと統合は、クロス評価の結果と集約の件数を受け取り、記録とグローバルモデルを更新する1つの関数にする。

### 新の既存部品との対応

| 旧 | 新 |
| --- | --- |
| クロス評価の表、`_last_pair_functional_stats` | `ModelCrossEvaluation`（`loss_sums_by_candidate_and_target_model_id`、`unique_correctness_counts_by_model_pair`） |
| `update_global_models`の戻り値（集約の件数） | `ClientModelAggregation.aggregated_training_sample_counts_by_model_id` |
| `get_model_concept_counts` | `ModelTrainingAndAssignmentCountsStore.get_model_assigned_sample_concept_counts`（clientの操作を足す） |
| グローバルモデルと統計の書換え・削除 | `GlobalModelRepository`の設定の操作（既存）と、外す操作（本specで足す） |
| `model_lineage`のクラスタリングの観測 | 診断の記録のowner（本specで足す） |
| `broadcast_models(id_mapping)`と受取り | `distribute_global_models_to_clients`、`apply_global_model_distribution`（ID対応つきで照合済み） |

### oracle

- 判定の計算: 実旧の`binomial_proportion_lower_bound`、`FunctionalPairStats.class_conditional_confidence_distance`、`cluster_models(…, "average")`へ、同じ入力を与えて照合する。
- ラウンド: tests/refactoring/test_post_aggregation_prediction_recalibration.pyの`run_round_with_recalibration_in_both`と同じ形で、旧は`legacy_server.run_round(round_index, clustering_enabled=<登録の前に、送信できるモデルを持つclientがいるか>)`、新は同期の関数を、同じ乱数の状態から行う。クラスタリングの診断の記録は、実旧の`model_lineage.clustering_pair_observations`・`clustering_observations`と照合する。

## Design Decisions

### Decision: 診断の記録の「なし」を、Noneで持つ

- **Context**: 旧は、最も近いモデルがないとき−1、距離がないときNaNを入れる。
- **Selected Approach**: 新の記録は、ない値をNoneにする。testが、旧の−1・NaNと対応づけて照合する。
- **Rationale**: NaNは、等値の比較ができず、−1は、モデルIDと紛れる。

### Decision: サーバの1ラウンドの同期を、1つの関数にする

- **Selected Approach**: `synchronize_models_in_server_round`が、登録→集約→［クロス評価→クラスタリングと統合］→配布→全clientの再較正を、旧の順に呼ぶ。クラスタリングが有効かどうかは、引数で受け取る（実行の枠の`new_model_registration_available`に当たる）。
- **Rationale**: 全部の段がそろい、旧の`run_round`と1対1で照合できる。実行の枠のサーバの操作の実体（次のspec）は、この関数を呼ぶだけになる。集約後の再較正のspecで見送った「全clientへ順に呼ぶ関数」は、この関数が担う。

### Decision: 判定の基準を、1つの設定型にまとめる

- **Selected Approach**: 閾値・評価の件数の下限・信頼水準を、`ModelClusteringCriteria`（methods/fedsdaのconsolidation）にまとめ、生成時に範囲を確かめる。1つのモデルを評価するclientの上限は、同期の関数の引数のままにする。
- **Rationale**: 3つは、いつも一緒に使う。置き場所（完全なrun設定）は、後のspecで決める。

### Synthesis

- **Build vs. Adopt**: 前後の段は既存の部品。新しく書くのは、判定の計算、対の走査、診断の記録、統合、同期の関数。
- **Simplification**: 最終構成でない判定・linkage・後処理は移植しない（設定`ModelConsolidationSettings`の選択肢は、最終構成の1つずつ）。

## Risks & Mitigations

- 加重平均と、平均損失の和の、演算の順が旧と違うと、末尾の桁が変わる — 旧と同じ順（メンバーの順、組み込みの和）で書き、実旧との対照で、全パラメータと統計の一致を確かめる。
- 統合の後、clientの状態（付け替え）と、次のラウンド以降が旧とずれる — ラウンドの対照を、統合の後も続ける。

## References

- [model-cross-evaluation](../model-cross-evaluation/)、[global-model-distribution](../global-model-distribution/)、[post-aggregation-prediction-recalibration](../post-aggregation-prediction-recalibration/)、[server-model-registration-and-aggregation](../server-model-registration-and-aggregation/)。
