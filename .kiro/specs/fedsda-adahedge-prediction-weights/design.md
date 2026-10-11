# Design Document: fedsda-adahedge-prediction-weights

## 1. 名前（値を変えない変更。commit `c345a88`）

対応は、naming.md。語の境界で、src・tests/refactoring・`docs/research/legacy-run-output-migration.md`を置き換えた。完了済みのspecの文書と、再開案内の過去の記述は、書き換えていない。ファイルも、名前に合わせて移した（証拠の2つ、学習帰属の変更での始め直し、Fixed-Shareの重み、対応するtest）。

考え方: 重みの出どころは3つある。

| 出どころ | 名前 | 役割 |
|---|---|---|
| Fixed-Shareの重み | `fixed_share_…` | 提案構成では、予測に使う。AdaHedgeの重みで予測する構成では、比較用 |
| AdaHedgeの重み（全標本） | `adahedge_…`（証拠の集まりの中では`all_sample_weight_evidence`） | 提案構成では、比較用。AdaHedgeの重みで予測する構成では、予測に使う |
| AdaHedgeの重み（真の概念別） | `true_concept_diagnostic_…`（変えない） | 真の概念を使うので、常に、比較用 |

「予測に使う重み」での値は、方式の名前を付けない（`combined_prediction_…`、`maximum_weight_model_id`、`maximum_prediction_weight`、`effective_model_count`、`prediction_weights_by_model_id`）。

## 2. 構成

- `PredictionCombinationSettings.prediction_combination_strategy`へ、値`adahedge_weighted_prediction`を足す。
- `predict_observed_sample_and_update_prediction_weights`（`runtime/observed_sample_prediction.py`）へ、引数`prediction_combination_strategy`を足す。予測に使う重みを、方式で選ぶ。Fixed-Shareの重みでの結合の正否・最大重みのモデル・実効モデル数を、どちらの方式でも計算して、記録へ入れる。Fixed-Shareの重みの更新には、Fixed-Shareの重みを渡す（これまでは、同じ値の「予測の重み」を渡していた）。確信度が同率のモデルの選び方は、予測に使う重みを使う（旧と同じ）。
- 標本1件の処理と、clientは、設定の方式を、そのまま渡す。
- `SamplePredictionRecord`へ、`fixed_share_weighted_prediction_is_correct`・`fixed_share_maximum_weight_model_id`・`fixed_share_effective_model_count`を足す。
- `FedsdaPredictionDiagnosticMetrics`へ、`fixed_share_weighted_prediction_correct_count`・`fixed_share_weighted_prediction_period_accuracies`・`mean_fixed_share_effective_model_count`を足す。
- 記録の表`sample_prediction`へ、3列を足す（130列→133列）。
- 掃引: 構成`fedsda-adahedge-prediction-weights`を足す。
- 集約後の再較正と、学習帰属の変更でのAdaHedgeの証拠の始め直しは、変えない（旧も、どちらの指定でも同じ）。

## 3. 検証

- 名前: 改名だけの時点で、全pytest（12506 passed / 4 skipped。改名の前の内容と同じ）。
- 構成: `tests/refactoring/test_fedsda_configuration_variants.py`へ構成を足す（旧の側は、`SOFT_ROUTING_CONTEXT`を`global`）。小さい条件の7つで、実旧の全体runと照合する。固有の経路のtestを足す。
- 実旧との対応（`tests/refactoring/test_fedsda_run_metric_derivation.py`）: 旧の`routing_switching_*`の指標と、`history_routing_switching_*`の列を、Fixed-Share側の項目と照合する形へ直す。旧の、Fixed-Shareの結合と実際の予測の差の3指標は、これまで「いつも0」と確かめていたが、新の指標から計算して照合する（この構成で、0でない値になる）。

## 4. 層の置き場所（変えていない）

AdaHedgeの証拠の2つのmoduleは、`evaluation/`にある。この構成では、予測に使う重みの出どころになるので、置き場所（評価の層）は、役割と合わなくなった。移すと、依存境界の登録と、多くのimportが変わる。今回は、名前だけを改め、置き場所は、改善候補として記録する（`docs/research/`）。
