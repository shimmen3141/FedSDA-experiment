# Design Document: fedsda-without-prediction-weight-recalibration

## 変更

- `PredictionCombinationSettings.prediction_weight_recalibration_after_aggregation_policy`へ、値`keep_weights_without_recalibration`を足す（これまでは、`recompute_buffer_losses_and_replay_weight_updates`だけ）。
- `FedsdaRunClient.recalibrate_prediction_state_after_aggregation`は、方針が`keep_weights_without_recalibration`のとき、何もせず、`None`を返す（戻り値の型は、`PostAggregationPredictionRecalibration | None`）。サーバの同期の結果（`ServerRoundSynchronization.prediction_recalibrations`）は、clientごとに、再較正の結果か、`None`を持つ。
- `sweep/experiment_configurations.py`へ、構成`fedsda-without-prediction-weight-recalibration`を足す（提案構成のrun設定の、上の方針だけを置き換える）。

サーバは、方針を見ない（clientが、自分の設定で決める）。旧は、サーバが、再較正を呼ばない。結果は同じ。

## 検証

- `tests/refactoring/test_fedsda_configuration_variants.py`（新規）: 既存の、小さい条件での実旧との照合（`compare_small_run_with_real_legacy`。指標の全部と、31の離散列）を、構成を指定して呼ぶ。旧の側は、設定`SHARED_BACKBONE_ROUTING_RECALIBRATION`を`none`にする。小さい条件の全部（7条件）で行い、同じ条件の提案構成の結果と比べて、固有の経路を通ったことを確かめる。
- 既存の照合の関数は、構成を受け取れる形に広げた（照合の内容は、変えていない。提案構成のtestは、そのまま通る）。
- 掃引: 構成の名前から作ったrun設定が、提案構成と、方針だけが違うこと。
