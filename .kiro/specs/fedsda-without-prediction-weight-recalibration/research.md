# Research & Design Decisions

- 旧の`none`は、サーバが、clientの再較正を呼ばない（`servers/shared_backbone.py`の`run_round`）。clientの`recalibrate_routing_after_aggregation`の中の、真の概念別・文脈別の重みの始め直しも、行われない。
- 新の再較正（`runtime/post_aggregation_prediction_recalibration.py`）は、予測重みとglobalの診断重みの作り直しと、真の概念別の診断重みの始め直しを、1つの関数で行う。方針が「再較正しない」のとき、その関数を呼ばなければ、旧の`none`と同じになる。
- 設定の値の名前は、既存の値（`recompute_buffer_losses_and_replay_weight_updates`）と同じく、何をするかを書く: `keep_weights_without_recalibration`。
