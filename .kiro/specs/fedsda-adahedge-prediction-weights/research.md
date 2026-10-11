# Research & Design Decisions

- 旧の`_record_prediction`: 保有モデルの出力から、AdaHedgeの重み（`proposal_probabilities`）、Fixed-Shareの重み（`switching_probabilities`）、真の概念別の重みでの結合を、毎標本で作る。`SOFT_ROUTING_CONTEXT`が`global`なら、予測の重み（`probabilities`）は、AdaHedgeの重み（基底の`_prediction_probabilities`は、そのまま返す）。`switching`なら、Fixed-Shareの重み。
- 予測の重みから作るもの: 予測の正否（`history_accuracy`）、最大重みのモデル（`history_model_id`。同率なら現在のモデル、次に最小のID）、実効モデル数、最大の重み、最大重みのモデルの正否、確信度が同率のときの選び方、クラス別の集計。
- Fixed-Shareの重みから、いつも作るもの: `history_routing_switching_correct`・`_leader_id`・`_effective_experts`、`routing_switching_diagnostics`（正解の数。実際の予測の正解の数と、AdaHedgeの結合の正解の数も、ここに持つ）。
- 更新は、どちらの指定でも、3つとも行う（AdaHedgeは、AdaHedgeの重みで。Fixed-Shareは、Fixed-Shareの重みで）。
- 集約後の再較正（`fifo_replay`）と、学習帰属の変更でのAdaHedgeの始め直しは、指定に依らない。
- 最初の実装で、小さい条件の7つ全部が、値では一致した。直したのは、testの側の対応（旧の3指標を「いつも0」としていた箇所、件数の期待）だけ。
