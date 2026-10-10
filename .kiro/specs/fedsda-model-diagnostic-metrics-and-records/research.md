# Research & Design Decisions

## Summary

- **Feature**: `fedsda-model-diagnostic-metrics-and-records`
- **Discovery Scope**: Extension（指標と、保存する列を足す）
- **Key Findings**:
  - 旧のモデル別の学習量（`client.model_training_examples`・`model_optimizer_steps`、`train_data_store`の長さ）は、新の`ModelTrainingAndAssignmentCountsStore`と`ModelTrainingSampleStore`にある（IDの確定・統合での付け替えも、移植済み）。
  - 旧のクラスタリングの観測（`ClusteringObservation`・`ClusteringPairObservation`）のfieldは、新の`ModelClusteringObservation`・`ModelPairClusteringObservation`に、全部ある（表には、まだ一部しか出していない）。
  - 旧の`clustering_oracle_diagnostic_summary`は、対の観測だけからの集計。
  - 旧の`model_registration_selection_counts`は、標本ごとの、最大重みのモデルのIDの列から数える。

## Design Decisions

### Decision: モデル別の診断の指標は、1つのまとまりとして、指標へ足す

- **Selected Approach**: `FedsdaModelDiagnosticMetrics`（モデル数、学習量、クラスタリングの比較）を、`FedsdaRunMetrics`の1つのfieldにする（応答の指標と同じ形）。

### Decision: 合計・平均・最小・変動係数は、1つの型にまとめる

- **Selected Approach**: `ValueDistributionSummary`を、3つの学習量で共用する。旧の12の指標が、3つのまとまりになる。

### Decision: サーバの同期のラウンドごとのモデル数の数え方を、導出のmoduleへ移す

- **Context**: 前のspecで、表のmoduleの中に書いた。指標（平均・最大・合計）も、同じ値を使う。
- **Selected Approach**: 導出のmoduleの関数にして、表のmoduleが使う（表のmoduleは、すでに、応答の指標の導出のmoduleを使っている）。値は、変えない。

### Decision: 保有モデルごとの学習量は、表と指標で、同じ関数から作る

- **Rationale**: 2箇所で数えると、食い違う余地が残る（前のspecのレビューの観点）。

## Risks & Mitigations

- 変動係数とAUCは、浮動小数点の計算の順で、最後の桁が変わりうる — 旧と同じ式（NumPyの`sum`・`mean`・`std`、同じ順の二重の繰返し）で計算し、`==`で照合する。
- 小さい条件の実旧の実行は、ラウンドごとのモデル数を記録しない — モデル数のまとめは、goldenの3ケース（sine2で、モデル数が変わる）で照合し、計算は、単体のtestで確かめる。
