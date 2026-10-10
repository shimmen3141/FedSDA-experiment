# Research & Design Decisions

## Summary

- **Feature**: `comparison-computation-metrics`
- **Discovery Scope**: Extension（計数と導出を足す。処理の結果は変えない）
- **Key Findings**:
  - サーバの計算は、パラメータの算術だけ（順伝播はない）。集約・統合・診断の距離の3箇所。
  - 集約で足すパラメータは、通信量の「上りのパラメータの値の数」と同じ（旧と一致済みの値で、照合できる）。
  - 保有モデル数の合計と、ラウンドごとの計算量は、旧の計数（予測の標本数、ラウンドごとの増分）で照合できる。サーバの計数は、旧にないので、記録と層の形からの手計算で確かめる。

## Research Log

### サーバの計算

- **Sources Consulted**: runtime/server_model_registration_and_aggregation.py（`aggregate_client_models_into_global_models`、`_add_weighted_parameters`、`_divide_parameters`、通信量の記録）、runtime/model_clustering_and_consolidation.py（`_compute_concept_specific_parameter_distance`、`_compute_sample_weighted_mean_parameters`、`ModelConsolidation`）、runtime/server_round_synchronization.py（`ServerRoundSynchronization`）、runtime/fedsda_run_server.py（同期の結果の保持）、runtime/model_cross_evaluation.py・global_model_distribution.py（通信量の記録）。
- **Findings**:
  - 集約: 参加するclientごとに、共有部を1回（`_add_weighted_parameters`）、参加モデルごとに概念固有部を1回、足す。足したものを、そのまま`uploaded_parameter_snapshots`へ入れて、通信量へ数える。上りのパラメータを数えるのは、ここだけ（クロス評価と配布は、下り）。
  - 統合: 統合するクラスタ（メンバー2つ以上）ごとに、重みが正のメンバーの、完全なパラメータを足す。重みがすべて0なら、足さない。
  - パラメータ距離: クラスタリングの対象のモデルの対ごとに、概念固有部の、`差×差`の和（値の数）、`左²`の和、`右²`の和。倍精度。診断の観測（`concept_specific_parameter_distance`）へ入るだけで、判定に使わない。
  - サーバは、同期の結果（`ServerRoundSynchronization`）を、ラウンドの順に保持している。集約の結果と、統合の結果が、その中にある。
- **Implications**: 計数は、集約の結果と、統合の結果へ、fieldとして足す。新しいownerや、引数の受渡しは要らない。ラウンドごとの値も、そのまま得られる。

### 旧の、ラウンドごとの計数と、保有モデル数

- **Sources Consulted**: federated_drift_experiment/experiment.py（`_record_round_telemetry`——ラウンドの前後の、clientの計数の差——、`_save_raw_run`の`round_client_*`・`round_client_held_model_count`）、clients/shared_backbone.py（`_routing_scores`——予測は、保有する全モデルを、1標本ずつ通す）。
- **Findings**:
  - 旧は、ラウンド（`run_timestep`）の前後で計数を読むので、ラウンドの値は、ローカルの処理と、同期の間のclientの計算の合計。用途別なので、同期の分（クロス評価、再較正）を分けられる。終端の処理（未完了の候補検証の回収）は、ラウンドの値に入らない（累計には入る）。
  - `round_client_held_model_count`は、ラウンドの後の保有モデル数（標本ごとではない）。標本ごとの保有モデル数の合計は、`prediction_examples`（累計とラウンドごと）に等しい。

### 実行の枠の、ラウンドの境界

- 区間の進行（execution/stream_protocol_execution_loop.py）は、ラウンドごとに、標本の処理→区間末の学習→サーバの`record_client_states_before_synchronization`→登録可能の照会→サーバの`synchronize_models`→送信待ちの進行、の順に呼ぶ。サーバの2つの操作が、「ローカルの処理の終わり」と「同期の終わり」の印になる。

## Design Decisions

### Decision: サーバの計数は、集約と統合の結果へ、fieldとして足す

- **Alternatives Considered**: (1)サーバに計数のownerを持たせ、集約と統合へ渡す——引数が増え、既存のtestの呼出しを、広く直す。(2)tensorの演算をhookで数える——全結合層のような、対象を絞る手がかりがない。
- **Selected Approach**: `ClientModelAggregation`と`ModelConsolidation`へ、計数のfieldを足す。合計は、サーバが保持する同期の結果から、指標の導出が足す。
- **Rationale**: 計算した関数が、自分の計算の量を返す。ラウンドごとの値が、追加の仕組みなしで残る。

### Decision: 積和演算は、「重みを掛けて足す」と「2乗して足す」だけを数える

- **Rationale**: clientの計数（全結合層の、重み行列の積和）と、同じ単位にする。割り算（モデルごとに、値の数ぶん）、写し、損失統計の平均は、積和ではないので、数えない（集約では、割り算は、足す回数より少ない）。

### Decision: 診断のパラメータ距離は、別の項目にする

- **Rationale**: 判定に使わない、診断のための計算である。「手法の計算量」へ含めるかを、分析で選べるようにする。

### Decision: 保有モデル数は、clientが、標本ごとの列として記録する

- **Alternatives Considered**: 合計だけを持つ——ラウンドごと・時点ごとの保有モデル数が、後から作れない。
- **Selected Approach**: 標本ごとの保有モデル数の列（clientごと）を持つ。合計と、ラウンドごとの値は、列から作る。
- **Rationale**: 主張(c)（モデル数と計算量の関係）に、時点ごとの保有モデル数が要る。標本数ぶんの整数で、負担は小さい。判定記録・検出器の計数と同じく、clientの操作が、標本の処理が成功した後に足す。

### Decision: ラウンドごとのモデルの計算は、計測つきの全体runが、サーバの操作を包んで控える

- **Selected Approach**: 計測つきの全体runが、準備で作られた参加者の、サーバの操作を、非公開の包みで置き換える。包みは、`record_client_states_before_synchronization`の入口（ローカルの処理の終わり）と、`synchronize_models`の出口（同期の終わり）で、その時点の計数を控える。
- **Rationale**: 実行の枠と、参加者を変えない。計測は、計測つきの全体runに閉じる。
- **Trade-offs**: 「ローカルの処理」には、サーバの2つの操作の間の、登録可能の照会（計算なし）を含めない——照会は、同期の側に入るが、計算がないので、値は変わらない。送信待ちの進行（計算なし）は、次のラウンドのローカルの側に入る。

### Synthesis

- **Build vs. Adopt**: 計測、同期の結果の保持、通信量の記録を、そのまま使う。
- **Simplification**: 回帰（傾き）、図、保存は、作らない。検出器の計数は、積和演算へ換算しない（別枠のまま）。

## Risks & Mitigations

- 計数のfieldの追加で、集約・統合の結果の等値の比較（既存のtest）が変わる — 結果の型を作るのは、それぞれ1箇所。既存のtestは、fieldを個別に読んでいる。全pytestで確かめる。
- ラウンドの境界の取り方が、旧のラウンドの値とずれる — 旧のラウンドごとの計数（用途別）と、ローカル・同期の別で照合する。

## References

- [model-computation-measurement](../model-computation-measurement/)（計測と、標本数・積和演算の定義）、[fedsda-run-metric-derivation](../fedsda-run-metric-derivation/)（指標の導出と、goldenの照合のtest）、[NEW-004](../../../docs/research/implementation-findings/new-004-legacy-computation-comparison-removal.md)（旧の計数との照合を、後で外すときの作業）。
