# Research & Design Decisions

## Summary

- **Feature**: `fedsda-run-metric-derivation`
- **Discovery Scope**: Extension（記録からの導出を足す。実行と記録は変えない）
- **Key Findings**:
  - goldenの条件（sine2、client 3、標本1500件、集約間隔50）で、新の全体runは、実旧の全体runと、全状態で一致する（下書きのtest、約160秒）。残るのは、記録からの計算だけ。
  - 26指標のうち、計算が要るのは、精度・定常精度・検出の4項目・最終のパラメータ量。ほかは、記録の値と、その合計。
  - 31の離散列は、新の記録の値の並べ替えで作れる。旧の保存形式（名前、型、「なし」の表し方）への写しは、旧との照合のためのもので、testに置く。

## Research Log

### 旧の指標と列

- **Sources Consulted**: federated_drift_experiment/metrics.py（`match_events`、`_stable_accuracy`、`compute_metrics`）、experiment.py（`run_random_drift_experiment` 2285〜2350行、`_add_model_diagnostic_results`、`_save_raw_run`）、data/schedules.py（`extract_true_drift_events`）、servers/shared_backbone.py（`final_parameter_footprint`）、models.py（`parameter_payload_size`）、tests/test_proposed_regression.py（`METRICS`・`TRACES`・`run_case`・`compare`）。
- **Findings**:
  - `match_events(true_positions, event_positions, delay_tolerance)`: 真の位置を順に見て、未使用の最初のイベントで、`true <= event <= true + tolerance` のものを対応づける。
  - `compute_metrics`: 精度は、全clientの`history_accuracy`の平均（なければ0）。検出は、`local_switch_positions`を昇順にして、clientごとに対応づける。適合率は 対応数÷切替の総数、再現率は 対応数÷真の変更の総数、F1は調和平均（分母0は0）。`total_detect`は切替の総数。
  - `_stable_accuracy`: 標本位置`idx`について、`idx`以下の最大の変更位置`p`があり、`idx < p + window`なら除く。残りの平均。なければNaN。
  - 真の変更位置は、概念列の全体（処理されない末尾を含む）から取る。末尾にある変更は、検出されないので、見逃しとして数えられる（旧の挙動）。
  - `final_model_count`は、終端でのグローバルモデルの数。`final_parameter_*`は、最初のグローバルモデルの共有部を1回と、各モデルの概念固有部。
  - `provisional_proposal_count`は、候補の判定の総数。`provisional_forward_count`は、検証の種別が`forward`の数（最終構成では、全部）。
  - `routing_soft_prediction_sample_count`は、混合予測を行った標本の数（常時有効なので、処理した標本の総数）。`routing_switching_recalibration_sample_count`・`routing_aggregation_recalibration_sample_count`は、Fixed-Shareと、globalの診断証拠の、再較正で再生した標本の数の、全clientの合計。
  - 離散列の型: 正誤はint8、モデルIDと位置はint32、文字列はstr、真偽はbool。`provisional_resolution_positions`は、確定位置（なければ提案位置）。クラスタリングの列は、来歴の観測の順。
- **Implications**: 計算の部品（手法に依存しない）と、FedSDAの参加者からの導出を分ける。

### 新の記録との対応（既存のhelperが照合済み）

- 標本ごとの記録`SamplePredictionRecord`: `combined_prediction_is_correct`（旧`history_accuracy`・`history_routing_switching_correct`）、`maximum_weight_model_id`（旧`history_model_id`・`history_routing_switching_leader_id`）。
- 適応記録: `training_model_switch_sample_indices`（旧`local_switch_positions`）、`adaptation_records`（旧`adaptation_events`。結果種別と旧の操作名の対応表は、testにある）。
- 判定記録の保持（旧`provisional_model_decisions`）。
- サーバ: `GlobalModelRepository`の登録の来歴とグローバルモデル、`CommunicationVolumeRecordStore`、`ModelClusteringRecordStore`。
- 再較正: `FixedSharePredictionWeightController.aggregation_recalibration_sample_count`、globalの診断証拠の`aggregation_recalibration_sample_count`。

## Design Decisions

### Decision: 指標だけをsourceに置き、旧の保存形式の列は、testで作る

- **Alternatives Considered**: (1)sourceが、旧と同じ名前・型の31列を作る——旧の保存形式（「なし」を−1で表す、旧の操作名ほか）を、新実装へ持ち込む。(2)列の型を新しく決めて、sourceが作る——保存の形（次の段階）を、先に決めることになる。
- **Selected Approach**: sourceは、指標（26項目に当たる値）だけを導出する。31列は、testが、新の記録の読取りから、旧の保存形式へ写して、旧の配列・goldenと照合する。
- **Rationale**: 列は、記録の値そのもので、計算がない。保存の形は、保存のspecで決める。照合によって、「記録から、旧の列がすべて作れる」ことは確かめられる。

### Decision: 計算の部品は、手法に依存しない層（evaluation）に置く

- **Rationale**: 変更位置の抽出、対応づけ、精度、検出の指標は、FedDriftほかの手法でも同じ。FedSDAの参加者から値を集める関数だけを、runtimeに置く。

### Decision: 指標の名前は、新実装の語で付け、旧の名前との対応は、testに置く

- **Rationale**: 既存の方針（旧の名前を、sourceへ持ち込まない）。対応表は、照合のために、testが持つ。

### Synthesis

- **Build vs. Adopt**: 通信量は、既存の読取り（`CommunicationVolumeSnapshot`）を、そのまま結果へ入れる（合計は、利用側が足す）。
- **Simplification**: goldenが比べない指標は、作らない。

## Risks & Mitigations

- 浮動小数の計算の順が、旧と違って、値がずれる — 精度と検出の指標は、整数の計数の比で、旧と同じ式で割る。実旧と、完全一致で照合する。
- goldenの照合が、環境に依存する — Windows用のgoldenとの照合は、Windowsのときだけ行うtestとして分ける（Linux用は、後のspec）。同じprocessの中の実旧との照合は、どの環境でも行う。

## References

- [candidate-validation-decision-record-retention](../candidate-validation-decision-record-retention/)（判定記録の保持と、調査の記録）、[fedsda-run-participant-preparation](../fedsda-run-participant-preparation/)（全体runの対照）。
