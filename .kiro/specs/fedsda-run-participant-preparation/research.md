# Research & Design Decisions

## Summary

- **Feature**: `fedsda-run-participant-preparation`
- **Discovery Scope**: Extension（移植済みの部品を、実行の枠の契約へつなぐ）
- **Key Findings**:
  - 全体runの接続は大きいので、「実行の枠への接続」（本spec）と、「指標の導出とgoldenとの照合」（次のspec）に分けた。
  - 実行の枠（`execute_stream_protocol_run`）は、初期準備→概念列→観測列→区間の進行→終端の順を、旧と同じ乱数の消費で行う（移植済み）。足りないのは、`RunParticipantFactory`と`RunServerOperations`の実体だけである。
  - 旧の全体runは、指標の計算より前の部分（準備→ループ→終端）を、旧の部品を旧の順に呼んで再現できる。その最終状態を、新の全体runの最終状態と照合する。
  - 実行の枠の契約は、clientへ真の概念を渡さない。旧のclientは、真の概念なしでは動かない（標本ごとの記録で`int(concept_id)`を読む）。

## Research Log

### 旧の全体の流れと、新の実行の枠

- **Sources Consulted**: 旧`experiment.py`（`run_random_drift_experiment` 2211〜2400行、`_setup_server_and_clients` 336〜358行、`_pretrain_initial_model` 294〜333行、`_process_per_sample_round`・`_run_per_sample_timestep` 82〜118行、`MODE_SPECS`の最終構成の定義）、`servers/base.py`（`record_client_state_summaries`、`finalize_protocol`）、`data/schedules.py`・`data/streams.py`、新`runtime/single_run_execution.py`、`execution/`の4 module、tests/refactoring/test_single_run_execution.py（旧の概念列・観測列との照合）、tests/test_proposed_regression.py（最終構成の設定と、goldenの比較項目）。
- **Findings**:

| 旧 | 新 |
| --- | --- |
| seedで3つの乱数を初期化 | `create_run_random_sources`、`isolated_cpu_torch_random_state` |
| `_setup_server_and_clients`（事前学習→サーバ→client） | `RunParticipantFactory.prepare_run`（本spec） |
| 概念列・観測列の生成 | `generate_random_client_concept_traces`、`build_sine_client_observed_streams` |
| `_run_per_sample_timestep` | `run_stream_protocol_intervals`（clientの5操作と、サーバの3操作を呼ぶ） |
| `record_client_state_summaries` | `RunServerOperations.record_client_states_before_synchronization`（本spec） |
| `run_round(t, clustering_enabled=has_new)` | `RunServerOperations.synchronize_models`→`synchronize_models_in_server_round`（本spec） |
| `finalize_incomplete_forward_validation` | `FedsdaRunClient.finalize_incomplete_candidate_validation` |
| `server.finalize_protocol`（最終構成のサーバは、基底の何もしない実装） | `RunServerOperations.finalize_started_communications`（本spec。何もしない） |
| `compute_metrics`、通信量・最終モデル数の集計、保存 | 次のspec |

  - 旧の`flush`を持つclient（FedDrift）の終端処理は、最終構成のclientには当たらない。
  - 最終構成のclientは、すべて状態を報告する（`reports_state_summary`）。
  - goldenの比較項目（指標と、標本ごとの列のhash）は、真の概念に依存する診断を含まない。計算量の指標（`compute_*`）は、新実装に記録がない（未移植）。次のspecで扱う。
- **Implications**: factoryとサーバの操作の実体は薄い。検証の中心は、全体runの最終状態の、実旧との照合である。

### 真の概念の扱い

- **Context**: 再開案内は、「実行の枠からclientへ真の概念を渡すか（契約を変えるか）」を、全体runを接続するspecで、goldenの比較項目を見て決める、としている。
- **Findings**: goldenの比較項目は、真の概念に依存する診断を含まない。実行の枠の契約には、「観測処理に真の概念を要求しない」ことを確かめるtestがある（single-run-executionの要求）。真の概念に依存するのは、診断だけである（真の概念別の診断証拠、割当概念の計数、標本ごとの記録の概念、クラスタリングの真の概念の一致）。
- **Implications**: 本specでは、契約を変えない。対照testは、test専用の中継（clientの操作を包み、標本位置から真の概念を引いて、clientの任意の引数へ渡す）で、真の概念を渡して、診断まで実旧と照合する。真の概念を渡さない全体runが、診断以外で同じ状態になることも確かめる。全体runで、真の概念に依存する診断を出すには、契約の変更（またはそれに代わる経路）が要る。その判断は、ユーザーへ確認する（再開案内の「ユーザー確認待ち」へ書く）。

### oracle

- 旧の全体runは、`run_random_drift_experiment`を呼ぶと、サーバとclientを返さない（指標のdictだけ）。その本体のうち、指標の計算より前の部分を、testが、旧の部品（`_setup_server_and_clients`、`make_concept_schedules`、`build_data_streams`、`_run_per_sample_timestep`、`finalize_incomplete_forward_validation`、`finalize_protocol`）を、旧と同じ順に呼んで実行する。
- 旧の設定の差し替え（`set_legacy_configuration`）に、全体runの条件（dataset、client数、標本数、集約間隔、概念列の条件、再較正の方式）を足す。

## Design Decisions

### Decision: 全体runの接続を、2つのspecに分ける

- **Rationale**: 実行の枠への接続（本spec）は、状態の照合で完結する。指標の導出は、旧の`compute_metrics`と保存の列の、新の記録との対応づけで、範囲が広い（計算量の指標の未移植を含む）。

### Decision: 設定の束を、runtimeに置く仮の形にする

- **Selected Approach**: `FedsdaRunParticipantSettings`が、clientの設定の束と、サーバ側・準備の設定をまとめる。旧で1つの値を2箇所で使う組（閾値）は、束が、同じ値であることを確かめる。
- **Rationale**: 完全なrun設定（保存表現、preset）は、後のspecで決める。clientの設定の束と同じ考え方。

### Decision: 同期の結果を、サーバが保持する

- **Selected Approach**: `FedsdaRunServer`が、ラウンドごとの`ServerRoundSynchronization`を、順に保持し、読取り用に返す。
- **Rationale**: 実行の枠の契約では、`synchronize_models`は何も返さない。結果（登録、クラスタ、ID対応、再較正）は、診断と検証に使う。

### Synthesis

- **Build vs. Adopt**: 新しく書くのは、サーバの操作の実体、設定の束、factory。
- **Simplification**: SINE以外のdatasetは扱わない（実行の枠が、SINEだけを受け入れる）。

## Risks & Mitigations

- 全体runで、部品の対照では通らなかった経路の違いが見つかる — 全体runの最終状態を、複数のseedと条件で、実旧と照合する。差が出たら、原因の部品を特定して直すか、旧の挙動として記録する。
- clientの組立てと、サーバの乱数の共有の順が、旧と違う — 最終状態と、runの乱数の最終状態の照合で確かめる。

## References

- [model-clustering-and-consolidation](../model-clustering-and-consolidation/)（サーバの1ラウンドの同期）、[fedsda-run-client-assembly](../fedsda-run-client-assembly/)、[initial-model-pretraining](../initial-model-pretraining/)。
