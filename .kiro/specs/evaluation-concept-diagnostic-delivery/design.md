# Design Document: evaluation-concept-diagnostic-delivery

## Overview

**Purpose**: 実行の枠が、標本ごとの真の概念を、診断専用の引数として、clientへ渡す。

**Users**: リファクタリングを進める研究者。全体runで、真の概念に依存する診断が得られる。

**Impact**: 実行の枠の、clientの標本処理の契約と、区間の進行の引数を変える（完了済みのmoduleの変更）。

### Goals

- 全体runが、test専用の中継なしで、診断まで実旧と一致する。
- 真の概念が、診断以外の状態へ影響しないことを、testで保証する。

### Non-Goals

- 診断の中身、保存と集計、真のドリフト位置を使う指標、サーバの操作の契約。

## Boundary Commitments

### This Spec Owns

- `RunClientOperations.process_observed_sample`の引数`evaluation_concept_id`。
- `run_stream_protocol_intervals`の引数`evaluation_concept_traces`と、その検査、受渡し。
- `execute_stream_protocol_run`が、概念列を区間の進行へ渡すこと。

### Out of Boundary

- `ObservedSample`（変えない。真の概念を持たない）。概念列と観測列の生成。clientの標本処理の中の処理。

### Allowed Dependencies

- `execution/stream_protocol_execution_loop.py`が、`data/observed_streams.py`の`ClientConceptTrace`を使う（既に`ClientObservedStream`を使っている）。新しい向きの依存はない。

### Revalidation Triggers

- 実行の枠の契約を満たすclientの実体（`FedsdaRunClient`、後で足す他の手法のclient）。区間の進行を直接呼ぶ箇所。

## File Structure Plan

### Modified Files

- `execution/run_participant_contracts.py` — `process_observed_sample`へ、`evaluation_concept_id: int`を足す。
- `execution/stream_protocol_execution_loop.py` — `run_stream_protocol_intervals`へ、`evaluation_concept_traces`を足す。検査と受渡し。
- `runtime/single_run_execution.py` — 概念列を、区間の進行へ渡す。
- `runtime/fedsda_run_client.py` — 説明だけ（実行の枠が、真の概念を渡すこと）。
- `tests/refactoring/test_single_run_execution.py` — 観測用のclient、区間の進行の呼出し、契約のtest、受渡しのtest。
- `tests/refactoring/test_fedsda_run_client.py`、`tests/refactoring/test_fedsda_stream_protocol_run.py`、`tests/refactoring/fresh_process_smoke.py`、`tests/refactoring/test_single_run_dependency_boundaries.py`（許可集合に変更があれば）。

## 過去の仕様との対応

- single-run-executionの要求1.4（真の概念・ドリフト位置を、評価用の情報として区別し、観測値と混同しない）は、変更の後も成り立つ: 真の概念は、観測標本の外の、名前で用途を示した引数（`evaluation_concept_id`）として渡し、clientは、診断にだけ使う。
- 変わるのは、契約の形である: 変更の前は、clientの標本処理が、真の概念を受け取らなかった。変更の後は、診断専用の引数として受け取る。single-run-executionの文書は、当時の合意の記録として、書き換えない。

## Requirements Traceability

| Requirement | Components |
|-------------|------------|
| 1.1, 1.2, 1.4 | `run_stream_protocol_intervals`、`RunClientOperations` |
| 1.3 | `execute_stream_protocol_run` |
| 2.1, 2.2 | 契約の説明、不干渉のtest |
| 3.1, 3.2 | 全体runの対照test（中継なし） |
| 4.1〜4.3 | 実行の枠のtest、共用script、依存境界test |

## Components and Interfaces

### execution: RunClientOperations

```python
def process_observed_sample(
    self, *, observed_sample: ObservedSample, sample_index: int, evaluation_concept_id: int
) -> None: ...
```

- `evaluation_concept_id`は、その標本の真の概念ID（評価用の真値）。clientは、診断にだけ使い、予測・検出・学習・モデルの選択に使わない（契約の説明に書く）。

### execution: run_stream_protocol_intervals

```python
def run_stream_protocol_intervals(
    *, participants: RunParticipants, observed_client_streams: tuple[ClientObservedStream, ...],
    evaluation_concept_traces: tuple[ClientConceptTrace, ...],
    server_aggregation_interval_per_client_samples: int,
) -> tuple[RunExecutionEvent, ...]: ...
```

- 最初に、概念列を確かめる（どのclientの操作も呼ぶ前。同じmoduleの`validate_evaluation_concept_traces_match_observed_streams`）: exact tuple、各要素がexact `ClientConceptTrace`、数が観測列と同じ、位置ごとに、clientのIDが観測列と同じ、標本の数が観測列と同じ。不正は`TypeError`／`ValueError`。
- 標本の処理: `process_observed_sample(observed_sample=…, sample_index=…, evaluation_concept_id=概念列[client][標本位置])`。記録する実行の記録（段、client、位置、ラウンド）は、変えない。

### runtime: execute_stream_protocol_run

- 生成した`evaluation_concept_traces`を、`run_stream_protocol_intervals`へ渡す。ほかは変えない。

## Error Handling

- 概念列の不正は、区間の進行の最初の検査で`TypeError`／`ValueError`。全体runの中では、概念列は、同じ実行の中で生成したものなので、起きない。

## Testing Strategy

### Unit Tests

- 受渡し（1.1, 4.1）: 観測用のclientが、受け取った真の概念を記録する。clientと位置で値が違う概念列で、渡された値が、概念列の値と一致すること。全体runでは、渡された値が、結果の概念列と一致すること。
- 契約（2.2）: `process_observed_sample`の引数が、`observed_sample`・`sample_index`・`evaluation_concept_id`（すべてkeyword専用）であること。観測標本が、真の概念を持たないこと。
- 拒否（1.4）: 概念列の型、数、clientのID、標本の数の不一致で、clientの操作が1回も呼ばれないこと。

### Integration Tests

- 全体runの対照（3.x）: fedsda-run-participant-preparationの対照の全条件を、test専用の中継なしで、実旧と照合する（各clientの全状態は、診断を含む）。
- 不干渉（2.1）: 真の概念を渡さずにclientを呼ぶ、test専用の中継で実行した全体runが、本来の全体runと、真の概念に依存する項目だけが違うこと。違う項目が、実際に違っていること（比較が働いていること）。

### fresh process・依存

- 共用script（4.2）: 全体runで、標本ごとの記録の概念が、概念列と一致し、真の概念別の診断証拠が作られること。clientを組み立てて区間の進行を呼ぶ流れは、概念列を渡す。
- 依存境界（4.3）。
