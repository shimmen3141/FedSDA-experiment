# Research & Design Decisions

## Summary

- **Feature**: `evaluation-concept-diagnostic-delivery`
- **Discovery Scope**: Extension（実行の枠の契約へ、引数を1つ足す）
- **Key Findings**:
  - 足りないのは、実行の枠が、概念列の値をclientへ渡す経路だけである。clientは、任意の引数で、すでに受け取れる。
  - single-run-executionの要求1.4（真の概念を、評価用の情報として区別し、観測値と混同しない）は、真の概念を、観測標本とは別の、診断専用の引数として渡す形で、守れる。
  - 「真の概念が判断へ影響しない」ことは、渡した全体runと、渡さない全体runの、診断以外の全状態の一致で確かめられる（fedsda-run-participant-preparationのtestを、向きを変えて使う）。

## Research Log

### 契約と、その利用箇所

- **Sources Consulted**: `execution/run_participant_contracts.py`、`execution/stream_protocol_execution_loop.py`、`runtime/single_run_execution.py`、`runtime/fedsda_run_client.py`（`process_observed_sample`）、`data/observed_streams.py`（`ClientConceptTrace`）、tests/refactoring/test_single_run_execution.py（区間の進行を直接呼ぶ7箇所、契約の引数のtest、全体runの呼出しの照合）、test_fedsda_run_client.py（区間の進行を呼ぶ1箇所）、test_fedsda_stream_protocol_run.py（test専用の中継）、fresh_process_smoke.py、single-run-executionのrequirements.md。
- **Findings**:
  - 区間の進行は、`participants`・`observed_client_streams`・集約間隔を受け取る。概念列は、全体runの実行が持っている。
  - 契約のtest（`test_run_participant_protocols_expose_only_declared_observation_operations`）は、`process_observed_sample`の引数が`("observed_sample", "sample_index")`であることを確かめ、説明に「観測処理に真の概念入力を要求しない」と書いている。
  - `ObservedSample`は、特徴とラベルだけを持つ。`ClientConceptTrace`は、clientのIDと、標本位置ごとの概念IDのtupleを持ち、生成時に型を確かめる。
  - 区間の進行は、「検証済みの参加者と観測列」を受け取る前提で、観測列を検査していない。
- **Implications**: 契約の`process_observed_sample`へ、`evaluation_concept_id`を足し、区間の進行へ、概念列の引数を足す。区間の進行は、概念列と観測列の対応（型、clientのIDと順、標本の数）を、最初に確かめる。

### 過去の仕様との関係

- single-run-executionの承認済みの文書は、書き換えない（規約）。要求1.4の文面は、本specの変更の後も成り立つ（真の概念は、評価用の情報として、観測値と区別して渡す）。契約の形（引数）と、testの説明だけが変わる。本specの設計に、その対応を書く。

## Design Decisions

### Decision: 真の概念を、観測標本とは別の、必須の引数として渡す

- **Alternatives Considered**: (1)`ObservedSample`へ概念を足す——観測値と評価用の真値が混ざる。要求1.4に反する。(2)clientの外で、同じ診断を計算する——clientの中の標本ごとの値（モデル別の損失、帰属）を外へ出す必要があり、変更が大きい。(3)任意の引数にする——渡し忘れが、黙って診断の欠落になる。
- **Selected Approach**: 契約の`process_observed_sample`へ、必須のkeyword引数`evaluation_concept_id: int`を足す。区間の進行は、概念列の値を、必ず渡す。
- **Rationale**: 実行の枠は、いつも概念列を持っている。必須にすると、欠落が起きない。`FedsdaRunClient`の側は、任意の引数のままにする（真の概念なしの単体の利用と、判断への不干渉のtestのため）。

### Decision: 判断への不干渉を、全体runの比較で保証する

- **Selected Approach**: test専用の中継（clientの操作を包み、真の概念を渡さずにclientを呼ぶ）で、「渡さない全体run」を作り、本来の全体runと、診断以外の全状態を比べる。
- **Rationale**: 個々の部品の検査ではなく、結果で保証する。真の概念を使う処理が、後から判断の経路へ足されても、このtestが検出する。

### Synthesis

- **Build vs. Adopt**: 新しいmoduleはない。3つのsourceの小さい変更。
- **Simplification**: サーバの操作の契約は変えない（クラスタリングの真の概念の一致は、clientの割当概念の計数から求まる）。

## Risks & Mitigations

- 契約の変更で、実行の枠の既存のtestが、弱くなる — 既存のtestの検査は、そのまま残し、真の概念の受渡しの検査（clientと位置の対応）を足す。
- 真の概念が、判断へ漏れる — 不干渉のtestを、恒久の保証として残す。

## References

- [single-run-execution](../single-run-execution/)（実行の枠。要求1.4）、[fedsda-run-participant-preparation](../fedsda-run-participant-preparation/)（全体runの対照と、test専用の中継）。
