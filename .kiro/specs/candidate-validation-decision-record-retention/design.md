# Design Document: candidate-validation-decision-record-retention

## Overview

**Purpose**: clientが、候補検証の判定記録を、起きた順に保持する。

**Users**: リファクタリングを進める研究者。次のspecで、候補の判定の列と指標を導出する。

**Impact**: clientのownerの束へ、ownerを1つ足す。clientの2つの操作が、戻り値の判定記録を、ownerへ足す。

### Goals

- 保持した一覧が、実旧の `provisional_model_decisions` と、全項目で一致する。
- 既存の処理の引数と挙動を変えない。

### Non-Goals

- 判定記録の中身、指標と列の導出、保存。最終構成で通らない、旧の判定の記録。

## Boundary Commitments

### This Spec Owns

- `CandidateValidationDecisionRecordStore`（新規）。
- `FedsdaRunClientOwners.candidate_validation_decision_record_store`、組立てでの生成、2つの操作での追加。

### Out of Boundary

- 2つの判定記録の型と、その生成。標本1件の処理、保持中の候補検証の進行と終端の回収、適応記録。

### Allowed Dependencies

- 新しいmodule（手法の層）→ 同じpackageの、2つの判定記録の型。
- `runtime/fedsda_run_client.py` → 新しいmodule（runtime→手法の層。既存の向き）。

### Revalidation Triggers

- 判定記録の型の変更。clientの2つの操作の戻り値の形の変更。

## File Structure Plan

### New Files

- `src/federated_learning_experiments/methods/fedsda/candidate_model_selection/candidate_validation_decision_record_store.py`
- `tests/refactoring/test_candidate_validation_decision_record_store.py`

### Modified Files

- `runtime/fedsda_run_client.py` — ownerの束のfield、組立て、2つの操作。
- `tests/refactoring/test_fedsda_run_client.py` — clientの全状態の新旧照合へ、候補の判定の照合を足す。組立てのtest（別々の空のowner）。
- `tests/refactoring/test_fedsda_stream_protocol_run.py` — 経路（判定の種類）と、不干渉のtestでの一致。
- `tests/refactoring/fresh_process_smoke.py`、`tests/refactoring/test_single_run_dependency_boundaries.py`。

## Requirements Traceability

| Requirement | Components |
|-------------|------------|
| 1.1〜1.3, 1.6 | `FedsdaRunClient`、`assemble_fedsda_run_client` |
| 1.4, 1.5 | `CandidateValidationDecisionRecordStore` |
| 2.1〜2.3 | clientの全状態の新旧照合、全体runの対照 |
| 3.1〜3.4 | 既存の対照、不干渉のtest、共用script、依存境界test |

## Components and Interfaces

### methods/fedsda: CandidateValidationDecisionRecordStore

```python
class CandidateValidationDecisionRecordStore:
    def append_candidate_validation_decision_record(
        self, *,
        decision_record: PostAlarmCandidateValidationDecisionRecord
        | IncompletePostAlarmCandidateValidationDecisionRecord,
    ) -> None: ...
    def snapshot_candidate_validation_decision_records(
        self,
    ) -> tuple[
        PostAlarmCandidateValidationDecisionRecord
        | IncompletePostAlarmCandidateValidationDecisionRecord, ...
    ]: ...
```

- 追加: exactに、2つの型のどちらか。違えば`TypeError`（保持は変えない）。記録は不変の値（frozen）なので、そのまま持つ。
- 読取り: 足した順のtuple。後からの追加で変わらない。

### runtime: FedsdaRunClient

- `process_observed_sample`: 標本1件の処理が返った後、結果の`held_validation_advance.validation_progress.completed_validation`が`None`でなければ、その`decision_record`を足す。戻り値は変えない。
- `finalize_incomplete_candidate_validation`: 終端の回収が返った後、結果が`None`でなければ、`incomplete_validation_finalization.decision_record`を足す。戻り値は変えない。
- `assemble_fedsda_run_client`: 空のownerを作って、ownerの束へ入れる。

## 失敗時の扱い（既知の制約）

- 標本1件の処理が、候補検証の確定の後の段で例外を出すと、適応記録は足されているが、判定記録は足されない。実行の枠は、その時点でrunを中断し、結果を返さないので、導出へは影響しない。

## 旧との対応（testが照合する）

| 旧 `ProvisionalModelDecision` | 確定の判定記録 | 未完了用の判定記録 |
|---|---|---|
| `position` | `proposal_sample_index` | `proposal_sample_index` |
| `resolution_position` | `resolution_sample_index` | `finalization_sample_index` |
| `detector` | `detector_name` | `detector_name` |
| `interval_count`・`training_count` | `candidate_training_interval_sample_count` | 同じ |
| `validation_count` | 評価の`validation_sample_count` | `validation_sample_count` |
| `accepted` | 評価の`candidate_accepted` | 偽 |
| `reason` | 評価の`decision_reason`（名前の対応表） | `insufficient_forward_data` |
| `reference_model_id` | 評価の`comparison_reference_model_id` | なし |
| 4つの平均損失 | 評価の4つの平均損失 | NaN |
| `reference_historical_mean` | 評価の`reference_historical_mean_loss`（なしはNaN） | NaN |
| `validation_source` | `forward` | `forward` |

## Testing Strategy

### Unit Tests

- owner（1.4, 1.5）: 足した順に読めること、読取りが後からの追加で変わらないこと、2つの型を受け入れること、ほかの値（`None`、辞書、適応記録、派生型）を、保持を変えずに拒否すること。
- 組立て（1.6）: 2つのclientが、別々の空のownerを持つこと。

### Integration Tests

- clientの全状態の新旧照合のhelper（2.1, 2.2）へ、保持した一覧と、実旧の`provisional_model_decisions`の照合を足す。このhelperは、clientの軌跡の対照と、全体runの対照の全条件で呼ばれる。
- 全体runの対照（2.3）: 通った経路へ、判定の種類（採用、区間の判定での棄却、再利用または維持、終端の回収）を足し、経路の網羅のtestで確かめる。
- 不干渉（3.2）: 真の概念を捨てる全体runと、本来の全体runで、保持した一覧が同じ。
- 既存の対照（3.1）: そのまま通る。

### fresh process・依存

- 共用script（3.3）: 全体runで、判定記録が1件以上保持され、適応記録の候補検証の結果の件数と一致すること。
- 依存境界（3.4）。
