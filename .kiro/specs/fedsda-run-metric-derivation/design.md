# Design Document: fedsda-run-metric-derivation

## Overview

**Purpose**: 新の全体runの結果から、goldenが比べる指標（計算量を除く26項目）を導出し、記録から作る離散列とあわせて、旧の結果・goldenと照合する。

**Users**: リファクタリングを進める研究者。

**Impact**: 新しいmoduleを2つ足す。既存のsourceは変えない。

### Goals

- 導出した指標が、実旧の結果と完全に一致し、Windows用のgoldenと一致する。
- 新の記録から、旧の保存形式の31列がすべて作れることを、照合で示す。

### Non-Goals

- 計算量の指標、goldenが比べない指標、保存と図、Linux用のgolden、sine2以外のdataset。

## Boundary Commitments

### This Spec Owns

- `evaluation/run_metric_calculations.py`（計算の部品と、指標の設定）。
- `runtime/fedsda_run_metric_derivation.py`（FedSDAの全体runからの導出と、結果の型）。

### Out of Boundary

- 全体runの実行、参加者、記録のowner（読むだけ）。固定旧実装とgolden。

### Allowed Dependencies

- `evaluation/run_metric_calculations.py` → 標準libraryだけ。
- `runtime/fedsda_run_metric_derivation.py` → evaluation（計算の部品、通信量の読取りの型）、execution（実行の枠の結果、参加者の型）、runtime（`FedsdaRunClient`、`FedsdaRunServer`、パラメータの分割）。

### Revalidation Triggers

- 記録のownerの読取りの形の変更。clientとサーバのownerの束の変更。

## File Structure Plan

### New Files

- `src/federated_learning_experiments/evaluation/run_metric_calculations.py`
- `src/federated_learning_experiments/runtime/fedsda_run_metric_derivation.py`
- `tests/refactoring/test_run_metric_calculations.py`
- `tests/refactoring/test_fedsda_run_metric_derivation.py`

### Modified Files

- `tests/refactoring/fresh_process_smoke.py`、`tests/refactoring/test_single_run_dependency_boundaries.py`

## Requirements Traceability

| Requirement | Components |
|-------------|------------|
| 1.1〜1.3 | `extract_concept_change_sample_indices`、`count_events_matched_to_concept_changes` |
| 2.1〜2.3 | `calculate_prediction_accuracy`、`calculate_stable_period_prediction_accuracy`、`calculate_detection_metrics` |
| 3.1〜3.3 | `derive_fedsda_run_metrics`、`FedsdaRunMetrics`、`RunMetricSettings` |
| 4.1〜4.5 | 2つのtest file |
| 5.1, 5.2 | 共用script、依存境界test |

## Components and Interfaces

### evaluation: run_metric_calculations

```python
@dataclass(frozen=True, kw_only=True)
class RunMetricSettings:
    maximum_detection_delay_sample_count: int        # 旧 DELAY_TOLERANCE
    post_change_recovery_window_sample_count: int    # 旧 STABLE_WINDOW

@dataclass(frozen=True, kw_only=True)
class EventMatchCounts:
    matched_event_count: int
    matched_concept_change_count: int

@dataclass(frozen=True, kw_only=True)
class DetectionMetrics:
    detection_precision: float
    detection_recall: float
    detection_f1: float
    detection_count: int
    concept_change_count: int
    matched_detection_count: int

def extract_concept_change_sample_indices(*, concept_ids_by_sample_index: tuple[int, ...]) -> tuple[int, ...]: ...
def count_events_matched_to_concept_changes(
    *, concept_change_sample_indices: tuple[int, ...], event_sample_indices: tuple[int, ...],
    maximum_delay_sample_count: int,
) -> EventMatchCounts: ...
def calculate_prediction_accuracy(*, prediction_correctness_by_client: tuple[tuple[bool, ...], ...]) -> float: ...
def calculate_stable_period_prediction_accuracy(
    *, prediction_correctness_by_client: tuple[tuple[bool, ...], ...],
    concept_change_sample_indices_by_client: tuple[tuple[int, ...], ...],
    recovery_window_sample_count: int,
) -> float: ...
def calculate_detection_metrics(
    *, concept_change_sample_indices_by_client: tuple[tuple[int, ...], ...],
    detection_sample_indices_by_client: tuple[tuple[int, ...], ...],
    maximum_delay_sample_count: int,
) -> DetectionMetrics: ...
```

- 対応づけ: 変更位置を与えられた順に見る。イベントは、与えられた順に、未使用の最初のもの。`calculate_detection_metrics`は、検出位置を昇順にしてから対応づける（旧と同じ）。
- 対応した変更位置の数と、対応したイベントの数は、1対1の対応なので等しい。旧は別々に数えるので、両方を返す。
- 定常精度: 変更位置を昇順にして、標本位置以下の最大の変更位置を二分探索で求める。
- 検査: 位置は、非負のbuiltin int（boolを除く）。列はexact tuple。clientごとの列の数は、互いに等しい。設定は、非負のbuiltin int。

### runtime: fedsda_run_metric_derivation

```python
@dataclass(frozen=True, kw_only=True)
class FedsdaRunMetrics:
    prediction_accuracy: float
    stable_period_prediction_accuracy: float
    training_model_switch_detection_metrics: DetectionMetrics
    final_global_model_count: int
    communication_volume: CommunicationVolumeSnapshot
    final_parameter_value_count: int
    final_parameter_byte_count: int
    candidate_validation_decision_count: int
    mixed_prediction_sample_count: int
    prediction_weight_recalibration_replayed_sample_count: int
    global_diagnostic_recalibration_replayed_sample_count: int

def derive_fedsda_run_metrics(
    *, run_result: StreamProtocolRunResult, participants: RunParticipants,
    run_metric_settings: RunMetricSettings,
) -> FedsdaRunMetrics: ...
```

- 検査（何も読む前）: `run_result`・`participants`・`run_metric_settings`のexact型、clientの操作が、すべてexact `FedsdaRunClient`、サーバの操作がexact `FedsdaRunServer`、clientの数が概念列の数と同じで、位置ごとにclientのIDが同じ。
- 正誤の列: 各clientの標本ごとの記録の`combined_prediction_is_correct`。
- 変更位置: 実行の枠の結果の概念列（全体。処理されない末尾を含む）から。
- 検出: 各clientの適応記録の`training_model_switch_sample_indices`。
- 最終のパラメータ量: グローバルモデルがなければ0。あれば、最初のグローバルモデルの共有部（`split_shared_and_concept_specific_parameters`）の値の数・バイト数と、各モデルの概念固有部の値の数・バイト数の合計。
- 混合予測を行った標本の数: 最終構成は常時有効なので、標本ごとの記録の総数。
- 状態を変えない: ownerの読取りだけを使う。

## 旧と違う点

- 真の変更位置を、概念列の全体から取る点は、旧と同じ（処理されない末尾の変更は、見逃しとして数える）。
- 旧の`provisional_forward_count`（検証の種別が`forward`の判定の数）は、最終構成では、判定の総数と同じ値なので、別の項目にしない。testは、旧の2つの指標の両方を、`candidate_validation_decision_count`と照合する。
- 通信量の合計（上り＋下り）は、結果に持たない。testが足して、旧の`comm_*_total`と照合する。

## Error Handling

- 不正な入力は`TypeError`／`ValueError`。導出は、状態を変えないので、失敗しても、後始末は要らない。

## Testing Strategy

### Unit Tests（test_run_metric_calculations.py）

- 実旧の関数をoracleにする（4.5）: `extract_true_drift_events`、`match_events`、`_stable_accuracy`・`compute_metrics`（旧のclientの代わりに、必要な属性だけを持つ値を渡す）。固定seedで作った多数の入力と、境界（イベントなし、変更なし、許容遅延0、同じ位置のイベント、窓0、窓が区間より長い、標本なし）で、完全一致。
- 拒否（1.3ほか）。

### Integration Tests（test_fedsda_run_metric_derivation.py）

- goldenの条件（4.1, 4.2）: moduleで1回だけ、(a)旧の設定の文脈の中で、実旧の`run_random_drift_experiment`を、保存つきで実行、(b)同じ条件の新の全体runを実行して導出。26指標の完全一致、31列の完全一致（形・型・値）。経路の件数（登録、採用、棄却、統合、切替）が、旧の回帰testと同じ下限を満たすこと。
- Windows用のgolden（4.3）: 同じ結果を、goldenのsine2の26指標（1e-9）と、31列（形とhash）に照合する。Windows以外ではskipする（理由を表示する）。計算量の7項目は、照合しないことを、testに明記する。
- 小さい条件（4.4）: 全体runの対照の条件から選んだものと、処理されない末尾がある条件で、実旧の全体run（既存のhelper）の後に、旧の`compute_metrics`ほかを呼んだ結果と、完全一致。
- 状態を変えない（3.2）、拒否（3.3）。

### fresh process・依存

- 共用script（5.1）: 全体runの後で導出し、値の範囲と、記録との整合（検出の数＝切替の位置の総数ほか）を確かめる。
- 依存境界（5.2）。
