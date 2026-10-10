# Design Document: comparison-computation-metrics

## Overview

**Purpose**: 手法の比較に使う計算量（clientとサーバ、標本あたり、保有モデルあたり、ラウンドごと）を、新実装の全体runから得られるようにする。

**Users**: リファクタリングを進める研究者。

**Impact**: 新しいmoduleを2つ足す。既存のmoduleを6つ変更する（集約、統合、client、計測つきの全体run、指標の導出）。処理の結果は変えない。

### Goals

- 主張(a)(b)(c)（requirements.mdの冒頭）に対応する量が、同じ式で得られる。
- サーバの計算が、clientと同じ単位（積和演算）で得られる。

### Non-Goals

- 保存、図、回帰。FedDriftの計測。検出器の計算の換算。実行時間。

## 主張と量の対応

| 主張 | 使う量 |
|---|---|
| (a) clientの計算が少ない | clientの積和演算の数（順伝播。逆伝播の見積りを含むものも）。標本1件あたりに直した値（ストリームの長さ・client数が違っても比べられる） |
| (b) 1モデルあたりの計算量 | 概念固有部の積和演算の数 ÷ Σ（client・標本ごとの保有モデル数）。保有していた期間に応じて重みがつく。共有部の計算は、保有モデル数に依らない（標本ごとに1回）ので、標本あたりの値として、別に示す。全体（共有部を含む）÷Σ保有モデル数、の値も持つが、保有モデルが増えると、共有部の分が薄まって小さくなる点に注意する（独立レビューの指摘で、概念固有部だけの値を足した） |
| (c) モデル数が増えても、増えにくい | 共有部と概念固有部の内訳（共有部は、保有モデル数に依らない）。ラウンドごとの計算量と、標本ごとの保有モデル数の列（関係を、分析で見る） |
| サーバの負荷 | サーバの積和演算の数（集約＋統合。診断のパラメータ距離は別） |

## Boundary Commitments

### This Spec Owns

- `evaluation/held_model_count_record_store.py`、`evaluation/computation_cost_summary.py`（新規）。
- `ClientModelAggregation`・`ModelConsolidation`の、計数のfield。
- `FedsdaRunClientOwners.held_model_count_record_store`と、標本の処理の後の追加。
- `FedsdaMeasuredRun`の、ラウンドごと・終端の、モデルの計算。
- `FedsdaRunMetrics`の、サーバの計数、標本数、保有モデル数の合計、計算量のまとめ。

### Out of Boundary

- 計測の数え方、集約・統合の計算、実行の枠、通信量。

### Allowed Dependencies

- 計算量のまとめ（evaluation）→ 計測の計数の型（learning/models）。
- 指標の導出（runtime）→ 計算量のまとめ、同期の結果。

### Revalidation Triggers

- 集約・統合の、パラメータの足し方の変更。実行の枠の、ラウンドの中の操作の順の変更。

## File Structure Plan

### New Files

- `src/federated_learning_experiments/evaluation/held_model_count_record_store.py`
- `src/federated_learning_experiments/evaluation/computation_cost_summary.py`
- `tests/refactoring/test_held_model_count_record_store.py`、`tests/refactoring/test_computation_cost_summary.py`

### Modified Files

- `runtime/server_model_registration_and_aggregation.py`、`runtime/model_clustering_and_consolidation.py` — 計数のfield。
- `runtime/fedsda_run_client.py` — 保有モデル数の記録。
- `runtime/fedsda_measured_run_execution.py` — ラウンドごとのモデルの計算。
- `runtime/fedsda_run_metric_derivation.py` — 指標の追加。
- 既存のtest（集約、統合、client、計測つきの全体run、指標の導出）、共用script、依存境界test。

## Requirements Traceability

| Requirement | Components |
|-------------|------------|
| 1.1, 1.3, 1.4 | `ClientModelAggregation.weighted_parameter_multiply_accumulate_count` |
| 1.2〜1.4 | `ModelConsolidation`の2つのfield |
| 2.1〜2.3 | `HeldModelCountRecordStore`、`FedsdaRunClient` |
| 3.1〜3.4 | `FedsdaMeasuredRun`、`RoundModelComputationCounts` |
| 4.1〜4.3 | `summarize_computation_cost`、`ComputationCostSummary`、`ServerComputationCounts` |
| 4.4 | `derive_fedsda_run_metrics` |
| 5.x | 既存のtestへの照合の追加、共用script、依存境界test |

## Components and Interfaces

### runtime: 集約と統合の結果

```python
class ClientModelAggregation:
    ...
    weighted_parameter_multiply_accumulate_count: int   # 重み付きの和へ足した、パラメータの値の数

class ModelConsolidation:
    ...
    consolidation_parameter_multiply_accumulate_count: int          # 統合の加重平均で足した、値の数
    diagnostic_parameter_distance_multiply_accumulate_count: int    # 対ごとに、概念固有部の値の数×3
```

- 集約: 重み付きの和へ足したパラメータ（`_add_weighted_parameters`へ渡したもの）は、上りとして数えるパラメータの一覧（`uploaded_parameter_snapshots`）と同じなので、その一覧の値の数（`numel`の合計）を、計数にする。
- 統合: `_compute_sample_weighted_mean_parameters`が、足したメンバーの値の数も返す（重みが0のメンバーと、重みがすべて0のクラスタは、0）。
- パラメータ距離: 距離を計算した対ごとに、概念固有部の値の数×3。

### evaluation: HeldModelCountRecordStore

```python
class HeldModelCountRecordStore:
    def append_held_model_count(self, *, held_model_count: int) -> None: ...   # 正のbuiltin int
    def snapshot_held_model_counts(self) -> tuple[int, ...]: ...               # 標本の順
```

- `FedsdaRunClient.process_observed_sample`が、処理の前に、保有モデル数を読み、処理が成功して返った後に、足す。

### runtime: 計測つきの全体run

```python
@dataclass(frozen=True, kw_only=True)
class RoundModelComputationCounts:
    round_index: int
    local_processing_model_computation_counts: ModelComputationCounts   # 標本の処理と、区間末の学習
    synchronization_model_computation_counts: ModelComputationCounts    # 同期の間の、clientの評価と再較正

class FedsdaMeasuredRun:
    ...
    round_model_computation_counts: tuple[RoundModelComputationCounts, ...]
    finalization_model_computation_counts: ModelComputationCounts       # 最後の同期の後（終端の処理）
```

- 包み（非公開）: 準備の後、サーバの操作を包む。`record_client_states_before_synchronization`の入口と、`synchronize_models`の出口で、計数を控える。ほかは、そのまま渡す。実行の枠が受け取る参加者は、包んだサーバの操作を持つ。結果の`participants`は、包む前の参加者（factoryが保持するもの）。
- ラウンドの値と、終端の値の合計が、`model_computation_counts`（準備の後から終わりまで）と一致する。

### evaluation: computation_cost_summary

```python
@dataclass(frozen=True, kw_only=True)
class ServerComputationCounts:
    aggregation_parameter_multiply_accumulate_count: int
    consolidation_parameter_multiply_accumulate_count: int
    diagnostic_parameter_distance_multiply_accumulate_count: int

@dataclass(frozen=True, kw_only=True)
class ComputationCostSummary:
    processed_sample_count: int
    held_model_sample_count: int
    mean_held_model_count: float
    client_forward_multiply_accumulate_count: int
    client_estimated_backward_multiply_accumulate_count: int
    client_shared_part_forward_multiply_accumulate_count: int
    client_concept_specific_part_forward_multiply_accumulate_count: int
    server_multiply_accumulate_count: int                       # 集約＋統合（診断を除く）
    server_diagnostic_multiply_accumulate_count: int
    client_forward_multiply_accumulate_count_per_processed_sample: float
    client_forward_multiply_accumulate_count_per_held_model_sample: float
    client_forward_and_backward_multiply_accumulate_count_per_processed_sample: float
    client_forward_and_backward_multiply_accumulate_count_per_held_model_sample: float
    client_shared_part_forward_multiply_accumulate_count_per_processed_sample: float
    client_concept_specific_part_forward_multiply_accumulate_count_per_held_model_sample: float

def summarize_computation_cost(
    *, model_computation_counts: ModelComputationCounts, server_computation_counts: ServerComputationCounts,
    processed_sample_count: int, held_model_sample_count: int,
) -> ComputationCostSummary: ...
```

- `processed_sample_count`は、全clientの、処理した標本数の合計。`held_model_sample_count`は、全client・全標本の、保有モデル数の合計。
- 比は、分母が0ならNaN。

### runtime: 指標の導出

- `FedsdaRunMetrics`へ足す: `server_computation_counts`、`processed_sample_count`、`held_model_sample_count`、`computation_cost_summary`（モデルの計算の計数が渡されなければ`None`）。

## 旧と違う点

- サーバの計数、積和演算の数、標本あたり・保有モデルあたりの値は、旧にない。
- 旧のラウンドごとの計数は、ローカルと同期の合計。新は、分けて返す。旧のラウンドごとの値は、用途別で、clientごと。新は、全clientの合計で、用途別ではない。

## Error Handling

- 計数のfieldは、非負のbuiltin int。まとめの入力の型・範囲の不正は、`TypeError`／`ValueError`。

## Testing Strategy

### Unit Tests

- 保有モデル数のowner（順、読取りの不変、拒否）。
- 計算量のまとめ: 手計算との照合、分母0のNaN、拒否。

### Integration Tests

- 集約: 既存の、実旧との対照の中で、計数が、その集約で通信量へ足された「上りのパラメータの値の数」と一致すること。
- 統合・パラメータ距離: 既存の、実旧との対照の中で、計数が、結果のクラスタ・診断の観測・層の形から計算した値と一致すること（統合したラウンドと、しなかったラウンド）。
- client: 全状態の新旧照合へ、保有モデル数の合計＝実旧の予測の計数、を足す（全条件で照合される）。標本ごとの値が、その時点の保有モデル数であること。
- 計測つきの全体run: ラウンドの値と終端の値の合計＝全体。同期の値は、学習を含まない。goldenの条件で、ラウンドごとの値が、実旧の保存結果（`round_client_*`）の、全clientの合計と一致すること（ローカル＝予測・検出・統計・初期化・学習、同期＝クロス評価・再較正）。
- 指標の導出: goldenの条件で、サーバの集約の計数の合計＝通信量の上りのパラメータの値の数、保有モデル数の合計＝旧の予測の計数、まとめの値が、計数からの手計算と一致すること。goldenの33指標の照合は、そのまま通す。

### fresh process・依存

- 共用script: 全体runの後で、計算量のまとめを得て、値の整合を確かめる。
- 依存境界。
