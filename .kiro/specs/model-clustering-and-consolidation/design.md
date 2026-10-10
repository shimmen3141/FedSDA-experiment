# Design Document: model-clustering-and-consolidation

## Overview

**Purpose**: クロス評価の結果から、モデルをクラスタリングして統合し、サーバの1ラウンドの同期の全体を、旧と同じ順序・数値で行う。

**Users**: リファクタリングを進める研究者。これで、サーバのラウンドを、クラスタリングを含めて、旧の`run_round`と照合できる。

**Impact**: 新しいmoduleを足す。グローバルモデルのownerへ操作を1つ、clientへ操作を1つ足す。既存の操作の契約は変えない。

### Goals

- 実旧のサーバの`run_round`（クラスタリングあり）と、ラウンドごとに、全状態・診断の記録・乱数が一致する。

### Non-Goals

- 最終構成でない判定・linkage・後処理。実行の枠のサーバの操作の実体、全clientとサーバの準備、全体runの接続。診断の記録の保存と集計。

## Boundary Commitments

### This Spec Owns

- 判定の計算と、判定の基準の型（methods/fedsda/consolidation `model_clustering_calculations`）。
- クラスタリングの診断の記録のowner（evaluation `model_clustering_record_store`）。
- クラスタリングと統合の関数と結果の型（runtime `model_clustering_and_consolidation`）。
- サーバの1ラウンドの同期の関数と結果の型（runtime `server_round_synchronization`）。
- `GlobalModelRepository.remove_global_model`、`FedsdaRunClient.get_model_assigned_sample_concept_counts`。

### Out of Boundary

- 登録、集約、クロス評価、配布、受取り、再較正の中の処理。変更しない。

### Allowed Dependencies

- 判定の計算は、`dataclasses`・`math`・`statistics`だけに依存する。
- 診断の記録は、`dataclasses`・`math`だけに依存する。
- クラスタリングと統合は、evaluation（記録）、learning（損失統計）、methods/fedsda（判定の計算、グローバルモデルのowner）、runtime（`FedsdaRunClient`、クロス評価の結果の型、パラメータの分割）、`torch`に依存する。
- 同期の関数は、runtimeの各段の関数と、そのowner・型に依存する。
- 旧実装をimportしない。

### Revalidation Triggers

- `ModelCrossEvaluation`・`ClientModelAggregation`の形の変更。同期の関数の引数（実行の枠のサーバの操作の実体が呼ぶ）。

## File Structure Plan

```
src/federated_learning_experiments/
├── evaluation/
│   └── model_clustering_record_store.py                 # 新規
├── methods/fedsda/consolidation/
│   └── model_clustering_calculations.py                 # 新規
└── runtime/
    ├── model_clustering_and_consolidation.py            # 新規
    └── server_round_synchronization.py                  # 新規
tests/refactoring/
├── test_model_clustering_calculations.py                # 新規: 実旧の関数との対照、拒否
├── test_model_clustering_record_store.py                # 新規
└── test_model_clustering_and_consolidation.py           # 新規: 実旧のrun_roundとの対照、拒否
```

### Modified Files

- `methods/fedsda/model_registration/global_model_repository.py` — `remove_global_model`を足す。
- `runtime/fedsda_run_client.py` — `get_model_assigned_sample_concept_counts`を足す。
- `tests/refactoring/test_global_model_repository_and_communication_volume.py`、`tests/refactoring/test_single_run_dependency_boundaries.py`、`tests/refactoring/fresh_process_smoke.py`。

## 旧処理との対応

research.mdの対応表のとおり。旧と違う点:

- **更新の順**: 旧は、クラスタごとに、パラメータと統計を書き換え、最後に、代表でないモデルを消す。新は、全クラスタの新しいパラメータと統計を計算してから、同じ順に書き換える。成功時の値は同じ。
- **診断の記録の「なし」**: 旧の−1（最も近いモデルなし）とNaN（距離なし）を、Noneで持つ。
- **入力の検査**: 新は、判定の基準、クロス評価の結果と集約の件数の対応（全部のモデルIDについて件数があること）を確かめる。旧は確かめない。
- **表示**（`verbose`）を行わない。最終構成でない判定・linkage・後処理を持たない。
- **パラメータの距離の、共有部の判定**: 旧は、名前が`backbone.`で始まるものを除く（なければ全部）。新は、分割の部品（共有部と概念固有部）の、概念固有部を使う。

## Requirements Traceability

| Requirement | Components |
|-------------|------------|
| 1.1〜1.4, 6.3 | `model_clustering_calculations`の3つの関数 |
| 2.1〜2.4, 3.1〜3.3, 4.1〜4.5, 7.1 | `cluster_and_consolidate_global_models`、`ModelClusteringRecordStore`、`GlobalModelRepository.remove_global_model` |
| 5.1〜5.3, 7.2 | `synchronize_models_in_server_round` |
| 6.1, 6.2 | 対照test |
| 8.1 | 共用script |
| 8.2 | 依存境界test |

## Components and Interfaces

### methods/fedsda/consolidation: model_clustering_calculations

```python
@dataclass(frozen=True, kw_only=True)
class ModelClusteringCriteria:
    maximum_same_cluster_decision_score: float   # 判定の値が、これ以下のモデルを同じクラスタにする（旧distance_threshold）
    minimum_pair_evaluation_sample_count: int    # 対の判定と、クラス別の集計に要る評価の件数（旧CLUSTER_MIN_EVAL_N）
    clustering_confidence_level: float           # 信頼水準（0.5より大きく1未満。旧FEDSDA_CLUSTERING_CONFIDENCE）

def compute_binomial_proportion_lower_confidence_bound(*, success_count: int, sample_count: int, confidence_level: float) -> float
def compute_classwise_unique_correctness_decision_score(
    *, overall_unique_correctness_counts: tuple[int, int, int],
    class_unique_correctness_counts: tuple[tuple[int, int, int, int], ...],
    minimum_class_sample_count: int, confidence_level: float) -> float
def cluster_model_ids_by_average_linkage(
    *, model_ids: tuple[int, ...], decision_scores_by_model_pair: dict[tuple[int, int], float],
    maximum_same_cluster_decision_score: float) -> tuple[tuple[int, ...], ...]
```

- 数の組は、（評価した標本数、小さいIDのモデルだけ正解、大きいIDのモデルだけ正解）。クラス別は、先頭にクラスを足した4つ組。runtimeの型に依存しないよう、builtinの値で受け取る。
- 式と、演算の順は、旧と同じ（`statistics.NormalDist().inv_cdf`、組み込みの`sum`）。
- average linkage: モデルIDを昇順にし、1モデル1クラスタから始める。クラスタの対（左の位置＜右の位置）ごとに、全部のモデル対（小さいID、大きいID）の値を引き、1つでもなければ候補にしない。値の平均と（左、右、左の位置、右の位置）の組の最小を選び、平均が閾値を超えたら終わる。統合したクラスタ（昇順）を列の末尾へ足し、列を並べ直す。戻り値は、クラスタ（昇順のtuple）の、並べ直した順のtuple。
- 検査: 型（builtin int・float。boolを除く）、範囲（標本数は正、成功数は0以上で標本数以下、信頼水準は0.5より大きく1未満、IDの重複なし、対のキーは（小さいID、大きいID）、値は有限）。

### evaluation: ModelClusteringRecordStore

```python
@dataclass(frozen=True, kw_only=True)
class ModelPairClusteringObservation:
    round_index: int; lower_model_id: int; higher_model_id: int
    loss_increase_distance: float; decision_score: float; assigned_to_same_cluster: bool
    true_concepts_match: bool | None; concept_specific_parameter_distance: float

@dataclass(frozen=True, kw_only=True)
class ModelClusteringObservation:
    round_index: int; model_id: int
    nearest_model_id: int | None; nearest_loss_increase_distance: float | None
    representative_model_id: int; cluster_model_count: int
    maximum_within_cluster_distance: float | None
    evaluated_within_cluster_pair_count: int; possible_within_cluster_pair_count: int
    merged_with_other_models: bool; absorbed_into_representative: bool

class ModelClusteringRecordStore:
    def append_clustering_observations(self, *, pair_observations: tuple[...], model_observations: tuple[...]) -> None
    def snapshot_pair_clustering_observations(self) -> tuple[ModelPairClusteringObservation, ...]
    def snapshot_model_clustering_observations(self) -> tuple[ModelClusteringObservation, ...]
```

- recordは、生成時に、型と、基本の範囲（非負のID、小さいID＜大きいID、非負の件数、NaNでない数）を確かめる。パラメータの距離は、無限大を許す。追加は、全部の検査の後で、2つの列へ足す。

### runtime: cluster_and_consolidate_global_models

```python
@dataclass(frozen=True, kw_only=True)
class ModelConsolidation:
    model_clusters: tuple[tuple[int, ...], ...]   # クラスタ（昇順）の列。最小のIDの昇順
    model_id_mapping: dict[int, int]              # 統合しなかったら空。統合したら、全メンバー→代表（自分自身への対応を含む）
    absorbed_model_ids: tuple[int, ...]           # 代表へ吸収されて、外されたモデルID（昇順）

def cluster_and_consolidate_global_models(
    *, run_clients: tuple[FedsdaRunClient, ...], model_cross_evaluation: ModelCrossEvaluation,
    aggregated_training_sample_counts_by_model_id: dict[int, int],
    global_model_repository: GlobalModelRepository, model_clustering_record_store: ModelClusteringRecordStore,
    model_clustering_criteria: ModelClusteringCriteria, round_index: int,
) -> ModelConsolidation: ...
```

処理順:

| 段 | 内容 | 状態の更新 |
| --- | --- | --- |
| 1 | 検査（要求7.1）。クロス評価したモデルIDの列を、クラスタリングの対象にする | なし |
| 2 | 診断値: モデルごとの真の概念のラベル（全clientの割当概念の計数を、clientの順に足す）。対ごとの、真の概念の一致と、パラメータの距離 | なし |
| 3 | 対ごとの距離と判定の値（要求2.1・2.2）。average linkageでクラスタを求める | なし |
| 4 | 診断の観測を作る（要求3.1・3.2）。統合するなら、クラスタごとの新しいパラメータと統計、ID対応を計算する | なし |
| 5 | 観測を、診断の記録へ足す | 診断の記録 |
| 6 | 統合するなら、クラスタの順に、代表のパラメータと統計を置き、最後に、代表でないモデルを外す | グローバルモデルのowner |

- 段4までに、全部の計算と、recordの生成を済ませる。段5・6は、段1〜4を通った値では拒否しない。
- クロス評価したモデルが1つ以下の結果は、拒否する（旧の`_cluster_and_consolidate`は、対象が1つ以下なら、クロス評価もクラスタリングも記録も行わない。新は、同期の関数が、対象が2つ以上のときだけ呼ぶ）。
- 加重平均: メンバーの順に、件数（負なら0）が正のメンバーだけを、`値×件数`で足し、合計で割る。合計が0以下なら、代表のパラメータのまま。
- 統計: 統計を持つメンバーの、件数の合計（組み込みの`sum`）が正なら、平均損失＝`sum(平均×件数)÷合計`、偏差平方和0、クラス別なし。

### runtime: synchronize_models_in_server_round

```python
@dataclass(frozen=True, kw_only=True)
class ServerRoundSynchronization:
    registered_client_models: tuple[RegisteredClientModel, ...]
    client_model_aggregation: ClientModelAggregation
    model_cross_evaluation: ModelCrossEvaluation | None      # クラスタリングを行わなかったらNone
    model_consolidation: ModelConsolidation | None           # 同上
    distribution_applications: tuple[GlobalModelDistributionApplication, ...]
    prediction_recalibrations: tuple[PostAggregationPredictionRecalibration, ...]

def synchronize_models_in_server_round(
    *, run_clients, global_model_repository, communication_volume_record_store,
    cross_evaluation_record_store, model_clustering_record_store, model_clustering_criteria,
    maximum_evaluating_client_count_per_model: int, python_random_generator: Random,
    round_index: int, model_clustering_enabled: bool,
) -> ServerRoundSynchronization: ...
```

- 各段の関数を、旧の順に呼ぶだけ。最初に、`model_clustering_enabled`がbuiltin boolであることと、判定の基準の型と範囲を確かめる（基準は、クロス評価が通信量と乱数を進めた後の段で使うので、不正なら、どの段より前に拒否する）。ほかは、各段の関数の検査。クラスタリングの対象は、集約の対象のモデルID（`aggregated_global_model_ids`）。2つ以上のときだけ、クロス評価とクラスタリングを行う。
- 途中の失敗では、済んだ段が残る（要求7.2）。

### 足す操作

- `GlobalModelRepository.remove_global_model(*, model_id: int) -> None`: パラメータと損失統計を外す（どちらも持たなければ`KeyError`）。次の正式IDと来歴は変えない。
- `FedsdaRunClient.get_model_assigned_sample_concept_counts(*, model_id: int) -> dict[int, int]`: モデルへ帰属させた標本の、真の概念ごとの件数（診断用）。

## Error Handling

- 不正な入力は`TypeError`／`ValueError`。

## Testing Strategy

実旧をoracleにする。式をtestへ複製しない。

### Unit Tests

- 判定の計算（1.x, 6.3）: 実旧の3つの関数へ、格子と、決まった擬似乱数の入力を与えて照合する（成功数0・全部、標本数1、クラスが下限未満だけ、対の値が欠ける、同じ平均、閾値ちょうど）。拒否。
- 診断の記録のowner、グローバルモデルを外す操作、clientの割当概念の計数。
- クラスタリングと統合の拒否（7.1）と、同期の途中の失敗（7.2）。

### Integration Tests

- ラウンドの対照（2.x〜6.x）: 再較正つきのラウンドの対照と同じ形で、旧は`run_round(round_index, clustering_enabled=<送信できるモデルを持つclientがいるか>)`、新は`synchronize_models_in_server_round`。ラウンドごとに、全状態、クラスタリングの診断の記録（実旧の`model_lineage`の2つの観測の列）、乱数を照合する。全条件を通して、要求6.2の経路を通ったことを確かめる。

### fresh process・依存

- 共用script（8.1）: サーバの代役の同期を、同期の関数の呼出しへ置き換える。
- 依存境界（8.2）。
