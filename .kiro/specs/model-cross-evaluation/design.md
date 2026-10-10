# Design Document: model-cross-evaluation

## Overview

**Purpose**: グローバルモデルのクロス評価（clientでの評価と、サーバでの集計・通信量・診断の記録）を、旧と同じ順序・数値・乱数の消費で行う。

**Users**: リファクタリングを進める研究者。結果は、次のspec（クラスタリングと統合）の入力になる。

**Impact**: 新しいmoduleを足す。clientへ操作を2つ、設定の束へfieldを1つ足す。既存の操作の契約は変えない。

### Goals

- 実旧のサーバの`_cross_evaluate`と、結果・診断の記録・通信量・乱数が一致する。

### Non-Goals

- 距離・判定・linkage・統合・ID対応・ラウンドへの接続（次のspec）。Cached方式、非劣性の検証、計算量の記録、記録の保存。

## Boundary Commitments

### This Spec Owns

- clientの評価の関数と結果の型（runtime `client_model_cross_evaluation`）。
- サーバのクロス評価の関数と結果の型（runtime `model_cross_evaluation`）。
- 診断の記録のowner（evaluation `cross_evaluation_record_store`）。
- `FedsdaRunClient`の2つの操作と、束のfield `maximum_cross_evaluation_sample_count`。

### Out of Boundary

- `evaluate_classifier_per_sample_bounded_losses`、`predict_class_labels_from_prediction_scores`、通信量のowner、グローバルモデルのownerの中の処理。変更しない。
- 表と集計を使う判定（次のspec）。

### Allowed Dependencies

- clientの評価のmoduleは、`dataclasses`・`random`・`torch`、evaluation（評価標本）、learning（分類器、損失と予測の部品、保有モデル、学習データ、現在の学習帰属）に依存する。`fedsda_run_client`をimportしない。
- サーバのクロス評価のmoduleは、evaluation（通信量、診断の記録）、methods/fedsda（グローバルモデルのowner）、runtime（`FedsdaRunClient`、clientの評価の結果の型、パラメータの分割）に依存する。
- 旧実装をimportしない。

### Revalidation Triggers

- 結果の型（表、対ごとの正誤の集計）の変更は、次のspecの入力を変える。束のfieldの追加。

## File Structure Plan

```
src/federated_learning_experiments/
├── evaluation/
│   └── cross_evaluation_record_store.py      # 新規: 診断の記録
└── runtime/
    ├── client_model_cross_evaluation.py      # 新規: clientの評価
    └── model_cross_evaluation.py             # 新規: サーバのクロス評価
tests/refactoring/
└── test_model_cross_evaluation.py            # 新規: 実旧のサーバ・clientとの対照、拒否
```

### Modified Files

- `runtime/fedsda_run_client_settings.py` — スカラーの設定へ`maximum_cross_evaluation_sample_count`（旧`EVAL_MAX_SAMPLES`）を足す。
- `runtime/fedsda_run_client.py` — 2つの操作を足す。
- 束を作っている既存のtestと共用script、`tests/refactoring/test_single_run_dependency_boundaries.py`。

## 旧処理との対応

| 旧 | 新 |
| --- | --- |
| `clients/base.py`の`get_held_model_ids` | `FedsdaRunClient.get_cross_evaluation_held_model_ids` |
| `_cross_evaluation_data`、`evaluate_model`、`evaluate_model_diagnostics`（クラス別つき） | `evaluate_candidate_model_on_target_model_samples`（正誤の比較の有無は引数） |
| `servers/clustering.py`の`_cross_evaluate`、`_record_functional_pair_stats` | `cross_evaluate_global_models` |
| `servers/shared_backbone.py`の`_begin_cross_evaluation_model_transfers`・`_record_cross_evaluation_model_transfer` | 同上（通信量の計上） |
| `pair_prediction_diagnostics`・`cross_evaluation_diagnostics`・`cross_evaluation_class_diagnostics` | `CrossEvaluationRecordStore`の記録1種類（3つの旧の記録は、記録から導く） |
| `clustering.py`の`FunctionalPairStats`（集計の部分） | `ModelPairUniqueCorrectnessCounts` |

旧と違う点:

- **入力の検査**: 新は、渡されたパラメータの名前・形・dtypeと、損失の部品の検査（有限のCPU float32、ラベルがクラスの範囲）を通す。旧は検査しない。
- **学習データの読取り**: 旧は、対象が現行モデルで、評価標本が足りないとき、学習データの辞書を既定値つきで読む（キーがなければ、空の列が作られる）。新は、状態を変えずに読む。[LEGACY-017](../../../docs/research/implementation-findings/legacy-017-cross-evaluation-creates-empty-training-collection.md)。空の列ができるのは、現行モデルが学習データを1件も持たないときだけで、実旧で再現したのは、生成直後のclient（まだ標本を帰属させていない）である。評価の結果と、保有とみなすモデルIDは変わらない（現行モデルのIDは、どちらでも保有に含まれる）。ラウンドの対照（毎ラウンドのクロス評価、8条件×15ラウンド）と、clientの対照では、この状態は起きず、評価の後の学習データの並びは実旧と一致した。
- **パラメータの非有限値**: 渡されたパラメータの値が有限であることは、乱数を消費する前には確かめない（損失の部品が、出力の検査で拒否する。そのとき、借りた乱数とtorchの乱数は進んでいるが、ownerは変わらない）。グローバルモデルのownerには、集約が非有限を拒否するので、入らない。
- **対象のモデルを保有していない場合**: 旧は、標本を抜き出した後で（乱数を消費した後で）件数0を返す。新も同じ順にする（乱数の消費を合わせる）。
- **計算量・所要時間**を記録しない。**Cached方式**と**非劣性の検証（損失差）**を持たない。対の診断の収集は、常に行う（旧の最終構成と同じ）。
- **診断の記録の形**: 旧の3つの記録を、評価1回につき1件の記録にまとめる。

## Requirements Traceability

| Requirement | Components |
|-------------|------------|
| 1.1〜1.5 | `evaluate_candidate_model_on_target_model_samples` |
| 1.6 | `FedsdaRunClient.get_cross_evaluation_held_model_ids` |
| 2.1〜2.6, 3.1〜3.3 | `cross_evaluate_global_models` |
| 3.3, 3.4 | `CrossEvaluationRecordStore` |
| 4.1〜4.3 | 対照test |
| 5.1〜5.4 | 2つの関数 |
| 6.1 | 共用script |
| 6.2 | 依存境界test |

## Components and Interfaces

### runtime: client_model_cross_evaluation

```python
@dataclass(frozen=True, kw_only=True)
class ModelPairCorrectnessCounts:
    evaluated_sample_count: int
    candidate_only_correct_count: int
    target_only_correct_count: int
    both_correct_count: int
    both_wrong_count: int

@dataclass(frozen=True, kw_only=True)
class ClientModelCrossEvaluation:
    evaluated_sample_count: int
    bounded_loss_sum: float
    squared_bounded_loss_sum: float
    correctness_counts: ModelPairCorrectnessCounts | None            # 正誤の比較を行わなかったらNone
    class_correctness_counts: tuple[tuple[int, ModelPairCorrectnessCounts], ...]  # クラスの昇順

def evaluate_candidate_model_on_target_model_samples(
    *, candidate_parameter_snapshot: dict[str, Tensor], target_model_id: int,
    compare_correctness_with_held_target_model: bool,
    held_model_training_state_registry, model_evaluation_sample_store, training_sample_store,
    current_training_model_assignment, maximum_evaluation_sample_count: int,
    python_random_generator: Random,
) -> ClientModelCrossEvaluation: ...
```

処理順（旧と同じ）:

1. 検査（ownerのexact型、乱数生成器、IDと上限の型と範囲、フラグがbuiltin bool、パラメータが現在の学習帰属のモデルの分類器と同じ名前・形のCPU float32）。
2. 標本の選択: 対象のモデルの評価標本が6件以上→それ。そうでなく、対象が現在の学習帰属で、学習データが11件以上→それ。どちらでもなければ空。上限を超えたら、借りた乱数生成器の`sample`で抜き出す。
3. 標本が5件未満、または（正誤の比較を求められて）対象のモデルを保有していない→件数0の結果。
4. 標本の特徴とラベルを連結し、新しい分類器を作って（torchの乱数を消費）、パラメータを載せる。評価モードにはしない（旧と同じ。分類器は、訓練の別で出力が変わる層を持たない）。
5. 損失: `evaluate_classifier_per_sample_bounded_losses`。件数、float32の配列の和、2乗の配列の和を、NumPyの配列の演算で求める（旧と同じ演算。torchの和は、足す順が違い、末尾の桁が変わりうる）。
6. 正誤の比較（求められたとき）: 渡されたモデルと、保有する対象のモデルの、それぞれの出力から、`predict_class_labels_from_prediction_scores`で予測クラスを求め、観測ラベルと比べる。全体と、標本に現れたクラスの昇順ごとに、4つの数を数える。

- どのownerも変えない。借りた乱数（標本の抜出し）と、torchの乱数（分類器の生成）だけが進む。

### evaluation: CrossEvaluationRecordStore

```python
@dataclass(frozen=True, kw_only=True)
class ClientCrossEvaluationRecord:
    round_index: int
    client_id: int
    candidate_model_id: int
    target_model_id: int
    evaluated_sample_count: int
    bounded_loss_sum: float
    squared_bounded_loss_sum: float
    correctness_counts: tuple[int, int, int, int] | None   # 候補だけ正解、対象だけ正解、両方正解、両方不正解
    class_correctness_counts: tuple[tuple[int, int, int, int, int, int], ...]  # クラス、件数、上の4つ

class CrossEvaluationRecordStore:
    def append_client_cross_evaluation_record(self, *, client_cross_evaluation_record) -> None
    def snapshot_client_cross_evaluation_records(self) -> tuple[ClientCrossEvaluationRecord, ...]
```

- evaluation層はruntimeに依存できないので、記録は、builtinの値だけを持つ。サーバのクロス評価が、clientの評価の結果から作る。recordは、生成時に、型と、数の整合（4つの数の和が件数）を確かめる。

### runtime: model_cross_evaluation

```python
@dataclass(frozen=True, kw_only=True)
class CrossEvaluationLossSums:
    evaluated_sample_count: int
    bounded_loss_sum: float
    squared_bounded_loss_sum: float

@dataclass(frozen=True, kw_only=True)
class ModelPairUniqueCorrectnessCounts:
    evaluated_sample_count: int
    lower_id_model_only_correct_count: int
    higher_id_model_only_correct_count: int
    class_counts: tuple[tuple[int, int, int, int], ...]   # クラス（初めて現れた順）、件数、小さいIDだけ正解、大きいIDだけ正解

@dataclass(frozen=True, kw_only=True)
class ModelCrossEvaluation:
    cross_evaluated_model_ids: tuple[int, ...]
    loss_sums_by_candidate_and_target_model_id: dict[tuple[int, int], CrossEvaluationLossSums]
    unique_correctness_counts_by_model_pair: dict[tuple[int, int], ModelPairUniqueCorrectnessCounts]

def cross_evaluate_global_models(
    *, run_clients: tuple[FedsdaRunClient, ...], global_model_repository, communication_volume_record_store,
    cross_evaluation_record_store, cross_evaluated_model_ids: tuple[int, ...], round_index: int,
    maximum_evaluating_client_count_per_model: int, python_random_generator: Random,
) -> ModelCrossEvaluation: ...
```

処理順:

1. 検査（要求5.1）。全モデルのパラメータを読み、共有部と概念固有部へ分ける（分けられなければ、何も変えない）。
2. 保有clientの列: 各clientの`get_cross_evaluation_held_model_ids`から、モデルIDごとに、clientの順で作る。
3. 評価する側`i`（外側）・対象`j`（内側）の順に:
   - 対象のclient: `j`の保有clientが上限を超えたら、乱数生成器の`sample`で抜き出す。
   - 軽量メッセージ: 下りと上りへ、対象のclientの数。
   - 通信量: 対象のclientごとに、共有部（未送信なら、モデル`i`の共有部を1回）、概念固有部（（`i`、client）が未送信なら、モデル転送数1と、モデル`i`の概念固有部を1回）。
   - 各clientへ評価を求め（`i`≠`j`なら正誤の比較つき）、記録を足し、件数・和・2乗和を足し合わせ、正誤の比較があれば、対（小さいID、大きいID）の集計へ足す。
4. 表（全部の組。対象のclientがいなければ、件数0）と、対の集計を返す。

- 表のキーは（評価する側、対象）。対の集計のキーは（小さいID、大きいID）で、正誤の比較が1回以上返った対だけを持つ（旧と同じ）。
- 途中の失敗では、済んだ分の通信量と記録が残る（要求5.3）。

### FedsdaRunClient の変更

- `get_cross_evaluation_held_model_ids() -> frozenset[int]`: 評価標本・学習データ・保有モデルのIDと、現在の学習帰属のIDの和集合。
- `evaluate_candidate_model_on_target_model_samples(*, candidate_parameter_snapshot, target_model_id, compare_correctness_with_held_target_model) -> ClientModelCrossEvaluation`: 自分のowner・束の上限・乱数生成器で、評価の関数を呼ぶ。

### 設定

- `FedsdaRunClientScalarSettings.maximum_cross_evaluation_sample_count`（正のbuiltin int。旧`EVAL_MAX_SAMPLES`）。
- 標本の件数の閾値（評価標本6件以上、学習データ11件以上、評価に5件以上）は、旧の固定値のまま、clientの評価のmoduleの定数にする。
- clientの上限（旧`CROSS_EVAL_MAX_CLIENTS`）は、サーバのクロス評価の引数にする（置き場所は、完全なrun設定を決めるspecで決める）。
- 乱数生成器: 旧は、サーバと全clientが、1つのPythonの乱数（module全体の`random`）を共有する。旧と同じ乱数の消費にするには、サーバのクロス評価と、全clientへ、同じ乱数生成器を渡す（対照testは、そうしている。全体runを接続するspecで、この渡し方を守る）。

## Error Handling

- 不正な入力は`TypeError`／`ValueError`（束の不正は`RunSettingsValidationError`）。

## Testing Strategy

実旧をoracleにする。式をtestへ複製しない。

### Unit Tests

- 診断の記録のowner: 追加の順、記録の検査、不正で不変。
- clientの評価の拒否（5.2）と、サーバのクロス評価の拒否（5.1）: 全状態と乱数が変わらないこと。途中の失敗（5.3）。
- 評価が変えない状態（1.5, 2.6）。

### Integration Tests

- サーバの対照（2.x, 3.x, 4.1, 4.2）: 再較正つきのラウンドの対照の途中で、登録→集約の後（配布の前）に、実旧の`_cross_evaluate(active_ids, round_index=t)`と、新の`cross_evaluate_global_models`を、同じ乱数の状態から行い、表、対の集計（実旧の`_last_pair_functional_stats`）、3つの診断の記録、通信量、乱数、clientの状態を照合する。その後、配布と再較正へ進む（どちらの実装も、クロス評価の結果を使わない）。評価標本の追加の件数、標本の上限、clientの上限を変えた条件で、要求4.2の経路を通す。
- clientの対照（1.x, 4.3）: 同じ状態で、実旧のclientの`evaluate_model`・`evaluate_model_diagnostics(include_class_correctness=True)`と、新のclientの操作を、モデルの全部の組について照合する。

### fresh process・依存

- 共用script（6.1）、依存境界（6.2）。
