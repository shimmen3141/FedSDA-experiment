# Design Document: global-model-distribution

## Overview

**Purpose**: グローバルモデルの配布と、clientでの受取りを、旧と同じ順序・数値・乱数の消費で行う。

**Users**: リファクタリングを進める研究者。これで、クラスタリングなしのラウンド（登録→集約→配布）が通る。

**Impact**: 既存の完了済みのmoduleへ、操作とfieldを足す（下の「Modified Files」）。既存の操作の契約は変えない。

### Goals

- 実旧のサーバ・複数のclientと、ラウンドごとに、配布の後の全状態と乱数が一致する。
- ID対応がある受取りを、実旧のclientの受取りと照合する。

### Non-Goals

- 集約後の再較正、ID対応を作る処理（クロス評価・クラスタリング・統合）、実行の枠のサーバの操作の実体。

## Boundary Commitments

### This Spec Owns

- 関数`distribute_global_models_to_clients`（サーバの配布）と`apply_global_model_distribution`（clientの受取り）、受取りの結果の記録（runtime）。
- `FedsdaRunClient.apply_global_model_distribution`（受取りを、自分のownerと設定で呼ぶ操作）。
- 共有部のoptimizerの状態の保持者`SharedParameterOptimizerStateHolder`。
- 既存のownerへ足す操作: 保有モデルの全置換え、損失統計の全置換え、グローバルの損失統計の一覧、サーバによる統合の適応記録。
- clientの設定の束へ足すfield（作り直すモデルのoptimizerの設定）。

### Out of Boundary

- `select_loss_statistics_after_model_id_mapping`、各ownerの`remap_*`、`reconnect_held_models_to_shared_feature_extractor`の中の処理。変更しない。
- 予測の重みと診断証拠（受取りでは触らない。再較正は次のspec）。

### Allowed Dependencies

- runtimeの新moduleは、`dataclasses`・`random`・`torch`、evaluation、learning、methods/fedsda（グローバルモデルのowner）、runtime（`FedsdaRunClient`）に依存する。旧実装をimportしない。
- `fedsda_run_client`と受取りの関数は、互いにimportしない向きにする: 受取りの関数はownerの記録と設定を引数で受け取り、clientの操作がそれを呼ぶ。サーバの配布は、clientの操作を呼ぶ。

### Revalidation Triggers

- `FedsdaRunClientOwners`のfield（共有部のoptimizerの状態の持ち方）と、束のfieldの変更。
- registry・損失統計のowner・適応記録のownerへ足した操作の変更。

## File Structure Plan

```
src/federated_learning_experiments/
├── learning/training/
│   └── shared_parameter_optimizer_state_holder.py   # 新規: SharedParameterOptimizerStateHolder
└── runtime/
    ├── global_model_distribution_application.py     # 新規: clientの受取り（apply_global_model_distribution、結果の記録）
    └── global_model_distribution.py                 # 新規: サーバの配布（distribute_global_models_to_clients）
tests/refactoring/
├── test_global_model_distribution.py                # 新規: 実旧のサーバ・clientとの対照、ID対応の受取り、拒否
└── （下の変更に対応する既存のtest）
```

### Modified Files

- `learning/training/held_model_training_state_registry.py` — `replace_held_model_training_states`を足す。
- `learning/loss_statistics/model_and_class_loss_statistics.py` — `ModelAndClassLossStatisticsStore.replace_model_loss_statistics`を足す。
- `methods/fedsda/model_registration/global_model_repository.py` — `snapshot_global_model_loss_statistics`を足す。
- `evaluation/adaptation_record_store.py` — 結果種別`server_consolidation_training_model_remapped`と、その位置の列を足す。
- `runtime/fedsda_run_client_settings.py` — `rebuilt_model_parameter_optimizer_settings`を足す。
- `runtime/fedsda_run_client.py` — ownerの記録の共有部のoptimizerの状態を保持者にする。共有部と構造の参照を、現在の学習帰属のモデルから読む。受取りの操作を足す。
- 上の変更に合わせる既存のtest（束を作る箇所、共有部のoptimizerの状態を読む箇所、適応記録の結果種別の対応表）と、共用script。

## 旧処理との対応

research.mdの対応表のとおり。旧と違う点:

- **更新の順**: 旧は、統計→評価標本→学習データ→計数→モデルの作り直し、の順に更新する。新は、モデルの生成（torchの乱数の消費を含む）を、ownerの更新より前に行う。成功時の値と乱数の消費は同じ（research.mdの「Decision」）。
- **現在の学習帰属の検査**: 旧は、受取りの後に現行モデルを保有していなくても受け取る。新は拒否する。
- **統計の型**: 旧は、サーバの統計を`deepcopy`して置く。新は、不変の値をそのまま置く。
- **Cached方式の控え**を持たない。**計算量**を記録しない。**表示**（`verbose`）を行わない。
- **適応記録の検出器名**: 旧は`"server"`。新も、同じ文字列を検出器名として記録する。

## Requirements Traceability

| Requirement | Components |
|-------------|------------|
| 1.1〜1.4 | `distribute_global_models_to_clients` |
| 2.1〜2.4, 3.1〜3.6 | `apply_global_model_distribution`、`FedsdaRunClient.apply_global_model_distribution`、足したownerの操作 |
| 3.4 | 束の`rebuilt_model_parameter_optimizer_settings` |
| 4.1〜4.3 | 全体（対照test） |
| 5.1〜5.5 | 2つの関数 |
| 6.1 | 共用script |
| 6.2 | 依存境界test |

## Components and Interfaces

### 足すownerの操作

- `HeldModelTrainingStateRegistry.replace_held_model_training_states(*, held_model_training_states: tuple[HeldModelTrainingState, ...])`: 全部の状態を、既存の登録と同じ検査（分類器とoptimizerの対応、IDの型）と、IDの重複の検査の後で、渡された順に置き換える。不正なら何も変えない。
- `ModelAndClassLossStatisticsStore.replace_model_loss_statistics(*, loss_statistics_by_model_id: tuple[tuple[int, ModelAndClassLossStatistics], ...])`: 全部の統計を、検査の後で、渡された順に置き換える。
- `GlobalModelRepository.snapshot_global_model_loss_statistics() -> tuple[tuple[int, ModelAndClassLossStatistics], ...]`: 統計を持つモデルの、（ID、統計）を、最初に置いた順で返す。
- 適応記録: `AdaptationOutcome`へ`"server_consolidation_training_model_remapped"`を足す。変更前後のIDが違う結果だが、警報による切替の位置（`training_model_switch_sample_indices`）には足さず、`AdaptationRecordSnapshot`の新しい列`server_remapped_sample_indices`へ位置を足す。「IDが違うのは切替の結果のときだけ」の検査は、「切替の結果か、サーバによる付け替えのときだけ」にする。
- `SharedParameterOptimizerStateHolder`（新規）: `__init__(*, shared_parameter_optimizer_state)`、property `held_shared_parameter_optimizer_state`、`replace_shared_parameter_optimizer_state(*, shared_parameter_optimizer_state)`。exact `ParameterOptimizerState`だけを受け入れる。

### clientの設定の束

- `FedsdaRunClientSettings.rebuilt_model_parameter_optimizer_settings`: 配布で作り直すモデルの、共有部と概念固有部のoptimizerの設定（旧`BASE_LR`。AdamかSGD）。既存の`parameter_optimizer_settings`（候補と、つなぎ直しで作り直す概念固有部。旧`NEW_MODEL_LR`）と、同じ種類（AdamどうしかSGDどうし）であることを確かめる（旧は、1つの`OPTIMIZER`の設定を両方に使う）。

### runtime: apply_global_model_distribution（clientの受取り）

```python
@dataclass(frozen=True, kw_only=True)
class GlobalModelDistributionApplication:
    training_assignment_change: TrainingModelAssignmentChange | None   # ID対応で変わった場合
    held_model_ids: tuple[int, ...]                                    # 受取りの後の保有モデル（配布の順、その後に一時ID）

def apply_global_model_distribution(
    *,
    model_id_mapping: dict[int, int],
    distributed_parameter_snapshots: tuple[tuple[int, dict[str, Tensor]], ...],
    distributed_loss_statistics: tuple[tuple[int, ModelAndClassLossStatistics], ...],
    （owner）held_model_training_state_registry, shared_parameter_optimizer_state_holder,
    loss_statistics_store, training_sample_store, model_evaluation_sample_store,
    model_training_and_assignment_counts_store, current_training_model_assignment,
    adaptation_record_store, pending_training_assignment_buffer,
    rebuilt_model_parameter_optimizer_settings, reconnected_model_parameter_optimizer_settings,
    python_random_generator: Random,
) -> GlobalModelDistributionApplication: ...
```

処理順:

| 段 | 内容 | 状態の更新 |
| --- | --- | --- |
| 1 | 入力の検査（下） | なし |
| 2 | 統計の選択（`select_loss_statistics_after_model_id_mapping`）。処理した標本数（保留位置のownerの最終観測位置＋1）と、付け替え後の現在の学習帰属を求める | なし |
| 3 | 配布されたモデルごとに、配布の順で、現在の学習帰属のモデルの分類器と同じ構造の、新しい分類器を作り（torchの乱数を消費）、配布された値を載せ、2つのoptimizerの状態を作る。一時IDの保有モデルは、既存の分類器と、既存のoptimizerの状態のまま、後ろへ並べる | torchの乱数だけ |
| 4 | 新しい分類器と、残す一時IDのモデルを、つなぎ先の共有部へつなぎ直す（既存の部品。全部の対応を検査してからつなぐ） | 分類器の共有部、概念固有部のoptimizer |
| 5 | 現在の学習帰属が変わるなら、適応記録を足す | 適応記録 |
| 6 | 統計を置き換える→評価標本を付け替える→学習データを付け替える→計数を付け替える | 各owner、借りた乱数（評価標本が上限を超えるとき） |
| 7 | 保有モデルを置き換える→共有部のoptimizerの状態の保持者を、つなぎ先のものへ置き換える | registry、保持者 |
| 8 | 現在の学習帰属を付け替える | 現在の学習帰属 |

検査（段1。どの更新・乱数の消費より前）: ownerのexact型。乱数生成器がexact `random.Random`。ID対応がexact dictで、キーと値がbuiltin int。配布されるパラメータと統計が、（ID、値）のexact tupleの列で、IDが非負のbuiltin intで重複せず、パラメータが、現在の学習帰属のモデルの分類器の`state_dict`と同じ名前・同じ形の、CPUのfloat32のtensor。配布されるモデルが1つ以上。付け替え後の現在の学習帰属が、配布されたIDか、保有している一時ID。2つのoptimizerの設定がexact型。

- 一時IDのモデルの概念固有部のoptimizerの状態は、つなぎ直し（既存の部品）が、その状態自身の設定で作り直す。候補の採用で登録された一時IDのモデルは、候補の設定（`NEW_MODEL_LR`に当たる）で作られているので、旧と同じ学習率になる。
- 配布で作り直すモデルの概念固有部のoptimizerの状態は、つなぎ先（非負のIDが最小）のものだけ、作り直すモデルの設定で作る。ほかは、つなぎ直しで作り直されるので、最初から、つなぎ直しの設定で作る（つなぎ直しの部品は、状態自身の設定で作り直す）。共有部のoptimizerの状態は、作り直す全モデルについて、作り直すモデルの設定で作る（つなぎ直しの部品が、各モデルの共有部とoptimizerの対応を検査するため。旧も、全モデルについて作る）。使われるのは、つなぎ先のものだけで、保持者へ置く。ほかのモデルの共有部とそのoptimizerの状態は、つなぎ直しで使われなくなる（optimizerの生成は乱数を使わない）。
- 段3までに、ownerへ渡す値を全部作る。段4のつなぎ直しは、最初の変更（一時IDのモデルの分類器の共有部の付け替え）より前に、全部の対応を検査するので、ここで拒否されれば、ownerは変わらない（torchの乱数は、段3で消費済み）。段5以降の検査（各ownerの操作の入力の検査）は、段1の検査を通った入力では拒否しない。段5より後の失敗では、済んだ段は残る。

### runtime: distribute_global_models_to_clients（サーバの配布）

```python
def distribute_global_models_to_clients(
    *, run_clients: tuple[FedsdaRunClient, ...], global_model_repository: GlobalModelRepository,
    communication_volume_record_store: CommunicationVolumeRecordStore, model_id_mapping: dict[int, int],
) -> tuple[GlobalModelDistributionApplication, ...]: ...
```

1. 検査: clientの列（exact tuple、exact `FedsdaRunClient`、IDの重複なし）、ownerのexact型、ID対応がexact dictで整数から整数。
2. 全グローバルモデルのパラメータ（置いた順）と、統計（置いた順）を読む。
3. 下りの通信量を足す: グローバルモデルがあれば、最初のモデルの共有部を、clientの数を転送回数として1回。全モデルの概念固有部を、それぞれclientの数を転送回数として。モデル転送数は、モデルの数×clientの数。ID対応が空でなければ、メッセージをclientの数。
4. 全clientへ、渡された順に、`apply_global_model_distribution`を呼ばせる。

- 3は4より前（旧と同じ）。4の途中の失敗では、通信量と、済んだclientの受取りが残る（要求5.4）。

### FedsdaRunClient の変更

- ownerの記録: `shared_parameter_optimizer_state`を、`shared_parameter_optimizer_state_holder`（保持者）にする。
- `process_observed_sample`・`flush_pending_local_updates`: 共有部を、現在の学習帰属のモデルの分類器の共有部から読み、共有部のoptimizerを、保持者から読む。候補の構造の参照も、現在の学習帰属のモデルの分類器にする。
- `apply_global_model_distribution(*, model_id_mapping, distributed_parameter_snapshots, distributed_loss_statistics) -> GlobalModelDistributionApplication`を足す（自分のowner・束の2つのoptimizerの設定・乱数生成器で、受取りの関数を呼ぶ）。

## Error Handling

- 不正な入力は`TypeError`／`ValueError`（束の不正は`RunSettingsValidationError`）で、どの更新・乱数の消費より前に拒否する。配布の途中の失敗は、済んだ分を残す。

## Testing Strategy

実旧をoracleにする。式をtestへ複製しない。

### Unit Tests

- 足したownerの操作: registryの全置換え（順、検査、不正で不変）、損失統計の全置換え、グローバルの統計の一覧、適応記録の新しい結果種別（位置の列、警報による切替の位置へ足さないこと、IDが同じなら拒否）、保持者、束の新しいfield（型、種類の不一致の拒否）。
- 拒否（5.1〜5.3）: 受取りと配布の各不正入力（ID対応、配布の列の型・重複・負のID、パラメータの名前・形・dtype、空の配布、現在の学習帰属を保有しなくなる入力、ownerと乱数生成器の型）で、全状態と3つの乱数が変わらないこと。
- 配布の途中の失敗（5.4）: 2つめのclientの受取りを失敗させ、通信量と1つめの受取りが残り、3つめへ進まないこと。
- 受取りが変えない状態（3.6）: 保留、監視、候補検証、送信保留、学習要求、一時IDの採番、予測の重みと記録、診断証拠が、受取りの前後で同じであること。

### Integration Tests

- ラウンドの対照（1.x, 2.x, 3.x, 4.1, 4.3）: 登録と集約の対照のoracleへ、実旧の`broadcast_models({})`と、新の配布を足し、ラウンドごとに、配布の後の全状態と乱数を照合する。2値・多クラス、標本列を複数、学習率の2つの設定を違う値にした条件。全条件を通して、要求4.3の経路を通ったことを確かめる。
- ID対応の受取り（2.x, 3.5, 4.2）: ラウンドの対照の途中の状態から、実旧のclientの`apply_server_mapping(id_mapping, global_models, global_stats)`と、新の受取りへ、作ったID対応（現在の学習帰属が付け替わるもの、複数のモデルが1つへ集まるもの）と、それに合わせたグローバルモデルを与えて、全状態と乱数を照合する。評価標本の上限を小さくした条件で、抜出しを通す。適応記録と位置の列を、実旧の`server_merge`のイベントと`mapping_change_positions`と照合する。

### fresh process・依存

- 共用script（6.1）: サーバの代役が、ラウンドごとに登録・集約・配布を行う。配布の後、全clientが、全グローバルモデルを同じ値で保有し、1つの共有部につながっていることを確かめる。
- 依存境界（6.2）: 新しい3 moduleの許可集合と、変更したmoduleの追加分を登録する。
