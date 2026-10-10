# Design Document: fedsda-run-participant-preparation

## Overview

**Purpose**: 最終構成のFedSDAの、サーバの操作の実体と、runごとの参加者の初期準備を作り、実行の枠で全体runを実行できるようにする。

**Users**: リファクタリングを進める研究者。結果は、次のspec（指標の導出とgoldenとの照合）が読む。

**Impact**: 新しいmoduleを足す。既存のmoduleは変更しない。

### Goals

- 新の全体run（`execute_stream_protocol_run`＋factory）の最終状態が、実旧の全体runの最終状態と一致する。

### Non-Goals

- 指標の導出・保存・goldenとの照合。実行の枠の契約の変更。完全なrun設定の組立て。SINE以外のdataset。

## Boundary Commitments

### This Spec Owns

- `FedsdaRunServer`と、そのownerの記録・組立て（runtime `fedsda_run_server`）。
- 設定の束`FedsdaRunParticipantSettings`と、`FedsdaRunParticipantFactory`（runtime `fedsda_run_participant_factory`）。

### Out of Boundary

- 実行の枠（`execution/`、`runtime/single_run_execution.py`）、clientの組立て、事前学習、サーバの1ラウンドの同期の中の処理。変更しない。

### Allowed Dependencies

- `fedsda_run_server`は、evaluation（通信量、2つの診断の記録）、learning（損失統計）、methods/fedsda（グローバルモデルのowner、判定の基準）、runtime（`FedsdaRunClient`、同期の関数）、`dataclasses`・`random`・`torch`に依存する。
- `fedsda_run_participant_factory`は、configuration（固定条件）、core（設定の検査の例外）、data（SINEの標本生成器）、execution（参加者の契約、乱数源）、learning（モデルの構造の設定、事前学習の条件、パラメータの写し）、methods/fedsda（判定の基準）、runtime（clientの組立てと束、事前学習、サーバの組立て）に依存する。
- 旧実装をimportしない。

### Revalidation Triggers

- 実行の枠の契約（`RunParticipantFactory`・`RunServerOperations`・`RunClientOperations`）の変更。`synchronize_models_in_server_round`の引数の変更。

## File Structure Plan

```
src/federated_learning_experiments/runtime/
├── fedsda_run_server.py                  # 新規
└── fedsda_run_participant_factory.py     # 新規
tests/refactoring/
├── test_fedsda_run_server.py             # 新規: サーバの操作、拒否
└── test_fedsda_stream_protocol_run.py    # 新規: 実旧の全体runとの対照、束、factory
```

### Modified Files

- `tests/refactoring/test_fedsda_run_client.py` — 旧の設定の差し替えへ、全体runの条件を足す（既定は、いまの値のまま）。
- `tests/refactoring/test_single_run_dependency_boundaries.py`、`tests/refactoring/fresh_process_smoke.py`。

## 旧処理との対応

research.mdの対応表のとおり。旧と違う点:

- **真の概念**: 旧のclientは、標本ごとに真の概念を受け取る。新の全体runでは、渡さない（実行の枠の契約）。真の概念に依存する診断は、行われない。
- **同期の結果の保持**: 新のサーバは、ラウンドごとの同期の結果を保持する。旧は持たない。
- **設定の検査**: 新は、設定の束と、準備の引数を、乱数を消費する前に確かめる。
- **計算量・所要時間・ラウンドごとの計測**（旧`telemetry`）を記録しない。**表示**を行わない。

## Requirements Traceability

| Requirement | Components |
|-------------|------------|
| 1.1〜1.4, 5.2 | `FedsdaRunServer`、`assemble_fedsda_run_server` |
| 2.1〜2.4, 5.1 | `FedsdaRunParticipantFactory` |
| 3.1〜3.3 | `FedsdaRunParticipantSettings` |
| 4.1〜4.4 | 対照test |
| 6.1 | 共用script |
| 6.2 | 依存境界test |

## Components and Interfaces

### runtime: fedsda_run_server

```python
@dataclass(frozen=True, kw_only=True)
class FedsdaRunServerOwners:
    global_model_repository: GlobalModelRepository
    communication_volume_record_store: CommunicationVolumeRecordStore
    cross_evaluation_record_store: CrossEvaluationRecordStore
    model_clustering_record_store: ModelClusteringRecordStore

class FedsdaRunServer:
    owners: FedsdaRunServerOwners   # property
    def record_client_states_before_synchronization(self, *, round_index: int) -> None
    def synchronize_models(self, *, round_index: int, new_model_registration_available: bool) -> None
    def finalize_started_communications(self, *, completed_round_count: int) -> None
    def snapshot_server_round_synchronizations(self) -> tuple[ServerRoundSynchronization, ...]

def assemble_fedsda_run_server(
    *, run_clients: tuple[FedsdaRunClient, ...], initial_model_id: int,
    initial_parameter_snapshot: dict[str, Tensor], initial_loss_statistics: ModelAndClassLossStatistics,
    model_clustering_criteria: ModelClusteringCriteria, maximum_evaluating_client_count_per_model: int,
    python_random_generator: Random,
) -> FedsdaRunServer
```

- 組立ては、引数を確かめてから（clientの列、判定の基準、clientの上限、乱数生成器。初期モデルは、グローバルモデルのownerの検査）、4つのownerを新しく作る。`FedsdaRunServer.__init__`は、受け取ったものを検査せずに持つ（`FedsdaRunClient`と同じ形）。
- `record_client_states_before_synchronization`: ラウンドの検査の後、上りの軽量メッセージへ、clientの数を足す。
- `synchronize_models`: ラウンドとフラグの検査は、同期の関数が行う。結果を、保持の列へ足す。
- `finalize_started_communications`: 完了したラウンド数（非負のbuiltin int）を確かめるだけ。

### runtime: fedsda_run_participant_factory

```python
@dataclass(frozen=True, kw_only=True)
class FedsdaRunParticipantSettings:
    run_client_settings: FedsdaRunClientSettings
    initial_model_pretraining_settings: InitialModelPretrainingSettings
    model_architecture_settings: ModelArchitectureSettings
    hidden_layer_widths: tuple[int, ...]
    model_clustering_criteria: ModelClusteringCriteria
    maximum_evaluating_client_count_per_model: int

class FedsdaRunParticipantFactory:
    def __init__(self, *, run_participant_settings: FedsdaRunParticipantSettings) -> None
    prepared_run_participants: RunParticipants | None    # property。直前に準備した参加者
    def validate_configuration(self) -> None
    def prepare_run(self, *, experiment_run_conditions, run_random_sources, sample_generator) -> RunParticipants
```

- 束の検査（生成時。不正は`RunSettingsValidationError`）: 各fieldのexact型、各設定の再検査、隠れ層の幅（正のbuiltin intの、空でないtuple）、clientの上限（正のbuiltin int）、クラスタリングの閾値＝clientの束の`maximum_tolerated_mean_loss_increase`。
- `prepare_run`の順:
  1. 検査: 束（再検査）、固定条件・乱数源・標本生成器のexact型、datasetが`sine2`。
  2. 事前学習`pretrain_initial_model`（クラス数は2。optimizerの設定は、clientの束の`rebuilt_model_parameter_optimizer_settings`。乱数は、runのPythonの乱数生成器と、渡された標本生成器）。
  3. clientを、IDが0から順に、`assemble_fedsda_run_client`で組み立てる（初期モデルIDは0。全clientへ、runの同じPythonの乱数生成器）。
  4. サーバを`assemble_fedsda_run_server`で組み立てる（初期のパラメータは、事前学習の分類器の写し。統計は、事前学習の統計）。
  5. `RunParticipants`を作り、保持して、返す。
- 旧は、サーバを作ってからclientを作る。サーバの生成は乱数を使わないので、clientを先に組み立てても、乱数の消費は同じ（サーバの組立てが、clientの列を受け取るため）。

## Error Handling

- 束の不正は`RunSettingsValidationError`。準備とサーバの操作の引数の不正は`TypeError`／`ValueError`（実行の枠が、`RunExecutionError`へ包む）。

## Testing Strategy

実旧をoracleにする。式をtestへ複製しない。

### Unit Tests

- サーバ（1.x, 5.2）: 3つの操作（軽量メッセージ、同期の関数の呼出しと結果の保持、終端で不変）、組立ての拒否、操作の拒否。
- 束（3.x）: 型、範囲、閾値の不一致の拒否。
- factory（2.x, 5.1）: 事前検査が乱数を消費しないこと、準備の拒否（乱数を消費しない）、参加者の公開、準備のたびに新しい参加者。

### Integration Tests

- 全体runの対照（4.1, 4.2）: 旧の全体の流れ（準備→概念列→観測列→ラウンド→終端）を、旧の部品を旧の順に呼んで実行し、新の`execute_stream_protocol_run`（test専用の中継で、真の概念をclientへ渡す）の結果と、概念列、観測列、サーバの全状態、2つの診断の記録、各clientの全状態、runの乱数の最終状態を照合する。複数のseedと条件。全条件を通して、要求4.2の経路を通ったことを確かめる。
- 真の概念なしの全体run（4.3）: 中継なしの全体runが、中継ありの全体runと、真の概念に依存する診断を除いて、同じ状態であること。
- 繰返し（4.4）: 同じfactoryで2回実行して、同じ結果・別の参加者であること。

### fresh process・依存

- 共用script（6.1）: factoryと実行の枠で、全体runを実行する流れを足す。
- 依存境界（6.2）。
