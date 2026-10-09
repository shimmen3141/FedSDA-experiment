# Design Document: server-model-registration-and-aggregation

## Overview

**Purpose**: サーバの1ラウンドの前半（新規モデルの登録と、clientのモデルの集約）を、旧と同じ順序と数値で行う。

**Users**: リファクタリングを進める研究者が、配布・統合・全体runの接続の前提として使う。

**Impact**: 既存のsourceは変更しない。

### Goals

- 実旧のサーバ・複数のclientと、ラウンドごとに、サーバとclientの全状態が一致する。
- サーバの状態を、後続のspecが読み書きできるownerにまとめる。

### Non-Goals

- クロス評価・クラスタリング・統合、配布、再較正、実行の枠のサーバの操作の実体。

## Boundary Commitments

### This Spec Owns

- `GlobalModelRepository`と来歴の記録`GlobalModelRegistrationRecord`（methods/fedsda/model_registration）。
- `CommunicationVolumeRecordStore`と`CommunicationVolumeSnapshot`（evaluation）。
- 関数`register_ready_client_models`・`aggregate_client_models_into_global_models`と、その結果の記録（runtime）。

### Out of Boundary

- `confirm_held_model_registration`、`snapshot_classifier_parameters`、`aggregate_participating_client_loss_means`、`FedsdaRunClient`の中の処理。変更しない。
- クラスタリングの診断の記録、最終的なモデルの容量の集計、配布。

### Allowed Dependencies

- evaluationの新moduleは、`dataclasses`と`torch`だけに依存する。
- methods/fedsdaの新moduleは、`dataclasses`・`torch`と、learningの損失統計の型に依存する。
- runtimeの新moduleは、`dataclasses`・`torch`、evaluation（通信量）、learning（パラメータの写し、損失統計と損失平均の集約）、methods/fedsda（グローバルモデルのowner）、runtime（`FedsdaRunClient`、正式IDの確認）に依存する。旧実装をimportしない。

### Revalidation Triggers

- `GlobalModelRepository`の操作と、パラメータの持ち方（モデルIDごとの完全なパラメータ）の変更。
- `FedsdaRunClientOwners`のfieldの変更。
- 集約の結果の記録のfieldの変更（統合のspecが、集約の件数を読む）。

## File Structure Plan

```
src/federated_learning_experiments/
├── evaluation/
│   └── communication_volume_record_store.py     # 新規: CommunicationVolumeSnapshot、CommunicationVolumeRecordStore
├── methods/fedsda/model_registration/
│   └── global_model_repository.py               # 新規: GlobalModelRegistrationRecord、GlobalModelRepository
└── runtime/
    └── server_model_registration_and_aggregation.py  # 新規: 登録と集約の関数、結果の記録
tests/refactoring/
├── test_global_model_repository_and_communication_volume.py  # 新規: 2つのownerの単独のtest
├── test_server_model_registration_and_aggregation.py         # 新規: 実旧のサーバとの対照、拒否
├── test_single_run_dependency_boundaries.py                  # 変更: 新moduleの許可集合の登録
└── fresh_process_smoke.py                                    # 変更: clientの流れで、ラウンドごとに登録と集約を行う
docs/research/implementation-findings/
└── legacy-016-registration-id-consumed-without-model.md      # 新規
```

## 旧処理との対応

| 旧 | 本spec |
| --- | --- |
| `global_models`（ID→完全なパラメータ）、`global_stats`、`next_model_id`、`model_lineage`の登録 | `GlobalModelRepository` |
| `comm_*`の8つの計数、`record_parameter_transfer`、`record_model_transfer` | `CommunicationVolumeRecordStore` |
| `register_model_params(0, …)`、`register_model_stats(0, …)`（`_setup_server_and_clients`） | `GlobalModelRepository`の生成（初期モデル） |
| `_register_new_models(t)` | `register_ready_client_models` |
| `update_global_models(active_ids)`（共有部を持つサーバの版）と、対象のIDの決定（`run_round`の中） | `aggregate_client_models_into_global_models` |
| `record_client_state_summaries()` | `CommunicationVolumeRecordStore.record_messages`を、呼出し側（後続のサーバの実体）がclientの数で呼ぶ |

旧と違う点:

- **反映の時点**: 旧は、clientを見ながら通信量を足し、最後にモデルと統計を置く。新は、全部の計算の後で、まとめて反映する。成功時の値は同じ。
- **統計の型**: 旧は、集約した統計を`{'n', 'mean', 'M2': 0.0}`の辞書にする（クラス別の統計は消える）。新は、クラス別の統計を持たない`ModelAndClassLossStatistics`にする。
- **来歴**: 旧の来歴の記録は、クラスタリングの診断も持つ。新は、登録だけを持つ。
- **採番の開始**: 旧は1で固定。新は、初期モデルのIDの次（初期モデルのIDが0なら、同じ1）。
- **非有限・float32以外のパラメータ**: 旧は、NaNや無限大を含むパラメータも、そのまま平均する。新は、パラメータの写しの部品（`snapshot_classifier_parameters`）が、非有限の値とCPUのfloat32以外を拒否するので、集約が、何も変えずに`ValueError`で止まる（学習が発散したrunは、旧は続行し、新は集約で止まる）。写しの部品の契約をそのまま使った結果で、発散したrunを続行させる必要が出たら、その時点で扱いを決める。
- **初期モデルの来歴**: 旧は、作成元の分からないモデルを「ラウンド-1・client -1」で記録する。新は、ラウンドとclientをNoneにする。
- **状態の報告の件数**: 旧は、状態を報告する種類のclient（`reports_state_summary`が真）の数を数える。最終構成のclientはすべて真なので、新は、clientの数で数える。
- **表示**: 旧の`verbose`の出力は行わない。

## Requirements Traceability

| Requirement | Components |
|-------------|------------|
| 1.1〜1.5 | `GlobalModelRepository` |
| 2.1〜2.3 | `CommunicationVolumeRecordStore` |
| 3.1〜3.5 | `register_ready_client_models` |
| 4.1〜4.7 | `aggregate_client_models_into_global_models` |
| 5.1, 5.2 | 全体（対照test） |
| 6.1〜6.4 | 2つの関数 |
| 7.1 | 共用script |
| 7.2 | 依存境界test |

## Components and Interfaces

### evaluation: CommunicationVolumeRecordStore

```python
@dataclass(frozen=True, kw_only=True)
class CommunicationVolumeSnapshot:
    uploaded_model_count: int
    downloaded_model_count: int
    uploaded_message_count: int
    downloaded_message_count: int
    uploaded_parameter_value_count: int
    downloaded_parameter_value_count: int
    uploaded_byte_count: int
    downloaded_byte_count: int

class CommunicationVolumeRecordStore:
    def record_model_transfers(self, *, transfer_direction: str, model_count: int) -> None: ...
    def record_messages(self, *, transfer_direction: str, message_count: int) -> None: ...
    def record_parameter_transfer(self, *, transfer_direction: str,
                                  parameter_snapshot: dict[str, Tensor], transfer_count: int = 1) -> None: ...
    def get_state_snapshot(self) -> CommunicationVolumeSnapshot: ...
```

- `transfer_direction`は`"upload"`（client→サーバ）か`"download"`（ValueError）。件数はbuiltin intで0以上（TypeError／ValueError）。`parameter_snapshot`はexact dictで、キーがstr、値がexact `torch.Tensor`（TypeError）。
- 値の数は各tensorの`numel()`の和、バイト数は`numel() × element_size()`の和。どちらも転送回数を掛ける（旧`parameter_payload_size`）。
- 旧の`record_model_transfer`（モデル転送数とパラメータの両方を足す）は、最終構成のサーバが使わないので持たない。

### methods/fedsda/model_registration: GlobalModelRepository

```python
@dataclass(frozen=True, kw_only=True)
class GlobalModelRegistrationRecord:
    model_id: int
    registered_round_index: int | None   # 初期モデルはNone
    registering_client_id: int | None    # 初期モデルはNone

class GlobalModelRepository:
    def __init__(self, *, initial_model_id: int, initial_parameter_snapshot: dict[str, Tensor],
                 initial_loss_statistics: ModelAndClassLossStatistics) -> None: ...
    @property
    def next_global_model_id(self) -> int: ...
    @property
    def global_model_ids(self) -> tuple[int, ...]: ...           # 設定した順
    def allocate_global_model_id(self) -> int: ...
    def record_model_registration(self, *, model_id: int, registered_round_index: int,
                                  registering_client_id: int) -> None: ...
    def snapshot_model_registration_records(self) -> tuple[GlobalModelRegistrationRecord, ...]: ...  # ID昇順
    def set_global_model_parameters(self, *, model_id: int, parameter_snapshot: dict[str, Tensor]) -> None: ...
    def get_global_model_parameters(self, *, model_id: int) -> dict[str, Tensor]: ...
    def set_global_model_loss_statistics(self, *, model_id: int,
                                         loss_statistics: ModelAndClassLossStatistics) -> None: ...
    def get_global_model_loss_statistics(self, *, model_id: int) -> ModelAndClassLossStatistics | None: ...
```

- モデルIDはbuiltin intで0以上（グローバルモデルは一時IDを持たない）。ラウンドとclientのIDはbuiltin intで0以上。`parameter_snapshot`は、空でないexact dictで、キーがstr、値がexact `torch.Tensor`。`loss_statistics`はexact `ModelAndClassLossStatistics`（TypeError／ValueError）。
- パラメータは、設定のときと取得のときに、各tensorを`detach().clone()`した新しい辞書にする。統計は不変の値。
- `get_global_model_parameters`は、持っていないIDで`KeyError`。統計は、持っていなければNone（旧の`global_stats`は既定値の辞書を返すが、新は「ない」を区別する）。
- 旧の`register_model_params`は、来歴に「作成元不明」を足す（`ensure_model`）。新は、初期モデルだけを、生成時に記録する。集約で初めて置かれるモデルは、その前に登録で来歴が記録されている。

### runtime: 登録と集約

```python
@dataclass(frozen=True, kw_only=True)
class RegisteredClientModel:
    client_id: int
    registered_global_model_id: int
    training_assignment_change: TrainingModelAssignmentChange | None   # 非負のIDへ戻っていた場合はNone

def register_ready_client_models(
    *, run_clients: tuple[FedsdaRunClient, ...], global_model_repository: GlobalModelRepository,
    round_index: int,
) -> tuple[RegisteredClientModel, ...]: ...

@dataclass(frozen=True, kw_only=True)
class ClientModelAggregation:
    aggregated_global_model_ids: tuple[int, ...]                  # 対象のID（昇順）
    aggregated_training_sample_counts_by_model_id: dict[int, int] # 対象のIDごとの、参加した学習データの総件数

def aggregate_client_models_into_global_models(
    *, run_clients: tuple[FedsdaRunClient, ...], global_model_repository: GlobalModelRepository,
    communication_volume_record_store: CommunicationVolumeRecordStore,
) -> ClientModelAggregation: ...
```

共通の検査（どの更新より前）: `run_clients`がexact tupleで、各要素がexact `FedsdaRunClient`、clientのIDが重複しない（TypeError／ValueError）。ownerがexact型（TypeError）。登録の`round_index`がbuiltin intで0以上。

登録の処理順: clientを渡された順に見て、`has_model_ready_for_server_registration()`が真のclientごとに、(1)`allocate_global_model_id`、(2)`record_model_registration`、(3)clientのownerで`confirm_held_model_registration`、(4)結果へ足す。学習帰属の変更は、診断へ通知しない（旧も、切替の通知を呼ばずに現行モデルを置き換える）。

集約の処理順:

1. 対象のID: 全clientの保有モデルのうち、非負のIDの昇順。
2. clientごと（渡された順）に、参加モデル（対象のIDの昇順で、学習データが1件以上のもの）と件数を求める。参加モデルがなければ、そのclientを飛ばす。参加モデルがあれば、最初の参加モデルの分類器から共有部の写しを、各参加モデルの分類器から概念固有部の写しを得て、共有部の和（重みは参加モデルの件数の合計）と、モデルIDごとの概念固有部の和（重みはそのモデルの件数）へ足す。損失統計があれば、そのモデルの統計の列へ足す。通信量の増分を数える。
3. 共有部の平均を求める。どのclientも参加していなければ、既存の最初のグローバルモデルの共有部を使う。
4. 対象のIDごとに、概念固有部（平均、または既存のグローバルモデルのもの）と、置くパラメータ、置く統計を決める。
5. 反映する: 通信量→グローバルモデルのパラメータ→統計。
6. 結果を返す。

- 和と平均の式の形は旧と同じにする: 最初は「値×重み」、以後は「和＋値×重み」、最後に「和÷重みの合計」。重みはPythonのint。
- 5より前は、どの状態も変えない。5の中の失敗は起きない（4までに、値を全部作ってある。ownerの検査を通る値だけを渡す）。
- clientの状態（保有モデル、学習データ、統計）は読むだけ。

## Error Handling

- 不正な入力は`TypeError`／`ValueError`で、どの更新より前に拒否する。登録の途中の失敗は、済んだ分を残す（要求6.3）。

## Testing Strategy

実旧をoracleにする。式をtestへ複製しない。

### Unit Tests

- グローバルモデルのowner（1.1〜1.5）: 初期状態、採番、設定と取得の写し（渡した辞書・返った辞書を後から変えても、内部が変わらない）、設定した順、来歴の昇順と上書き、各不正入力の拒否と不変。
- 通信量のowner（2.1〜2.3）: 各操作の加算、値の数とバイト数（dtypeの違い、転送回数）、各不正入力の拒否と不変。
- 関数の拒否（6.1, 6.2）: clientの列の別の型、重複するclientのID、ownerの別の型・派生型、ラウンドの不正で、サーバとclientの全状態が変わらないこと。集約の計算の途中の失敗（clientの分類器の形の不一致）で、何も変わらないこと。
- 登録の途中の失敗（6.3）: 2つめのclientへの確認を失敗させ、1つめの登録が残り、3つめへ進まないこと。

### Integration Tests

- 学習データを持たないモデルの集約（4.2, 4.4, 5.2）: 生成直後（どのclientも学習データを持たない）の集約を、実旧と照合する。参加するclientがいる集約の中に、学習データを持たない非負のIDのモデルが混ざる状態は、配布が未移植なので、複数ラウンドの対照では起きない（配布のspecの対照で通る）。
- 実旧のサーバとの対照（3.x, 4.x, 5.1, 5.2）: 実旧の事前学習→実旧のサーバ（初期モデルと統計の登録）→実`__init__`で作ったclient 3つ。新側は、同じ初期モデルから組み立てたclient 3つ（1つの乱数生成器を共有する）と、2つのowner。ラウンドごとに、全clientの標本処理（標本ごとにclient順）→保留中の学習→状態の報告の計数→登録→集約→送信待ちの進行、を両実装で行い、登録の後と集約の後に、要求5.1の全項目を照合する。2値・多クラス、標本列を複数。全条件を通して、要求5.2の経路を通ったことを確かめる。

### fresh process・依存

- 共用script（7.1）: clientの流れの、何もしないサーバの代役を、ラウンドごとに登録と集約を行う代役に替える（配布は行わない）。
- 依存境界（7.2）: 新しい3 moduleの許可集合を登録する。
