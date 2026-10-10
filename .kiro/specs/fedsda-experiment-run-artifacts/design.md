# Design Document: fedsda-experiment-run-artifacts

## Overview

**Purpose**: 完全なrun設定から、最終構成のFedSDAの1 runを実行して、結果を、新実装の名前で、ファイルへ保存する。

**Impact**: 新しいmoduleを3つ足す。保存用の辞書への変換へ、非有限の値の扱いの指定を足す。全体runの実行と、指標の導出は、変えない。

### Non-Goals

- 未照合の残りの項目、旧の結果の変換、保存先の名前・manifest・掃引・コマンド・図、未移植の診断、FedDrift。

## Boundary Commitments

### This Spec Owns

- `runtime/fedsda_experiment_run.py`、`runtime/fedsda_run_record_tables.py`、`runtime/fedsda_run_artifact_files.py`（新規）。
- `core/settings_serialization.py`の、非有限の値の扱いの指定。

### Out of Boundary

- 全体runの実行、指標の導出、各owner（読取りだけを使う）。

### Allowed Dependencies

- `fedsda_experiment_run` → 完全なrun設定、計測つきの全体run、指標の導出、参加者のfactory。
- `fedsda_run_record_tables` → numpy、実行の結果の型、参加者とownerの読取り。
- `fedsda_run_artifact_files` → 標準ライブラリ（json、pathlib、os、platform）、numpy、torch（版の記録）、上の2つ、保存用の辞書への変換。

## File Structure Plan

- 新規: `src/federated_learning_experiments/runtime/fedsda_experiment_run.py`、`fedsda_run_record_tables.py`、`fedsda_run_artifact_files.py`
- 新規test: `tests/refactoring/test_fedsda_experiment_run.py`、`test_fedsda_run_record_tables.py`、`test_fedsda_run_artifact_files.py`
- 変更: `core/settings_serialization.py`とそのtest、共用script、依存境界test。

## Requirements Traceability

| Requirement | Components |
|-------------|------------|
| 1.1〜1.4 | `execute_fedsda_experiment_run`、`FedsdaExperimentRun` |
| 2.1〜2.4 | `build_fedsda_run_record_tables` |
| 3.1〜3.5 | `save_fedsda_experiment_run`、`load_fedsda_run_artifacts`、`convert_settings_to_plain_mapping` |
| 4.1〜4.3 | goldenの条件のtest、共用script、依存境界test |

## Components and Interfaces

### runtime: fedsda_experiment_run

```python
@dataclass(frozen=True, kw_only=True)
class FedsdaExperimentRun:
    run_settings: FedsdaRunSettings
    measured_run: FedsdaMeasuredRun
    run_metrics: FedsdaRunMetrics

def execute_fedsda_experiment_run(*, run_settings: FedsdaRunSettings) -> FedsdaExperimentRun: ...
```

- `run_settings`の型（exact）を確かめ、各部分の検証をもう一度行ってから、`execute_fedsda_stream_protocol_run_with_computation_measurement`と`derive_fedsda_run_metrics`を呼ぶ。

### runtime: fedsda_run_record_tables

```python
def build_fedsda_run_record_tables(*, experiment_run: FedsdaExperimentRun) -> dict[str, np.ndarray]: ...
```

キーは`表の名前.列の名前`。表と列（型）:

| 表 | 行 | 列 |
|---|---|---|
| `sample_prediction` | client×処理した標本 | `combined_prediction_is_correct`(bool)、`maximum_weight_model_id`(int64)、`held_model_count`(int64) |
| `concept_change` | 概念の変更 | `client_id`、`sample_index` |
| `training_model_switch` | 学習帰属の切替 | `client_id`、`sample_index` |
| `adaptation` | 適応の記録 | `client_id`、`adaptation_sample_index`、`adaptation_outcome`(str)、`adaptation_action`(str)、`previous_training_model_id`、`current_training_model_id` |
| `candidate_validation_decision` | 候補検証の判定 | `client_id`、`proposal_sample_index`、`is_incomplete_at_run_end`(bool)、`candidate_accepted`(bool)、`decision_reason`(str。終端の回収は空文字)、`resolution_sample_index` |
| `model_registration` | 登録したモデル | `model_id`、`registered_round_index`(初期モデルは−1)、`is_final_global_model`(bool) |
| `model_clustering` | クラスタリングの、モデルごとの観測 | `round_index`、`model_id`、`representative_model_id`、`merged_with_other_models`(bool)、`absorbed_into_representative`(bool) |
| `model_pair_clustering` | クラスタリングの、対ごとの観測 | `round_index`、`lower_model_id`、`higher_model_id`、`assigned_to_same_cluster`(bool) |
| `round` | ラウンド | `global_model_count`、`client_held_model_count`(ラウンド×client) |
| `round_local_processing_computation`・`round_synchronization_computation` | ラウンド | `ModelComputationCounts`の12のfield |

- 整数は`int64`、真偽は`bool`、文字列はNumPyのUnicode。事象の表は、client順・起きた順。
- `adaptation_action`: 結果種別からの区分（`defer_during_candidate_validation`・`skip_too_short_change_interval`・`reuse_held_model`・`maintain_current_model`・`start_candidate_validation`・`adopt_candidate_model`・`reject_candidate_model`・`remap_by_server_consolidation`の8つ。旧の`action`の8つの値と、1対1）。対応の表を、moduleの定数として持ち、未知の結果種別は拒否する。
- ラウンドごとのモデル数は、research.mdの求め方。

### runtime: fedsda_run_artifact_files

```python
RUN_ARTIFACT_FORMAT_VERSION = 1

@dataclass(frozen=True, kw_only=True)
class FedsdaRunArtifactPaths:
    summary_file_path: Path          # <file_stem>.json
    record_table_file_path: Path     # <file_stem>.npz

def save_fedsda_experiment_run(
    *, experiment_run: FedsdaExperimentRun, output_directory: Path, file_stem: str
) -> FedsdaRunArtifactPaths: ...
def load_fedsda_run_artifacts(
    *, summary_file_path: Path
) -> tuple[dict[str, object], dict[str, np.ndarray]]: ...
```

- 要約のキー: `format_version`、`run_settings`、`run_metrics`、`environment`（OS、機種、Python・NumPy・torchの版）、`record_table_file_name`。
- 保存: 出力のディレクトリは、あること（作らない）。`file_stem`は、区切り文字を含まない名前。どちらかのファイルがあれば`FileExistsError`。一時の名前へ書いてから、名前を変える。
- 読込み: 要約を読み、形式の版を確かめ、同じディレクトリの表のファイルを、`allow_pickle=False`で読む。

### core: settings_serialization

- `convert_settings_to_plain_mapping(settings, *, non_finite_float_as_none=False)`: 指定したとき、非有限のfloatを`None`にする（指定しなければ、これまでどおり拒否）。指標の辞書に使う。

## 検証の方針

- goldenの3ケース: `execute_fedsda_experiment_run`で実行→保存→読込み→表だけから旧の31の配列を作る→実旧が保存した配列と照合。要約の指標から、旧の33指標を作って照合。ラウンドごとのモデル数を、実旧の3つの配列と照合。
- 小さい条件（登録・統合を通る）: ラウンドごとのモデル数を、実旧の全体run（`run_real_legacy_whole_run`に、ラウンドごとの記録を足す）と照合。
