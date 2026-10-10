# Design Document: fedsda-complete-run-settings

## Overview

**Purpose**: 最終構成のFedSDAの1 runの条件を、1つの検証済みの値として作り、渡し、保存できるようにする。

**Impact**: 新しいmoduleを3つ足す。既存の設定型と、全体runの実行は、変えない。

### Goals

- 完全なrun設定の型と、最終構成の既定値の関数と、保存用の辞書への変換。
- 既定値が、旧の設定の値と、全部の項目で一致する。goldenの3ケースの条件を、既定値から作れる。

### Non-Goals

- 1 runの実行と保存、掃引、コマンド、辞書から設定への変換、設定の束の作り直し、FedDrift。

## Boundary Commitments

### This Spec Owns

- `core/settings_serialization.py`（新規）。
- `runtime/fedsda_run_settings.py`（新規）。
- `runtime/fedsda_final_configuration_run_settings.py`（新規）。

### Out of Boundary

- 既存の設定型（機能別、束、実行の枠、指標）の定義と検証（clientの設定の束への、検出器の表示名の検査の追加を除く）。全体runの実行。

### Allowed Dependencies

- `core/settings_serialization` → 標準ライブラリだけ。
- `runtime/fedsda_run_settings` → configuration（部分型）、execution（実行の枠の設定）、evaluation（指標の設定）、methods（統合の設定）、runtime（参加者の設定の束）。
- `runtime/fedsda_final_configuration_run_settings` → 上の型と、機能別の設定型、datasetの定義。

## File Structure Plan

### New Files

- `src/federated_learning_experiments/core/settings_serialization.py`
- `src/federated_learning_experiments/runtime/fedsda_run_settings.py`
- `src/federated_learning_experiments/runtime/fedsda_final_configuration_run_settings.py`
- `tests/refactoring/test_settings_serialization.py`、`tests/refactoring/test_fedsda_run_settings.py`、`tests/refactoring/test_fedsda_final_configuration_run_settings.py`

### Modified Files

- 依存境界test（登録）。

## Requirements Traceability

| Requirement | Components |
|-------------|------------|
| 1.1〜1.4 | `FedsdaRunSettings` |
| 2.1〜2.5 | `build_final_configuration_fedsda_run_settings` |
| 3.1〜3.3 | `convert_settings_to_plain_mapping` |

## Components and Interfaces

### core: settings_serialization

```python
def convert_settings_to_plain_mapping(settings: object) -> dict[str, object]: ...
```

- 入力は、dataclassの値（型そのものではない）。戻り値は、`{"settings_type": 型の名前, field名: 変換した値, ...}`（fieldの宣言の順）。
- 値の変換: dataclass→同じ規則の辞書。tuple・list→list（要素を変換）。`str`・`int`・`bool`・`None`→そのまま。`float`→有限なら、そのまま。ほかは、`TypeError`（非有限のfloatは`ValueError`）。field名`settings_type`を持つdataclassは、拒否する。

### runtime: fedsda_run_settings

```python
@dataclass(frozen=True, kw_only=True)
class FedsdaRunSettings:
    execution_settings: StreamProtocolExecutionSettings
    run_participant_settings: FedsdaRunParticipantSettings
    model_consolidation_settings: ModelConsolidationSettings
    run_metric_settings: RunMetricSettings
```

- 生成時: 4つの部分の型（exact）を確かめ、各部分の検証をもう一度行う。既存の部分型`ValidatedExperimentRunSettingsSubset`を、部分から組み立てて、機能の組合せを検証する（手法の名前は`fedsda`）。
- 部分型へ渡す値: 実行条件（実行の枠の設定から）、モデルの構造・検出・予測の結合・学習・帰属・候補の方針（参加者の設定の束と、その中のclientの設定の束から）、統合の設定。モデルの構造は、参加者の設定の束が持つ1つだけなので、食い違いは起きない。
- 失敗は、既存の`RunSettingsValidationError`。
- 一致すべき値の食い違い（要求1.3）は、値を両方持つ束が確かめる: 検出の方式と検出器の表示名は、clientの設定の束（このspecで足した。対応は、`("e_sr", "overall_and_true_class_losses")`→`"overall + class-conditional e-SR mixture"`の1つ）。再利用の許容量とクラスタリングの判定の上限は、参加者の設定の束（既存）。完全なrun設定は、各部分の検証をもう一度行うので、frozenを回避して組み立てた食い違いも、ここで拒否される。

### runtime: fedsda_final_configuration_run_settings

```python
def build_final_configuration_fedsda_run_settings(
    *, experiment_run_conditions: ExperimentRunConditions
) -> FedsdaRunSettings: ...
```

- 既定値は、moduleの定数（research.mdの「Key Findings」の値）。
- datasetごとのモデルの既定: 隠れ層の幅（mnist2・mnist4は(1568,)、ほかは(32, 32)）と、学習率（mnist2・mnist4は1e-3、ほかは1e-2。2つのoptimizerの設定の両方に使う）。定義のないdatasetは、実行条件の検証と、ここの表の両方で拒否される。
- 概念の変更（最小の間隔300件、確率0.0015）と、指標の設定（遅れの許容100件、回復の窓200件）も、既定値として持つ。

## 検証の方針

- 既定値: 旧の設定を、最終構成の固定設定で有効化して読んだ値を、既存のtestの対応（`make_golden_condition_settings`）で新の設定へ写したものと、`==`で照合する（sine2とmnist2。全datasetで、隠れ層の幅と学習率を、旧のdatasetの定義と照合する）。
- goldenの条件: 既定値から、旧の回帰testが縮小した項目だけを`replace`した設定が、既存の（goldenを再現する）設定と、`==`で一致する（3ケース）。
