# Design Document: synthetic-dataset-sample-generation

## Overview

**Purpose**: 新実装の全体runを、合成データのdataset（sine2・sea2・sea4・circle2）で実行できるようにする。

**Users**: リファクタリングを進める研究者。

**Impact**: 新しいmoduleを4つ足す。SINEに固定した前提を持つ、完了済みのmoduleを変更する（観測標本と概念列の型、概念列の生成、実行条件と実行設定、参加者の契約、実行の枠、factory、事前学習、client、計測つきの全体run）。sine2の結果は変えない。

### Goals

- sea2で、goldenの33指標と31の離散列が一致する。
- sea4・circle2で、全体runの最終状態が、実旧と一致する。

### Non-Goals

- MNIST、blobs、固定の概念列、datasetごとの学習率と隠れ層の幅、Linux用のgolden。

## Boundary Commitments

### This Spec Owns

- `data/dataset_definitions.py`、`data/observed_sample_generation.py`、`data/sea/sea_sample_generation.py`、`data/circle/circle_sample_generation.py`（新規）。
- 観測標本と概念列の型の一般化、概念列の生成の一般化。
- 実行の枠・factory・事前学習・clientの、datasetの定義への切替え。

### Out of Boundary

- 予測、損失、検出器、学習、サーバの処理（特徴数・クラス数に依らない。変えない）。

### Allowed Dependencies

- 生成器（data）→ numpy、観測標本の型。
- 概念列の生成（data）→ datasetの定義。
- 実行の枠・契約（execution）→ 生成器の共通の型（data）。
- runtime → datasetの定義、生成器を作る関数。

### Revalidation Triggers

- datasetの追加（MNIST）。観測標本の特徴の持ち方の変更。

## File Structure Plan

### New Files

- `src/federated_learning_experiments/data/dataset_definitions.py`
- `src/federated_learning_experiments/data/observed_sample_generation.py`
- `src/federated_learning_experiments/data/sea/sea_sample_generation.py`、`data/sea/__init__.py`
- `src/federated_learning_experiments/data/circle/circle_sample_generation.py`、`data/circle/__init__.py`
- `tests/refactoring/test_dataset_definitions.py`、`tests/refactoring/test_synthetic_sample_generation.py`

### Modified Files

- `data/observed_streams.py`、`data/sine/sine_sample_generation.py`（観測列を作る関数を、共通のmoduleへ移す）、`data/concept_schedules/random_concept_schedule_generation.py`
- `configuration/experiment_run_conditions.py`（dataset名の許す値へ、sea4・circle2を足す）、`execution/stream_protocol_execution_settings.py`（対応するdatasetの検査）、`execution/run_participant_contracts.py`
- `runtime/single_run_execution.py`、`runtime/fedsda_run_participant_factory.py`、`runtime/initial_model_pretraining.py`、`runtime/fedsda_run_client.py`、`runtime/fedsda_measured_run_execution.py`
- 既存のtest（型の一般化、名前の変更への追随、datasetを選べる対照）、共用script、依存境界test。

## Requirements Traceability

| Requirement | Components |
|-------------|------------|
| 1.1〜1.3 | `DatasetDefinition`、`get_dataset_definition` |
| 2.1, 2.3, 2.4, 2.6 | `SeaSampleGenerator` |
| 2.2, 2.3, 2.4, 2.6 | `CircleSampleGenerator` |
| 2.5 | `create_observed_sample_generator`、`ObservedSampleGenerator` |
| 3.1, 3.2 | `ObservedSample`、`ClientConceptTrace` |
| 3.3 | `generate_random_client_concept_traces` |
| 4.1〜4.4 | 実行の枠、factory、事前学習、client、実行設定 |
| 5.x | goldenの照合のtest、全体runの対照、共用script、依存境界test |

## Components and Interfaces

### data: dataset_definitions

```python
@dataclass(frozen=True, kw_only=True)
class DatasetDefinition:
    dataset_name: str
    input_feature_count: int
    concept_count: int
    class_count: int

def get_dataset_definition(*, dataset_name: str) -> DatasetDefinition: ...   # 定義がなければ ValueError
```

### data: 生成器

```python
class SeaSampleGenerator:
    def __init__(self, *, numpy_random_generator: np.random.RandomState, concept_count: int) -> None: ...
    def generate_sample(self, *, concept_id: int) -> ObservedSample: ...

class CircleSampleGenerator:
    def __init__(self, *, numpy_random_generator: np.random.RandomState) -> None: ...
    def generate_sample(self, *, concept_id: int) -> ObservedSample: ...
```

- SEA: `uniform(0.0, 10.0, size=3)`→`特徴0＋特徴1 <= 閾値[概念]`でラベル→`rand() < 0.10`なら反転。閾値は(9.0, 8.0, 7.0, 9.5)。`concept_count`は2（sea2）か4（sea4）で、受け付ける概念IDの範囲を決める。
- CIRCLE-2: `uniform(0.0, 1.0, size=2)`→`(x0−cx)²＋(x1−cy)²−r² > 0`ならラベル1。
- どちらも、概念IDの検査（builtin int、範囲）を、乱数より前に行う。特徴は、float32へ直してから、Pythonのfloatのtupleにする。

### data: observed_sample_generation

```python
class ObservedSampleGenerator(Protocol):
    def generate_sample(self, *, concept_id: int) -> ObservedSample: ...

OBSERVED_SAMPLE_GENERATOR_TYPES: tuple[type, ...]   # このmoduleが作る生成器の型

def create_observed_sample_generator(
    *, dataset_name: str, numpy_random_generator: np.random.RandomState
) -> ObservedSampleGenerator: ...
def build_client_observed_streams(
    *, evaluation_concept_traces: tuple[ClientConceptTrace, ...], sample_generator: ObservedSampleGenerator
) -> tuple[ClientObservedStream, ...]: ...
```

- `build_client_observed_streams`は、SINE専用だった`build_sine_client_observed_streams`を、名前を変えて、ここへ移す（処理は同じ）。

### data: 型と概念列

- `ObservedSample`: `feature_values`は、長さ1以上の、floatのtuple。`class_label`は、0以上のbuiltin int。
- `ClientConceptTrace`: 概念IDは、0以上のbuiltin int。
- `generate_random_client_concept_traces`: 概念数を、実行条件のdataset名の定義から取る。候補は、`range(概念数)`から現在の概念を除いたもの（昇順）。

### execution・runtime

- 実行設定の検査: dataset名が、datasetの定義にあること（mnist2は、まだ拒否）。
- 契約`RunParticipantFactory.prepare_run`の`sample_generator`の型を、`ObservedSampleGenerator`にする。
- 実行の枠: `create_observed_sample_generator`で生成器を作り、`build_client_observed_streams`で観測列を作る。
- factory: dataset名の検査を、定義の有無にする。初期モデルの、入力の特徴数とクラス数を、定義から取る。生成器の型の検査は、`OBSERVED_SAMPLE_GENERATOR_TYPES`で行う。
- 事前学習: 入力の特徴数を、引数で受け取る（定数をやめる）。
- client: 初期モデルの入力の特徴数を、定数と比べる検査をやめる（分類器と標本の特徴数の不一致は、標本の処理の、入力の検査が拒否する）。

## 旧と違う点

- SEAの閾値・雑音率と、CIRCLEの円は、設定ではなく、定数（research.mdの「Decision」）。
- blobs、MNIST、固定の概念列は、対象外。

## Error Handling

- 定義のないdataset名、範囲外の概念IDは、`ValueError`。型の不正は、`TypeError`。実行設定の不正は、既存の`RunSettingsValidationError`。

## Testing Strategy

### Unit Tests

- datasetの定義: 実旧の`DATASET_SPECS`との一致、拒否。
- 生成器: 実旧の`generate_data`と、同じ乱数の状態から、特徴・ラベル・生成の後の乱数の状態が一致すること（全dataset・全概念、多数の標本。SEAは、雑音で反転した標本を含むこと、CIRCLEは、両方のラベルを含むこと）。概念IDの拒否（乱数を進めない）。作る関数が、datasetごとの型を返すこと。
- 型: 観測標本（1特徴・多特徴、クラスラベル2以上）、概念列（2以上の概念ID）、拒否。
- 概念列の生成: 4概念のdatasetで、実旧の`make_random_schedules`と、概念列・乱数の状態が一致すること。3つ以上の概念が現れること。

### Integration Tests

- goldenの条件（sea2）: 指標の導出のtestのfixtureを、datasetごとに回せる形にして、33指標・31の離散列を、実旧の実行結果と、Windows用のgoldenへ照合する（sine2は、そのまま）。
- 全体runの対照: 既存のhelperへ、datasetを渡せるようにして、sea2・sea4・circle2の小さい条件で、実旧の全体runと、全状態を照合する。sea4は、3つ以上の概念が現れる条件。
- 既存のsine2の対照・goldenの照合・計算量の照合が、そのまま通ること。

### fresh process・依存

- 共用script: sine2以外のdatasetの全体run。
- 依存境界。
