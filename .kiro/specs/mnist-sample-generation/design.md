# Design Document: mnist-sample-generation

## Overview

**Purpose**: 新実装の全体runを、MNIST（mnist2・mnist4）で実行できるようにする。

**Users**: リファクタリングを進める研究者。

**Impact**: 新しいmoduleを2つ足す。datasetの定義、生成器を作る関数、実行条件の許す値へ、MNISTを足す。合成データの結果は変えない。

### Goals

- mnist2で、goldenの33指標と31の離散列が一致する。
- mnist4で、全体runの最終状態が、実旧と一致する。

### Non-Goals

- ファイルの取得、blobs、固定の概念列、datasetごとの学習率と隠れ層の幅の既定、観測標本の持ち方の変更、Linux用のgolden。

## Boundary Commitments

### This Spec Owns

- `data/mnist/mnist_training_data.py`、`data/mnist/mnist_sample_generation.py`（新規）。
- `data/dataset_definitions.py`・`data/observed_sample_generation.py`・`configuration/experiment_run_conditions.py`への、MNISTの追加。

### Out of Boundary

- 実行の枠、factory、事前学習、client、予測、損失、検出器、学習、サーバの処理（変えない）。

### Allowed Dependencies

- 読込み（data）→ 標準ライブラリ（gzip、struct、os、pathlib、dataclasses）、numpy。
- 生成器（data）→ numpy、観測標本の型、読込みの結果の型。

### Revalidation Triggers

- 観測標本の特徴の持ち方の変更。置き場所の決め方の変更。

## File Structure Plan

### New Files

- `src/federated_learning_experiments/data/mnist/__init__.py`
- `src/federated_learning_experiments/data/mnist/mnist_training_data.py`
- `src/federated_learning_experiments/data/mnist/mnist_sample_generation.py`
- `tests/refactoring/test_mnist_sample_generation.py`

### Modified Files

- `data/dataset_definitions.py`（mnist2・mnist4）、`data/observed_sample_generation.py`（表と、生成）、`configuration/experiment_run_conditions.py`（mnist4）
- 既存のtest（「定義のないdataset」の例にmnist2を使っている箇所、全体runの対照、goldenの照合）、共用script、依存境界test。

## Requirements Traceability

| Requirement | Components |
|-------------|------------|
| 1.1〜1.3 | `get_dataset_definition`、`ExperimentRunConditions` |
| 2.1〜2.5 | `load_mnist_training_data`、`MnistTrainingData` |
| 2.6 | `resolve_mnist_data_directory` |
| 3.1〜3.3, 3.5, 3.6 | `MnistSampleGenerator` |
| 3.4 | `create_observed_sample_generator`、`is_observed_sample_generator_of_dataset` |
| 4.x | goldenの照合のtest、全体runの対照、共用script、依存境界test |

## Components and Interfaces

### data/mnist: mnist_training_data

```python
@dataclass(frozen=True, kw_only=True, eq=False)
class MnistTrainingData:
    pixel_values: np.ndarray   # uint8、(件数, 784)、書込み不可
    digit_labels: np.ndarray   # int64、(件数,)、書込み不可

def resolve_mnist_data_directory() -> Path: ...
def load_mnist_training_data(*, data_directory: Path) -> MnistTrainingData: ...
```

- `resolve_mnist_data_directory`: `FDE_MNIST_DATA_DIR`が空でなければ、その値。なければ、リポジトリ直下（このmoduleから4つ上）の`data/mnist`。
- `load_mnist_training_data`: `train-images-idx3-ubyte.gz`（識別の数値2051）と`train-labels-idx1-ubyte.gz`（識別の数値2049）を読む。ファイルがなければ、`FileNotFoundError`（足りないファイルの名前と、`FDE_MNIST_DATA_DIR`を書く）。形式の不正は、`ValueError`。解決したディレクトリごとに、結果を保持して使い回す。
- 画素は、uint8のまま持つ（float32への変換は、生成器が、256個の値の表で行う）。

### data/mnist: mnist_sample_generation

```python
class MnistSampleGenerator:
    def __init__(
        self, *, numpy_random_generator: np.random.RandomState, concept_count: int,
        mnist_training_data: MnistTrainingData,
    ) -> None: ...
    def generate_sample(self, *, concept_id: int) -> ObservedSample: ...
```

- 概念IDの検査（builtin int、`0 <= concept_id < concept_count`）を、乱数より前に行う。`concept_count`は、1以上4以下。
- `randint(0, 件数, size=1)`で位置を選ぶ→その行の画素を、moduleの定数の表（`float32(画素) / 255.0`を、Pythonのfloatにした256個）で引いて、tupleにする→ラベルを、概念で交換する（概念kは、`2k−1`と`2k`を入れ替える。概念0は、そのまま）。

### data: dataset_definitions・observed_sample_generation

- 定義へ、mnist2（784・2・10）、mnist4（784・4・10）を足す。
- dataset名から生成器の型への表へ、mnist2・mnist4→`MnistSampleGenerator`を足す。`create_observed_sample_generator`は、MNISTのとき、`load_mnist_training_data(data_directory=resolve_mnist_data_directory())`の結果と、定義の概念数を渡す。`is_observed_sample_generator_of_dataset`は、MNISTでも、概念数の一致を確かめる。

## 旧と違う点

- ファイルを取得しない（research.mdの「Decision」）。
- 画素を、uint8のまま持つ。標本の特徴の値は、旧と同じ。
- 学習率と隠れ層の幅は、datasetの定義に入れない。
