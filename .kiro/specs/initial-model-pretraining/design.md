# Design Document: initial-model-pretraining

## Overview

**Purpose**: 初期モデルの事前学習を、旧と同じ順序・数値・乱数の消費で行い、clientの組立てが受け取る形で返す。

**Users**: リファクタリングを進める研究者が、全clientとサーバの準備（`prepare_run`）を接続する前提として使う。

**Impact**: 既存のsourceは変更しない。

### Goals

- 実旧の事前学習と、パラメータ・2つのoptimizerの状態・損失統計・3つの乱数の状態が一致する。
- 結果を、そのまま`assemble_fedsda_run_client`へ渡せる。

### Non-Goals

- 全clientとサーバの準備、サーバへの登録、SINE以外のdataset、モデルの寸法の設定の置き場所。

## Boundary Commitments

### This Spec Owns

- 設定型`InitialModelPretrainingSettings`。
- 関数`pretrain_initial_model`と、結果の記録`PretrainedInitialModel`。

### Out of Boundary

- 分類器の生成、標本生成器、共同更新、損失の評価、損失統計の逐次更新の、中の処理。
- 実行の枠、clientの組立て。変更しない。

### Allowed Dependencies

- learningの新module（設定型）は、`dataclasses`とcoreの検査だけに依存する。
- runtimeの新moduleは、`dataclasses`・`random`・`torch`、data（SINEの生成器、観測標本）、learning（分類器、モデル構造の設定、optimizerの設定と状態、共同更新、損失の評価、損失統計、学習の設定）に依存する。旧実装をimportしない。

### Revalidation Triggers

- `PretrainedInitialModel`のfieldの変更（clientの組立てと、`prepare_run`のfactoryが読む）。
- 分類器の生成の乱数の消費順、標本生成器、共同更新の変更。

## File Structure Plan

```
src/federated_learning_experiments/
├── learning/training/
│   └── initial_model_pretraining_settings.py   # 新規: InitialModelPretrainingSettings
└── runtime/
    └── initial_model_pretraining.py            # 新規: PretrainedInitialModel、pretrain_initial_model
tests/refactoring/
├── test_initial_model_pretraining.py           # 新規: 設定の検査、実旧の事前学習との対照、拒否、clientの組立てへの接続
├── test_single_run_dependency_boundaries.py    # 変更: 新moduleの許可集合の登録
└── fresh_process_smoke.py                      # 変更: clientの流れの初期モデルを、事前学習で作る
```

## 旧処理との対応

research.mdの対応表のとおり。旧と違う点:

- **乱数**: 旧は、module全体の`random`と`numpy.random`を使う。新は、借りた`Random`と、標本生成器が持つ`RandomState`を使う（同じseedなら同じ列）。torchは、どちらも全体の乱数を使う。
- **設定**: 旧は`config`の`PRETRAIN_SAMPLES`・`PRETRAIN_EPOCHS`・`PRETRAIN_BATCH_SIZE`、`BASE_LR`ほか、datasetの定義を読む。新は、設定型と引数で受け取る。
- **統計の初期値**: 旧は、クラス別の統計を持つ辞書を返す。新は、不変の`ModelAndClassLossStatistics`を返す。
- **勾配**: 旧のモデルには、最後の更新の勾配が残る。新の分類器にも残る（clientの組立ては、写しを作るとき、旧の`deepcopy`と同じく勾配を引き継がない）。
- 計算量の記録は行わない（旧の事前学習も行わない）。

## Requirements Traceability

| Requirement | Components |
|-------------|------------|
| 1.1〜1.5 | `pretrain_initial_model`、`PretrainedInitialModel` |
| 2.1 | `pretrain_initial_model`（実旧の事前学習との対照test） |
| 2.2 | `pretrain_initial_model`と`assemble_fedsda_run_client`（接続のtest） |
| 3.1 | `InitialModelPretrainingSettings` |
| 3.2, 3.3 | `pretrain_initial_model` |
| 4.1 | 共用script |
| 4.2 | 依存境界test |

## Components and Interfaces

### learning/training: InitialModelPretrainingSettings

| field | 型・範囲 | 旧 |
| --- | --- | --- |
| `pretraining_sample_count` | int、1以上 | `PRETRAIN_SAMPLES` |
| `pretraining_epoch_count` | int、0以上 | `PRETRAIN_EPOCHS` |
| `pretraining_batch_sample_count` | int、1以上 | `PRETRAIN_BATCH_SIZE` |

既存の検査の仕組み（fieldのmetadata）で確かめ、builtin intだけを受け入れる（`RunSettingsValidationError`）。

### runtime: pretrain_initial_model

```python
@dataclass(frozen=True, kw_only=True)
class PretrainedInitialModel:
    classifier: ResidualAdapterClassifier
    concept_specific_parameter_optimizer_state: ParameterOptimizerState
    shared_parameter_optimizer_state: ParameterOptimizerState
    loss_statistics: ModelAndClassLossStatistics

def pretrain_initial_model(
    *,
    initial_model_pretraining_settings: InitialModelPretrainingSettings,
    model_architecture_settings: ModelArchitectureSettings,
    hidden_layer_widths: tuple[int, ...],
    class_count: int,
    parameter_optimizer_settings: AdamParameterOptimizerSettings | SgdParameterOptimizerSettings,
    sample_generator: SineSampleGenerator,
    python_random_generator: Random,
) -> PretrainedInitialModel: ...
```

検査（すべて、分類器を作る前・乱数を消費する前）:

| 検査（例外） | 値の出所 |
| --- | --- |
| 事前学習の設定・モデル構造の設定・optimizerの設定がexact型（TypeError）。それぞれの`__post_init__`をもう一度呼ぶ | 引数 |
| `sample_generator`がexact `SineSampleGenerator`で、持っている乱数がexact `numpy.random.RandomState`。`python_random_generator`がexact `random.Random`（TypeError） | 引数 |
| 隠れ層の幅とクラス数が、分類器の生成の条件に合う（`SharedFeatureExtractor.validate_feature_dimensions`と、クラス数がbuiltin intで2以上。ValueError／TypeError） | 引数。分類器の生成が拒否する条件を、乱数を使わない検査で先に確かめる |
| 隠れ層が1層以上（ValueError） | 引数。隠れ層がないと、共有部にパラメータがなく、共有部のoptimizerの生成が、分類器の生成の後で拒否する（旧も、空の共有部ではoptimizerを作れない。LEGACY-010） |
| epoch数が1以上なら、勾配の計算が有効（ValueError） | 呼出し時の状態。無効だと、共同更新が、分類器と標本の生成の後で拒否する |

処理順（検査の後）:

1. 分類器を作る（入力の特徴数は2。torchの全体の乱数を消費する）。概念固有部（アダプタ→分類層の順のパラメータ）と、共有部のoptimizerの状態を作る。
2. 標本生成器から、概念0の観測標本を標本数だけ得る。
3. epoch数だけ繰り返す: 標本のlistをshuffleする。先頭からbatchの件数ずつ、特徴を`[B, 2]`、ラベルを`[B, 1]`のfloat32のtensorにして、参加するモデルが1つの共同更新（共有部も更新する）を行う。
4. 一時的な損失統計のownerへ、最後の並びの順に、1件ずつ、有界損失と観測クラスを記録する。
5. 分類器、2つのoptimizerの状態、取り出した統計を返す。

- 1より後の失敗では、済んだ段（乱数の消費を含む）は残る。検査を通った入力で、後の段が拒否する条件はない（標本生成器は0/1のラベルを返し、クラス数は2以上）。
- 共同更新の設定は、関数の中で固定する（共有部・アダプタ・分類層の共同学習。候補の学習と同じ）。
- 共同更新は、損失を「モデルの損失×件数÷総件数」で求める。参加するモデルが1つのとき、旧の更新（損失の平均をそのまま使う）と式の形が違うので、batchの件数1〜33のすべてと、旧の既定の設定（500標本・10 epoch・batch 32。最後のbatchは20件）で、全パラメータとoptimizerの状態が実旧と一致することを、対照testで確かめる。34件以上のbatchは照合していない。

## Error Handling

- 設定の不正は`RunSettingsValidationError`、引数の不正は`TypeError`／`ValueError`。どれも、何も作る前・乱数を消費する前に拒否する。

## Testing Strategy

実旧をoracleにする。式をtestへ複製しない。

### Unit Tests

- 設定（3.1）: 各fieldの境界と、型・範囲の不正。
- 拒否（3.2）: 各引数の別の型・派生型、不正な寸法とクラス数で、3つの乱数の状態が変わらないこと。拒否の後に、正しい引数で実行できること。
- 結果の形（1.1, 1.5）: 2つのoptimizerが、返した分類器の概念固有部・共有部のパラメータを、決まった順で指すこと。epoch数0では、分類器が初期化のままで、統計だけが求まること。

### Integration Tests

- batchの件数ごとの対照（1.3, 2.1）: batchの件数1〜33のそれぞれで、その件数のbatchを1つだけ作る設定にして3 epoch更新し、実旧と全パラメータ・2つのoptimizerの状態・損失統計を照合する（2値・多クラス、AdamとSGD）。
- 実旧の事前学習との対照（1.2〜1.4, 2.1）: 旧の設定を差し替え、3つの乱数を同じseedで初期化して、実`_pretrain_initial_model(ResidualAdapterMLP)`と新の関数を実行する。2値・多クラス、標本数・epoch数・batchの件数の組（2の冪でない件数、最後のbatchが端数になる組、batchが標本数より大きい組、epoch数0を含む）、旧の既定の設定（500標本・10 epoch・batch 32）、optimizerの種類（Adamの2種、SGD）で、全パラメータ、2つのoptimizerの状態、損失統計、実行後の3つの乱数の状態を照合する。
- clientの組立てへの接続（2.2）: 事前学習の結果から組み立てたclientを、実旧の事前学習の結果から実`__init__`で作った実旧のclientと、生成直後と、続く標本列・ラウンド境界の処理の後に、既存の照合（`assert_run_client_matches_legacy`）で照合する。

### fresh process・依存

- 共用script（4.1）: clientの流れの初期モデルを、事前学習で作る形に替える。
- 依存境界（4.2）: 新しい2 moduleの許可集合を登録する。
