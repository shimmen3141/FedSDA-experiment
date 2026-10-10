# Research & Design Decisions

## Summary

- **Feature**: `synthetic-dataset-sample-generation`
- **Discovery Scope**: Extension（datasetを足す）＋ 完了済みのmoduleの、SINEに固定した前提の一般化
- **Key Findings**:
  - SINEに固定しているのは、観測標本の型（2特徴・二値）、概念列の型（0/1）、概念列の生成（2概念）、生成器の型の名指し（実行の枠、参加者の契約とfactory、事前学習、計測つきの全体run）、入力の特徴数の定数（事前学習、client）、クラス数の定数（factory）、dataset名の検査（実行設定、factory）。
  - 予測・損失・検出器・学習・クロス評価・統合は、特徴数とクラス数に依らない形で、移植済み（4クラスの対照がある）。
  - SEAとCIRCLE-2の定数（閾値、雑音率、円）は、掃引の対象ではなく、datasetを決める値。

## Research Log

### 旧のdatasetと生成

- **Sources Consulted**: federated_drift_experiment/data/specs.py（`DATASET_SPECS`）、names.py、synthetic.py（`generate_sea4`・`generate_circle2`・`generate_sine2`）、streams.py（`generate_data`・`build_data_streams`）、schedules.py（`make_random_schedules`・`make_concept_schedules`）、config.py（`SEA_THRESHOLDS`・`SEA_LABEL_NOISE`・`CIRCLE_PARAMS`・`num_concepts`・`dataset_spec`）、experiment.py（`_pretrain_initial_model`——事前学習の標本は、概念0から生成する）、tests/test_proposed_regression.py（`CASES`の`sea2`: 標本600件。`COMMON`が、SEAの定数を既定値で固定）。
- **Findings**:
  - `generate_data(concept_id)`は、標本1件を生成し、特徴をfloat32のtensor（1次元）、ラベルをfloat32のtensor（要素1つ）にする。
  - SEA: `np.random.uniform(0.0, 10.0, size=3)`→ラベル→`np.random.rand()`（雑音。毎回引く）。CIRCLE-2: `np.random.uniform(0.0, 1.0, size=2)`→`(x0−cx)²＋(x1−cy)²−r² > 0`ならラベル1。
  - `sea2`は、`generate_sea4`を、概念0・1で使う（閾値9.0・8.0）。
  - 概念列: `random.random() < 確率`の後、`random.choice([現在以外の概念])`。
- **Implications**: 生成器は、datasetごとに、借りたNumPyの乱数を、旧と同じ順で使う。概念列の生成は、概念数を、datasetの定義から取る。

### 新の、SINEに固定した箇所

- **Sources Consulted**: data/observed_streams.py、data/sine/sine_sample_generation.py、data/concept_schedules/random_concept_schedule_generation.py、configuration/experiment_run_conditions.py（dataset名の許す値は`sine2`・`sea2`・`mnist2`）、execution/stream_protocol_execution_settings.py（`sine2`以外を拒否）、execution/run_participant_contracts.py（`prepare_run`の`sample_generator: SineSampleGenerator`）、runtime/single_run_execution.py、runtime/fedsda_run_participant_factory.py（`_SUPPORTED_DATASET_NAME`・`_SINE_CLASS_COUNT`・生成器のexact型の検査）、runtime/initial_model_pretraining.py（`_OBSERVED_SAMPLE_FEATURE_COUNT`・生成器のexact型の検査）、runtime/fedsda_run_client.py（`_OBSERVED_SAMPLE_FEATURE_COUNT`——初期モデルの入力の特徴数の検査）、runtime/fedsda_measured_run_execution.py（包みの引数の型）。
- **Findings**: 上の箇所を、datasetの定義と、生成器の共通の型へ置き換えれば、ほかの部品は変えずに済む。

## Design Decisions

### Decision: datasetの定義を、data層の1つのmoduleに置く

- **Selected Approach**: `data/dataset_definitions.py`に、`DatasetDefinition`（dataset名、入力の特徴数、概念数、クラス数）と、名前から引く関数を置く。対応するdatasetは、このspecでは、sine2・sea2・sea4・circle2。
- **Rationale**: 特徴数・概念数・クラス数を、生成器、概念列の生成、参加者の準備が、同じ所から取る。隠れ層の幅と学習率は、手法の設定の束が持つ（datasetの定義には入れない。MNISTのspecで、datasetごとの既定の扱いを決める）。

### Decision: SEAとCIRCLE-2の定数は、生成器のmoduleの定数にする

- **Alternatives Considered**: 設定の型を作って、実行設定から渡す——旧では設定だが、掃引の対象ではなく、値を変えると、別のdatasetになる。
- **Selected Approach**: 閾値（9.0、8.0、7.0、9.5）、雑音率（0.10）、円（(0.2, 0.5, 0.15)、(0.6, 0.5, 0.25)）を、生成器のmoduleの定数にする。
- **Rationale**: dataset名が、生成の規則を一意に決める。変えたいときは、新しいdataset名を足す。
- **Follow-up**: 雑音率を変える実験が要るなら、そのときに、設定へ出す。

### Decision: 生成器は、共通の型（Protocol）で受け渡し、作るのは1つの関数

- **Selected Approach**: `data/observed_sample_generation.py`に、`ObservedSampleGenerator`（`generate_sample(concept_id)`を持つProtocol）、`create_observed_sample_generator(dataset_name, numpy_random_generator)`、`build_client_observed_streams`（SINE専用だった関数の一般化）を置く。受け取る側のexact型の検査は、「このmoduleが作る生成器の型のどれか」で行う。
- **Rationale**: 実行の枠・契約・事前学習が、datasetごとの型を名指ししない。

### Decision: 観測標本の特徴は、floatのtupleのままにする

- **Alternatives Considered**: float32の配列・tensorで持つ——MNIST（784次元）で、メモリと変換の負担が小さい。ただし、観測標本・実行の結果の等値の比較（多くのtestと、結果の型の検査が使う）が、配列では成り立たない。
- **Selected Approach**: このspecでは、tupleのまま、長さの固定だけを外す。
- **Follow-up**: MNISTのspecで、784次元のときのメモリと時間を測り、必要なら、持ち方を見直す。

### Synthesis

- **Build vs. Adopt**: SINEの生成器の形（借りた`RandomState`を持ち、標本1件を返す）を、SEAとCIRCLE-2でも使う。
- **Simplification**: datasetごとの設定の型は作らない。

## Risks & Mitigations

- 型の一般化で、SINEの前提に依存していた検査（「2特徴」「0/1」）が弱くなる — 観測標本は、長さ1以上・クラスラベル0以上、を確かめる。datasetと合わない標本は、分類器の入力の検査（特徴数、クラスラベルの範囲）が拒否する。生成器は、自分のdatasetの概念IDだけを受け付ける。
- sine2の結果が変わる — goldenの条件と、全体runの対照（9条件）、計算量の照合を、そのまま通す。

## References

- [single-run-execution](../single-run-execution/)（実行の枠と、SINEの供給）、[fedsda-run-participant-preparation](../fedsda-run-participant-preparation/)（参加者の準備）、[fedsda-run-metric-derivation](../fedsda-run-metric-derivation/)（goldenの照合のtest）。
