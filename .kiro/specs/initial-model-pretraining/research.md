# Research & Design Decisions

## Summary

- **Feature**: `initial-model-pretraining`
- **Discovery Scope**: Extension（移植済みの部品を、旧の事前学習の順につなぐ）
- **Key Findings**:
  - 旧の事前学習の各段は、移植済みの部品に1対1で対応する。新しく書く数値の処理はない。
  - 旧のモデルの更新1回（`ResidualAdapterMLP.update`）は、参加するモデルが1つの共同更新と同じである（候補の学習が、この形で実旧と照合済み）。
  - 損失統計は、batchの平均・分散ではなく、1件ずつの逐次更新（Welford）で求める。移植済みのbatch用の初期化（`initialize_model_and_class_loss_statistics_from_batch`）とは数値が違うので、使わない。

## Research Log

### 旧`_pretrain_initial_model`と、対応する新部品

- **Sources Consulted**: 旧`experiment.py`（294〜358行、2236〜2241行）、`models.py`（291〜324行、236〜244行、97〜104行）、`data/streams.py`（27〜43行）、`data/synthetic.py`（62〜75行）。新`data/sine/sine_sample_generation.py`、`learning/training/joint_model_parameter_update.py`、`learning/training/candidate_epoch_training.py`（130〜175行）、`learning/prediction/classifier_bounded_loss_evaluation.py`、`learning/loss_statistics/model_and_class_loss_statistics.py`（`record_assigned_loss`）。
- **Findings**:

| 旧 | 新部品 |
| --- | --- |
| `model0 = model_cls()`（共有部→アダプタ→分類層の順に初期化。2つのoptimizerを作る。学習率は`BASE_LR`） | `ResidualAdapterClassifier`の生成と、`ParameterOptimizerState`を2つ |
| `generate_data(0)`をN回（SINEは、一様乱数2つ→境界の判定→float32のtensor） | `SineSampleGenerator.generate_sample(concept_id=0)`をN回 |
| `random.shuffle(replay_buf)` | 借りた`Random`の`shuffle` |
| `torch.stack`でbatchを作り、`model0.update(bx, by)`（両optimizerの`zero_grad`→損失→`backward`→共有部の`step`→概念固有部の`step`） | 参加するモデルが1つの`perform_joint_model_parameter_update`（共有部を更新する） |
| `get_absolute_error(x, y)`を1件ずつ | `evaluate_classifier_per_sample_bounded_losses`を1件ずつ |
| 全体とクラス別の逐次更新（クラスは`setdefault`の順） | `ModelAndClassLossStatisticsStore.record_assigned_loss`（クラスは到着順） |

  - 共同更新は、損失を「モデルごとの損失×件数の和÷総件数」で求める。参加するモデルが1つのとき、旧の損失と同じ値になるかは、件数に依存しうる（×件数÷件数の丸め）。候補の学習のspecが、この形で実旧と全値を照合している。本specの対照では、batchの件数1〜33のすべてと、旧の既定の設定（最後のbatchが20件）で照合し、全部一致した（2026-10-10、Windows基準環境）。
  - 旧の統計の初期値は、件数0・平均0・偏差平方和0で、新の`record_assigned_loss`の、統計のないモデルへの最初の記録と同じである。
- **Implications**: 事前学習は、既存の部品を旧の順に呼ぶ関数1つにする。

### oracle

- 実旧の`experiment._pretrain_initial_model(ResidualAdapterMLP)`を、旧の設定（`DATASET="sine2"`と、小さい寸法）を差し替えて実行する。3つの乱数を同じseedで初期化する。新側は、`Random(seed)`、`RandomState(seed)`から作った標本生成器、`torch.manual_seed(seed)`で実行する。実旧の実行は、fedsda-run-client-assemblyの対照testで確認済み（`build_run_client_oracle`）。

## Design Decisions

### Decision: 事前学習の条件を、learningの設定型にする

- **Selected Approach**: `InitialModelPretrainingSettings`（標本数、epoch数、batchの件数）を、他の学習の設定型と同じ場所・同じ検査の仕組みで置く。
- **Rationale**: 1つの機能の値だけを持つ型で、置き場所が明確である（clientの値の束のような仮の形にしない）。

### Decision: モデルの寸法（隠れ層の幅、クラス数）は引数で受け取る

- **Context**: 旧は、datasetごとの定義（`dataset_spec`）から読む。新には、まだ置き場所がない。入力の特徴数は、観測標本の2で固定である。
- **Selected Approach**: 隠れ層の幅とクラス数を、事前学習の引数にする。置き場所は、完全なrun設定を決めるspecで決める。

### Decision: 統計は、一時的な損失統計のownerで求める

- **Selected Approach**: 関数の中で`ModelAndClassLossStatisticsStore`を1つ作り、1件ずつ`record_assigned_loss`で足して、最後に統計を取り出す。
- **Rationale**: 逐次更新とクラスの到着順は、このownerが実旧と照合済みである。式を書き直さない。

### Synthesis

- **Build vs. Adopt**: すべて既存の部品を使う。
- **Simplification**: 標本の出所を差し替える抽象は作らない（SINEの生成器を直接受け取る）。

## Risks & Mitigations

- 共同更新の「×件数÷件数」で、旧の更新と末尾の桁が違う — 2の冪でないbatchの件数を含む条件で、全パラメータとoptimizerの状態を実旧と照合する。違いが出た場合は、実測を記録して、設計を見直す。
