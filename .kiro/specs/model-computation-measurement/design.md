# Design Document: model-computation-measurement

## Overview

**Purpose**: 新実装の全体runの計算量を、実行中に外側から数える方法で計測し、goldenの計算量の7項目と照合する。新実装の重複した計算を3箇所なくして、実際の計算量を、旧と同じにする。

**Users**: リファクタリングを進める研究者。

**Impact**: 新しいmoduleを3つ足す。完了済みのmoduleを5つ変更する（有界損失の評価、クロス評価、警報時の区間の準備、集約後の再較正、client、指標の導出）。結果（状態、指標、乱数）は変えない。

### Goals

- goldenの33指標のすべてが、新実装で照合される。
- 計算量の計測が、部品の数え方に依存しない。
- 旧の計数の検査が、恒久のtestとして残る。

### Non-Goals

- 用途別・clientごと・ラウンドごとの内訳、実行時間、モデルの大きさを反映した計算量、事前学習の計算を指標へ含めること。

## Boundary Commitments

### This Spec Owns

- `learning/models/model_computation_measurement.py`（計測）。
- `evaluation/loss_monitoring_computation_count_store.py`（検出器の計算の計数）。
- `runtime/fedsda_measured_run_execution.py`（計測つきの全体run）。
- 有界損失の評価の、3つの新しい入口。3つの利用箇所の、重複の解消。
- `FedsdaRunMetrics`の、計算量の2項目。

### Out of Boundary

- 損失の式、検出器、予測、学習の中身。固定旧実装とgolden。

### Allowed Dependencies

- 計測 → torch、`learning/models`の2つのmoduleの型。
- 計測つきの全体run → 計測、実行の枠、参加者のfactory。
- 指標の導出 → 計測の計数の型、検出器の計数の型。

### Revalidation Triggers

- 共有部・概念固有部のmoduleの型や、呼び方（`__call__`を通らない呼出し）の変更。勾配つきの推論を行う部品の追加。

## File Structure Plan

### New Files

- `src/federated_learning_experiments/learning/models/model_computation_measurement.py`
- `src/federated_learning_experiments/evaluation/loss_monitoring_computation_count_store.py`
- `src/federated_learning_experiments/runtime/fedsda_measured_run_execution.py`
- `tests/refactoring/test_legacy_computation_count_audit.py`、`test_model_computation_measurement.py`、`test_loss_monitoring_computation_count_store.py`、`test_fedsda_measured_run_execution.py`

### Modified Files

- `learning/prediction/classifier_bounded_loss_evaluation.py` — 3つの入口。
- `runtime/client_model_cross_evaluation.py`、`runtime/alarm_training_interval_preparation.py`、`runtime/post_aggregation_prediction_recalibration.py` — 重複の解消。
- `runtime/fedsda_run_client.py` — 検出器の計数のownerと、標本の処理の後の加算。
- `runtime/fedsda_run_metric_derivation.py` — 計算量の2項目。
- 既存のtest（有界損失の評価、3つの利用箇所、client、指標の導出）、共用script、依存境界test。

## Requirements Traceability

| Requirement | Components |
|-------------|------------|
| 1.1, 1.2 | `test_legacy_computation_count_audit.py` |
| 2.1〜2.5 | `measure_model_computation`、`ModelComputationMeter`、`ModelComputationCounts`、`subtract_model_computation_counts` |
| 3.1, 3.2 | `LossMonitoringComputationCountStore`、`FedsdaRunClient` |
| 4.1〜4.5 | 有界損失の評価の3つの入口、3つの利用箇所 |
| 5.1〜5.4 | `execute_fedsda_stream_protocol_run_with_computation_measurement`、`derive_fedsda_run_metrics` |
| 6.1, 6.2 | 共用script、依存境界test |

## Components and Interfaces

### learning/models: model_computation_measurement

```python
@dataclass(frozen=True, kw_only=True)
class ModelComputationCounts:
    shared_part_training_example_count: int
    shared_part_inference_example_count: int
    concept_specific_part_training_example_count: int
    concept_specific_part_inference_example_count: int
    shared_parameter_optimizer_step_count: int
    concept_specific_parameter_optimizer_step_count: int

class ModelComputationMeter:
    def get_model_computation_counts(self) -> ModelComputationCounts: ...

def measure_model_computation() -> ContextManager[ModelComputationMeter]: ...
def subtract_model_computation_counts(
    *, later_counts: ModelComputationCounts, earlier_counts: ModelComputationCounts
) -> ModelComputationCounts: ...
```

- 順伝播: moduleの型がexact `SharedFeatureExtractor`なら共有部、exact `NonlinearResidualAdapter`なら概念固有部。足す数は、最初の入力のtensorの、先頭の次元の大きさ（標本数）。`torch.is_grad_enabled()`が真なら学習、偽なら推論。ほかのmoduleは数えない。
- optimizerの更新: そのoptimizerが、「これまでに順伝播した共有部のパラメータ」を1つでも持てば共有部、そうでなければ概念固有部。共有部のパラメータは、弱参照の集合で覚える（計測が、モデルを生かし続けない）。
- `measure_model_computation`は、入るときに2つのhookを登録し、出るとき（例外を含む）に外す。meterは、区間を出た後も、最後の計数を読める。
- 差の計算は、各項目の引き算。どれかが負になる組は拒否する。

### evaluation: LossMonitoringComputationCountStore

```python
@dataclass(frozen=True, kw_only=True)
class LossMonitoringComputationCounts:
    detector_component_update_count: int
    evaluated_candidate_bet_count: int

class LossMonitoringComputationCountStore:
    def record_loss_monitoring_computation(
        self, *, detector_component_update_count: int, evaluated_candidate_bet_count: int
    ) -> None: ...
    def get_loss_monitoring_computation_counts(self) -> LossMonitoringComputationCounts: ...
```

- `FedsdaRunClient.process_observed_sample`が、標本1件の処理が返った後で、結果の`loss_monitoring_observation`の2つの値を足す（判定記録の保持と同じ形）。ownerは、clientのownerの束の`loss_monitoring_computation_count_store`。

### learning/prediction: 有界損失の評価の入口

```python
def validate_classifier_bounded_loss_inputs(*, classifier, input_features, observed_class_labels) -> None: ...
def evaluate_classifier_per_sample_bounded_losses_and_outputs(
    *, classifier, input_features, observed_class_labels
) -> tuple[Tensor, Tensor]: ...   # (標本ごとの損失, 分類器の出力)
def evaluate_classifiers_per_sample_bounded_losses_from_shared_features(
    *, classifiers: tuple[ResidualAdapterClassifier, ...], input_features, observed_class_labels
) -> tuple[Tensor, ...]: ...      # 分類器の順の、標本ごとの損失
```

- 既存の`evaluate_classifier_per_sample_bounded_losses`は、2つめの関数の損失だけを返す形にする（検査、順伝播、式は同じ）。
- 3つめ: 全部の分類器の入力を検査し、全部が、同じ共有部（同じオブジェクト）につながっていることを確かめてから、最初の分類器で共有部の特徴を1回計算し、各分類器の概念固有部を計算する。出力の検査と損失の式は、既存と同じ。分類器が1つもなければ拒否する。

### runtime: 3つの利用箇所

- クロス評価: 損失と出力を、2つめの関数で得る。渡されたモデルの予測は、その出力から作る。対象のモデルの順伝播は、これまでどおり1回。
- 警報時の区間の準備: 変更区間より前の観測の検査を、1つめの関数（順伝播なし）と、「1標本であること」の確認にする。損失は、取込みが1回だけ計算する。
- 集約後の再較正: 3つめの関数で、モデルIDの昇順の分類器の損失を得る。

### runtime: 計測つきの全体run

```python
@dataclass(frozen=True, kw_only=True)
class FedsdaMeasuredRun:
    run_result: StreamProtocolRunResult
    participants: RunParticipants
    model_computation_counts: ModelComputationCounts              # 準備の後から、終わりまで
    preparation_model_computation_counts: ModelComputationCounts  # 準備の間（事前学習ほか）

def execute_fedsda_stream_protocol_run_with_computation_measurement(
    *, execution_settings: StreamProtocolExecutionSettings,
    run_participant_settings: FedsdaRunParticipantSettings,
) -> FedsdaMeasuredRun: ...
```

- 中で、`FedsdaRunParticipantFactory`を作り、それを包む非公開のfactory（準備が終わった時点の計数を控える）を、実行の枠へ渡す。全体を、計測の区間で囲む。
- 実行の枠の例外は、そのまま伝わる（計測の登録は外れる）。

### runtime: 指標の導出

- `derive_fedsda_run_metrics`へ、任意の引数`model_computation_counts: ModelComputationCounts | None = None`を足す。`FedsdaRunMetrics`へ、`model_computation_counts`（なしは`None`）と、`loss_monitoring_computation_counts`（全clientの合計）を足す。

## 旧の7項目との対応（testが照合する）

| 旧の指標 | 新 |
|---|---|
| `compute_inference_examples_total` | 概念固有部の推論の標本数 |
| `compute_training_examples_total` | 概念固有部の学習の標本数 |
| `compute_optimizer_steps_total` | 概念固有部のoptimizerの更新回数 |
| `compute_backbone_examples_total` | 共有部の、学習＋推論の標本数 |
| `compute_head_examples_total` | 概念固有部の、学習＋推論の標本数 |
| `compute_drift_detector_updates_total` | 検出器の更新回数（全clientの合計） |
| `compute_drift_detector_hypotheses_total` | 評価した候補×賭け率の数（全clientの合計） |

## 旧と違う点

- 数え方（部品が足す→外側から数える）。値は、同じになることを、照合で確かめる。
- 用途別の内訳、clientごと・ラウンドごとの時系列は、作らない。
- 準備の間の計算（事前学習）を、別の項目として返す（旧は、数えない）。指標には含めない。

## Error Handling

- 計測の区間の中の例外は、登録を外してから、そのまま伝える。
- 新しい入口の不正な入力は、既存と同じ`TypeError`／`ValueError`。3つめの関数は、分類器の共有部が同じオブジェクトでなければ`ValueError`。

## Testing Strategy

### 旧の計数の検査（1.x）

- goldenの3ケースで、旧の全体run（旧の回帰testと同じ設定の文脈。準備の後から終端まで）を、hookつきで実行し、旧の計数の合計と照合する。旧の指標の集計（`_add_telemetry_results`）の7項目のうち、モデルの5項目とも照合する。

### 計測（2.x）

- 小さい分類器で、共有部を使い回す順伝播、勾配つき・なし、複数の概念固有部、optimizerの更新（共有部・概念固有部）を行い、計数を、手計算の値と照合する。対象外のmoduleを数えないこと。区間の後と、例外の後に、登録が残らないこと（区間の外の計算が、数えられないこと）。入れ子。差の計算と拒否。計測の有無で、出力と乱数が変わらないこと。

### 検出器の計数（3.x）

- owner（加算、読取り、拒否）。clientの全状態の新旧照合へ、実旧の`drift_detector_updates`・`drift_detector_hypotheses`との照合を足す（全条件で照合される）。

### 重複の解消（4.x）

- 新しい入口: 既存の関数との一致（損失、出力）、共有部を使い回した損失が、分類器ごとの計算と、完全に一致すること、検査と拒否。
- 3つの利用箇所: 既存の、実旧との対照と拒否のtestを、そのまま通す。加えて、それぞれの対照の中で、新の順伝播の標本数（計測）が、実旧の同じ処理の計数の増分と一致することを確かめる。

### 計測つきの全体runと、指標（5.x）

- goldenの条件: 指標の導出のtestのfixtureを、計測つきの全体runへ替え、7項目を、実旧の実行結果・Windows用のgoldenと照合する（照合しない項目の一覧を、空にする）。
- 小さい条件: 実旧の全体runの計数の合計と照合する。
- 計測なしの導出（5.4）。準備の間の計数が、事前学習の計算であること。

### fresh process・依存

- 共用script（6.1）、依存境界（6.2）。
