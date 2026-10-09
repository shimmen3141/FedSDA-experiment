# Design Document: post-aggregation-prediction-recalibration

## Overview

**Purpose**: 配布の後の、予測の重みと診断証拠の再較正を、旧と同じ順序・数値で行う。

**Users**: リファクタリングを進める研究者。これで、クラスタリングなしのラウンドを、旧のサーバの`run_round`そのものと照合できる。

**Impact**: 単一の診断証拠（`AdaHedgeDiagnosticEvidence`）へ操作と計数を足し、clientへ操作を足す。既存の操作の契約は変えない。

### Goals

- 実旧のサーバの`run_round`（クラスタリングなし、`fifo_replay`）と、ラウンドごとに、全状態と乱数が一致する。

### Non-Goals

- 再較正の方式の選択、統合、実行の枠のサーバの操作の実体、計算量の記録。

## Boundary Commitments

### This Spec Owns

- 関数`recalibrate_prediction_state_after_aggregation`と、結果の記録（runtime）。
- `FedsdaRunClient.recalibrate_prediction_state_after_aggregation`。
- `AdaHedgeDiagnosticEvidence`へ足す操作（集約後の再生、集約後の再始動）と、3つの計数。

### Out of Boundary

- `FixedSharePredictionWeightController`、`evaluate_classifier_per_sample_bounded_losses`、診断証拠の既存の操作の中の処理。変更しない。
- 全clientへ順に呼ぶこと（呼出し側）。

### Allowed Dependencies

- runtimeの新moduleは、`dataclasses`・`torch`、evaluation（診断証拠）、learning（保有モデル、損失の計算）、methods/fedsda（保留標本、Fixed-Share）に依存する。旧実装と、`fedsda_run_client`をimportしない（clientの操作が、新moduleを呼ぶ）。

### Revalidation Triggers

- 診断証拠の計数の追加（診断の保存を決めるspecが読む）。`FedsdaRunClient`の操作の追加。

## File Structure Plan

```
src/federated_learning_experiments/
└── runtime/
    └── post_aggregation_prediction_recalibration.py   # 新規
tests/refactoring/
└── test_post_aggregation_prediction_recalibration.py  # 新規: 実旧のサーバのrun_roundとの対照、拒否、変えない状態
```

### Modified Files

- `evaluation/adahedge_diagnostic_evidence.py` — 集約後の再生・再始動と、3つの計数を足す。
- `runtime/fedsda_run_client.py` — 再較正の操作を足す。
- `tests/refactoring/test_adahedge_diagnostic_evidence.py` — 足した操作の、実旧のAdaHedgeとの対照と拒否。照合のhelperへ3つの計数を足す。
- `tests/refactoring/test_fedsda_run_client.py` — 旧の設定の差し替えへ、再較正の方式を足す（既定は`none`のまま）。
- `tests/refactoring/test_single_run_dependency_boundaries.py`、`tests/refactoring/fresh_process_smoke.py`。

## 旧処理との対応

research.mdの対応表のとおり。旧と違う点:

- **共有部の特徴の計算**: 旧は、モデルIDが最小のモデルで1回だけ計算する。新は、既存の損失の部品が、モデルごとに分類器全体を呼ぶ。値は同じ。
- **入力の検査**: 新は、損失の部品の検査（パラメータ・入力・出力が有限のCPU float32、ラベルがクラスの範囲）を通す。旧は検査しない。
- **再較正の方式**: 新は、この方式だけ。旧の`none`・集約後の再始動だけ・首位の変化による再生は、移植しない。
- **最終構成で空の状態**（予測クラス別・shadowのAdaHedge）と**計算量**を持たない。最終構成で増えない計数（`check_count`・`skip_count`）を持たない。

## Requirements Traceability

| Requirement | Components |
|-------------|------------|
| 1.1〜1.3 | `recalibrate_prediction_state_after_aggregation`（損失の列） |
| 2.1〜2.4 | `AdaHedgeDiagnosticEvidence`の足す操作、再較正の関数 |
| 3.1, 3.2 | 再較正の関数 |
| 4.1〜4.3 | 対照test |
| 5.1〜5.4 | 再較正の関数、診断証拠の足す操作 |
| 6.1 | 共用script |
| 6.2 | 依存境界test |

## Components and Interfaces

### AdaHedgeDiagnosticEvidence へ足すもの

- property `aggregation_restart_count`、`aggregation_recalibration_count`、`aggregation_recalibration_sample_count`（初期値0）。
- `restart_evidence_after_aggregation() -> None`: 証拠（累積損失とmixability gap）を消し、`aggregation_restart_count`と`aggregation_recalibration_count`を1ずつ足す。証拠が空でも数える。
- `replay_observed_losses_after_aggregation(*, observed_loss_sequence: Iterable[Mapping[int, float]]) -> None`: 列を固定して、全行を既存の損失の検査（対応であること、IDがbuiltin int、空でない、値が有限の数）で確かめる。列が空なら何もしない。空でなければ、`aggregation_recalibration_count`を1、`aggregation_recalibration_sample_count`を列の長さだけ足し、証拠を消して、列の順に、`get_diagnostic_weights_before_loss_observation`（行のモデルID）と`update_evidence_after_loss_observation`を行う。
  - モデル集合の変化の回数は、既存の同期の規則のまま（証拠を消した直後の最初の行では増えない。途中の行でモデル集合が変われば増える。旧と同じ）。概念操作による再始動の回数は変えない。

### runtime: recalibrate_prediction_state_after_aggregation

```python
@dataclass(frozen=True, kw_only=True)
class PostAggregationPredictionRecalibration:
    replayed_sample_count: int                 # 再生した損失の列の長さ（空なら0）
    replayed_model_ids: tuple[int, ...]        # 列に含めたモデルID（昇順。空なら()）
    restarted_true_concept_ids: tuple[int, ...]  # 再始動した真の概念別の診断証拠（作成順）

def recalibrate_prediction_state_after_aggregation(
    *,
    held_model_training_state_registry: HeldModelTrainingStateRegistry,
    pending_sample_observation_store: PendingSampleObservationStore,
    fixed_share_prediction_weight_controller: FixedSharePredictionWeightController,
    diagnostic_evidence_collection: AdaHedgeDiagnosticEvidenceCollection,
) -> PostAggregationPredictionRecalibration: ...
```

処理順:

| 段 | 内容 | 状態の更新 |
| --- | --- | --- |
| 1 | ownerのexact型の検査 | なし |
| 2 | 損失の列: 保留中の標本と、保有モデル（IDの昇順）を読む。標本がないか、モデルが1つ以下なら空。そうでなければ、標本の特徴とラベルを観測順に連結し、モデルごとに`evaluate_classifier_per_sample_bounded_losses`を呼び、標本ごとに、モデルIDの昇順の`{モデルID: 損失（builtin float）}`を作る | なし |
| 3 | globalの診断証拠を、列で再生する | globalの診断証拠 |
| 4 | 作成済みの真の概念別の診断証拠を、作成順に全部、再始動する | 真の概念別の診断証拠 |
| 5 | Fixed-Shareの重みを、列で再生する | Fixed-Shareの重みと計数 |

- 段2は勾配なしで、乱数を使わない。訓練の別を変えない（既存の損失の部品の契約）。
- 段3〜5は、段2の値（有限のbuiltin floatと、builtin intのID）を受け取るので、段1・2を通った入力では拒否しない。

### FedsdaRunClient の変更

- `recalibrate_prediction_state_after_aggregation() -> PostAggregationPredictionRecalibration`を足す（自分のownerで、再較正の関数を呼ぶ）。

## Error Handling

- 不正なownerは`TypeError`。保有モデルや保留標本が損失の部品の契約に合わなければ、その部品の例外（`TypeError`／`ValueError`）。どちらも、どの更新より前。

## Testing Strategy

実旧をoracleにする。式をtestへ複製しない。

### Unit Tests

- 診断証拠（2.x, 4.3, 5.2）: 実旧の`AdaHedgeRouter`と、通常の観測・概念操作による再始動・集約後の再生・集約後の再始動を混ぜた列で、各操作の後に、累積損失・gap・全計数を照合する（空の列、途中でモデル集合が変わる列、モデルが1つの列を含む）。不正な列（対応でない行、bool・strのID、空の行、非有限・数でない値、列挙の途中で失敗する反復）で、証拠と計数が変わらないこと。
- 拒否（5.1, 5.3）: 4つのownerの型で、全状態が変わらないこと。
- 変えない状態（1.3, 3.2）: 再較正の前後で、予測の重み・診断証拠のほかの全owner、モデルの値・勾配・optimizer・訓練の別、乱数が同じであること。

### Integration Tests

- ラウンドの対照（1.x, 2.x, 3.1, 4.1, 4.2）: 配布の対照のoracleで、旧の設定の再較正の方式を`fifo_replay`にして、ラウンドごとに、旧は`legacy_server.run_round(round_index, clustering_enabled=False)`、新は登録→集約→配布→全clientの再較正、を行い、全状態と乱数を照合する。2値・多クラス、学習率の2つの設定が違う条件を含む。全条件を通して、要求4.2の経路を通ったことを確かめる。
- clientの再較正だけの対照: ラウンドの途中の状態で、実旧のclientの`recalibrate_routing_after_aggregation`と、新のclientの操作を呼び、結果の記録（標本数、モデルID、再始動した概念）を、実旧の状態と照合する。

### fresh process・依存

- 共用script（6.1）: サーバの代役が、配布の後に、全clientの再較正を行う。空でない列での再較正が1回以上あることを確かめる。
- 依存境界（6.2）: 新しいmoduleの許可集合と、clientの追加分を登録する。
