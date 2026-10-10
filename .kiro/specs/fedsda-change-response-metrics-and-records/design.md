# Design Document: fedsda-change-response-metrics-and-records

## Overview

**Purpose**: 概念の変更への応答（検出、警報、適応、候補の判定）の指標と記録を、新実装で導出・保存し、旧実装と照合する。保存する表・列の名前を、時点と意味が分かる名前へ直す。

**Impact**: 計算のmoduleを1つ足す。指標の型へ、まとまりを1つ足す。記録の表の、名前を変え、列を足す。全体runの実行は、変えない。

## Boundary Commitments

### This Spec Owns

- `evaluation/change_response_metric_calculations.py`（新規）。
- `runtime/fedsda_change_response_metric_derivation.py`（新規。参加者の記録から、指標のまとまりを作る）。
- `runtime/fedsda_run_metric_derivation.py`（`FedsdaRunMetrics`へ、fieldを1つ足す）。
- `runtime/fedsda_run_record_tables.py`（名前の変更と、列の追加）。

### Out of Boundary

- 各owner（読取りだけ）。既存の指標の計算。

## 名前の見直し（Requirement 1）

| 前 | 後 | 理由 |
|---|---|---|
| 表`round`、`round_local_processing_computation`、`round_synchronization_computation` | 表`server_round`（1つ） | どのラウンドかが分からない。行は同じ（サーバの同期のラウンド）なので、1つの表にする |
| `round.global_model_count` | `server_round.global_model_count_after_synchronization` | ラウンドの中の、どの時点の値かが分からない |
| `round.client_held_model_count` | `server_round.client_held_model_count_after_synchronization` | 同上 |
| `round_local_processing_computation.<項目>` | `server_round.local_processing_<項目>` | 表をまとめる |
| `round_synchronization_computation.<項目>` | `server_round.synchronization_<項目>` | 同上 |
| `sample_prediction.held_model_count` | `sample_prediction.held_model_count_at_prediction` | その標本の予測に使った保有モデルの数（処理の前の値）であることを表す |
| `concept_change.sample_index` | `concept_change.first_new_concept_sample_index` | 変更の前後の、どちらの標本かが分からない |

| `training_model_switch.sample_index` | `training_model_switch.switch_sample_index` | ほかの表の位置の列（`alarm_sample_index`、`adaptation_sample_index`、`proposal_sample_index`）とそろえる（独立レビューの指摘） |
| `model_clustering.round_index`、`model_pair_clustering.round_index` | `server_round_index` | 表`round`を直したのと同じ理由（どのラウンドかが分からない）。表`server_round`の行を指す（独立レビューの指摘） |
| `model_registration.registered_round_index` | `model_registration.registered_server_round_index` | 同上 |

ほかの表と列は、記録の型とfieldの名前のままで、十分と判断した（`adaptation.adaptation_sample_index`、`candidate_validation_decision.proposal_sample_index`ほか）。記録の型のfieldの名前（`round_index`、`registered_round_index`）は、変えていない（保存する列の名前だけを直した）。

列の読み方の注意:

- `candidate_validation_decision.resolution_sample_index`は、判定が確定した標本の位置。終端で回収した未完了の判定（`is_incomplete_at_run_end`が真）では、回収した位置（提案の位置と、最後に処理した標本の位置の、大きいほう。旧と同じ）。
- `candidate_validation_decision.decision_reason`が「2つの区間とも通過」でも、`candidate_accepted`が偽のことがある（採否は、平均の損失を倍精度で比べ、理由は、単精度の差で決めるので、境界の丸めで食い違いうる。旧と同じ構造で、件数は旧と一致する）。

## Components

### evaluation: change_response_metric_calculations

```python
@dataclass(frozen=True, kw_only=True)
class ConceptChangeEventMatching:
    matched_event_positions: tuple[int, ...]          # イベントの列の中の位置（対応づいた順）
    matched_concept_change_positions: tuple[int, ...] # 変更位置の列の中の位置（同じ順）
    matched_delay_sample_counts: tuple[int, ...]      # イベントの位置−変更位置

@dataclass(frozen=True, kw_only=True)
class UnmatchedEventCounts:
    duplicate_event_count: int   # 変更位置以上・変更位置＋許容遅延以下
    early_event_count: int       # 変更位置−許容遅延以上・変更位置未満
    late_event_count: int        # 変更位置＋許容遅延より後・変更位置＋2×許容遅延以下
    isolated_event_count: int    # どれでもない

def match_events_to_concept_changes(*, concept_change_sample_indices, event_sample_indices, maximum_delay_sample_count) -> ConceptChangeEventMatching
def count_unmatched_events_by_timing(*, concept_change_sample_indices, event_sample_indices, matched_event_positions, maximum_delay_sample_count) -> UnmatchedEventCounts
def calculate_change_point_estimation_errors(*, concept_change_sample_indices, alarm_sample_indices, estimated_change_point_sample_indices, maximum_delay_sample_count) -> tuple[int, ...]
```

- 対応づけの規則は、既存の`count_events_matched_to_concept_changes`と同じ（与えられた順）。
- 変化点の推定の誤差: 変更位置を昇順に見て、警報（与えられた順）を対応づけ、対応した警報の、推定−変更位置。

### runtime: fedsda_change_response_metric_derivation

```python
@dataclass(frozen=True, kw_only=True)
class TrainingModelSwitchTimingMetrics:
    mean_matched_switch_delay_sample_count: float
    unmatched_switch_counts: UnmatchedEventCounts

@dataclass(frozen=True, kw_only=True)
class ChangePointEstimationMetrics:
    matched_alarm_count: int
    mean_absolute_error_sample_count: float
    mean_signed_error_sample_count: float

@dataclass(frozen=True, kw_only=True)
class AdaptationActionMetrics:
    <8つの区分>_count: int ...
    held_model_reuse_precision: float
    candidate_model_adoption_precision: float
    current_model_fit_selection_count: int
    alternative_model_reuse_selection_count: int

@dataclass(frozen=True, kw_only=True)
class CandidateValidationDecisionMetrics: ...   # 採用率、対応した数、平均、理由ごとの数

@dataclass(frozen=True, kw_only=True)
class FedsdaChangeResponseMetrics:
    training_model_switch_timing_metrics: TrainingModelSwitchTimingMetrics
    loss_change_alarm_detection_metrics: DetectionMetrics
    change_point_estimation_metrics: ChangePointEstimationMetrics
    adaptation_action_metrics: AdaptationActionMetrics
    candidate_validation_decision_metrics: CandidateValidationDecisionMetrics
    maximum_monitored_log_e_value: float

def derive_fedsda_change_response_metrics(*, run_clients, concept_change_sample_indices_by_client, maximum_delay_sample_count) -> FedsdaChangeResponseMetrics
```

- 適応の「再利用」「採用」の適合率: clientごとに、区分が再利用か採用の適応を、記録の順に並べ、変更位置と対応づける。区分ごとに、対応した数÷件数。
- 判定の「変更に対応したか」: clientごとに、判定の提案の位置（記録の順）を、変更位置と対応づける。
- 平均は、有限の値だけの平均。値がなければ0（旧と同じ）。終端で回収した未完了の判定は、損失を持たない（平均に入らない）。
- `FedsdaRunMetrics`へ、`change_response_metrics: FedsdaChangeResponseMetrics`を足す。

### runtime: fedsda_run_record_tables（列の追加）

| 表 | 足す列 |
|---|---|
| `loss_change_alarm`（新） | `client_id`、`alarm_sample_index`、`estimated_change_point_sample_index`、`detector_candidate_start_sample_index` |
| `sample_loss_monitoring`（新。client×処理した標本） | `monitored_log_e_value`（float64） |
| `adaptation` | `detector_name`、`estimated_change_point_sample_index`（ないときは−1） |
| `candidate_validation_decision` | `detector_name`、`candidate_training_interval_sample_count`、`validation_sample_count`、`comparison_reference_model_id`（ないときは−1）、`candidate_full_interval_mean_loss`・`reference_full_interval_mean_loss`・`candidate_second_segment_mean_loss`・`reference_second_segment_mean_loss`・`reference_historical_mean_loss`（ないときはNaN）、`matched_to_concept_change`(bool) |

## 旧と違う点

- 変化点の推定の誤差の計算は、警報の位置の列と、推定の位置の列の長さが違うと、`ValueError`を出す（旧の`_change_point_errors`は、そのclientを黙って飛ばす）。2つの記録は、同じ場所で同時に足すので、実行では、長さは食い違わない。

## 検証の方針

- 計算: 乱数で作った変更位置とイベントの列で、実旧の`match_events`・`_classify_unmatched_events`・`_change_point_errors`と照合する。
- 指標: goldenの3ケースの実旧の実行結果（263の指標の辞書）と、小さい条件の実旧の`compute_metrics`の戻り値と、全部の項目で照合する。新→旧の対応を、testの`derive_legacy_metric_values`へ足す。
- 表: goldenの3ケースの、実旧が保存した配列と照合する。
- 対応表の取り違え: 照合のたびに、旧の値を積み、実行した全部の条件（goldenの3ケースと、小さい条件）で旧の値が同じになる指標の対と、0のままの指標を、最後のtestで数える（取り違えを検出できない対が、期待の一覧と同じであること、0のままの指標がないこと。独立レビューの指摘で足した）。
