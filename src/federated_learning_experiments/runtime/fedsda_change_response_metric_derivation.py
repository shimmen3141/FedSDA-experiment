"""最終構成のFedSDAの参加者の記録から、概念の変更への応答（検出、警報、適応、候補の判定）の指標を導出する。

読取りだけを行い、参加者の状態と乱数を変えない。比の分母が0のとき、平均する値がないときは、0。
"""

from dataclasses import dataclass
from math import isfinite

from federated_learning_experiments.evaluation.change_response_metric_calculations import (
    UnmatchedEventCounts,
    calculate_change_point_estimation_errors,
    count_unmatched_events_by_timing,
    match_events_to_concept_changes,
)
from federated_learning_experiments.evaluation.run_metric_calculations import (
    DetectionMetrics,
    calculate_detection_metrics,
)
from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient

# 適応の結果種別から、粗い区分への対応。旧の結果（粗い区分しか持たない）を変換した表と、同じ列を持てるようにする。
ADAPTATION_ACTION_BY_ADAPTATION_OUTCOME: dict[str, str] = {
    "alarm_change_interval_too_short": "skip_too_short_change_interval",
    "alarm_interval_held_model_reused": "reuse_held_model",
    "alarm_interval_current_model_maintained": "maintain_current_model",
    "alarm_interval_candidate_adopted_as_new_model": "adopt_candidate_model",
    "server_consolidation_training_model_remapped": "remap_by_server_consolidation",
}
# 学習帰属を、別のモデルへ切り替える区分（変更位置と対応づけて、適合率を求める）。
_MODEL_CHANGING_ADAPTATION_ACTIONS: tuple[str, ...] = ("reuse_held_model", "adopt_candidate_model")


@dataclass(frozen=True, kw_only=True)
class TrainingModelSwitchTimingMetrics:
    """学習帰属の切替の、変更位置に対する時機。"""

    # 変更に対応した切替の、切替の位置−変更位置の平均。
    mean_matched_switch_delay_sample_count: float
    # 変更に対応しなかった切替の、時機ごとの数。
    unmatched_switch_counts: UnmatchedEventCounts


@dataclass(frozen=True, kw_only=True)
class ChangePointEstimationMetrics:
    """変更に対応した警報の、検出器が推定した変化点の誤差（推定した位置−変更位置）。"""

    matched_alarm_count: int
    mean_absolute_error_sample_count: float
    mean_signed_error_sample_count: float


@dataclass(frozen=True, kw_only=True)
class AdaptationActionMetrics:
    """適応の記録の、区分ごとの件数と、モデルを切り替える区分の適合率。

    適合率は、clientごとに、再利用と採用の適応を記録の順に並べて、変更位置と対応づけたときの、
    区分ごとの、対応した数÷件数。
    """

    maintain_current_model_count: int
    reuse_held_model_count: int
    adopt_candidate_model_count: int
    skip_too_short_change_interval_count: int
    remap_by_server_consolidation_count: int
    reuse_held_model_precision: float
    adopt_candidate_model_precision: float
    # 警報の区間の再利用の評価で、現行モデルが合うと判定した数と、別の保有モデルを選んだ数。
    current_model_fit_selection_count: int
    alternative_model_reuse_selection_count: int


@dataclass(frozen=True, kw_only=True)
class FedsdaChangeResponseMetrics:
    """概念の変更への応答の指標（全clientの合計・平均）。"""

    training_model_switch_timing_metrics: TrainingModelSwitchTimingMetrics
    # 警報を検出とみなしたときの、検出の指標。
    loss_change_alarm_detection_metrics: DetectionMetrics
    change_point_estimation_metrics: ChangePointEstimationMetrics
    adaptation_action_metrics: AdaptationActionMetrics
    # 損失の監視の値（log e）の、有限の値の最大。有限の値が1つもなければ、負の無限大。
    maximum_monitored_log_e_value: float


def _ratio_or_zero(*, numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def derive_fedsda_change_response_metrics(
    *,
    run_clients: tuple[FedsdaRunClient, ...],
    concept_change_sample_indices_by_client: tuple[tuple[int, ...], ...],
    maximum_delay_sample_count: int,
) -> FedsdaChangeResponseMetrics:
    """clientの記録（切替、警報、適応、監視の値）と、変更位置から、応答の指標を導出する。"""
    if type(run_clients) is not tuple:
        raise TypeError("run_clients must be builtin tuple")
    for run_client in run_clients:
        if type(run_client) is not FedsdaRunClient:
            raise TypeError("run_clients must hold exact FedsdaRunClient")
    if type(concept_change_sample_indices_by_client) is not tuple:
        raise TypeError("concept_change_sample_indices_by_client must be builtin tuple")
    if len(concept_change_sample_indices_by_client) != len(run_clients):
        raise ValueError("concept change indices must be given for every client")

    matched_switch_delay_sample_counts: list[int] = []
    duplicate_switch_count = 0
    early_switch_count = 0
    late_switch_count = 0
    isolated_switch_count = 0
    alarm_sample_indices_by_client: list[tuple[int, ...]] = []
    change_point_estimation_errors: list[int] = []
    adaptation_action_counts = dict.fromkeys(
        dict.fromkeys(ADAPTATION_ACTION_BY_ADAPTATION_OUTCOME.values()), 0
    )
    matched_model_changing_action_counts: dict[str, int] = dict.fromkeys(
        _MODEL_CHANGING_ADAPTATION_ACTIONS, 0
    )
    current_model_fit_selection_count = 0
    alternative_model_reuse_selection_count = 0
    finite_monitored_log_e_values: list[float] = []

    for run_client, concept_change_sample_indices in zip(
        run_clients, concept_change_sample_indices_by_client, strict=True
    ):
        owners = run_client.owners
        adaptation_snapshot = owners.adaptation_record_store.get_state_snapshot()
        alarm_snapshot = owners.loss_change_alarm_record_store.get_state_snapshot()

        # ---- 学習帰属の切替（位置の昇順で、変更位置と対応づける） ----
        switch_sample_indices = tuple(
            sorted(adaptation_snapshot.training_model_switch_sample_indices)
        )
        switch_matching = match_events_to_concept_changes(
            concept_change_sample_indices=concept_change_sample_indices,
            event_sample_indices=switch_sample_indices,
            maximum_delay_sample_count=maximum_delay_sample_count,
        )
        matched_switch_delay_sample_counts.extend(switch_matching.matched_delay_sample_counts)
        unmatched_switch_counts = count_unmatched_events_by_timing(
            concept_change_sample_indices=concept_change_sample_indices,
            event_sample_indices=switch_sample_indices,
            matched_event_positions=switch_matching.matched_event_positions,
            maximum_delay_sample_count=maximum_delay_sample_count,
        )
        duplicate_switch_count += unmatched_switch_counts.duplicate_event_count
        early_switch_count += unmatched_switch_counts.early_event_count
        late_switch_count += unmatched_switch_counts.late_event_count
        isolated_switch_count += unmatched_switch_counts.isolated_event_count

        # ---- 警報と、変化点の推定 ----
        alarm_sample_indices_by_client.append(alarm_snapshot.alarm_sample_indices)
        change_point_estimation_errors.extend(
            calculate_change_point_estimation_errors(
                concept_change_sample_indices=concept_change_sample_indices,
                alarm_sample_indices=alarm_snapshot.alarm_sample_indices,
                estimated_change_point_sample_indices=(
                    alarm_snapshot.estimated_change_point_sample_indices
                ),
                maximum_delay_sample_count=maximum_delay_sample_count,
            )
        )
        finite_monitored_log_e_values.extend(
            monitored_log_e_value
            for monitored_log_e_value in alarm_snapshot.monitored_log_e_values
            if isfinite(monitored_log_e_value)
        )

        # ---- 適応の区分 ----
        model_changing_adaptations: list[tuple[int, str]] = []
        for adaptation_record in adaptation_snapshot.adaptation_records:
            if adaptation_record.adaptation_outcome not in ADAPTATION_ACTION_BY_ADAPTATION_OUTCOME:
                raise ValueError(
                    "adaptation outcome has no action category: "
                    f"{adaptation_record.adaptation_outcome!r}"
                )
            adaptation_action = ADAPTATION_ACTION_BY_ADAPTATION_OUTCOME[
                adaptation_record.adaptation_outcome
            ]
            adaptation_action_counts[adaptation_action] += 1
            if adaptation_action in _MODEL_CHANGING_ADAPTATION_ACTIONS:
                model_changing_adaptations.append(
                    (adaptation_record.adaptation_sample_index, adaptation_action)
                )
        model_changing_matching = match_events_to_concept_changes(
            concept_change_sample_indices=concept_change_sample_indices,
            event_sample_indices=tuple(
                adaptation_sample_index for adaptation_sample_index, _ in model_changing_adaptations
            ),
            maximum_delay_sample_count=maximum_delay_sample_count,
        )
        for matched_adaptation_position in model_changing_matching.matched_event_positions:
            matched_model_changing_action_counts[
                model_changing_adaptations[matched_adaptation_position][1]
            ] += 1
        current_model_fit_selection_count += adaptation_snapshot.current_model_fit_count
        alternative_model_reuse_selection_count += adaptation_snapshot.alternative_model_reuse_count

    return FedsdaChangeResponseMetrics(
        training_model_switch_timing_metrics=TrainingModelSwitchTimingMetrics(
            mean_matched_switch_delay_sample_count=(
                sum(matched_switch_delay_sample_counts) / len(matched_switch_delay_sample_counts)
                if matched_switch_delay_sample_counts
                else 0.0
            ),
            unmatched_switch_counts=UnmatchedEventCounts(
                duplicate_event_count=duplicate_switch_count,
                early_event_count=early_switch_count,
                late_event_count=late_switch_count,
                isolated_event_count=isolated_switch_count,
            ),
        ),
        loss_change_alarm_detection_metrics=calculate_detection_metrics(
            concept_change_sample_indices_by_client=concept_change_sample_indices_by_client,
            detection_sample_indices_by_client=tuple(alarm_sample_indices_by_client),
            maximum_delay_sample_count=maximum_delay_sample_count,
        ),
        change_point_estimation_metrics=ChangePointEstimationMetrics(
            matched_alarm_count=len(change_point_estimation_errors),
            mean_absolute_error_sample_count=(
                sum(abs(error) for error in change_point_estimation_errors)
                / len(change_point_estimation_errors)
                if change_point_estimation_errors
                else 0.0
            ),
            mean_signed_error_sample_count=(
                sum(change_point_estimation_errors) / len(change_point_estimation_errors)
                if change_point_estimation_errors
                else 0.0
            ),
        ),
        adaptation_action_metrics=AdaptationActionMetrics(
            maintain_current_model_count=adaptation_action_counts["maintain_current_model"],
            reuse_held_model_count=adaptation_action_counts["reuse_held_model"],
            adopt_candidate_model_count=adaptation_action_counts["adopt_candidate_model"],
            skip_too_short_change_interval_count=adaptation_action_counts[
                "skip_too_short_change_interval"
            ],
            remap_by_server_consolidation_count=adaptation_action_counts[
                "remap_by_server_consolidation"
            ],
            reuse_held_model_precision=_ratio_or_zero(
                numerator=matched_model_changing_action_counts["reuse_held_model"],
                denominator=adaptation_action_counts["reuse_held_model"],
            ),
            adopt_candidate_model_precision=_ratio_or_zero(
                numerator=matched_model_changing_action_counts["adopt_candidate_model"],
                denominator=adaptation_action_counts["adopt_candidate_model"],
            ),
            current_model_fit_selection_count=current_model_fit_selection_count,
            alternative_model_reuse_selection_count=alternative_model_reuse_selection_count,
        ),
        maximum_monitored_log_e_value=max(finite_monitored_log_e_values, default=float("-inf")),
    )
