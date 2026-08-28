"""実験指標の正規ID・分類・適用範囲を一元管理する。"""

from dataclasses import dataclass


METRIC_SCHEMA_VERSION = 18

PRIMARY = "primary"
SECONDARY = "secondary"
DIAGNOSTIC = "diagnostic"

ALL_METHODS = "all_methods"
ADAPTIVE_METHODS = "adaptive_methods"
CHANGE_POINT_METHODS = "change_point_estimators"
FEDSDA_METHODS = "fedsda_methods"
FORWARD_POLICIES = "forward_creation_policies"


@dataclass(frozen=True)
class MetricSpec:
    """一つのスカラー指標について、利用目的と有効な手法範囲を表す。"""

    id: str
    group: str
    tier: str
    applicability: str
    higher_is_better: bool | None
    description: str
    storage: str = "csv"


def _humanize(metric_id):
    return metric_id.replace("_", " ")


def _make(ids, group, tier, applicability, higher_is_better=None, descriptions=None):
    descriptions = descriptions or {}
    return tuple(
        MetricSpec(
            id=metric_id,
            group=group,
            tier=tier,
            applicability=applicability,
            higher_is_better=higher_is_better,
            description=descriptions.get(metric_id, _humanize(metric_id)),
        )
        for metric_id in ids
    )


METRICS = (
    *_make(
        ("stable_accuracy", "accuracy"), "predictive_performance", PRIMARY,
        ALL_METHODS, True,
        {
            "accuracy": "全期間のprequential精度",
            "stable_accuracy": "真のドリフト直後の回復窓を除いたprequential精度",
        },
    ),
    *_make(
        ("comm_models_up", "comm_models_down", "comm_models_total"),
        "communication", PRIMARY, ALL_METHODS, False,
    ),
    *_make(
        ("comm_messages_up", "comm_messages_down", "comm_messages_total"),
        "communication", SECONDARY, ALL_METHODS, False,
    ),
    *_make(
        (
            "comm_parameter_values_up", "comm_parameter_values_down",
            "comm_parameter_values_total", "comm_bytes_up", "comm_bytes_down",
            "comm_bytes_total",
        ),
        "communication_volume", PRIMARY, ALL_METHODS, False,
    ),
    *_make(
        (
            "versioned_cache_parameter_values_saved",
            "versioned_cache_bytes_saved",
            "versioned_cache_down_reduction_rate",
            "versioned_cache_backbone_hit_rate",
            "versioned_cache_personalized_hit_rate",
        ),
        "communication_cache", DIAGNOSTIC, FEDSDA_METHODS, None,
        {
            "versioned_cache_parameter_values_saved": (
                "完全配布と比べてversioned cacheが省略した下りパラメータ値数"
            ),
            "versioned_cache_bytes_saved": (
                "完全配布と比べてversioned cacheが省略した下りバイト数"
            ),
            "versioned_cache_down_reduction_rate": (
                "完全配布を基準とした下りパラメータ値数の削減率"
            ),
            "versioned_cache_backbone_hit_rate": (
                "共有部の配布候補のうち前回版を再利用できた割合"
            ),
            "versioned_cache_personalized_hit_rate": (
                "概念別部の配布候補のうち前回版を再利用できた割合"
            ),
        },
    ),
    *_make(
        ("final_model_count",), "model_population", PRIMARY, ALL_METHODS, False,
        {"final_model_count": "実験終了時のサーバモデル数またはクライアント平均保持モデル数"},
    ),
    *_make(
        ("precision", "recall", "f1"), "detection", PRIMARY,
        ADAPTIVE_METHODS, True,
    ),
    *_make(
        ("avg_delay", "total_detect"), "detection", SECONDARY,
        ADAPTIVE_METHODS, False,
    ),
    *_make(
        ("change_point_mae", "change_point_bias", "change_point_estimate_count"),
        "change_point", DIAGNOSTIC, CHANGE_POINT_METHODS, None,
    ),
    *_make(
        ("alarm_precision", "alarm_recall", "alarm_f1", "alarm_total"),
        "alarm_episode", DIAGNOSTIC, FEDSDA_METHODS, None,
    ),
    *_make(
        ("switch_fp_early", "switch_fp_late", "switch_fp_duplicate", "switch_fp_isolated"),
        "false_positive", DIAGNOSTIC, FEDSDA_METHODS, False,
    ),
    *_make(
        (
            "adaptation_reuse_count", "adaptation_reuse_precision",
            "adaptation_create_count", "adaptation_create_precision",
            "adaptation_create_rejected_count", "model_reuse_current_fit_count",
            "model_reuse_alternative_fit_count", "adaptation_maintain_count",
            "adaptation_episode_suppressed_count",
        ),
        "adaptation_action", DIAGNOSTIC, FEDSDA_METHODS, None,
    ),
    *_make(
        (
            "provisional_proposal_count", "provisional_acceptance_rate",
            "provisional_matched_true_count", "provisional_accepted_matched_true_count",
            "provisional_rejected_matched_true_count", "provisional_accepted_precision",
            "provisional_interval_count_mean", "provisional_training_count_mean",
            "provisional_validation_count_mean", "provisional_forward_count",
            "provisional_resolution_delay_mean", "provisional_accepted_full_margin_mean",
            "provisional_accepted_recent_margin_mean", "provisional_rejected_full_margin_mean",
            "provisional_rejected_recent_margin_mean", "provisional_matched_full_margin_mean",
            "provisional_matched_recent_margin_mean", "provisional_unmatched_full_margin_mean",
            "provisional_unmatched_recent_margin_mean", "provisional_reject_insufficient_data_count",
            "provisional_reject_insufficient_forward_data_count",
            "provisional_reject_full_interval_count", "provisional_reject_recent_interval_count",
            "provisional_reject_full_and_recent_count", "provisional_reject_first_interval_count",
            "provisional_reject_second_interval_count", "provisional_reject_first_and_second_count",
            "provisional_reject_reference_refit_count", "provisional_reject_current_refit_count",
            "provisional_reject_alternative_refit_count", "provisional_reference_excess_mean",
        ),
        "provisional_model", DIAGNOSTIC, FORWARD_POLICIES, None,
    ),
    *_make(
        ("server_mapping_change_count",), "server_mapping", DIAGNOSTIC,
        FEDSDA_METHODS, None,
    ),
    *_make(
        ("runtime_seconds", "client_compute_seconds_sum", "client_compute_seconds_max"),
        "runtime", SECONDARY, ALL_METHODS, False,
    ),
    *_make(
        (
            "compute_inference_examples_total", "compute_training_examples_total",
            "compute_model_examples_total", "compute_optimizer_steps_total",
            "compute_drift_detector_updates_total", "compute_drift_detector_hypotheses_total",
            "compute_backbone_examples_total", "compute_head_examples_total",
            "compute_backbone_optimizer_steps_total",
            "compute_head_optimizer_steps_total",
        ),
        "compute", SECONDARY, ALL_METHODS, False,
    ),
    *_make(
        (
            "mean_model_count", "max_model_count", "model_count_auc",
            "final_parameter_values", "final_parameter_bytes",
        ),
        "model_population", SECONDARY, ALL_METHODS, False,
    ),
    *_make(
        (
            "model_assigned_samples_total", "model_assigned_samples_mean",
            "model_assigned_samples_min", "model_assigned_samples_cv",
            "model_training_examples_total", "model_training_examples_mean",
            "model_training_examples_min", "model_training_examples_cv",
            "model_optimizer_steps_total", "model_optimizer_steps_mean",
            "model_optimizer_steps_min", "model_optimizer_steps_cv",
        ),
        "model_learning", DIAGNOSTIC, ALL_METHODS, None,
    ),
    *_make(
        (
            "model_pair_evaluation_count", "model_pair_sample_count",
            "model_pair_correctness_disagreement_rate",
            "model_pair_oracle_gain_rate", "model_pair_both_correct_rate",
        ),
        "model_complementarity", DIAGNOSTIC, FEDSDA_METHODS, None,
    ),
    *_make(
        (
            "clustering_noninferiority_candidate_count",
            "clustering_noninferiority_accepted_count",
            "clustering_noninferiority_rejected_count",
            "clustering_noninferiority_comparison_count",
            "clustering_noninferiority_sample_count",
            "clustering_noninferiority_acceptance_rate",
        ),
        "clustering_noninferiority", DIAGNOSTIC, FEDSDA_METHODS, None,
    ),
    *_make(
        (
            "clustering_oracle_pair_count",
            "clustering_oracle_same_pair_count",
            "clustering_oracle_merge_tp",
            "clustering_oracle_merge_fp",
            "clustering_oracle_merge_fn",
            "clustering_oracle_merge_tn",
            "clustering_oracle_merge_precision",
            "clustering_oracle_merge_recall",
            "clustering_oracle_merge_f1",
            "clustering_oracle_loss_distance_auc",
            "clustering_oracle_parameter_distance_auc",
        ),
        "clustering_oracle_diagnostic", DIAGNOSTIC, FEDSDA_METHODS, None,
    ),
    *_make(
        (
            "backbone_gradient_pair_count",
            "backbone_gradient_conflict_count",
            "backbone_gradient_conflict_rate",
            "backbone_gradient_cosine_mean",
            "backbone_gradient_negative_cosine_mean",
            "backbone_gradient_applied_pair_count",
            "backbone_gradient_applied_conflict_count",
            "backbone_gradient_applied_conflict_rate",
            "backbone_gradient_applied_cosine_mean",
            "backbone_gradient_applied_negative_cosine_mean",
            "backbone_gradient_update_comparison_count",
            "backbone_gradient_update_cosine_mean",
            "backbone_gradient_update_norm_ratio_mean",
            "backbone_gradient_update_delta_ratio_mean",
        ),
        "shared_gradient_conflict", DIAGNOSTIC, FEDSDA_METHODS, None,
    ),
    *_make(
        (
            "routing_sample_count", "routing_oracle_accuracy",
            "routing_mixture_accuracy", "routing_leader_accuracy",
            "routing_confidence_leader_accuracy",
            "routing_oracle_gain_rate", "routing_oracle_recovery_rate",
            "routing_missed_oracle_count", "routing_aggregation_restart_count",
            "routing_confidence_leader_oracle_recovery_rate",
            "routing_confidence_leader_missed_oracle_count",
            "routing_class_macro_oracle_accuracy",
            "routing_class_macro_mixture_accuracy",
            "routing_class_macro_leader_accuracy",
            "routing_class_macro_confidence_leader_accuracy",
            "routing_class_oracle_gap_mean",
            "routing_class_oracle_gap_std",
            "routing_class_oracle_recovery_rate_mean",
            "routing_class_oracle_recovery_rate_min",
            "routing_meta_accuracy",
            "routing_meta_gain_rate",
            "routing_meta_global_accuracy",
            "routing_meta_context_mixture_accuracy",
            "routing_meta_context_leader_accuracy",
            "routing_meta_best_candidate_gain_rate",
            "routing_meta_context_leader_weight_mean",
            "routing_meta_context_leader_preferred_rate",
            "routing_class_macro_meta_accuracy",
            "routing_class_macro_meta_global_accuracy",
            "routing_class_macro_meta_context_mixture_accuracy",
            "routing_class_macro_meta_context_leader_accuracy",
            "routing_switching_accuracy",
            "routing_switching_gain_rate",
            "routing_switching_global_gain_rate",
            "routing_switching_stable_accuracy",
            "routing_switching_stable_gain_rate",
            "routing_switching_recovery_accuracy",
            "routing_switching_recovery_gain_rate",
            "routing_switching_effective_experts_mean",
            "routing_switching_leader_switch_count",
            "routing_switching_pool_reset_count",
            "routing_switching_recalibration_sample_count",
            "routing_meta_switching_accuracy",
            "routing_meta_switching_meta_gain_rate",
            "routing_meta_switching_switching_gain_rate",
            "routing_meta_switching_selected_switching_rate",
            "routing_meta_switching_leader_switch_count",
            "routing_aggregation_recalibration_count",
            "routing_aggregation_recalibration_sample_count",
            "routing_aggregation_recalibration_check_count",
            "routing_aggregation_recalibration_skip_count",
        ),
        "soft_routing", DIAGNOSTIC, FEDSDA_METHODS, None,
    ),
    *_make(
        (
            "routing_loo_evaluation_count",
            "routing_loo_bounded_delta_mean",
            "routing_loo_zero_one_delta_mean",
            "routing_loo_positive_rate",
            "routing_loo_fallback_count",
            "routing_loo_active_model_count",
            "routing_loo_active_evaluable_model_count",
            "routing_loo_active_joint_nonpositive_model_count",
            "routing_loo_active_joint_nonpositive_rate",
            "routing_loo_active_unassigned_model_count",
            "routing_loo_active_unassigned_evaluable_model_count",
            "routing_loo_active_unassigned_nonpositive_model_count",
            "routing_loo_active_unassigned_nonpositive_rate",
            "routing_loo_active_unassigned_joint_nonpositive_model_count",
            "routing_loo_active_unassigned_joint_nonpositive_rate",
            "routing_archive_shadow_sample_count",
            "routing_archive_shadow_accuracy",
            "routing_archive_shadow_accuracy_delta",
            "routing_archive_shadow_bounded_delta_mean",
            "routing_archive_shadow_retained_global_model_rate",
            "routing_archive_shadow_reconfiguration_count",
            "routing_active_set_sample_count",
            "routing_active_set_probe_sample_count",
            "routing_active_set_retained_global_model_rate",
            "routing_active_set_apply_retained_global_model_rate",
            "routing_active_set_reconfiguration_count",
            "routing_active_set_failure_probe_count",
        ),
        "routing_contribution", DIAGNOSTIC, FEDSDA_METHODS, None,
        {
            "routing_loo_bounded_delta_mean": (
                "モデル除外で増える有界損失の平均（正なら寄与あり）"
            ),
            "routing_loo_zero_one_delta_mean": (
                "モデル除外で増える0/1損失の平均（正なら寄与あり）"
            ),
            "routing_loo_active_unassigned_nonpositive_rate": (
                "最終activeかつ未割当で、有界損失の除外寄与が非正の評価可能モデル割合"
            ),
            "routing_loo_active_joint_nonpositive_rate": (
                "有界損失と0/1損失の除外寄与がともに非正の最終activeモデル割合"
            ),
            "routing_loo_active_unassigned_joint_nonpositive_rate": (
                "二損失の除外寄与がともに非正の最終active・未割当モデル割合"
            ),
            "routing_archive_shadow_accuracy_delta": (
                "前通信区間の二損失寄与でローカル保持集合を絞ったshadow精度差"
            ),
            "routing_archive_shadow_retained_global_model_rate": (
                "前通信区間ベースのshadowがローカルに保持するグローバルモデル割合"
            ),
            "routing_active_set_retained_global_model_rate": (
                "実予測でforwardしたグローバルモデル割合（probe区間を含む）"
            ),
            "routing_active_set_apply_retained_global_model_rate": (
                "probe後の適用区間でforwardしたグローバルモデル割合"
            ),
            "routing_active_set_failure_probe_count": (
                "適用区間で全active expertが誤り、全expert probeへ戻した回数"
            ),
        },
    ),
)

METRICS_BY_ID = {metric.id: metric for metric in METRICS}
SCALAR_METRIC_IDS = tuple(metric.id for metric in METRICS)

METRIC_PROFILES = {
    "core": (
        "accuracy", "stable_accuracy", "precision", "recall", "f1", "avg_delay",
        "comm_models_total", "comm_messages_total", "final_model_count",
        "compute_model_examples_total", "runtime_seconds",
    ),
    "detection": tuple(metric.id for metric in METRICS if metric.group in {
        "detection", "change_point", "alarm_episode", "false_positive",
    }),
    "adaptation": tuple(metric.id for metric in METRICS if metric.group in {
        "adaptation_action", "provisional_model", "server_mapping",
        "soft_routing",
    }),
    "resource": tuple(metric.id for metric in METRICS if metric.group in {
        "communication", "communication_volume", "runtime", "compute",
        "communication_cache", "model_population",
    }),
    "model_diagnostics": tuple(metric.id for metric in METRICS if metric.group in {
        "model_learning", "model_complementarity",
        "soft_routing", "routing_contribution", "shared_gradient_conflict",
        "clustering_noninferiority",
        "clustering_oracle_diagnostic",
    }),
    "all": SCALAR_METRIC_IDS,
}


def metric(metric_id):
    """正規IDから指標定義を返す。"""
    try:
        return METRICS_BY_ID[metric_id]
    except KeyError as exc:
        raise KeyError(f"Unknown metric id: {metric_id}") from exc


def metrics_in_group(group):
    return tuple(item for item in METRICS if item.group == group)


def metrics_in_profile(profile):
    try:
        return tuple(metric(metric_id) for metric_id in METRIC_PROFILES[profile])
    except KeyError as exc:
        raise KeyError(f"Unknown metric profile: {profile}") from exc
