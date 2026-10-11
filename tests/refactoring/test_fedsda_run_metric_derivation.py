"""全体runの指標の導出: 実旧の実行と保存結果、Windows用のgolden、小さい条件の実旧の指標との照合。

31の離散列は、sourceではなく、このtestが、新の記録の読取りから、旧の保存形式へ写して作る
（新の記録から、旧の列がすべて作れることを確かめる）。
"""

import hashlib
import io
import json
import random
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from itertools import combinations
from math import inf, isnan
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_fedsda_run_client as client_test_module
import test_fedsda_stream_protocol_run as whole_run_test_module
import test_proposed_regression as legacy_regression
from test_environment_proposed_regression import describe_missing_golden, find_golden_path
from test_fedsda_run_client import (
    LEGACY_ACTION_BY_ADAPTATION_OUTCOME,
    LEGACY_REASON_BY_DECISION_REASON,
    snapshot_run_client_state,
)
from test_fedsda_stream_protocol_run import (
    assert_server_parameter_computation_counts_match_records,
    make_execution_settings,
    make_run_participant_settings,
    run_real_legacy_whole_run,
)
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

from federated_drift_experiment import config, experiment
from federated_drift_experiment.data.schedules import extract_true_drift_events
from federated_drift_experiment.experiment_spec.configuration import (
    ExperimentConfiguration,
    ParameterAssignment,
    temporary_config,
)
from federated_drift_experiment.metrics import compute_metrics
from federated_learning_experiments.evaluation.communication_volume_record_store import (
    CommunicationVolumeSnapshot,
)
from federated_learning_experiments.evaluation.computation_cost_summary import (
    ComputationCostSummary,
    ServerComputationCounts,
    summarize_computation_cost,
)
from federated_learning_experiments.evaluation.run_metric_calculations import (
    DetectionMetrics,
    RunMetricSettings,
)
from federated_learning_experiments.execution.run_participant_contracts import RunParticipants
from federated_learning_experiments.methods.fedsda.candidate_model_selection.incomplete_post_alarm_candidate_validation_decision_record import (
    IncompletePostAlarmCandidateValidationDecisionRecord,
)
from federated_learning_experiments.methods.fedsda.consolidation.model_clustering_calculations import (
    ModelClusteringCriteria,
)
from federated_learning_experiments.runtime.fedsda_measured_run_execution import (
    execute_fedsda_stream_protocol_run_with_computation_measurement,
)
from federated_learning_experiments.runtime.fedsda_run_metric_derivation import (
    FedsdaRunMetrics,
    derive_fedsda_run_metrics,
)

# goldenの条件を照合するdataset（旧の回帰testの3ケースの全部）。
GOLDEN_DATASET_NAMES = ("sine2", "sea2", "mnist2")
# goldenが比べる33指標のうち、導出していないもの（なし。計算量の7項目も、計測つきの全体runから導出する）。
UNDERIVED_LEGACY_METRIC_NAMES = ()
# goldenの条件ごとの、通る経路（照合が、中身のある比較になっていることを確かめるための期待）。
# sine2（1500件）: 複数のグローバルモデル、統合、候補の判定、学習帰属の切替を、すべて通る。
# sea2（600件）: 警報のたびに現行モデルを維持し、モデルが1つのまま終わる。
# mnist2（client 2、100件）: 候補の採用と棄却、学習帰属の切替を通るが、グローバルモデルは1つのまま終わる。
GOLDEN_CONDITION_PATHS_BY_DATASET_NAME = {
    "sine2": dict(
        multiple_global_models=True,
        candidate_decisions=True,
        multiple_held_models=True,
        synchronization_computation=True,
    ),
    "sea2": dict(
        multiple_global_models=False,
        candidate_decisions=False,
        multiple_held_models=False,
        synchronization_computation=False,
    ),
    "mnist2": dict(
        multiple_global_models=False,
        candidate_decisions=True,
        multiple_held_models=True,
        synchronization_computation=True,
    ),
}


def derive_legacy_metric_values(run_metrics):
    """導出した指標を、旧の指標の名前へ写す（照合のための対応表）。"""
    detection_metrics = run_metrics.training_model_switch_detection_metrics
    communication_volume = run_metrics.communication_volume
    model_computation_counts = run_metrics.model_computation_counts
    loss_monitoring_computation_counts = run_metrics.loss_monitoring_computation_counts
    return dict(
        # 計算量: 旧の「推論」は、学習以外の用途の、概念固有部を通った標本数。「optimizerの更新」は、概念固有部の更新。
        compute_inference_examples_total=(
            model_computation_counts.concept_specific_part_inference_example_count
        ),
        compute_training_examples_total=(
            model_computation_counts.concept_specific_part_training_example_count
        ),
        compute_optimizer_steps_total=(
            model_computation_counts.concept_specific_parameter_optimizer_step_count
        ),
        compute_backbone_examples_total=(
            model_computation_counts.shared_part_training_example_count
            + model_computation_counts.shared_part_inference_example_count
        ),
        compute_head_examples_total=(
            model_computation_counts.concept_specific_part_training_example_count
            + model_computation_counts.concept_specific_part_inference_example_count
        ),
        compute_drift_detector_updates_total=(
            loss_monitoring_computation_counts.detector_component_update_count
        ),
        compute_drift_detector_hypotheses_total=(
            loss_monitoring_computation_counts.evaluated_candidate_bet_count
        ),
        accuracy=run_metrics.prediction_accuracy,
        stable_accuracy=run_metrics.stable_period_prediction_accuracy,
        final_model_count=run_metrics.final_global_model_count,
        precision=detection_metrics.detection_precision,
        recall=detection_metrics.detection_recall,
        f1=detection_metrics.detection_f1,
        total_detect=detection_metrics.detection_count,
        comm_models_up=communication_volume.uploaded_model_count,
        comm_models_down=communication_volume.downloaded_model_count,
        comm_models_total=(
            communication_volume.uploaded_model_count + communication_volume.downloaded_model_count
        ),
        comm_messages_up=communication_volume.uploaded_message_count,
        comm_messages_down=communication_volume.downloaded_message_count,
        comm_messages_total=(
            communication_volume.uploaded_message_count
            + communication_volume.downloaded_message_count
        ),
        comm_parameter_values_up=communication_volume.uploaded_parameter_value_count,
        comm_parameter_values_down=communication_volume.downloaded_parameter_value_count,
        comm_parameter_values_total=(
            communication_volume.uploaded_parameter_value_count
            + communication_volume.downloaded_parameter_value_count
        ),
        comm_bytes_up=communication_volume.uploaded_byte_count,
        comm_bytes_down=communication_volume.downloaded_byte_count,
        comm_bytes_total=(
            communication_volume.uploaded_byte_count + communication_volume.downloaded_byte_count
        ),
        final_parameter_values=run_metrics.final_parameter_value_count,
        final_parameter_bytes=run_metrics.final_parameter_byte_count,
        # 最終構成の候補の判定は、すべて、警報後の標本での検証（旧の2つの指標は同じ値）。
        provisional_proposal_count=run_metrics.candidate_validation_decision_count,
        provisional_forward_count=run_metrics.candidate_validation_decision_count,
        routing_soft_prediction_sample_count=run_metrics.mixed_prediction_sample_count,
        routing_switching_recalibration_sample_count=(
            run_metrics.fixed_share_weight_recalibration_replayed_sample_count
        ),
        routing_aggregation_recalibration_sample_count=(
            run_metrics.adahedge_weight_recalibration_replayed_sample_count
        ),
    )


# 旧の`compute_metrics`が返す指標のうち、新に対応する値がないもの（最終構成では使わない選択肢の指標。値は0）。
LEGACY_METRIC_NAMES_OF_UNUSED_OPTIONS = (
    "adaptation_episode_suppressed_count",
    "provisional_reject_insufficient_data_count",
    "provisional_reject_full_interval_count",
    "provisional_reject_recent_interval_count",
    "provisional_reject_full_and_recent_count",
)


# 照合のたびに、旧の指標の値（応答の指標の、旧の名前→値）を積む。対応表の取り違えを検出できるかを、
# 最後のtestが、この観測から確かめる。
LEGACY_CHANGE_RESPONSE_METRIC_VALUE_OBSERVATIONS = []
# 実行した全部の条件で、旧の値が同じになる、指標の対（対応表で、この2つを取り違えても、照合は検出できない）。
INDISTINGUISHABLE_LEGACY_CHANGE_RESPONSE_METRIC_NAME_PAIRS = {
    # 最終構成では、候補の学習区間の全部を、学習に使う（新は、1つの値で持つ）。
    ("provisional_interval_count_mean", "provisional_training_count_mean"),
    # 最終構成の検証は、提案の次の標本から、1標本ずつ進む（判定までの標本数と、検証に使った標本数が、同じになる）。
    # 新の2つの値は、別の記録（判定の位置と、検証の標本数）から作り、表の列で、別々に、旧の配列と照合している。
    ("provisional_resolution_delay_mean", "provisional_validation_count_mean"),
}


def derive_legacy_change_response_metric_values(run_metrics):
    """応答の指標を、旧の指標（`compute_metrics`と、監視の値の最大）の名前へ写す（照合のための対応表）。

    旧の指標のうち、ほかの値から計算できるもの（見逃し率、誤検出率、真の変更の数ほか）は、ここで計算する。
    """
    detection_metrics = run_metrics.training_model_switch_detection_metrics
    change_response_metrics = run_metrics.change_response_metrics
    switch_timing_metrics = change_response_metrics.training_model_switch_timing_metrics
    unmatched_switch_counts = switch_timing_metrics.unmatched_switch_counts
    alarm_metrics = change_response_metrics.loss_change_alarm_detection_metrics
    change_point_metrics = change_response_metrics.change_point_estimation_metrics
    action_metrics = change_response_metrics.adaptation_action_metrics
    decision_metrics = change_response_metrics.candidate_validation_decision_metrics
    false_switch_count = (
        detection_metrics.detection_count - detection_metrics.matched_detection_count
    )
    missed_change_count = (
        detection_metrics.concept_change_count - detection_metrics.matched_detection_count
    )
    return dict(
        # 検出の指標の計数から計算できる値。
        total_true=detection_metrics.concept_change_count,
        tp=detection_metrics.matched_detection_count,
        fp=false_switch_count,
        fn=missed_change_count,
        miss_rate=(
            missed_change_count / detection_metrics.concept_change_count
            if detection_metrics.concept_change_count
            else 0.0
        ),
        fdr=(
            false_switch_count / detection_metrics.detection_count
            if detection_metrics.detection_count
            else 0.0
        ),
        # 切替の時機。
        avg_delay=switch_timing_metrics.mean_matched_switch_delay_sample_count,
        switch_fp_duplicate=unmatched_switch_counts.duplicate_event_count,
        switch_fp_early=unmatched_switch_counts.early_event_count,
        switch_fp_late=unmatched_switch_counts.late_event_count,
        switch_fp_isolated=unmatched_switch_counts.isolated_event_count,
        # 警報。
        alarm_precision=alarm_metrics.detection_precision,
        alarm_recall=alarm_metrics.detection_recall,
        alarm_f1=alarm_metrics.detection_f1,
        alarm_total=alarm_metrics.detection_count,
        # 変化点の推定。
        change_point_mae=change_point_metrics.mean_absolute_error_sample_count,
        change_point_bias=change_point_metrics.mean_signed_error_sample_count,
        change_point_estimate_count=change_point_metrics.matched_alarm_count,
        # 適応の区分。
        adaptation_reuse_count=action_metrics.reuse_held_model_count,
        adaptation_reuse_precision=action_metrics.reuse_held_model_precision,
        adaptation_create_count=action_metrics.adopt_candidate_model_count,
        adaptation_create_precision=action_metrics.adopt_candidate_model_precision,
        adaptation_create_rejected_count=action_metrics.reject_candidate_model_count,
        adaptation_maintain_count=action_metrics.maintain_current_model_count,
        server_mapping_change_count=action_metrics.remap_by_server_consolidation_count,
        model_reuse_current_fit_count=action_metrics.current_model_fit_selection_count,
        model_reuse_alternative_fit_count=action_metrics.alternative_model_reuse_selection_count,
        # 候補の判定。
        provisional_acceptance_rate=decision_metrics.accepted_decision_rate,
        provisional_matched_true_count=decision_metrics.matched_decision_count,
        provisional_accepted_matched_true_count=decision_metrics.matched_accepted_decision_count,
        provisional_rejected_matched_true_count=decision_metrics.matched_rejected_decision_count,
        provisional_accepted_precision=decision_metrics.accepted_decision_precision,
        # 最終構成では、候補の学習区間の全部を、学習に使う（旧の2つの指標は同じ値）。
        provisional_interval_count_mean=(
            decision_metrics.mean_candidate_training_interval_sample_count
        ),
        provisional_training_count_mean=(
            decision_metrics.mean_candidate_training_interval_sample_count
        ),
        provisional_validation_count_mean=decision_metrics.mean_validation_sample_count,
        provisional_resolution_delay_mean=decision_metrics.mean_resolution_delay_sample_count,
        provisional_accepted_full_margin_mean=(
            decision_metrics.accepted_mean_full_interval_loss_margin
        ),
        provisional_accepted_recent_margin_mean=(
            decision_metrics.accepted_mean_second_segment_loss_margin
        ),
        provisional_rejected_full_margin_mean=(
            decision_metrics.rejected_mean_full_interval_loss_margin
        ),
        provisional_rejected_recent_margin_mean=(
            decision_metrics.rejected_mean_second_segment_loss_margin
        ),
        provisional_matched_full_margin_mean=decision_metrics.matched_mean_full_interval_loss_margin,
        provisional_matched_recent_margin_mean=(
            decision_metrics.matched_mean_second_segment_loss_margin
        ),
        provisional_unmatched_full_margin_mean=(
            decision_metrics.unmatched_mean_full_interval_loss_margin
        ),
        provisional_unmatched_recent_margin_mean=(
            decision_metrics.unmatched_mean_second_segment_loss_margin
        ),
        provisional_reject_insufficient_forward_data_count=(
            decision_metrics.incomplete_at_run_end_decision_count
        ),
        provisional_reject_first_interval_count=(
            decision_metrics.first_segment_margin_failed_decision_count
        ),
        provisional_reject_second_interval_count=(
            decision_metrics.second_segment_margin_failed_decision_count
        ),
        provisional_reject_first_and_second_count=(
            decision_metrics.both_segment_margins_failed_decision_count
        ),
        provisional_reject_current_refit_count=(
            decision_metrics.current_reference_within_historical_loss_tolerance_decision_count
        ),
        provisional_reject_alternative_refit_count=(
            decision_metrics.alternative_reference_within_historical_loss_tolerance_decision_count
        ),
        # 旧の「参照の再適合」は、現行と、別の保有モデルの合計。
        provisional_reject_reference_refit_count=(
            decision_metrics.current_reference_within_historical_loss_tolerance_decision_count
            + decision_metrics.alternative_reference_within_historical_loss_tolerance_decision_count
        ),
        provisional_reference_excess_mean=(
            decision_metrics.mean_reference_loss_excess_over_historical_mean
        ),
    )


def assert_change_response_metrics_match_legacy_metrics(*, run_metrics, legacy_metric_values):
    """応答の指標が、旧の指標の値（`compute_metrics`の戻り値を含む辞書）と、全部の項目で一致する。

    旧の`compute_metrics`が返す指標の全部が、照合済みの33指標・応答の指標・最終構成では使わない指標の
    どれかに入ることも確かめる。
    """
    change_response_values = derive_legacy_change_response_metric_values(run_metrics)
    for legacy_metric_name, metric_value in change_response_values.items():
        assert metric_value == legacy_metric_values[legacy_metric_name], legacy_metric_name
        assert type(metric_value) in (int, float), legacy_metric_name
    for legacy_metric_name in LEGACY_METRIC_NAMES_OF_UNUSED_OPTIONS:
        assert legacy_metric_values[legacy_metric_name] == 0, legacy_metric_name
    legacy_compute_metrics_names = set(compute_metrics([], []))
    assert legacy_compute_metrics_names == (
        (set(derive_legacy_metric_values(run_metrics)) & legacy_compute_metrics_names)
        | set(change_response_values)
        | set(LEGACY_METRIC_NAMES_OF_UNUSED_OPTIONS)
    )
    assert len(change_response_values) == 52
    LEGACY_CHANGE_RESPONSE_METRIC_VALUE_OBSERVATIONS.append(
        {
            legacy_metric_name: legacy_metric_values[legacy_metric_name]
            for legacy_metric_name in change_response_values
        }
    )


# 旧の、ラウンドごとのグローバルモデル数のまとめ（`_add_telemetry_results`）。
LEGACY_GLOBAL_MODEL_COUNT_METRIC_NAMES = ("mean_model_count", "max_model_count", "model_count_auc")
# 旧の、2つの距離のAUC（定義できないとき、旧はNaN、新はNone）。
LEGACY_AUC_METRIC_NAMES = (
    "clustering_oracle_loss_distance_auc",
    "clustering_oracle_parameter_distance_auc",
)
# 照合のたびに、旧の指標の値（モデル別の診断の指標の、旧の名前→値。NaNはNone）を積む（対応表の取り違えの検査に使う）。
LEGACY_MODEL_DIAGNOSTIC_METRIC_VALUE_OBSERVATIONS = []
# 実行した全部の条件で、旧の値が同じになる、指標の対（対応表で、この2つを取り違えても、照合は検出できない）。
# クラスタの割当の適合率・再現率・F1は、実行した条件では、いつも同じ値になる（統合した対が、すべて、真の概念の一致する対で、
# 一致する対を、すべて統合したか、統合が1つもないか）。この3つは、4通りの数（条件の間で、互いに違う値を取る）から、
# 計算のmoduleが作り、乱数の列で、旧と同じ式と照合してある（test_model_diagnostic_metric_calculations.py）。
INDISTINGUISHABLE_LEGACY_MODEL_DIAGNOSTIC_METRIC_NAME_PAIRS = {
    ("clustering_oracle_merge_f1", "clustering_oracle_merge_precision"),
    ("clustering_oracle_merge_f1", "clustering_oracle_merge_recall"),
    ("clustering_oracle_merge_precision", "clustering_oracle_merge_recall"),
}


def derive_legacy_model_diagnostic_metric_values(run_metrics):
    """モデル別の診断の指標を、旧の指標の名前へ写す（照合のための対応表）。"""
    model_diagnostic_metrics = run_metrics.model_diagnostic_metrics
    global_model_count_metrics = model_diagnostic_metrics.global_model_count_metrics
    training_amount_metrics = model_diagnostic_metrics.held_model_training_amount_metrics
    clustering_metrics = model_diagnostic_metrics.clustering_true_concept_agreement_metrics
    cluster_assignment_agreement = clustering_metrics.cluster_assignment_agreement
    legacy_metric_values = dict(
        mean_model_count=global_model_count_metrics.mean_global_model_count_after_synchronization,
        max_model_count=global_model_count_metrics.maximum_global_model_count_after_synchronization,
        model_count_auc=global_model_count_metrics.total_global_model_count_over_server_rounds,
    )
    for legacy_prefix, summary in (
        ("model_assigned_samples", training_amount_metrics.stored_training_sample_count_summary),
        ("model_training_examples", training_amount_metrics.trained_sample_count_summary),
        ("model_optimizer_steps", training_amount_metrics.parameter_update_step_count_summary),
    ):
        legacy_metric_values[f"{legacy_prefix}_total"] = summary.total_value
        legacy_metric_values[f"{legacy_prefix}_mean"] = summary.mean_value
        legacy_metric_values[f"{legacy_prefix}_min"] = summary.minimum_value
        legacy_metric_values[f"{legacy_prefix}_cv"] = summary.coefficient_of_variation
    legacy_metric_values.update(
        clustering_oracle_pair_count=clustering_metrics.true_concept_known_pair_count,
        clustering_oracle_same_pair_count=clustering_metrics.same_true_concept_pair_count,
        clustering_oracle_merge_tp=cluster_assignment_agreement.same_cluster_same_concept_pair_count,
        clustering_oracle_merge_fp=(
            cluster_assignment_agreement.same_cluster_different_concept_pair_count
        ),
        clustering_oracle_merge_fn=(
            cluster_assignment_agreement.different_cluster_same_concept_pair_count
        ),
        clustering_oracle_merge_tn=(
            cluster_assignment_agreement.different_cluster_different_concept_pair_count
        ),
        clustering_oracle_merge_precision=cluster_assignment_agreement.same_cluster_precision,
        clustering_oracle_merge_recall=cluster_assignment_agreement.same_cluster_recall,
        clustering_oracle_merge_f1=cluster_assignment_agreement.same_cluster_f1,
        clustering_oracle_loss_distance_auc=clustering_metrics.loss_increase_distance_same_concept_auc,
        clustering_oracle_parameter_distance_auc=(
            clustering_metrics.concept_specific_parameter_distance_same_concept_auc
        ),
    )
    return legacy_metric_values


def assert_model_diagnostic_metrics_match_legacy_metrics(
    *, run_metrics, legacy_metric_values, global_model_counts_are_recorded
):
    """モデル別の診断の指標が、旧の指標の値と、全部の項目で一致する（旧のNaN——AUCを定義できない——は、新ではNone）。

    小さい条件の実旧の実行は、ラウンドごとのモデル数を記録しないので、そのまとめの3項目は、比べない。
    """
    model_diagnostic_values = derive_legacy_model_diagnostic_metric_values(run_metrics)
    assert len(model_diagnostic_values) == 26
    compared_values = {}
    for legacy_metric_name, metric_value in model_diagnostic_values.items():
        if (
            legacy_metric_name in LEGACY_GLOBAL_MODEL_COUNT_METRIC_NAMES
            and not global_model_counts_are_recorded
        ):
            continue
        legacy_metric_value = legacy_metric_values[legacy_metric_name]
        if isnan(legacy_metric_value):
            assert legacy_metric_name in LEGACY_AUC_METRIC_NAMES, legacy_metric_name
            assert metric_value is None, legacy_metric_name
        else:
            assert type(metric_value) in (int, float), legacy_metric_name
            assert metric_value == legacy_metric_value, legacy_metric_name
        compared_values[legacy_metric_name] = metric_value
    LEGACY_MODEL_DIAGNOSTIC_METRIC_VALUE_OBSERVATIONS.append(compared_values)


# 照合のたびに、旧の指標の値（モデル対の予測の相補性の、旧の名前→値）を積む（対応表の取り違えの検査に使う）。
LEGACY_MODEL_PAIR_METRIC_VALUE_OBSERVATIONS = []
# 実行した全部の条件で、旧の値が同じになる、指標の対（実測で決める）。
INDISTINGUISHABLE_LEGACY_MODEL_PAIR_METRIC_NAME_PAIRS = set()


def derive_legacy_model_pair_metric_values(run_metrics):
    """モデル対の予測の相補性の指標を、旧の指標の名前へ写す（照合のための対応表）。"""
    complementarity_metrics = run_metrics.model_pair_prediction_complementarity_metrics
    return dict(
        model_pair_evaluation_count=complementarity_metrics.correctness_compared_evaluation_count,
        model_pair_sample_count=complementarity_metrics.correctness_compared_sample_count,
        model_pair_correctness_disagreement_rate=(
            complementarity_metrics.correctness_disagreement_rate
        ),
        model_pair_oracle_gain_rate=complementarity_metrics.per_sample_selection_gain_rate,
        model_pair_both_correct_rate=complementarity_metrics.both_correct_rate,
    )


def assert_model_pair_metrics_match_legacy_metrics(*, run_metrics, legacy_metric_values):
    """モデル対の予測の相補性の指標が、旧の指標の値と、全部の項目で一致する。"""
    model_pair_values = derive_legacy_model_pair_metric_values(run_metrics)
    assert len(model_pair_values) == 5
    for legacy_metric_name, metric_value in model_pair_values.items():
        assert type(metric_value) in (int, float), legacy_metric_name
        assert metric_value == legacy_metric_values[legacy_metric_name], legacy_metric_name
    LEGACY_MODEL_PAIR_METRIC_VALUE_OBSERVATIONS.append(
        {
            legacy_metric_name: legacy_metric_values[legacy_metric_name]
            for legacy_metric_name in model_pair_values
        }
    )


# 旧の予測の集計の指標のうち、Fixed-Shareの重みで予測する構成では、いつも0のもの（Fixed-Shareの結合と、実際の予測の差）。
# 値そのものは、どの構成でも、新の指標から計算して照合する。
LEGACY_PREDICTION_DIAGNOSTIC_METRIC_NAMES_ZERO_FOR_FIXED_SHARE_PREDICTION = (
    "routing_switching_gain_rate",
    "routing_switching_stable_gain_rate",
    "routing_switching_recovery_gain_rate",
)
# 照合のたびに、旧の指標の値（予測の診断の指標の、旧の名前→値）を積む（対応表の取り違えの検査に使う）。
LEGACY_PREDICTION_DIAGNOSTIC_METRIC_VALUE_OBSERVATIONS = []
# 実行した全部の条件で、旧の値が同じになる、指標の対（実測で決める）。
INDISTINGUISHABLE_LEGACY_PREDICTION_DIAGNOSTIC_METRIC_NAME_PAIRS = {
    # 最終構成では、Fixed-Shareの結合が、実際の予測（新は、1つの値で持つ）。
    ("routing_mixture_accuracy", "routing_switching_accuracy"),
}


def derive_legacy_prediction_diagnostic_metric_values(run_metrics):
    """予測の診断の指標を、旧の指標の名前へ写す（照合のための対応表）。

    旧の指標のうち、数から計算できるもの（正解の割合、差、比）は、ここで、旧と同じ式で計算する。
    """
    diagnostic_metrics = run_metrics.prediction_diagnostic_metrics
    correct_counts = diagnostic_metrics.prediction_correct_counts
    sample_count = correct_counts.predicted_sample_count
    combined_correct_count = correct_counts.combined_prediction_correct_count
    any_correct_count = correct_counts.any_model_or_combined_prediction_correct_count
    maximum_weight_correct_count = correct_counts.maximum_weight_model_prediction_correct_count
    highest_confidence_correct_count = (
        correct_counts.highest_confidence_model_prediction_correct_count
    )
    combined_period = diagnostic_metrics.combined_prediction_period_accuracies
    any_period = diagnostic_metrics.any_model_or_combined_prediction_period_accuracies
    maximum_weight_period = diagnostic_metrics.maximum_weight_model_prediction_period_accuracies
    true_concept_period = diagnostic_metrics.true_concept_diagnostic_prediction_period_accuracies
    class_metrics = diagnostic_metrics.class_macro_prediction_metrics

    def divide_or_zero(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    return dict(
        routing_sample_count=sample_count,
        routing_oracle_accuracy=divide_or_zero(any_correct_count, sample_count),
        routing_mixture_accuracy=divide_or_zero(combined_correct_count, sample_count),
        routing_leader_accuracy=divide_or_zero(maximum_weight_correct_count, sample_count),
        routing_confidence_leader_accuracy=divide_or_zero(
            highest_confidence_correct_count, sample_count
        ),
        routing_oracle_gain_rate=divide_or_zero(
            any_correct_count - combined_correct_count, sample_count
        ),
        routing_oracle_recovery_rate=divide_or_zero(combined_correct_count, any_correct_count),
        routing_oracle_stable_accuracy=any_period.stable_period_accuracy,
        routing_oracle_recovery_accuracy=any_period.recovery_period_accuracy,
        routing_oracle_concept_accuracy=divide_or_zero(
            diagnostic_metrics.true_concept_diagnostic_prediction_correct_count,
            diagnostic_metrics.true_concept_known_sample_count,
        ),
        routing_oracle_concept_stable_accuracy=true_concept_period.stable_period_accuracy,
        routing_oracle_concept_recovery_accuracy=true_concept_period.recovery_period_accuracy,
        routing_oracle_concept_stable_gain_rate=(
            true_concept_period.stable_period_accuracy - combined_period.stable_period_accuracy
        ),
        routing_oracle_concept_recovery_gain_rate=(
            true_concept_period.recovery_period_accuracy - combined_period.recovery_period_accuracy
        ),
        routing_leader_stable_accuracy=maximum_weight_period.stable_period_accuracy,
        routing_leader_recovery_accuracy=maximum_weight_period.recovery_period_accuracy,
        routing_stable_oracle_gap=(
            any_period.stable_period_accuracy - combined_period.stable_period_accuracy
        ),
        routing_recovery_oracle_gap=(
            any_period.recovery_period_accuracy - combined_period.recovery_period_accuracy
        ),
        routing_missed_oracle_count=any_correct_count - combined_correct_count,
        routing_confidence_leader_oracle_recovery_rate=divide_or_zero(
            highest_confidence_correct_count, any_correct_count
        ),
        routing_confidence_leader_missed_oracle_count=(
            any_correct_count - highest_confidence_correct_count
        ),
        routing_class_macro_oracle_accuracy=(
            class_metrics.macro_any_model_or_combined_prediction_accuracy
        ),
        routing_class_macro_mixture_accuracy=class_metrics.macro_combined_prediction_accuracy,
        routing_class_macro_leader_accuracy=(
            class_metrics.macro_maximum_weight_model_prediction_accuracy
        ),
        routing_class_macro_confidence_leader_accuracy=(
            class_metrics.macro_highest_confidence_model_prediction_accuracy
        ),
        routing_class_oracle_gap_mean=class_metrics.mean_any_model_or_combined_accuracy_gap,
        routing_class_oracle_gap_std=(
            class_metrics.any_model_or_combined_accuracy_gap_standard_deviation
        ),
        routing_class_oracle_recovery_rate_mean=(
            class_metrics.mean_combined_to_any_model_or_combined_correct_ratio
        ),
        routing_class_oracle_recovery_rate_min=(
            class_metrics.minimum_combined_to_any_model_or_combined_correct_ratio
        ),
        # 旧の`routing_switching_*`は、Fixed-Shareの重みでの結合の指標（提案構成では、実際の予測と同じ値）。
        routing_switching_accuracy=divide_or_zero(
            diagnostic_metrics.fixed_share_weighted_prediction_correct_count, sample_count
        ),
        # Fixed-Shareの結合と、実際の予測の差。
        routing_switching_gain_rate=divide_or_zero(
            diagnostic_metrics.fixed_share_weighted_prediction_correct_count
            - combined_correct_count,
            sample_count,
        ),
        routing_switching_stable_gain_rate=(
            diagnostic_metrics.fixed_share_weighted_prediction_period_accuracies.stable_period_accuracy
            - combined_period.stable_period_accuracy
        ),
        routing_switching_recovery_gain_rate=(
            diagnostic_metrics.fixed_share_weighted_prediction_period_accuracies.recovery_period_accuracy
            - combined_period.recovery_period_accuracy
        ),
        routing_switching_global_gain_rate=divide_or_zero(
            diagnostic_metrics.fixed_share_weighted_prediction_correct_count
            - diagnostic_metrics.adahedge_weighted_prediction_correct_count,
            sample_count,
        ),
        routing_switching_stable_accuracy=(
            diagnostic_metrics.fixed_share_weighted_prediction_period_accuracies.stable_period_accuracy
        ),
        routing_switching_recovery_accuracy=(
            diagnostic_metrics.fixed_share_weighted_prediction_period_accuracies.recovery_period_accuracy
        ),
        routing_switching_effective_experts_mean=(
            diagnostic_metrics.mean_fixed_share_effective_model_count
        ),
        routing_switching_leader_switch_count=(
            diagnostic_metrics.fixed_share_weight_leader_switch_count
        ),
        routing_switching_pool_reset_count=(
            diagnostic_metrics.fixed_share_weight_model_pool_reset_count
        ),
        routing_aggregation_recalibration_count=(
            diagnostic_metrics.adahedge_weight_aggregation_recalibration_count
        ),
    )


def assert_prediction_diagnostic_metrics_match_legacy_metrics(
    *, run_metrics, legacy_metric_values, fixed_share_weights_are_used_for_prediction=True
):
    """予測の診断の指標が、旧の指標の値（旧の40の指標）と、全部の項目で一致する。

    Fixed-Shareの重みで予測する構成では、Fixed-Shareの結合と実際の予測の差の指標が、いつも0であることも確かめる。
    """
    prediction_diagnostic_values = derive_legacy_prediction_diagnostic_metric_values(run_metrics)
    assert len(prediction_diagnostic_values) == 40
    for legacy_metric_name, metric_value in prediction_diagnostic_values.items():
        assert type(metric_value) in (int, float), legacy_metric_name
        assert metric_value == legacy_metric_values[legacy_metric_name], legacy_metric_name
    if fixed_share_weights_are_used_for_prediction:
        for (
            legacy_metric_name
        ) in LEGACY_PREDICTION_DIAGNOSTIC_METRIC_NAMES_ZERO_FOR_FIXED_SHARE_PREDICTION:
            assert legacy_metric_values[legacy_metric_name] == 0, legacy_metric_name
    # 実際の予測の正解の割合と、定常の区間の正解の割合は、既存の指標と同じ値。
    assert (
        prediction_diagnostic_values["routing_mixture_accuracy"] == run_metrics.prediction_accuracy
    )
    LEGACY_PREDICTION_DIAGNOSTIC_METRIC_VALUE_OBSERVATIONS.append(
        {
            legacy_metric_name: legacy_metric_values[legacy_metric_name]
            for legacy_metric_name in prediction_diagnostic_values
        }
    )


def derive_legacy_trace_arrays(*, run_result, participants):
    """新の記録の読取りから、旧の保存形式（`_save_raw_run`）の31の離散列を作る。"""
    run_clients = participants.client_operations
    server_owners = participants.server_operations.owners
    prediction_records_by_client = [
        run_client.owners.sample_prediction_record_store.snapshot_sample_prediction_records()
        for run_client in run_clients
    ]
    combined_correctness = [
        [record.combined_prediction_is_correct for record in prediction_records]
        for prediction_records in prediction_records_by_client
    ]
    maximum_weight_model_ids = [
        [record.maximum_weight_model_id for record in prediction_records]
        for prediction_records in prediction_records_by_client
    ]
    concept_changes = [
        (client_id, sample_index)
        for client_id, concept_trace in enumerate(run_result.evaluation_concept_traces)
        for sample_index in range(1, len(concept_trace.concept_ids_by_sample_index))
        if concept_trace.concept_ids_by_sample_index[sample_index]
        != concept_trace.concept_ids_by_sample_index[sample_index - 1]
    ]
    adaptation_snapshots = [
        run_client.owners.adaptation_record_store.get_state_snapshot() for run_client in run_clients
    ]
    switches = [
        (client_id, sample_index)
        for client_id, adaptation_snapshot in enumerate(adaptation_snapshots)
        for sample_index in adaptation_snapshot.training_model_switch_sample_indices
    ]
    adaptations = [
        (client_id, adaptation_record)
        for client_id, adaptation_snapshot in enumerate(adaptation_snapshots)
        for adaptation_record in adaptation_snapshot.adaptation_records
    ]
    decisions = [
        (client_id, decision_record)
        for client_id, run_client in enumerate(run_clients)
        for decision_record in run_client.owners.candidate_validation_decision_record_store.snapshot_candidate_validation_decision_records()
    ]

    def decision_is_incomplete(decision_record):
        return type(decision_record) is IncompletePostAlarmCandidateValidationDecisionRecord

    registration_records = (
        server_owners.global_model_repository.snapshot_model_registration_records()
    )
    final_global_model_ids = set(server_owners.global_model_repository.global_model_ids)
    model_observations = (
        server_owners.model_clustering_record_store.snapshot_model_clustering_observations()
    )
    pair_observations = (
        server_owners.model_clustering_record_store.snapshot_pair_clustering_observations()
    )
    return dict(
        history_accuracy=np.asarray(combined_correctness, dtype=np.int8),
        history_model_id=np.asarray(maximum_weight_model_ids, dtype=np.int32),
        drift_client_ids=np.asarray(
            [client_id for client_id, _ in concept_changes], dtype=np.int32
        ),
        drift_positions=np.asarray(
            [sample_index for _, sample_index in concept_changes], dtype=np.int32
        ),
        switch_client_ids=np.asarray([client_id for client_id, _ in switches], dtype=np.int32),
        switch_positions=np.asarray([sample_index for _, sample_index in switches], dtype=np.int32),
        adaptation_client_ids=np.asarray(
            [client_id for client_id, _ in adaptations], dtype=np.int32
        ),
        adaptation_positions=np.asarray(
            [adaptation_record.adaptation_sample_index for _, adaptation_record in adaptations],
            dtype=np.int32,
        ),
        adaptation_actions=np.asarray(
            [
                LEGACY_ACTION_BY_ADAPTATION_OUTCOME[adaptation_record.adaptation_outcome]
                for _, adaptation_record in adaptations
            ],
            dtype=np.str_,
        ),
        adaptation_old_model_ids=np.asarray(
            [adaptation_record.previous_training_model_id for _, adaptation_record in adaptations],
            dtype=np.int32,
        ),
        adaptation_new_model_ids=np.asarray(
            [adaptation_record.current_training_model_id for _, adaptation_record in adaptations],
            dtype=np.int32,
        ),
        provisional_client_ids=np.asarray(
            [client_id for client_id, _ in decisions], dtype=np.int32
        ),
        provisional_positions=np.asarray(
            [decision_record.proposal_sample_index for _, decision_record in decisions],
            dtype=np.int32,
        ),
        # 採否と理由は、記録のとおりに写す（理由から採否を逆算しない。LEGACY-001）。
        provisional_accepted=np.asarray(
            [
                False
                if decision_is_incomplete(decision_record)
                else decision_record.post_alarm_candidate_loss_evaluation.candidate_accepted
                for _, decision_record in decisions
            ],
            dtype=np.bool_,
        ),
        provisional_reasons=np.asarray(
            [
                "insufficient_forward_data"
                if decision_is_incomplete(decision_record)
                else LEGACY_REASON_BY_DECISION_REASON[
                    decision_record.post_alarm_candidate_loss_evaluation.decision_reason
                ]
                for _, decision_record in decisions
            ],
            dtype=np.str_,
        ),
        provisional_resolution_positions=np.asarray(
            [
                decision_record.finalization_sample_index
                if decision_is_incomplete(decision_record)
                else decision_record.resolution_sample_index
                for _, decision_record in decisions
            ],
            dtype=np.int32,
        ),
        model_registration_ids=np.asarray(
            [registration_record.model_id for registration_record in registration_records],
            dtype=np.int32,
        ),
        # 旧は、初期モデルの登録ラウンドを−1で表す。
        model_registration_rounds=np.asarray(
            [
                -1
                if registration_record.registered_round_index is None
                else registration_record.registered_round_index
                for registration_record in registration_records
            ],
            dtype=np.int32,
        ),
        model_registration_final_active=np.asarray(
            [
                registration_record.model_id in final_global_model_ids
                for registration_record in registration_records
            ],
            dtype=np.bool_,
        ),
        clustering_rounds=np.asarray(
            [model_observation.round_index for model_observation in model_observations],
            dtype=np.int32,
        ),
        clustering_model_ids=np.asarray(
            [model_observation.model_id for model_observation in model_observations],
            dtype=np.int32,
        ),
        clustering_representative_model_ids=np.asarray(
            [model_observation.representative_model_id for model_observation in model_observations],
            dtype=np.int32,
        ),
        clustering_participated_in_merge=np.asarray(
            [
                model_observation.merged_with_other_models
                for model_observation in model_observations
            ],
            dtype=np.bool_,
        ),
        clustering_absorbed=np.asarray(
            [
                model_observation.absorbed_into_representative
                for model_observation in model_observations
            ],
            dtype=np.bool_,
        ),
        # 旧の`history_routing_switching_*`は、Fixed-Shareの重みでの結合の列（提案構成では、実際の予測と同じ値）。
        history_routing_switching_correct=np.asarray(
            [
                [
                    prediction_record.fixed_share_weighted_prediction_is_correct
                    for prediction_record in prediction_records
                ]
                for prediction_records in prediction_records_by_client
            ],
            dtype=np.bool_,
        ),
        history_routing_switching_leader_id=np.asarray(
            [
                [
                    prediction_record.fixed_share_maximum_weight_model_id
                    for prediction_record in prediction_records
                ]
                for prediction_records in prediction_records_by_client
            ],
            dtype=np.int32,
        ),
        # 混合予測は常時有効。
        history_routing_soft_active=np.asarray(
            [
                [True] * len(prediction_records)
                for prediction_records in prediction_records_by_client
            ],
            dtype=np.bool_,
        ),
        clustering_pair_rounds=np.asarray(
            [pair_observation.round_index for pair_observation in pair_observations], dtype=np.int32
        ),
        clustering_pair_left_model_ids=np.asarray(
            [pair_observation.lower_model_id for pair_observation in pair_observations],
            dtype=np.int32,
        ),
        clustering_pair_right_model_ids=np.asarray(
            [pair_observation.higher_model_id for pair_observation in pair_observations],
            dtype=np.int32,
        ),
        clustering_pair_same_cluster=np.asarray(
            [pair_observation.assigned_to_same_cluster for pair_observation in pair_observations],
            dtype=np.bool_,
        ),
    )


def assert_multiply_accumulate_counts_match_example_counts(
    *, model_computation_counts, participants, models_share_feature_extractor=True
):
    """積和演算の数が、標本数の計数×（部品の全結合層の、入力の次元×出力の次元の合計）と一致する。

    逆伝播の見積りは、共有部の最初の層だけ、入力へ勾配を戻さない（重みの勾配だけ）。
    共有部を持たない構造では、共有部の計数は、全部0で、最初の層は、概念固有部（モデルごとの隠れ層）の最初の層。
    """
    run_client = participants.client_operations[0]
    classifier = run_client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()[
        0
    ].classifier

    def linear_layer_sizes(module):
        return [
            layer.in_features * layer.out_features
            for layer in module.modules()
            if type(layer) is torch.nn.Linear
        ]

    shared_layer_sizes = linear_layer_sizes(classifier.feature_extractor)
    concept_specific_size = sum(linear_layer_sizes(classifier.residual_adapter)) + sum(
        linear_layer_sizes(classifier.classification_layer)
    )
    assert concept_specific_size > 0
    counts = model_computation_counts
    if not models_share_feature_extractor:
        assert shared_layer_sizes == []
        assert (
            counts.shared_part_training_example_count,
            counts.shared_part_inference_example_count,
            counts.shared_parameter_optimizer_step_count,
            counts.shared_part_training_forward_multiply_accumulate_count,
            counts.shared_part_inference_forward_multiply_accumulate_count,
            counts.shared_part_estimated_backward_multiply_accumulate_count,
        ) == (0, 0, 0, 0, 0, 0)
        assert counts.concept_specific_part_inference_forward_multiply_accumulate_count == (
            counts.concept_specific_part_inference_example_count * concept_specific_size
        )
        assert counts.concept_specific_part_training_forward_multiply_accumulate_count == (
            counts.concept_specific_part_training_example_count * concept_specific_size
        )
        first_layer_size = linear_layer_sizes(classifier.residual_adapter)[0]
        assert counts.concept_specific_part_estimated_backward_multiply_accumulate_count == (
            counts.concept_specific_part_training_example_count
            * (2 * concept_specific_size - first_layer_size)
        )
        assert counts.concept_specific_part_training_example_count > 0
        assert counts.concept_specific_part_inference_example_count > 0
        return
    assert len(shared_layer_sizes) >= 1
    assert counts.shared_part_inference_forward_multiply_accumulate_count == (
        counts.shared_part_inference_example_count * sum(shared_layer_sizes)
    )
    assert counts.shared_part_training_forward_multiply_accumulate_count == (
        counts.shared_part_training_example_count * sum(shared_layer_sizes)
    )
    assert counts.concept_specific_part_inference_forward_multiply_accumulate_count == (
        counts.concept_specific_part_inference_example_count * concept_specific_size
    )
    assert counts.concept_specific_part_training_forward_multiply_accumulate_count == (
        counts.concept_specific_part_training_example_count * concept_specific_size
    )
    assert counts.shared_part_estimated_backward_multiply_accumulate_count == (
        counts.shared_part_training_example_count
        * (shared_layer_sizes[0] + 2 * sum(shared_layer_sizes[1:]))
    )
    assert counts.concept_specific_part_estimated_backward_multiply_accumulate_count == (
        counts.concept_specific_part_training_example_count * 2 * concept_specific_size
    )
    assert counts.shared_part_training_example_count > 0
    assert counts.shared_part_inference_example_count > 0


def legacy_result_client_count(dataset_name):
    """goldenの条件の、clientの数（旧の回帰testの条件から）。"""
    return {**legacy_regression.COMMON, **legacy_regression.CASES[dataset_name]}["N_CLIENTS"]


def make_golden_condition_settings(
    *, dataset_name, legacy_values, hidden_layer_widths, monkeypatch
):
    """goldenの条件の、旧の設定の値を、新の実行設定と参加者の設定の束へ写す。

    既存の組立て（testの定数を読む）を使うので、定数を、旧の設定の値へ差し替える。
    """
    for constant_name, constant_value in dict(
        PENDING_CAPACITY=legacy_values["FIFO_BUFFER_SIZE"],
        UPLOAD_DELAY_ROUND_COUNT=legacy_values["FEDSDA_MODEL_UPLOAD_DELAY_ROUNDS"],
        BATCH_SAMPLE_COUNT=legacy_values["CLIENT_BATCH_SIZE"],
        MINIMUM_CHANGE_INTERVAL_SAMPLE_COUNT=legacy_values["MIN_DRIFT_DATA"],
        FALSE_ALARM_CONTROL_ALPHA=legacy_values["E_DETECTOR_ALPHA"],
        MAXIMUM_RETAINED_CANDIDATE_COUNT=legacy_values["ADWIN_MAX_WINDOW"],
        MAXIMUM_TOLERATED_MEAN_LOSS_INCREASE=legacy_values["FEDSDA_DISTANCE_THRESHOLD"],
        MINIMUM_CANDIDATE_MEAN_LOSS_IMPROVEMENT=legacy_values["NEW_MODEL_EARLY_STOPPING_MIN_DELTA"],
        CANDIDATE_EPOCH_COUNT=legacy_values["NEW_MODEL_EPOCHS"],
        EARLY_STOPPING_PATIENCE=legacy_values["NEW_MODEL_EARLY_STOPPING_PATIENCE"],
        VALIDATION_SAMPLE_FRACTION=legacy_values["NEW_MODEL_VALIDATION_FRACTION"],
        WEIGHT_DECAY=legacy_values["WEIGHT_DECAY"],
    ).items():
        assert hasattr(client_test_module, constant_name), constant_name
        monkeypatch.setattr(client_test_module, constant_name, constant_value)
    for constant_name, constant_value in dict(
        ADAPTER_RANK=legacy_values["SHARED_ADAPTER_RANK"],
        HIDDEN_LAYER_WIDTHS=hidden_layer_widths,
        PRETRAINING_VALUES=dict(
            pretraining_sample_count=legacy_values["PRETRAIN_SAMPLES"],
            pretraining_epoch_count=legacy_values["PRETRAIN_EPOCHS"],
            pretraining_batch_sample_count=legacy_values["PRETRAIN_BATCH_SIZE"],
        ),
        MODEL_CLUSTERING_CRITERIA=ModelClusteringCriteria(
            maximum_same_cluster_decision_score=legacy_values["FEDSDA_DISTANCE_THRESHOLD"],
            minimum_pair_evaluation_sample_count=legacy_values["CLUSTER_MIN_EVAL_N"],
            clustering_confidence_level=legacy_values["FEDSDA_CLUSTERING_CONFIDENCE"],
            model_pair_comparison_strategy="classwise_unique_correctness_lower_confidence_bound",
        ),
        CROSS_EVALUATION_CLIENT_LIMIT=legacy_values["CROSS_EVAL_MAX_CLIENTS"],
    ).items():
        assert hasattr(whole_run_test_module, constant_name), constant_name
        monkeypatch.setattr(whole_run_test_module, constant_name, constant_value)
    execution_settings = make_execution_settings(
        random_seed=0,
        client_count=legacy_values["N_CLIENTS"],
        per_client_sample_count=legacy_values["TOTAL_DATA_POINTS"],
        aggregation_interval=legacy_values["AGGREGATION_INTERVAL"],
        minimum_change_gap=legacy_values["MIN_STABLE_PERIOD"],
        concept_change_probability=legacy_values["DRIFT_PROB"],
        dataset_name=dataset_name,
    )
    run_participant_settings = make_run_participant_settings(
        valid_run_settings_mapping.__wrapped__(),
        update_interval=legacy_values["LOCAL_UPDATE_INTERVAL"],
        validation_sample_count=legacy_values["NEW_MODEL_FORWARD_VALIDATION_SAMPLES"],
        base_learning_rate=legacy_values["BASE_LR"],
        new_model_learning_rate=legacy_values["NEW_MODEL_LR"],
        stored_evaluation_sample_limit=legacy_values["STORED_DATA_LIMIT"],
        routing_recalibration=legacy_values["SHARED_BACKBONE_ROUTING_RECALIBRATION"],
        added_evaluation_sample_count=legacy_values["EVAL_STORE_SAMPLE_SIZE"],
        cross_evaluation_sample_limit=legacy_values["EVAL_MAX_SAMPLES"],
        cross_evaluation_client_limit=legacy_values["CROSS_EVAL_MAX_CLIENTS"],
    )
    run_metric_settings = RunMetricSettings(
        maximum_detection_delay_sample_count=legacy_values["DELAY_TOLERANCE"],
        post_change_recovery_window_sample_count=legacy_values["STABLE_WINDOW"],
    )
    return execution_settings, run_participant_settings, run_metric_settings


@pytest.fixture(scope="module", params=GOLDEN_DATASET_NAMES)
def golden_condition_runs(request, tmp_path_factory):
    """goldenの条件（datasetごと）で、実旧の`run_random_drift_experiment`（保存つき）と、新の全体run＋導出を、1回ずつ実行する。

    旧の実行は、旧の回帰test（`run_case`）と同じ設定の文脈・同じ引数で行う。呼出し側の乱数は進めない。
    """
    dataset_name = request.param
    assert dataset_name in legacy_regression.CASES
    configuration = ExperimentConfiguration(
        mode=legacy_regression.MODE,
        dataset=dataset_name,
        seed=0,
        concept_schedule="random",
        series="proposed_regression",
        sweep_parameter=None,
        sweep_value=None,
        parameters=(
            ParameterAssignment("aggregation_interval", 50),
            ParameterAssignment("fedsda_distance_threshold", 0.1),
        ),
        algorithm=legacy_regression.ALGORITHM,
    )
    raw_path = tmp_path_factory.mktemp("legacy-raw") / f"{dataset_name}.npz"
    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    previous_thread_count = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            with (
                temporary_config(
                    {
                        **legacy_regression.COMMON,
                        **legacy_regression.CASES[dataset_name],
                    }
                ),
                configuration.activated(),
                redirect_stdout(io.StringIO()),
            ):
                legacy_result = experiment.run_random_drift_experiment(
                    mode=legacy_regression.MODE,
                    random_seed=0,
                    verbose=False,
                    show_plot=False,
                    raw_path=str(raw_path),
                )
                legacy_values = {
                    setting_name: getattr(config, setting_name)
                    for setting_name in dir(config)
                    if setting_name.isupper()
                }
                hidden_layer_widths = tuple(config.dataset_spec().hidden_dims)
                # 旧のdatasetの定義の学習率（mnistは1e-3）は、通常の学習率と、新規モデルの学習率の両方に優先する。
                dataset_learning_rate = config.dataset_spec().learning_rate
                if dataset_learning_rate is not None:
                    legacy_values = legacy_values | dict(
                        BASE_LR=dataset_learning_rate, NEW_MODEL_LR=dataset_learning_rate
                    )
        with np.load(raw_path, allow_pickle=False) as legacy_arrays:
            legacy_trace_arrays = {
                trace_name: legacy_arrays[trace_name] for trace_name in legacy_regression.TRACES
            }
            # 実旧が保存した配列の全部（名前→配列）。
            legacy_saved_arrays = {
                array_name: legacy_arrays[array_name] for array_name in legacy_arrays.files
            }
            # 旧の、ラウンドごとの、モデル数（同期の後の、グローバルモデル数と、クラスタリング待ちのモデル数）。
            legacy_round_model_counts = dict(
                global_model_count=legacy_arrays["round_global_model_count"],
                pending_clustering_count=legacy_arrays["round_pending_clustering_count"],
            )
            # 旧の、ラウンドごとの、clientの計数の増分（形は、ラウンド×client）。
            legacy_round_client_counts = {
                array_name.removeprefix("round_client_"): legacy_arrays[array_name]
                for array_name in legacy_arrays.files
                if array_name.startswith("round_client_") and not array_name.endswith("_seconds")
            }
        # 定数の差し替えは、設定の束を作る間だけ（同じmoduleの、小さい条件のtestへ残さない）。
        with pytest.MonkeyPatch.context() as monkeypatch:
            execution_settings, run_participant_settings, run_metric_settings = (
                make_golden_condition_settings(
                    dataset_name=dataset_name,
                    legacy_values=legacy_values,
                    hidden_layer_widths=hidden_layer_widths,
                    monkeypatch=monkeypatch,
                )
            )
        measured_run = execute_fedsda_stream_protocol_run_with_computation_measurement(
            execution_settings=execution_settings,
            run_participant_settings=run_participant_settings,
        )
        run_result = measured_run.run_result
        participants = measured_run.participants
        yield SimpleNamespace(
            dataset_name=dataset_name,
            measured_run=measured_run,
            legacy_result=legacy_result,
            legacy_trace_arrays=legacy_trace_arrays,
            legacy_round_client_counts=legacy_round_client_counts,
            legacy_round_model_counts=legacy_round_model_counts,
            legacy_saved_arrays=legacy_saved_arrays,
            run_result=run_result,
            participants=participants,
            run_metric_settings=run_metric_settings,
            run_metrics=derive_fedsda_run_metrics(
                run_result=run_result,
                participants=participants,
                run_metric_settings=run_metric_settings,
                model_computation_counts=measured_run.model_computation_counts,
            ),
        )
    finally:
        torch.set_num_threads(previous_thread_count)
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)


def test_golden_condition_metrics_match_real_legacy_experiment_result(golden_condition_runs):
    """goldenの条件で、導出した指標が、実旧の実行結果の33項目（計算量の7項目を含む）と、完全に一致する。"""
    run_metrics = golden_condition_runs.run_metrics
    assert type(run_metrics) is FedsdaRunMetrics
    legacy_metric_values = derive_legacy_metric_values(run_metrics)
    assert set(legacy_metric_values) | set(UNDERIVED_LEGACY_METRIC_NAMES) == set(
        legacy_regression.METRICS
    )
    assert len(legacy_metric_values) == 33
    assert len(UNDERIVED_LEGACY_METRIC_NAMES) == 0
    for legacy_metric_name, metric_value in legacy_metric_values.items():
        assert metric_value == golden_condition_runs.legacy_result[legacy_metric_name], (
            legacy_metric_name
        )
    # 応答の指標（検出の遅れ、警報、変化点の推定、適応の区分、候補の判定）も、実旧の実行結果と一致する。
    assert_change_response_metrics_match_legacy_metrics(
        run_metrics=run_metrics, legacy_metric_values=golden_condition_runs.legacy_result
    )
    assert (
        run_metrics.change_response_metrics.maximum_monitored_log_e_value
        == golden_condition_runs.legacy_result["e_detector_max_log_e"]
    )
    # モデル別の診断の指標（モデル数のまとめ、モデル別の学習量、クラスタリングの判定と真の概念の一致）も、一致する。
    assert_model_diagnostic_metrics_match_legacy_metrics(
        run_metrics=run_metrics,
        legacy_metric_values=golden_condition_runs.legacy_result,
        global_model_counts_are_recorded=True,
    )
    # モデル対の予測の相補性の指標も、一致する。
    assert_model_pair_metrics_match_legacy_metrics(
        run_metrics=run_metrics, legacy_metric_values=golden_condition_runs.legacy_result
    )
    # 予測の診断の指標も、一致する。
    assert_prediction_diagnostic_metrics_match_legacy_metrics(
        run_metrics=run_metrics, legacy_metric_values=golden_condition_runs.legacy_result
    )
    # 旧の回帰testが求める経路（複数のモデル、候補の採用、統合、再較正）を通っている。
    # 旧の回帰testと同じく、すべての経路を求めるのはsine2だけ（datasetごとの期待は、上の表）。
    expected_paths = GOLDEN_CONDITION_PATHS_BY_DATASET_NAME[golden_condition_runs.dataset_name]
    detection_metrics = run_metrics.training_model_switch_detection_metrics
    assert detection_metrics.concept_change_count > 0
    if expected_paths["multiple_global_models"]:
        assert run_metrics.final_global_model_count > 1
        assert run_metrics.fixed_share_weight_recalibration_replayed_sample_count > 0
        assert 0 < detection_metrics.matched_detection_count < detection_metrics.detection_count
    else:
        assert run_metrics.final_global_model_count == 1
    assert (run_metrics.candidate_validation_decision_count > 0) == (
        expected_paths["candidate_decisions"]
    )
    assert (detection_metrics.detection_count > 0) == expected_paths["candidate_decisions"]
    assert (detection_metrics.matched_detection_count > 0) == (
        expected_paths["candidate_decisions"]
    )
    # 計算量: 積和演算の数が、標本数の計数と、層の形から計算した値と一致する。
    assert_multiply_accumulate_counts_match_example_counts(
        model_computation_counts=run_metrics.model_computation_counts,
        participants=golden_condition_runs.participants,
    )
    # 共有部の特徴を使い回すので、共有部を通った標本数は、概念固有部より少ない
    # （clientの保有モデルが1つのままのsea2では、同じ数）。
    shared_part_count = run_metrics.model_computation_counts.shared_part_inference_example_count
    concept_specific_part_count = (
        run_metrics.model_computation_counts.concept_specific_part_inference_example_count
    )
    if expected_paths["multiple_held_models"]:
        assert shared_part_count < concept_specific_part_count
    else:
        assert shared_part_count == concept_specific_part_count
    # 準備の間の計算（事前学習）は、指標に含めない。
    preparation_counts = golden_condition_runs.measured_run.preparation_model_computation_counts
    golden_condition = {
        **legacy_regression.COMMON,
        **legacy_regression.CASES[golden_condition_runs.dataset_name],
    }
    assert preparation_counts.concept_specific_part_training_example_count == (
        golden_condition["PRETRAIN_SAMPLES"] * golden_condition["PRETRAIN_EPOCHS"]
    )
    assert preparation_counts.concept_specific_parameter_optimizer_step_count > 0
    # 定常精度は、精度と違う値（回復の窓が、実際に標本を除いている）。
    assert run_metrics.stable_period_prediction_accuracy != run_metrics.prediction_accuracy


def test_golden_condition_comparison_computation_metrics_match_real_legacy_counts(
    golden_condition_runs,
):
    """goldenの条件で、比較用の計算量（サーバ、保有モデル数、まとめ、ラウンドごと）を、実旧の値と照合する。"""
    run_metrics = golden_condition_runs.run_metrics
    legacy_result = golden_condition_runs.legacy_result
    measured_run = golden_condition_runs.measured_run
    # 集約の積和演算の数は、上りのパラメータの値の数（実旧の通信量）。
    server_computation_counts = run_metrics.server_computation_counts
    assert (
        server_computation_counts.aggregation_parameter_multiply_accumulate_count
        == legacy_result["comm_parameter_values_up"]
    )
    # 統合と、診断のパラメータ距離は、記録と層の形からの計算。goldenの条件は、統合を通る。
    aggregation_count, consolidation_count, diagnostic_distance_count = (
        assert_server_parameter_computation_counts_match_records(
            run_server=golden_condition_runs.participants.server_operations,
            legacy_server=SimpleNamespace(
                comm_parameter_values_up=legacy_result["comm_parameter_values_up"]
            ),
        )
    )
    assert server_computation_counts == ServerComputationCounts(
        aggregation_parameter_multiply_accumulate_count=aggregation_count,
        consolidation_parameter_multiply_accumulate_count=consolidation_count,
        diagnostic_parameter_distance_multiply_accumulate_count=diagnostic_distance_count,
    )
    expected_paths = GOLDEN_CONDITION_PATHS_BY_DATASET_NAME[golden_condition_runs.dataset_name]
    assert (consolidation_count > 0) == expected_paths["multiple_global_models"]
    assert (diagnostic_distance_count > 0) == expected_paths["multiple_global_models"]
    # 処理した標本数と、保有モデル×標本の延べ数（旧の、予測でモデルへ入力した標本の数）。
    per_client_sample_count = legacy_regression.CASES[golden_condition_runs.dataset_name][
        "TOTAL_DATA_POINTS"
    ]
    client_count = len(golden_condition_runs.participants.client_operations)
    assert client_count == legacy_result_client_count(golden_condition_runs.dataset_name)
    processed_sample_count = client_count * per_client_sample_count
    assert run_metrics.processed_sample_count == processed_sample_count
    assert run_metrics.held_model_sample_count == legacy_result["compute_prediction_examples_total"]
    # clientの保有モデルが1つのままなら、延べ数は、処理した標本数と同じ。
    assert (
        run_metrics.held_model_sample_count > run_metrics.processed_sample_count
    ) == expected_paths["multiple_held_models"]
    assert run_metrics.held_model_sample_count >= run_metrics.processed_sample_count
    # まとめ: 計数からの計算と一致する。
    model_computation_counts = run_metrics.model_computation_counts
    computation_cost_summary = run_metrics.computation_cost_summary
    assert type(computation_cost_summary) is ComputationCostSummary
    assert computation_cost_summary == summarize_computation_cost(
        model_computation_counts=model_computation_counts,
        server_computation_counts=server_computation_counts,
        processed_sample_count=run_metrics.processed_sample_count,
        held_model_sample_count=run_metrics.held_model_sample_count,
    )
    forward_count = (
        model_computation_counts.shared_part_training_forward_multiply_accumulate_count
        + model_computation_counts.shared_part_inference_forward_multiply_accumulate_count
        + model_computation_counts.concept_specific_part_training_forward_multiply_accumulate_count
        + model_computation_counts.concept_specific_part_inference_forward_multiply_accumulate_count
    )
    assert computation_cost_summary.client_forward_multiply_accumulate_count == forward_count
    assert (
        computation_cost_summary.client_forward_multiply_accumulate_count_per_processed_sample
        == (forward_count / processed_sample_count)
    )
    assert (
        computation_cost_summary.client_forward_multiply_accumulate_count_per_held_model_sample
        == forward_count / run_metrics.held_model_sample_count
    )
    assert computation_cost_summary.mean_held_model_count == (
        run_metrics.held_model_sample_count / processed_sample_count
    )
    # 概念固有部だけの、保有モデル×標本あたりの値: 予測では、保有モデル×標本ごとに、概念固有部を1回通す。
    # ほかの処理（検出、統計、学習ほか）の分があるので、1回ぶんの積和演算の数より大きい。
    concept_specific_macs_per_example = (
        model_computation_counts.concept_specific_part_inference_forward_multiply_accumulate_count
        // model_computation_counts.concept_specific_part_inference_example_count
    )
    assert (
        computation_cost_summary.client_concept_specific_part_forward_multiply_accumulate_count_per_held_model_sample
        > concept_specific_macs_per_example
    )
    assert (
        computation_cost_summary.client_shared_part_forward_multiply_accumulate_count_per_processed_sample
        == computation_cost_summary.client_shared_part_forward_multiply_accumulate_count
        / processed_sample_count
    )
    assert computation_cost_summary.server_multiply_accumulate_count == (
        aggregation_count + consolidation_count
    )
    # ラウンドごとのモデルの計算: 実旧の、ラウンドごとの計数の増分の、全clientの合計。
    legacy_round_counts = {
        counter_name: counter_values.sum(axis=1).tolist()
        for counter_name, counter_values in golden_condition_runs.legacy_round_client_counts.items()
    }
    round_counts = measured_run.round_model_computation_counts
    assert (
        len(round_counts)
        == len(legacy_round_counts["head_examples"])
        == per_client_sample_count // 50
    )
    for each_round in round_counts:
        round_index = each_round.round_index
        local_counts = each_round.local_processing_model_computation_counts
        synchronization_counts = each_round.synchronization_model_computation_counts

        def legacy_round_sum(*counter_names, round_index=round_index):
            return sum(
                legacy_round_counts[counter_name][round_index] for counter_name in counter_names
            )

        # ローカルの処理: 予測・検出・統計・初期化（推論）と、学習。
        assert local_counts.concept_specific_part_inference_example_count == legacy_round_sum(
            "prediction_examples",
            "detection_examples",
            "statistics_examples",
            "initialization_examples",
        )
        assert local_counts.concept_specific_part_training_example_count == legacy_round_sum(
            "training_examples"
        )
        assert local_counts.concept_specific_parameter_optimizer_step_count == legacy_round_sum(
            "optimizer_steps"
        )
        assert local_counts.shared_parameter_optimizer_step_count == legacy_round_sum(
            "backbone_optimizer_steps"
        )
        # 同期: クロス評価と、再較正（どちらも推論）。
        assert (
            synchronization_counts.concept_specific_part_inference_example_count
            == legacy_round_sum("cross_evaluation_examples", "routing_recalibration_examples")
        )
        assert synchronization_counts.concept_specific_part_training_example_count == 0
        # 共有部と概念固有部の合計（旧は、用途別に分けていない）。
        assert (
            local_counts.shared_part_training_example_count
            + local_counts.shared_part_inference_example_count
            + synchronization_counts.shared_part_inference_example_count
        ) == legacy_round_sum("backbone_examples")
        assert (
            local_counts.concept_specific_part_training_example_count
            + local_counts.concept_specific_part_inference_example_count
            + synchronization_counts.concept_specific_part_inference_example_count
        ) == legacy_round_sum("head_examples")
    # 同期の間のclientの計算が、実際にある（クロス評価か再較正。clientの保有モデルが1つのままのsea2では、ない）。
    assert (
        any(
            each_round.synchronization_model_computation_counts.concept_specific_part_inference_example_count
            > 0
            for each_round in round_counts
        )
        == expected_paths["synchronization_computation"]
    )
    # 終端の処理（旧のラウンドの値に入らない分）: 旧の累計−ラウンドの合計。
    assert (
        measured_run.finalization_model_computation_counts.concept_specific_part_inference_example_count
        == legacy_result["compute_head_examples_total"] - sum(legacy_round_counts["head_examples"])
    )
    # 標本ごとの保有モデル数の、ラウンドごとの合計は、旧の、ラウンドごとの予測の計数（clientごと）。
    clients_with_changing_held_model_count = 0
    for client_index, run_client in enumerate(golden_condition_runs.participants.client_operations):
        held_model_counts = (
            run_client.owners.held_model_count_record_store.snapshot_held_model_counts()
        )
        assert [
            sum(held_model_counts[round_start : round_start + 50])
            for round_start in range(0, per_client_sample_count, 50)
        ] == golden_condition_runs.legacy_round_client_counts["prediction_examples"][
            :, client_index
        ].tolist()
        clients_with_changing_held_model_count += len(set(held_model_counts)) > 1
    assert (clients_with_changing_held_model_count > 0) == expected_paths["multiple_held_models"]


def test_golden_condition_trace_arrays_match_real_legacy_saved_arrays(golden_condition_runs):
    """goldenの条件で、新の記録から作った31の離散列が、実旧が保存した配列と、形・型・値で一致する。"""
    trace_arrays = derive_legacy_trace_arrays(
        run_result=golden_condition_runs.run_result,
        participants=golden_condition_runs.participants,
    )
    assert tuple(trace_arrays) == legacy_regression.TRACES
    assert len(trace_arrays) == 31
    for trace_name, trace_values in trace_arrays.items():
        legacy_trace_values = golden_condition_runs.legacy_trace_arrays[trace_name]
        assert trace_values.shape == legacy_trace_values.shape, trace_name
        assert trace_values.dtype == legacy_trace_values.dtype, trace_name
        assert np.array_equal(trace_values, legacy_trace_values), trace_name
    assert trace_arrays["drift_positions"].size > 0
    # 列が、中身のある比較になっている（旧の回帰testの経路の件数。datasetごとの期待は、上の表）。
    expected_paths = GOLDEN_CONDITION_PATHS_BY_DATASET_NAME[golden_condition_runs.dataset_name]
    if expected_paths["multiple_global_models"]:
        assert trace_arrays["model_registration_ids"].size > 1
        assert trace_arrays["clustering_absorbed"].sum() > 0
        assert trace_arrays["clustering_pair_same_cluster"].size > 0
    else:
        assert trace_arrays["model_registration_ids"].size == 1
    if expected_paths["candidate_decisions"]:
        assert trace_arrays["provisional_accepted"].sum() > 0
        assert (~trace_arrays["provisional_accepted"]).sum() > 0
        assert trace_arrays["switch_positions"].size > 0
        assert len(set(trace_arrays["adaptation_actions"].tolist())) > 2
        assert len(set(trace_arrays["provisional_reasons"].tolist())) > 1
    else:
        # 警報のたびに現行モデルを維持する（候補の判定まで進まない）。
        assert set(trace_arrays["adaptation_actions"].tolist()) == {"maintain"}


def test_golden_condition_metrics_and_traces_match_environment_golden(golden_condition_runs):
    """導出した指標と列が、実行環境のgoldenと一致する。

    goldenは、実行環境に依存する（旧実装の結果自体が、OSや、Python・NumPyの版で違う）。実行環境の記録と
    完全に一致するgolden（固定のWindows用のgoldenか、環境ごとのgolden）を選ぶ。なければ、照合しない。
    同じprocessの中の実旧との照合（上の2つ）は、どの環境でも行う。
    """
    environment = legacy_regression.environment()
    golden_path = find_golden_path(environment)
    if golden_path is None:
        pytest.skip(describe_missing_golden(environment))
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert golden["_env"] == environment
    golden_case = golden["cases"][golden_condition_runs.dataset_name]
    legacy_metric_values = derive_legacy_metric_values(golden_condition_runs.run_metrics)
    assert set(golden_case["metrics"]) == set(legacy_metric_values) | set(
        UNDERIVED_LEGACY_METRIC_NAMES
    )
    for legacy_metric_name, metric_value in legacy_metric_values.items():
        assert float(metric_value) == pytest.approx(
            golden_case["metrics"][legacy_metric_name], rel=0, abs=1e-9
        ), legacy_metric_name
    trace_arrays = derive_legacy_trace_arrays(
        run_result=golden_condition_runs.run_result,
        participants=golden_condition_runs.participants,
    )
    assert set(golden_case["traces"]) == set(trace_arrays)
    for trace_name, trace_values in trace_arrays.items():
        # 旧の回帰test（`run_case`）と同じ、形と、正規化したJSONのSHA-256。
        serialized_values = json.dumps(
            trace_values.tolist(), separators=(",", ":"), ensure_ascii=True
        )
        assert golden_case["traces"][trace_name] == dict(
            shape=list(trace_values.shape),
            sha256=hashlib.sha256(serialized_values.encode()).hexdigest(),
        ), trace_name


def test_golden_condition_decision_kinds_are_reported(golden_condition_runs):
    """goldenの条件の全体runが通った、候補の判定の種類（別の保有モデルの再利用を含むか）を確かめる。"""
    trace_arrays = derive_legacy_trace_arrays(
        run_result=golden_condition_runs.run_result,
        participants=golden_condition_runs.participants,
    )
    observed_reasons = set(trace_arrays["provisional_reasons"].tolist())
    assert (
        observed_reasons
        == GOLDEN_CONDITION_DECISION_REASONS_BY_DATASET_NAME[golden_condition_runs.dataset_name]
    ), sorted(observed_reasons)


# goldenの条件の全体runが通る、候補の判定の理由（実旧の名前）。
# sine2は、別の保有モデルの再利用（alternative_reference_refit）を含む。sea2は、警報のたびに現行モデルを維持し、候補の判定まで進まない。
# mnist2は、候補の採用と、後半の区間での棄却。
GOLDEN_CONDITION_DECISION_REASONS_BY_DATASET_NAME = {
    "sine2": {"accepted", "alternative_reference_refit", "first_interval"},
    "sea2": set(),
    "mnist2": {"accepted", "second_interval"},
}


def test_derivation_does_not_change_participant_state_and_is_repeatable(golden_condition_runs):
    """導出は、参加者の状態と乱数を変えず、同じ結果を返す。"""
    participants = golden_condition_runs.participants

    def snapshot_all_states():
        server_owners = participants.server_operations.owners
        return (
            [
                repr(
                    snapshot_run_client_state(
                        run_client=run_client,
                        python_random_generator=run_client._python_random_generator,
                    )
                )
                for run_client in participants.client_operations
            ],
            [
                run_client.owners.candidate_validation_decision_record_store.snapshot_candidate_validation_decision_records()
                for run_client in participants.client_operations
            ],
            server_owners.communication_volume_record_store.get_state_snapshot(),
            server_owners.global_model_repository.global_model_ids,
            server_owners.global_model_repository.snapshot_model_registration_records(),
            server_owners.model_clustering_record_store.snapshot_model_clustering_observations(),
            random.getstate(),
            torch.get_rng_state().tolist(),
        )

    states_before = snapshot_all_states()
    repeated_run_metrics = derive_fedsda_run_metrics(
        run_result=golden_condition_runs.run_result,
        participants=participants,
        run_metric_settings=golden_condition_runs.run_metric_settings,
        model_computation_counts=golden_condition_runs.measured_run.model_computation_counts,
    )
    assert snapshot_all_states() == states_before
    assert repeated_run_metrics == golden_condition_runs.run_metrics
    # 計測なしの導出は、モデルの計算の計数を「なし」として、ほかの指標を同じに返す。
    unmeasured_run_metrics = derive_fedsda_run_metrics(
        run_result=golden_condition_runs.run_result,
        participants=participants,
        run_metric_settings=golden_condition_runs.run_metric_settings,
    )
    assert unmeasured_run_metrics.model_computation_counts is None
    assert unmeasured_run_metrics.computation_cost_summary is None
    assert unmeasured_run_metrics == replace(
        golden_condition_runs.run_metrics,
        model_computation_counts=None,
        computation_cost_summary=None,
    )


# 小さい条件: (seed, client数, 標本数, 集約間隔, 最小の変更間隔, 変更確率, 学習の間隔)。
# 統合が起きる条件、終端で候補検証が回収される条件、処理されない末尾がある条件（末尾に変更がありうる）、
# 候補が区間の判定で棄却される条件（全体runの対照の条件から、実旧だけの実行で理由を調べて選んだ）。
SMALL_RUN_CONDITIONS = [
    (7, 3, 300, 10, 30, 0.05, 2),
    (17, 3, 250, 10, 30, 0.05, 2),
    (23, 2, 300, 10, 25, 0.06, 2),
    (1, 3, 330, 50, 30, 0.05, 1),
    (3, 2, 137, 30, 3, 0.2, 1),
    (2, 5, 300, 10, 50, 0.03, 1),
    (0, 3, 500, 25, 60, 0.03, 2),
]
SMALL_RUN_OBSERVATIONS = {}
# 条件ごとの、実旧の全体runと、新の全体run（指標の設定を変えたtestで、使い回す）。
SMALL_RUNS_BY_CONDITION = {}


# 旧の離散列のうち、重みで結合して予測するclientだけが保存するもの。
LEGACY_TRACE_NAMES_RECORDED_ONLY_WITH_PREDICTION_WEIGHTS = (
    "history_routing_switching_correct",
    "history_routing_switching_leader_id",
    "history_routing_soft_active",
)


# 実験の構成（照合の単位）: 名前、旧の構成名、旧の設定の上書き、新の参加者の設定の変更。提案構成は、何も変えない。
PROPOSED_CONFIGURATION_VARIANT = SimpleNamespace(
    name="fedsda",
    legacy_mode_name=whole_run_test_module.LEGACY_MODE_NAME,
    legacy_setting_overrides={},
    modify_run_participant_settings=lambda run_participant_settings: run_participant_settings,
)
# 照合のたびに、旧の値を積む一覧（対応表の取り違えの検査が読む）。
LEGACY_METRIC_VALUE_OBSERVATION_LISTS = (
    LEGACY_CHANGE_RESPONSE_METRIC_VALUE_OBSERVATIONS,
    LEGACY_MODEL_DIAGNOSTIC_METRIC_VALUE_OBSERVATIONS,
    LEGACY_MODEL_PAIR_METRIC_VALUE_OBSERVATIONS,
    LEGACY_PREDICTION_DIAGNOSTIC_METRIC_VALUE_OBSERVATIONS,
)


def compare_small_run_with_real_legacy(
    *,
    monkeypatch,
    tmp_path,
    valid_run_settings_mapping,
    random_seed,
    client_count,
    per_client_sample_count,
    aggregation_interval,
    minimum_change_gap,
    concept_change_probability,
    update_interval,
    maximum_delay,
    recovery_window,
    configuration_variant=None,
):
    """小さい条件で、導出した指標と離散列が、実旧の全体runの後に、旧の指標の計算と保存を呼んだ結果と一致する。

    `configuration_variant`（名前、旧の構成名、旧の設定の上書き、新の参加者の設定の変更）で、提案構成以外の構成を照合できる。
    戻り値: 指標、離散列、通った場合の観測、実旧の全体run、新の参加者。
    """
    if configuration_variant is None:
        configuration_variant = PROPOSED_CONFIGURATION_VARIANT
    # 提案構成以外の照合は、対応表の取り違えの検査（提案構成の条件の数を前提にしている）の観測へ、足さない。
    value_observation_counts = [
        len(value_observations) for value_observations in LEGACY_METRIC_VALUE_OBSERVATION_LISTS
    ]
    condition = (
        random_seed,
        client_count,
        per_client_sample_count,
        aggregation_interval,
        minimum_change_gap,
        concept_change_probability,
        update_interval,
    )
    run_key = (configuration_variant.name, condition)
    if run_key not in SMALL_RUNS_BY_CONDITION:
        execution_settings = make_execution_settings(
            random_seed=random_seed,
            client_count=client_count,
            per_client_sample_count=per_client_sample_count,
            aggregation_interval=aggregation_interval,
            minimum_change_gap=minimum_change_gap,
            concept_change_probability=concept_change_probability,
        )
        legacy_run = run_real_legacy_whole_run(
            monkeypatch=monkeypatch,
            execution_settings=execution_settings,
            update_interval=update_interval,
            legacy_mode_name=configuration_variant.legacy_mode_name,
            legacy_setting_overrides=configuration_variant.legacy_setting_overrides,
        )
        measured_run = execute_fedsda_stream_protocol_run_with_computation_measurement(
            execution_settings=execution_settings,
            run_participant_settings=configuration_variant.modify_run_participant_settings(
                make_run_participant_settings(
                    valid_run_settings_mapping, update_interval=update_interval
                )
            ),
        )
        run_result, participants = measured_run.run_result, measured_run.participants
        # 実旧の保存処理を、実旧の全体runの結果へ呼んで、旧の保存形式の配列を得る（設定の差し替えが有効な間に）。
        legacy_raw_path = tmp_path / "legacy-small-run.npz"
        experiment._save_raw_run(
            str(legacy_raw_path),
            legacy_run["legacy_clients"],
            legacy_run["legacy_server"],
            extract_true_drift_events(legacy_run["legacy_concept_schedules"]),
            configuration_variant.legacy_mode_name,
            configuration_variant.legacy_mode_name,
            random_seed,
            experiment._new_round_telemetry(),
            client_test_module.MAXIMUM_TOLERATED_MEAN_LOSS_INCREASE,
        )
        with np.load(legacy_raw_path, allow_pickle=False) as legacy_arrays:
            # 重みを持たない構成の実旧は、重みでの結合の列を、保存しない（下で、その列だけが無いことを確かめる）。
            legacy_trace_arrays = {
                trace_name: legacy_arrays[trace_name]
                for trace_name in legacy_regression.TRACES
                if trace_name in legacy_arrays.files
            }
        SMALL_RUNS_BY_CONDITION[run_key] = (
            legacy_run,
            run_result,
            participants,
            legacy_trace_arrays,
            measured_run.model_computation_counts,
        )
    legacy_run, run_result, participants, legacy_trace_arrays, model_computation_counts = (
        SMALL_RUNS_BY_CONDITION[run_key]
    )
    # 新の記録から作った31の離散列が、実旧の保存処理が作った配列と、形・型・値で一致する。
    trace_arrays = derive_legacy_trace_arrays(run_result=run_result, participants=participants)
    assert tuple(trace_arrays) == legacy_regression.TRACES
    # 構成が指定しなければ、実旧は、重みと、予測の内訳の記録を持つ。
    legacy_records_prediction_weight_diagnostics = getattr(
        configuration_variant, "legacy_records_prediction_weight_diagnostics", True
    )
    assert set(legacy_regression.TRACES) - set(legacy_trace_arrays) == (
        set()
        if legacy_records_prediction_weight_diagnostics
        else set(LEGACY_TRACE_NAMES_RECORDED_ONLY_WITH_PREDICTION_WEIGHTS)
    )
    for trace_name, trace_values in trace_arrays.items():
        if trace_name not in legacy_trace_arrays:
            continue
        legacy_trace_values = legacy_trace_arrays[trace_name]
        assert trace_values.shape == legacy_trace_values.shape, trace_name
        # 空の文字列の配列は、要素の長さが決まらないので、型は、種類で比べる。
        assert trace_values.dtype.kind == legacy_trace_values.dtype.kind, trace_name
        if trace_values.size:
            assert trace_values.dtype == legacy_trace_values.dtype, trace_name
        assert np.array_equal(trace_values, legacy_trace_values), trace_name
    legacy_clients = legacy_run["legacy_clients"]
    legacy_server = legacy_run["legacy_server"]
    legacy_true_drift_events = extract_true_drift_events(legacy_run["legacy_concept_schedules"])
    legacy_metrics = compute_metrics(
        legacy_clients,
        legacy_true_drift_events,
        delay_tolerance=maximum_delay,
        stable_window=recovery_window,
    )
    legacy_parameter_value_count, legacy_parameter_byte_count = (
        legacy_server.final_parameter_footprint()
    )
    run_metrics = derive_fedsda_run_metrics(
        run_result=run_result,
        participants=participants,
        run_metric_settings=RunMetricSettings(
            maximum_detection_delay_sample_count=maximum_delay,
            post_change_recovery_window_sample_count=recovery_window,
        ),
        model_computation_counts=model_computation_counts,
    )
    # 計算量の7項目: 実旧の指標の集計（全clientの計数の合計から）。
    legacy_computation_results = {}
    experiment._add_telemetry_results(
        legacy_computation_results, legacy_clients, experiment._new_round_telemetry()
    )
    # 監視の値の最大が、実旧の集計と一致する。構成が指定しなければ、実旧は、監視の値（e値）を持つ。
    if getattr(configuration_variant, "legacy_records_monitored_log_e_values", True):
        assert (
            run_metrics.change_response_metrics.maximum_monitored_log_e_value
            == legacy_computation_results["e_detector_max_log_e"]
        )
    else:
        # e値を持たない検出器（ADWIN）の実旧は、この指標を作らない。新は、負の無限大。
        assert "e_detector_max_log_e" not in legacy_computation_results
        assert all(
            not hasattr(legacy_client, "history_detector_log_e") for legacy_client in legacy_clients
        )
        assert run_metrics.change_response_metrics.maximum_monitored_log_e_value == -inf
    # 応答の指標が、実旧の`compute_metrics`の戻り値と、全部の項目で一致する。
    assert_change_response_metrics_match_legacy_metrics(
        run_metrics=run_metrics, legacy_metric_values=legacy_metrics
    )
    # モデル対の予測の相補性の指標が、実旧のサーバの集計と一致する。
    assert_model_pair_metrics_match_legacy_metrics(
        run_metrics=run_metrics, legacy_metric_values=legacy_server.pair_diagnostic_summary()
    )
    # clientの保有モデルの順が、実旧の`client.models`の順と同じ（モデル別の学習量を、モデルの間でまとめるときの、値の順）。
    for run_client, legacy_client in zip(
        participants.client_operations, legacy_clients, strict=True
    ):
        assert [
            held_model_training_state.model_id
            for held_model_training_state in run_client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        ] == list(legacy_client.models)
    # モデル別の学習量と、クラスタリングの判定と真の概念の一致が、実旧の集計と一致する。
    # 旧の、回復の区間の窓は、設定の値から取る（この条件の値にする）。
    monkeypatch.setattr(config, "STABLE_WINDOW", recovery_window)
    legacy_model_diagnostic_results = {}
    experiment._add_model_diagnostic_results(
        legacy_model_diagnostic_results, legacy_clients, legacy_server, legacy_true_drift_events
    )
    assert_model_diagnostic_metrics_match_legacy_metrics(
        run_metrics=run_metrics,
        legacy_metric_values=legacy_model_diagnostic_results,
        global_model_counts_are_recorded=False,
    )
    if legacy_records_prediction_weight_diagnostics:
        # 予測の診断の指標が、実旧の集計と一致する。
        assert_prediction_diagnostic_metrics_match_legacy_metrics(
            run_metrics=run_metrics,
            legacy_metric_values=legacy_model_diagnostic_results,
            # 構成が指定しなければ、Fixed-Shareの重みで予測する構成。
            fixed_share_weights_are_used_for_prediction=getattr(
                configuration_variant, "fixed_share_weights_are_used_for_prediction", True
            ),
        )
    else:
        # 実旧は、予測の内訳を数えていない（旧の指標は、標本数0からの値）。新の、予測の内訳の指標は、照合しない。
        # 予測の正否そのもの（`history_accuracy`）と、予測したモデル（`history_model_id`）は、上の離散列で照合している。
        assert legacy_model_diagnostic_results["routing_sample_count"] == 0
        assert all(
            not hasattr(legacy_client, "routing_diagnostics") for legacy_client in legacy_clients
        )
    legacy_metric_values = derive_legacy_metric_values(run_metrics)
    # 構成が指定しなければ、実旧のモデルは、共有部を持つ。
    legacy_models_share_feature_extractor = getattr(
        configuration_variant, "legacy_models_share_feature_extractor", True
    )
    if not legacy_models_share_feature_extractor:
        # 共有部を持たない実旧のモデルは、共有部の標本数として、概念固有部と同じ数（モデル×標本の延べ数）を数える。
        # 新は、共有部が無いので、共有部の計数は0。旧の値は、新の、概念固有部の標本数と照合する。
        assert (
            legacy_computation_results["compute_backbone_examples_total"]
            == legacy_computation_results["compute_head_examples_total"]
        )
        assert legacy_metric_values["compute_backbone_examples_total"] == 0
        legacy_metric_values["compute_backbone_examples_total"] = legacy_metric_values[
            "compute_head_examples_total"
        ]
    for legacy_metric_name in legacy_regression.METRICS:
        if legacy_metric_name.startswith("compute_"):
            assert (
                legacy_metric_values[legacy_metric_name]
                == legacy_computation_results[legacy_metric_name]
            ), legacy_metric_name
    assert_multiply_accumulate_counts_match_example_counts(
        model_computation_counts=model_computation_counts,
        participants=participants,
        models_share_feature_extractor=legacy_models_share_feature_extractor,
    )
    # サーバの計数（集約＝実旧の上りのパラメータの値の数。統合と距離は、記録と層の形から）と、保有モデル数。
    aggregation_count, consolidation_count, diagnostic_distance_count = (
        assert_server_parameter_computation_counts_match_records(
            run_server=participants.server_operations, legacy_server=legacy_server
        )
    )
    assert run_metrics.server_computation_counts == ServerComputationCounts(
        aggregation_parameter_multiply_accumulate_count=aggregation_count,
        consolidation_parameter_multiply_accumulate_count=consolidation_count,
        diagnostic_parameter_distance_multiply_accumulate_count=diagnostic_distance_count,
    )
    # 保有モデル×標本の延べ数。重みで結合して予測する実旧では、予測で評価した標本の数と同じ。
    # 重みを持たない実旧は、標本ごとに、現在のモデルだけを評価する。
    legacy_prediction_example_count = sum(
        legacy_client.compute_counters["prediction_examples"] for legacy_client in legacy_clients
    )
    if legacy_records_prediction_weight_diagnostics:
        assert run_metrics.held_model_sample_count == legacy_prediction_example_count
    else:
        assert legacy_prediction_example_count == (
            client_count * run_result.processed_sample_count_per_client
        )
        assert run_metrics.held_model_sample_count == sum(
            sum(run_client.owners.held_model_count_record_store.snapshot_held_model_counts())
            for run_client in participants.client_operations
        )
    assert run_metrics.processed_sample_count == (
        client_count * run_result.processed_sample_count_per_client
    )
    assert run_metrics.computation_cost_summary == summarize_computation_cost(
        model_computation_counts=model_computation_counts,
        server_computation_counts=run_metrics.server_computation_counts,
        processed_sample_count=run_metrics.processed_sample_count,
        held_model_sample_count=run_metrics.held_model_sample_count,
    )
    assert run_metrics.prediction_accuracy == legacy_metrics["accuracy"]
    if isnan(legacy_metrics["stable_accuracy"]):
        assert isnan(run_metrics.stable_period_prediction_accuracy)
    else:
        assert run_metrics.stable_period_prediction_accuracy == legacy_metrics["stable_accuracy"]
    assert run_metrics.training_model_switch_detection_metrics == DetectionMetrics(
        detection_precision=legacy_metrics["precision"],
        detection_recall=legacy_metrics["recall"],
        detection_f1=legacy_metrics["f1"],
        detection_count=legacy_metrics["total_detect"],
        concept_change_count=legacy_metrics["total_true"],
        matched_detection_count=legacy_metrics["tp"],
    )
    assert run_metrics.final_global_model_count == len(legacy_server.global_models)
    assert run_metrics.communication_volume == CommunicationVolumeSnapshot(
        uploaded_model_count=legacy_server.comm_models_up,
        downloaded_model_count=legacy_server.comm_models_down,
        uploaded_message_count=legacy_server.comm_messages_up,
        downloaded_message_count=legacy_server.comm_messages_down,
        uploaded_parameter_value_count=legacy_server.comm_parameter_values_up,
        downloaded_parameter_value_count=legacy_server.comm_parameter_values_down,
        uploaded_byte_count=legacy_server.comm_bytes_up,
        downloaded_byte_count=legacy_server.comm_bytes_down,
    )
    assert run_metrics.final_parameter_value_count == legacy_parameter_value_count
    assert run_metrics.final_parameter_byte_count == legacy_parameter_byte_count
    assert (
        run_metrics.candidate_validation_decision_count
        == legacy_metrics["provisional_proposal_count"]
        == legacy_metrics["provisional_forward_count"]
    )
    if legacy_records_prediction_weight_diagnostics:
        assert run_metrics.mixed_prediction_sample_count == sum(
            legacy_client.soft_routing_activation.soft_sample_count
            for legacy_client in legacy_clients
        )
        assert run_metrics.fixed_share_weight_recalibration_replayed_sample_count == sum(
            legacy_client.switching_expert_router.aggregation_recalibration_sample_count
            for legacy_client in legacy_clients
        )
        assert run_metrics.adahedge_weight_recalibration_replayed_sample_count == sum(
            legacy_client.expert_router.aggregation_recalibration_sample_count
            for legacy_client in legacy_clients
        )
    else:
        # 重みを持たない実旧: 結合して予測した標本も、重みの再較正も、無い。
        assert all(
            not hasattr(legacy_client, "soft_routing_activation")
            and not hasattr(legacy_client, "switching_expert_router")
            and not hasattr(legacy_client, "expert_router")
            for legacy_client in legacy_clients
        )
        assert run_metrics.mixed_prediction_sample_count == 0
        assert run_metrics.fixed_share_weight_recalibration_replayed_sample_count == 0
        assert run_metrics.adahedge_weight_recalibration_replayed_sample_count == 0
    processed_sample_count = run_result.processed_sample_count_per_client
    small_run_observation = dict(
        unprocessed_tail=run_result.unprocessed_tail_sample_count_per_client > 0,
        concept_change_in_unprocessed_tail=any(
            sample_index >= processed_sample_count
            for sample_indices in legacy_true_drift_events.values()
            for sample_index in sample_indices
        ),
        multiple_global_models=run_metrics.final_global_model_count > 1,
        matched_detection=run_metrics.training_model_switch_detection_metrics.matched_detection_count
        > 0,
        unmatched_detection=(
            run_metrics.training_model_switch_detection_metrics.matched_detection_count
            < run_metrics.training_model_switch_detection_metrics.detection_count
        ),
        missed_concept_change=(
            run_metrics.training_model_switch_detection_metrics.matched_detection_count
            < run_metrics.training_model_switch_detection_metrics.concept_change_count
        ),
        stable_accuracy_differs=(
            run_metrics.stable_period_prediction_accuracy != run_metrics.prediction_accuracy
        ),
        # 終端で回収された、未完了の候補検証の判定（列では、理由が「前向きの標本が足りない」）。
        incomplete_candidate_validation_decision=(
            "insufficient_forward_data" in trace_arrays["provisional_reasons"].tolist()
        ),
        candidate_rejected_by_segment_margin=bool(
            {"first_interval", "second_interval", "first_and_second"}
            & set(trace_arrays["provisional_reasons"].tolist())
        ),
        models_absorbed=bool(trace_arrays["clustering_absorbed"].any()),
        server_remapped_adaptation="server_merge" in trace_arrays["adaptation_actions"].tolist(),
    )
    if configuration_variant is PROPOSED_CONFIGURATION_VARIANT:
        SMALL_RUN_OBSERVATIONS[
            (random_seed, per_client_sample_count, maximum_delay, recovery_window)
        ] = small_run_observation
    else:
        for value_observations, value_observation_count in zip(
            LEGACY_METRIC_VALUE_OBSERVATION_LISTS, value_observation_counts, strict=True
        ):
            del value_observations[value_observation_count:]
    return SimpleNamespace(
        run_metrics=run_metrics,
        trace_arrays=trace_arrays,
        small_run_observation=small_run_observation,
        legacy_run=legacy_run,
        participants=participants,
        run_result=run_result,
    )


@pytest.mark.parametrize(
    (
        "random_seed",
        "client_count",
        "per_client_sample_count",
        "aggregation_interval",
        "minimum_change_gap",
        "concept_change_probability",
        "update_interval",
    ),
    SMALL_RUN_CONDITIONS,
)
@pytest.mark.parametrize(("maximum_delay", "recovery_window"), [(100, 50), (8, 5)])
def test_small_run_metrics_match_real_legacy_metric_computation(
    monkeypatch,
    tmp_path,
    valid_run_settings_mapping,
    random_seed,
    client_count,
    per_client_sample_count,
    aggregation_interval,
    minimum_change_gap,
    concept_change_probability,
    update_interval,
    maximum_delay,
    recovery_window,
):
    """小さい条件で、導出した指標が、実旧の全体runの後に、旧の指標の計算を呼んだ結果と一致する（提案構成）。"""
    compare_small_run_with_real_legacy(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        valid_run_settings_mapping=valid_run_settings_mapping,
        random_seed=random_seed,
        client_count=client_count,
        per_client_sample_count=per_client_sample_count,
        aggregation_interval=aggregation_interval,
        minimum_change_gap=minimum_change_gap,
        concept_change_probability=concept_change_probability,
        update_interval=update_interval,
        maximum_delay=maximum_delay,
        recovery_window=recovery_window,
    )


def test_small_run_conditions_cover_required_cases():
    """上の対照が、処理されない末尾（その中の変更を含む）、対応する検出・しない検出、見逃し、未完了の候補検証の判定、区間の判定での棄却、統合を通っている。"""
    if len(SMALL_RUN_OBSERVATIONS) < 2 * len(SMALL_RUN_CONDITIONS):
        pytest.skip("対照の全条件を実行したときだけ確かめる")
    for observation_name in next(iter(SMALL_RUN_OBSERVATIONS.values())):
        assert any(
            observation[observation_name] for observation in SMALL_RUN_OBSERVATIONS.values()
        ), observation_name


def test_derivation_rejects_invalid_arguments(golden_condition_runs):
    """参加者・実行の結果・設定の型、clientの数と順の不一致を拒否する。"""
    run_result = golden_condition_runs.run_result
    participants = golden_condition_runs.participants
    valid_arguments = dict(
        run_result=run_result,
        participants=participants,
        run_metric_settings=golden_condition_runs.run_metric_settings,
        model_computation_counts=golden_condition_runs.measured_run.model_computation_counts,
    )
    invalid_argument_cases = [
        ("run_result", SimpleNamespace(), TypeError),
        ("run_result", None, TypeError),
        ("participants", SimpleNamespace(), TypeError),
        ("run_metric_settings", dict(maximum_detection_delay_sample_count=1), TypeError),
        ("run_metric_settings", None, TypeError),
        ("model_computation_counts", dict(shared_part_training_example_count=1), TypeError),
        ("model_computation_counts", 0, TypeError),
        # clientの操作が、FedSDAのclientそのものでない。
        (
            "participants",
            RunParticipants(
                client_operations=(
                    SimpleNamespace(client_id=0),
                    *participants.client_operations[1:],
                ),
                server_operations=participants.server_operations,
            ),
            TypeError,
        ),
        # サーバの操作が、FedSDAのサーバそのものでない。
        (
            "participants",
            RunParticipants(
                client_operations=participants.client_operations,
                server_operations=SimpleNamespace(),
            ),
            TypeError,
        ),
        # clientの数が、概念列の数と合わない。
        (
            "participants",
            RunParticipants(
                client_operations=participants.client_operations[:-1],
                server_operations=participants.server_operations,
            ),
            ValueError,
        ),
        # clientの順が、概念列の順と合わない。
        (
            "participants",
            RunParticipants(
                client_operations=(
                    participants.client_operations[1],
                    participants.client_operations[0],
                    *participants.client_operations[2:],
                ),
                server_operations=participants.server_operations,
            ),
            ValueError,
        ),
    ]
    for argument_name, invalid_value, expected_exception_type in invalid_argument_cases:
        with pytest.raises(expected_exception_type):
            derive_fedsda_run_metrics(**valid_arguments | {argument_name: invalid_value})
    with pytest.raises(TypeError):
        derive_fedsda_run_metrics(
            run_result, participants, golden_condition_runs.run_metric_settings
        )
    # 設定を変えると、検出の指標と、応答の指標と、定常精度だけが変わる。
    strict_run_metrics = derive_fedsda_run_metrics(
        **valid_arguments
        | dict(
            run_metric_settings=RunMetricSettings(
                maximum_detection_delay_sample_count=0, post_change_recovery_window_sample_count=0
            )
        )
    )
    run_metrics = golden_condition_runs.run_metrics
    assert strict_run_metrics.stable_period_prediction_accuracy == run_metrics.prediction_accuracy
    # 検出が1件もないsea2では、検出の指標は変わらない。
    assert (
        strict_run_metrics.training_model_switch_detection_metrics
        != run_metrics.training_model_switch_detection_metrics
    ) == (
        GOLDEN_CONDITION_PATHS_BY_DATASET_NAME[golden_condition_runs.dataset_name][
            "candidate_decisions"
        ]
    )
    assert (
        replace(
            strict_run_metrics,
            stable_period_prediction_accuracy=run_metrics.stable_period_prediction_accuracy,
            training_model_switch_detection_metrics=run_metrics.training_model_switch_detection_metrics,
            change_response_metrics=run_metrics.change_response_metrics,
            model_diagnostic_metrics=run_metrics.model_diagnostic_metrics,
            model_pair_prediction_complementarity_metrics=(
                run_metrics.model_pair_prediction_complementarity_metrics
            ),
            prediction_diagnostic_metrics=run_metrics.prediction_diagnostic_metrics,
        )
        == run_metrics
    )


def test_change_response_metric_correspondence_distinguishes_metric_pairs():
    """応答の指標の対応表で、2つの指標を取り違えたら、goldenと小さい条件の照合が検出する。

    実行した全部の条件で、旧の値が同じになる対（取り違えを検出できない対）が、期待の一覧と同じであることを確かめる。
    """
    observations = LEGACY_CHANGE_RESPONSE_METRIC_VALUE_OBSERVATIONS
    if len(observations) < len(GOLDEN_DATASET_NAMES) + 2 * len(SMALL_RUN_CONDITIONS):
        pytest.skip("goldenと、対照の全条件を実行したときだけ確かめる")
    legacy_metric_names = sorted(observations[0])
    indistinguishable_name_pairs = {
        (first_name, second_name)
        for first_name, second_name in combinations(legacy_metric_names, 2)
        if all(observation[first_name] == observation[second_name] for observation in observations)
    }
    assert (
        indistinguishable_name_pairs == INDISTINGUISHABLE_LEGACY_CHANGE_RESPONSE_METRIC_NAME_PAIRS
    )
    # どの条件でも0のままの指標（値の照合が、中身のない比較になるもの）は、ない。
    assert not [
        legacy_metric_name
        for legacy_metric_name in legacy_metric_names
        if all(observation[legacy_metric_name] == 0 for observation in observations)
    ]


def test_model_diagnostic_metric_correspondence_distinguishes_metric_pairs():
    """モデル別の診断の指標の対応表で、2つの指標を取り違えたら、goldenと小さい条件の照合が検出する。

    実行した全部の条件（2つとも比べた条件）で、旧の値が同じになる対が、期待の一覧と同じであることを確かめる。
    """
    observations = LEGACY_MODEL_DIAGNOSTIC_METRIC_VALUE_OBSERVATIONS
    if len(observations) < len(GOLDEN_DATASET_NAMES) + 2 * len(SMALL_RUN_CONDITIONS):
        pytest.skip("goldenと、対照の全条件を実行したときだけ確かめる")
    legacy_metric_names = sorted({name for observation in observations for name in observation})
    assert len(legacy_metric_names) == 26
    indistinguishable_name_pairs = {
        (first_name, second_name)
        for first_name, second_name in combinations(legacy_metric_names, 2)
        if all(
            observation[first_name] == observation[second_name]
            for observation in observations
            if first_name in observation and second_name in observation
        )
    }
    assert (
        indistinguishable_name_pairs == INDISTINGUISHABLE_LEGACY_MODEL_DIAGNOSTIC_METRIC_NAME_PAIRS
    )
    # どの条件でも同じ値のままの指標（値の照合が、取り違え以外の誤りも検出しにくいもの）は、ない。
    assert not [
        legacy_metric_name
        for legacy_metric_name in legacy_metric_names
        if len(
            {
                observation[legacy_metric_name]
                for observation in observations
                if legacy_metric_name in observation
            }
        )
        == 1
    ]


def test_model_pair_metric_correspondence_distinguishes_metric_pairs():
    """モデル対の予測の相補性の指標の対応表で、2つの指標を取り違えたら、goldenと小さい条件の照合が検出する。"""
    observations = LEGACY_MODEL_PAIR_METRIC_VALUE_OBSERVATIONS
    if len(observations) < len(GOLDEN_DATASET_NAMES) + 2 * len(SMALL_RUN_CONDITIONS):
        pytest.skip("goldenと、対照の全条件を実行したときだけ確かめる")
    legacy_metric_names = sorted(observations[0])
    indistinguishable_name_pairs = {
        (first_name, second_name)
        for first_name, second_name in combinations(legacy_metric_names, 2)
        if all(observation[first_name] == observation[second_name] for observation in observations)
    }
    assert indistinguishable_name_pairs == INDISTINGUISHABLE_LEGACY_MODEL_PAIR_METRIC_NAME_PAIRS
    # どの条件でも0のままの指標は、ない。
    assert not [
        legacy_metric_name
        for legacy_metric_name in legacy_metric_names
        if all(observation[legacy_metric_name] == 0 for observation in observations)
    ]


def test_prediction_diagnostic_metric_correspondence_distinguishes_metric_pairs():
    """予測の診断の指標の対応表で、2つの指標を取り違えたら、goldenと小さい条件の照合が検出する。"""
    observations = LEGACY_PREDICTION_DIAGNOSTIC_METRIC_VALUE_OBSERVATIONS
    if len(observations) < len(GOLDEN_DATASET_NAMES) + 2 * len(SMALL_RUN_CONDITIONS):
        pytest.skip("goldenと、対照の全条件を実行したときだけ確かめる")
    # Fixed-Shareの結合と実際の予測の差の指標は、ここで見る提案構成の観測では、いつも0なので、除く
    # （AdaHedgeの重みで予測する構成の照合——test_fedsda_configuration_variants.py——が、0でない値で確かめる）。
    legacy_metric_names = sorted(
        set(observations[0])
        - set(LEGACY_PREDICTION_DIAGNOSTIC_METRIC_NAMES_ZERO_FOR_FIXED_SHARE_PREDICTION)
    )
    indistinguishable_name_pairs = {
        (first_name, second_name)
        for first_name, second_name in combinations(legacy_metric_names, 2)
        if all(observation[first_name] == observation[second_name] for observation in observations)
    }
    assert (
        indistinguishable_name_pairs
        == INDISTINGUISHABLE_LEGACY_PREDICTION_DIAGNOSTIC_METRIC_NAME_PAIRS
    )
    # どの条件でも0のままの指標は、ない。
    assert not [
        legacy_metric_name
        for legacy_metric_name in legacy_metric_names
        if all(observation[legacy_metric_name] == 0 for observation in observations)
    ]
