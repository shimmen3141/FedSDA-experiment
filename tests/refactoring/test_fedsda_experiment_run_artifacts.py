"""完全なrun設定からの1 runの実行、記録の表、保存と読込みを、実旧と照合する。"""

import hashlib
import json
import random
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest
import torch
from run_provenance_examples import TEST_RUN_PROVENANCE
from test_fedsda_final_configuration_run_settings import make_golden_condition_run_settings
from test_fedsda_run_metric_derivation import (
    GOLDEN_CONDITION_PATHS_BY_DATASET_NAME,
    client_test_module,
    derive_legacy_metric_values,
    experiment,
    extract_true_drift_events,
    legacy_regression,
    whole_run_test_module,
)
from test_fedsda_run_metric_derivation import (
    golden_condition_runs as golden_condition_runs,
)
from test_fedsda_stream_protocol_run import (
    make_execution_settings,
    make_run_participant_settings,
    run_real_legacy_whole_run,
)
from test_held_candidate_validation_progress import make_subclass_copy
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.core.run_provenance import convert_run_provenance_to_mapping
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.evaluation.model_clustering_record_store import (
    ModelClusteringObservation,
    ModelPairClusteringObservation,
)
from federated_learning_experiments.evaluation.run_metric_calculations import RunMetricSettings
from federated_learning_experiments.learning.models.model_computation_measurement import (
    ModelComputationCounts,
)
from federated_learning_experiments.runtime import fedsda_run_artifact_files
from federated_learning_experiments.runtime.fedsda_change_response_metric_derivation import (
    ADAPTATION_ACTION_BY_ADAPTATION_OUTCOME,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    FedsdaExperimentRun,
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_final_configuration_run_settings import (
    build_final_configuration_fedsda_run_settings,
)
from federated_learning_experiments.runtime.fedsda_model_diagnostic_metric_derivation import (
    derive_fedsda_model_diagnostic_metrics,
)
from federated_learning_experiments.runtime.fedsda_run_artifact_files import (
    RUN_ARTIFACT_FORMAT_VERSION,
    FedsdaRunArtifactPaths,
    load_fedsda_run_artifacts,
    save_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_run_record_tables import (
    build_fedsda_run_record_tables,
)
from federated_learning_experiments.runtime.fedsda_run_settings import FedsdaRunSettings

# 表の名前から、列の名前と型の種類（i: int64、b: bool、U: 文字列、f: float64）への対応。
COMPUTATION_COLUMNS = {
    f"{phase_name}_{count_field.name}": "i"
    for phase_name in ("local_processing", "synchronization")
    for count_field in fields(ModelComputationCounts)
}
EXPECTED_TABLE_COLUMNS = {
    "sample_prediction": dict(
        combined_prediction_is_correct="b",
        maximum_weight_model_id="i",
        fixed_share_maximum_weight_model_id="i",
        held_model_count_at_prediction="i",
        observed_class_id="i",
        observed_concept_id="i",
        maximum_prediction_weight="f",
        effective_model_count="f",
        fixed_share_effective_model_count="f",
        any_model_or_combined_prediction_is_correct="b",
        maximum_weight_model_prediction_is_correct="b",
        highest_confidence_model_prediction_is_correct="b",
        adahedge_weighted_prediction_is_correct="b",
        fixed_share_weighted_prediction_is_correct="b",
        true_concept_diagnostic_prediction_is_correct="i",
    ),
    "concept_change": dict(client_id="i", first_new_concept_sample_index="i"),
    "training_model_switch": dict(client_id="i", switch_sample_index="i"),
    "loss_change_alarm": dict(
        client_id="i",
        alarm_sample_index="i",
        estimated_change_point_sample_index="i",
        detector_candidate_start_sample_index="i",
    ),
    # 最小記述長の原理による選択（今の扱いでは、空の表）。
    "alarm_interval_assignment_selection": dict(
        client_id="i",
        alarm_sample_index="i",
        selected_hypothesis="U",
        change_interval_sample_count="i",
        destination_model_id="i",
        description_length="f",
        alarm_raised="b",
        new_model_learner_name="U",
    ),
    "sample_loss_monitoring": dict(monitored_log_e_value="f"),
    "adaptation": dict(
        client_id="i",
        adaptation_sample_index="i",
        adaptation_outcome="U",
        adaptation_action="U",
        detector_name="U",
        estimated_change_point_sample_index="i",
        previous_training_model_id="i",
        current_training_model_id="i",
    ),
    "candidate_validation_decision": dict(
        client_id="i",
        proposal_sample_index="i",
        is_incomplete_at_run_end="b",
        candidate_accepted="b",
        decision_reason="U",
        resolution_sample_index="i",
        detector_name="U",
        candidate_training_interval_sample_count="i",
        validation_sample_count="i",
        comparison_reference_model_id="i",
        candidate_full_interval_mean_loss="f",
        reference_full_interval_mean_loss="f",
        candidate_second_segment_mean_loss="f",
        reference_second_segment_mean_loss="f",
        reference_historical_mean_loss="f",
        matched_to_concept_change="b",
    ),
    "model_registration": dict(
        model_id="i",
        registered_server_round_index="i",
        is_final_global_model="b",
        registering_client_id="i",
        maximum_weight_prediction_sample_count="i",
    ),
    "model_clustering": dict(
        server_round_index="i",
        model_id="i",
        representative_model_id="i",
        merged_with_other_models="b",
        absorbed_into_representative="b",
        nearest_model_id="i",
        nearest_loss_increase_distance="f",
        cluster_model_count="i",
        maximum_within_cluster_distance="f",
        evaluated_within_cluster_pair_count="i",
        possible_within_cluster_pair_count="i",
    ),
    "model_pair_clustering": dict(
        server_round_index="i",
        lower_model_id="i",
        higher_model_id="i",
        assigned_to_same_cluster="b",
        loss_increase_distance="f",
        decision_score="f",
        true_concepts_match="i",
        concept_specific_parameter_distance="f",
    ),
    "client_cross_evaluation": dict(
        server_round_index="i",
        client_id="i",
        candidate_model_id="i",
        target_model_id="i",
        evaluated_sample_count="i",
        bounded_loss_sum="f",
        squared_bounded_loss_sum="f",
        candidate_only_correct_sample_count="i",
        target_only_correct_sample_count="i",
        both_correct_sample_count="i",
        both_wrong_sample_count="i",
    ),
    "client_cross_evaluation_class": dict(
        server_round_index="i",
        client_id="i",
        candidate_model_id="i",
        target_model_id="i",
        class_id="i",
        class_evaluated_sample_count="i",
        candidate_only_correct_sample_count="i",
        target_only_correct_sample_count="i",
        both_correct_sample_count="i",
        both_wrong_sample_count="i",
    ),
    "client_weight_update_counts": dict(
        client_id="i",
        adahedge_weight_model_pool_reset_count="i",
        adahedge_weight_concept_operation_restart_count="i",
        adahedge_weight_aggregation_recalibration_count="i",
        adahedge_weight_aggregation_recalibration_replayed_sample_count="i",
        fixed_share_weight_model_pool_reset_count="i",
        fixed_share_weight_leader_switch_count="i",
        fixed_share_weight_aggregation_recalibration_count="i",
        fixed_share_weight_aggregation_recalibration_replayed_sample_count="i",
    ),
    "client_held_model": dict(
        client_id="i",
        model_id="i",
        stored_training_sample_count="i",
        trained_sample_count="i",
        parameter_update_step_count="i",
    ),
    "server_round": dict(
        global_model_count_after_synchronization="i",
        client_held_model_count_after_synchronization="i",
    )
    | COMPUTATION_COLUMNS,
}
# 新の粗い区分から、旧の`action`への対応（照合のため。旧の結果の変換では、逆向きに使う）。
LEGACY_ACTION_BY_ADAPTATION_ACTION = {
    "defer_during_candidate_validation": "forward_validation_pending",
    "skip_too_short_change_interval": "insufficient_data",
    "reuse_held_model": "reuse",
    "maintain_current_model": "maintain",
    "start_candidate_validation": "create_pending",
    "adopt_candidate_model": "create",
    "reject_candidate_model": "create_rejected",
    "remap_by_server_consolidation": "server_merge",
}
LEGACY_REASON_BY_DECISION_REASON = {
    "current_reference_within_historical_loss_tolerance": "current_reference_refit",
    "alternative_reference_within_historical_loss_tolerance": "alternative_reference_refit",
    "first_segment_margin_failed": "first_interval",
    "second_segment_margin_failed": "second_interval",
    "both_segment_margins_failed": "first_and_second",
    "both_segment_margins_passed": "accepted",
}


def derive_legacy_trace_arrays_from_record_tables(record_tables):
    """保存する表だけから、旧の保存形式の31の配列を作る（表が、旧の配列を作るのに足りることの確認）。"""
    correctness = record_tables["sample_prediction.combined_prediction_is_correct"]
    maximum_weight_model_ids = record_tables["sample_prediction.maximum_weight_model_id"]
    is_incomplete = record_tables["candidate_validation_decision.is_incomplete_at_run_end"]
    return dict(
        history_accuracy=correctness.astype(np.int8),
        history_model_id=maximum_weight_model_ids.astype(np.int32),
        drift_client_ids=record_tables["concept_change.client_id"].astype(np.int32),
        drift_positions=record_tables["concept_change.first_new_concept_sample_index"].astype(
            np.int32
        ),
        switch_client_ids=record_tables["training_model_switch.client_id"].astype(np.int32),
        switch_positions=record_tables["training_model_switch.switch_sample_index"].astype(
            np.int32
        ),
        adaptation_client_ids=record_tables["adaptation.client_id"].astype(np.int32),
        adaptation_positions=record_tables["adaptation.adaptation_sample_index"].astype(np.int32),
        adaptation_actions=np.asarray(
            [
                LEGACY_ACTION_BY_ADAPTATION_ACTION[adaptation_action]
                for adaptation_action in record_tables["adaptation.adaptation_action"].tolist()
            ],
            dtype=np.str_,
        ),
        adaptation_old_model_ids=record_tables["adaptation.previous_training_model_id"].astype(
            np.int32
        ),
        adaptation_new_model_ids=record_tables["adaptation.current_training_model_id"].astype(
            np.int32
        ),
        provisional_client_ids=record_tables["candidate_validation_decision.client_id"].astype(
            np.int32
        ),
        provisional_positions=record_tables[
            "candidate_validation_decision.proposal_sample_index"
        ].astype(np.int32),
        provisional_accepted=record_tables["candidate_validation_decision.candidate_accepted"],
        provisional_reasons=np.asarray(
            [
                "insufficient_forward_data"
                if incomplete
                else LEGACY_REASON_BY_DECISION_REASON[decision_reason]
                for incomplete, decision_reason in zip(
                    is_incomplete.tolist(),
                    record_tables["candidate_validation_decision.decision_reason"].tolist(),
                    strict=True,
                )
            ],
            dtype=np.str_,
        ),
        provisional_resolution_positions=record_tables[
            "candidate_validation_decision.resolution_sample_index"
        ].astype(np.int32),
        model_registration_ids=record_tables["model_registration.model_id"].astype(np.int32),
        model_registration_rounds=record_tables[
            "model_registration.registered_server_round_index"
        ].astype(np.int32),
        model_registration_final_active=record_tables["model_registration.is_final_global_model"],
        clustering_rounds=record_tables["model_clustering.server_round_index"].astype(np.int32),
        clustering_model_ids=record_tables["model_clustering.model_id"].astype(np.int32),
        clustering_representative_model_ids=record_tables[
            "model_clustering.representative_model_id"
        ].astype(np.int32),
        clustering_participated_in_merge=record_tables["model_clustering.merged_with_other_models"],
        clustering_absorbed=record_tables["model_clustering.absorbed_into_representative"],
        # 旧の`history_routing_switching_*`は、Fixed-Shareの重みでの結合の列。
        history_routing_switching_correct=record_tables[
            "sample_prediction.fixed_share_weighted_prediction_is_correct"
        ],
        history_routing_switching_leader_id=record_tables[
            "sample_prediction.fixed_share_maximum_weight_model_id"
        ].astype(np.int32),
        history_routing_soft_active=np.ones(correctness.shape, dtype=np.bool_),
        clustering_pair_rounds=record_tables["model_pair_clustering.server_round_index"].astype(
            np.int32
        ),
        clustering_pair_left_model_ids=record_tables["model_pair_clustering.lower_model_id"].astype(
            np.int32
        ),
        clustering_pair_right_model_ids=record_tables[
            "model_pair_clustering.higher_model_id"
        ].astype(np.int32),
        clustering_pair_same_cluster=record_tables[
            "model_pair_clustering.assigned_to_same_cluster"
        ],
    )


def assert_record_tables_are_well_formed(record_tables, *, client_count, round_count):
    """表と列の名前・型・行の数が、決めた形である。"""
    assert set(record_tables) == {
        f"{table_name}.{column_name}"
        for table_name, columns in EXPECTED_TABLE_COLUMNS.items()
        for column_name in columns
    }
    for table_name, columns in EXPECTED_TABLE_COLUMNS.items():
        row_counts = set()
        for column_name, kind in columns.items():
            column = record_tables[f"{table_name}.{column_name}"]
            assert type(column) is np.ndarray
            assert column.dtype == {"i": np.int64, "b": np.bool_, "f": np.float64}.get(
                kind, column.dtype
            ), (
                table_name,
                column_name,
            )
            assert column.dtype.kind == kind, (table_name, column_name, column.dtype)
            row_counts.add(column.shape[0])
            expected_dimension_count = (
                2
                if table_name in ("sample_prediction", "sample_loss_monitoring")
                or (table_name, column_name)
                == ("server_round", "client_held_model_count_after_synchronization")
                else 1
            )
            assert column.ndim == expected_dimension_count, (table_name, column_name)
        assert len(row_counts) == 1, table_name
    assert (
        record_tables["sample_prediction.held_model_count_at_prediction"].shape[0] == client_count
    )
    assert record_tables["server_round.client_held_model_count_after_synchronization"].shape == (
        round_count,
        client_count,
    )
    assert record_tables[
        "server_round.local_processing_shared_part_training_example_count"
    ].shape == (round_count,)
    assert set(record_tables["adaptation.adaptation_action"].tolist()) <= set(
        ADAPTATION_ACTION_BY_ADAPTATION_OUTCOME.values()
    )


def get_random_states():
    return random.getstate(), repr(np.random.get_state()), torch.get_rng_state().clone()


def assert_random_states_equal(random_states, other_random_states):
    assert random_states[0] == other_random_states[0]
    assert random_states[1] == other_random_states[1]
    assert torch.equal(random_states[2], other_random_states[2])


def make_small_run_settings(**conditions):
    return build_final_configuration_fedsda_run_settings(
        experiment_run_conditions=ExperimentRunConditions(
            **dict(
                dataset_name="sine2",
                random_seed=5,
                client_count=2,
                per_client_sample_count=75,
                server_aggregation_interval_per_client_samples=10,
            )
            | conditions
        )
    )


@pytest.fixture(scope="module")
def small_experiment_run():
    return execute_fedsda_experiment_run(run_settings=make_small_run_settings())


@pytest.fixture(scope="module")
def golden_experiment_runs(golden_condition_runs):
    """goldenの条件を、完全なrun設定から実行する（実旧の結果は、指標の導出のtestのfixtureのもの）。"""
    run_settings = make_golden_condition_run_settings(golden_condition_runs.dataset_name)
    random_states = get_random_states()
    experiment_run = execute_fedsda_experiment_run(run_settings=run_settings)
    assert_random_states_equal(get_random_states(), random_states)
    return experiment_run


# ---- 1 runの実行 ----


def test_experiment_run_holds_settings_measured_run_and_metrics(small_experiment_run):
    experiment_run = small_experiment_run
    assert type(experiment_run) is FedsdaExperimentRun
    assert tuple(run_field.name for run_field in fields(FedsdaExperimentRun)) == (
        "run_settings",
        "measured_run",
        "run_metrics",
    )
    assert experiment_run.run_settings == make_small_run_settings()
    assert experiment_run.measured_run.run_result.processed_sample_count_per_client == 70
    assert experiment_run.run_metrics.processed_sample_count == 140
    assert (
        experiment_run.run_metrics.model_computation_counts
        == experiment_run.measured_run.model_computation_counts
    )
    with pytest.raises(FrozenInstanceError):
        experiment_run.run_metrics = experiment_run.run_metrics  # type: ignore[misc]


def test_experiment_run_is_repeatable_and_leaves_caller_random_state(small_experiment_run):
    """同じ設定からは、同じ結果（指標、表）を返す。呼出し側の乱数を進めない。"""
    random_states = get_random_states()
    repeated_experiment_run = execute_fedsda_experiment_run(run_settings=make_small_run_settings())
    assert_random_states_equal(get_random_states(), random_states)
    assert repeated_experiment_run.run_metrics == small_experiment_run.run_metrics
    record_tables = build_fedsda_run_record_tables(experiment_run=small_experiment_run)
    repeated_record_tables = build_fedsda_run_record_tables(experiment_run=repeated_experiment_run)
    assert set(record_tables) == set(repeated_record_tables)
    for table_column_name, column in record_tables.items():
        assert np.array_equal(column, repeated_record_tables[table_column_name]), table_column_name
    # seedを変えると、結果が変わる（seedが、実行へ渡っている）。
    other_experiment_run = execute_fedsda_experiment_run(
        run_settings=make_small_run_settings(random_seed=6)
    )
    assert not np.array_equal(
        build_fedsda_run_record_tables(experiment_run=other_experiment_run)[
            "sample_prediction.combined_prediction_is_correct"
        ],
        record_tables["sample_prediction.combined_prediction_is_correct"],
    )


def test_experiment_run_rejects_invalid_settings_before_running(monkeypatch):
    run_settings = make_small_run_settings()
    executed = []
    monkeypatch.setattr(
        "federated_learning_experiments.runtime.fedsda_experiment_run."
        "execute_fedsda_stream_protocol_run_with_computation_measurement",
        lambda **arguments: executed.append(arguments),
    )
    mutated_run_settings = replace(run_settings)
    object.__setattr__(mutated_run_settings, "run_metric_settings", None)
    for invalid_run_settings in (
        None,
        run_settings.execution_settings,
        make_subclass_copy(run_settings),
        mutated_run_settings,
    ):
        with pytest.raises(RunSettingsValidationError):
            execute_fedsda_experiment_run(run_settings=invalid_run_settings)
    with pytest.raises(TypeError):
        execute_fedsda_experiment_run(run_settings)  # type: ignore[misc]
    assert executed == []


# ---- 記録の表 ----


def test_record_tables_of_small_run_are_well_formed_and_do_not_change_the_run(small_experiment_run):
    random_states = get_random_states()
    record_tables = build_fedsda_run_record_tables(experiment_run=small_experiment_run)
    assert_random_states_equal(get_random_states(), random_states)
    assert_record_tables_are_well_formed(record_tables, client_count=2, round_count=7)
    assert record_tables["sample_prediction.combined_prediction_is_correct"].shape == (2, 70)
    # 概念の変更は、処理しなかった末尾の標本（70〜74）も含む、概念列の全体から求める。
    concept_traces = small_experiment_run.measured_run.run_result.evaluation_concept_traces
    assert len(concept_traces[0].concept_ids_by_sample_index) == 75
    # 2回めの読取りも、同じ表（読取りは、参加者の状態を変えない）。
    repeated_record_tables = build_fedsda_run_record_tables(experiment_run=small_experiment_run)
    for table_column_name, column in record_tables.items():
        assert np.array_equal(column, repeated_record_tables[table_column_name]), table_column_name
        assert column is not repeated_record_tables[table_column_name]
    with pytest.raises(TypeError):
        build_fedsda_run_record_tables(experiment_run=small_experiment_run.measured_run)
    with pytest.raises(TypeError):
        build_fedsda_run_record_tables(small_experiment_run)  # type: ignore[misc]


def test_adaptation_actions_cover_every_adaptation_outcome():
    """粗い区分は、全部の結果種別に決まっていて、旧の`action`と1対1に対応する。"""
    from typing import get_args

    from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationOutcome

    assert set(ADAPTATION_ACTION_BY_ADAPTATION_OUTCOME) == set(get_args(AdaptationOutcome))
    assert set(ADAPTATION_ACTION_BY_ADAPTATION_OUTCOME.values()) == set(
        LEGACY_ACTION_BY_ADAPTATION_ACTION
    )
    assert len(set(LEGACY_ACTION_BY_ADAPTATION_ACTION.values())) == len(
        LEGACY_ACTION_BY_ADAPTATION_ACTION
    )
    # 粗い区分→旧の`action`が、既存の対照の対応（結果種別→旧の`action`）と一致する。
    from test_observed_sample_processing import LEGACY_ACTION_BY_ADAPTATION_OUTCOME

    assert {
        adaptation_outcome: LEGACY_ACTION_BY_ADAPTATION_ACTION[adaptation_action]
        for adaptation_outcome, adaptation_action in ADAPTATION_ACTION_BY_ADAPTATION_OUTCOME.items()
    } == LEGACY_ACTION_BY_ADAPTATION_OUTCOME


@pytest.mark.slow
def test_golden_condition_experiment_run_matches_separately_executed_run(
    golden_condition_runs, golden_experiment_runs
):
    """設定からの実行は、計測つきの全体runと指標の導出を別々に呼んだ場合と、同じ指標になる。"""
    assert golden_experiment_runs.run_metrics == golden_condition_runs.run_metrics
    assert golden_experiment_runs.run_metrics is not golden_condition_runs.run_metrics
    assert (
        golden_experiment_runs.measured_run.round_model_computation_counts
        == golden_condition_runs.measured_run.round_model_computation_counts
    )


def test_golden_condition_record_tables_rebuild_legacy_saved_arrays(
    golden_condition_runs, golden_experiment_runs
):
    """goldenの条件で、表だけから作った31の配列が、実旧が保存した配列と、形・型・値で一致する。"""
    client_count = len(golden_experiment_runs.measured_run.participants.client_operations)
    round_count = golden_experiment_runs.measured_run.run_result.synchronization_interval_count
    record_tables = build_fedsda_run_record_tables(experiment_run=golden_experiment_runs)
    assert_record_tables_are_well_formed(
        record_tables, client_count=client_count, round_count=round_count
    )
    legacy_arrays = derive_legacy_trace_arrays_from_record_tables(record_tables)
    assert tuple(legacy_arrays) == legacy_regression.TRACES
    for trace_name, trace_values in legacy_arrays.items():
        legacy_trace_values = golden_condition_runs.legacy_trace_arrays[trace_name]
        assert trace_values.shape == legacy_trace_values.shape, trace_name
        assert trace_values.dtype == legacy_trace_values.dtype, trace_name
        assert np.array_equal(trace_values, legacy_trace_values), trace_name


def test_golden_condition_round_model_counts_match_legacy_saved_arrays(
    golden_condition_runs, golden_experiment_runs
):
    """goldenの条件で、ラウンドごとのモデル数（グローバル、clientごとの保有、クラスタリング待ち）が、実旧と一致する。"""
    record_tables = build_fedsda_run_record_tables(experiment_run=golden_experiment_runs)
    legacy_round_model_counts = golden_condition_runs.legacy_round_model_counts
    assert (
        record_tables["server_round.global_model_count_after_synchronization"].tolist()
        == legacy_round_model_counts["global_model_count"].tolist()
    )
    assert (
        record_tables["server_round.client_held_model_count_after_synchronization"].tolist()
        == golden_condition_runs.legacy_round_client_counts["held_model_count"].tolist()
    )
    # 旧の「クラスタリング待ちのモデル数」は、最終構成のサーバでは、いつも0（表に入れない）。
    assert not legacy_round_model_counts["pending_clustering_count"].any()
    # ラウンドごとのモデルの計算は、計測つきの全体runの値。
    round_counts = golden_experiment_runs.measured_run.round_model_computation_counts
    for count_field in fields(ModelComputationCounts):
        assert record_tables[f"server_round.local_processing_{count_field.name}"].tolist() == [
            getattr(each_round.local_processing_model_computation_counts, count_field.name)
            for each_round in round_counts
        ]
        assert record_tables[f"server_round.synchronization_{count_field.name}"].tolist() == [
            getattr(each_round.synchronization_model_computation_counts, count_field.name)
            for each_round in round_counts
        ]
    # 標本ごとの保有モデル数は、clientの記録。
    assert record_tables["sample_prediction.held_model_count_at_prediction"].sum() == (
        golden_experiment_runs.run_metrics.held_model_sample_count
    )


# 旧の配列の名前から、このspecで足した、新の表の列への対応（値が同じもの）。
NEW_COLUMN_BY_LEGACY_CHANGE_RESPONSE_ARRAY_NAME = {
    "estimated_drift_start_client_ids": "loss_change_alarm.client_id",
    "estimated_drift_alarm_positions": "loss_change_alarm.alarm_sample_index",
    "estimated_drift_start_positions": "loss_change_alarm.estimated_change_point_sample_index",
    "detector_candidate_start_positions": "loss_change_alarm.detector_candidate_start_sample_index",
    "history_detector_log_e": "sample_loss_monitoring.monitored_log_e_value",
    "adaptation_detectors": "adaptation.detector_name",
    "adaptation_estimated_change_points": "adaptation.estimated_change_point_sample_index",
    "provisional_detectors": "candidate_validation_decision.detector_name",
    # 最終構成では、候補の学習区間の全部を、学習に使う（旧の2つの配列は同じ値）。
    "provisional_interval_counts": "candidate_validation_decision.candidate_training_interval_sample_count",
    "provisional_training_counts": "candidate_validation_decision.candidate_training_interval_sample_count",
    "provisional_validation_counts": "candidate_validation_decision.validation_sample_count",
    "provisional_reference_model_ids": "candidate_validation_decision.comparison_reference_model_id",
    "provisional_candidate_mean_losses": "candidate_validation_decision.candidate_full_interval_mean_loss",
    "provisional_reference_mean_losses": "candidate_validation_decision.reference_full_interval_mean_loss",
    "provisional_candidate_recent_losses": "candidate_validation_decision.candidate_second_segment_mean_loss",
    "provisional_reference_recent_losses": "candidate_validation_decision.reference_second_segment_mean_loss",
    "provisional_reference_historical_means": "candidate_validation_decision.reference_historical_mean_loss",
    "provisional_matched_true": "candidate_validation_decision.matched_to_concept_change",
}


def test_golden_condition_change_response_columns_match_legacy_saved_arrays(
    golden_condition_runs, golden_experiment_runs
):
    """goldenの条件で、警報・監視の値・適応・判定の、足した列が、実旧が保存した配列と、値で一致する。"""
    record_tables = build_fedsda_run_record_tables(experiment_run=golden_experiment_runs)
    legacy_saved_arrays = golden_condition_runs.legacy_saved_arrays
    for (
        legacy_array_name,
        table_column_name,
    ) in NEW_COLUMN_BY_LEGACY_CHANGE_RESPONSE_ARRAY_NAME.items():
        legacy_array = legacy_saved_arrays[legacy_array_name]
        column = record_tables[table_column_name]
        assert column.shape == legacy_array.shape, legacy_array_name
        assert column.dtype.kind == legacy_array.dtype.kind or not column.size, legacy_array_name
        # 欠損（未完了の判定の損失）は、新旧ともNaN。整数の欠損は、新旧とも−1。
        assert np.array_equal(column, legacy_array, equal_nan=column.dtype.kind == "f"), (
            legacy_array_name
        )
    # 最終構成の判定は、すべて、警報後の標本での検証。検出episodeは、使わない。
    assert set(legacy_saved_arrays["provisional_validation_sources"].tolist()) <= {"forward"}
    assert set(legacy_saved_arrays["adaptation_episode_ids"].tolist()) <= {-1}
    # 中身のある照合になっている（警報があり、監視の値が変わる）。
    assert record_tables["loss_change_alarm.alarm_sample_index"].size > 0
    assert np.unique(record_tables["sample_loss_monitoring.monitored_log_e_value"]).size > 10
    expected_paths = GOLDEN_CONDITION_PATHS_BY_DATASET_NAME[golden_condition_runs.dataset_name]
    assert (
        record_tables["candidate_validation_decision.candidate_full_interval_mean_loss"].size > 0
    ) == expected_paths["candidate_decisions"]


# 条件ごとの、足した列が通った場合の観測（goldenの3ケースと、小さい条件）。
MODEL_DIAGNOSTIC_COLUMN_OBSERVATIONS = []
# どの条件も通らない場合（実測）。真の概念の一致が分からない対（割当の最多の真の概念が一意でないモデルを含む対）は、
# 実旧との照合では通らない。表での表し方（−1）は、下の、記録を足したtestで確かめる。
MODEL_DIAGNOSTIC_COLUMN_CASES_NOT_COVERED = {"true_concepts_unknown_pair"}
# 旧の配列の名前から、モデル別の診断・登録・クラスタリングの、新の表の列への対応（値が同じもの）。
NEW_COLUMN_BY_LEGACY_MODEL_DIAGNOSTIC_ARRAY_NAME = {
    "model_registration_client_ids": "model_registration.registering_client_id",
    "model_registration_selection_counts": "model_registration.maximum_weight_prediction_sample_count",
    "clustering_nearest_model_ids": "model_clustering.nearest_model_id",
    "clustering_nearest_distances": "model_clustering.nearest_loss_increase_distance",
    "clustering_cluster_sizes": "model_clustering.cluster_model_count",
    "clustering_cluster_max_distances": "model_clustering.maximum_within_cluster_distance",
    "clustering_evaluated_pair_counts": "model_clustering.evaluated_within_cluster_pair_count",
    "clustering_possible_pair_counts": "model_clustering.possible_within_cluster_pair_count",
    "clustering_pair_distances": "model_pair_clustering.loss_increase_distance",
    "clustering_pair_decision_scores": "model_pair_clustering.decision_score",
    "clustering_pair_oracle_same_concept": "model_pair_clustering.true_concepts_match",
    "clustering_pair_personalized_parameter_distances": (
        "model_pair_clustering.concept_specific_parameter_distance"
    ),
    "model_diagnostic_client_ids": "client_held_model.client_id",
    "model_diagnostic_model_ids": "client_held_model.model_id",
    "model_diagnostic_assigned_samples": "client_held_model.stored_training_sample_count",
    "model_diagnostic_training_examples": "client_held_model.trained_sample_count",
    "model_diagnostic_optimizer_steps": "client_held_model.parameter_update_step_count",
}


# 旧の配列の名前の末尾から、clientの評価ごとの表・クラスごとの表の、列の名前への対応。
CROSS_EVALUATION_COLUMN_BY_LEGACY_ARRAY_NAME_SUFFIX = {
    "round_index": "server_round_index",
    "client_id": "client_id",
    "candidate_model_id": "candidate_model_id",
    "target_model_id": "target_model_id",
    "class_id": "class_id",
    "n": "evaluated_sample_count",
    "sum": "bounded_loss_sum",
    "sum_sq": "squared_bounded_loss_sum",
    "candidate_only_correct": "candidate_only_correct_sample_count",
    "target_only_correct": "target_only_correct_sample_count",
    "both_correct": "both_correct_sample_count",
    "both_wrong": "both_wrong_sample_count",
}
# 条件ごとの、クロス評価の表が通った場合の観測（goldenの3ケースと、小さい条件）。
CROSS_EVALUATION_TABLE_OBSERVATIONS = []
# どの条件も通らない場合（実測で決める）。
CROSS_EVALUATION_TABLE_CASES_NOT_COVERED = set()


def assert_cross_evaluation_tables_match_legacy_saved_arrays(record_tables, legacy_saved_arrays):
    """clientの評価ごとの表と、クラスごとの表が、実旧が保存した配列と、形と値で一致する。"""
    compared_legacy_array_names = set()
    for table_name, legacy_prefix in (
        ("client_cross_evaluation", "cross_evaluation_"),
        ("client_cross_evaluation_class", "cross_evaluation_class_"),
    ):
        column_names = EXPECTED_TABLE_COLUMNS[table_name]
        for (
            legacy_suffix,
            column_name,
        ) in CROSS_EVALUATION_COLUMN_BY_LEGACY_ARRAY_NAME_SUFFIX.items():
            # クラスごとの表では、標本数は、そのクラスの標本数。
            if table_name == "client_cross_evaluation_class" and legacy_suffix == "n":
                column_name = "class_evaluated_sample_count"
            if column_name not in column_names:
                continue
            legacy_array_name = legacy_prefix + legacy_suffix
            legacy_array = legacy_saved_arrays[legacy_array_name]
            column = record_tables[f"{table_name}.{column_name}"]
            assert column.shape == legacy_array.shape, legacy_array_name
            assert column.dtype == legacy_array.dtype, legacy_array_name
            assert np.array_equal(column, legacy_array), legacy_array_name
            compared_legacy_array_names.add(legacy_array_name)
    # 旧の`model_pair_*`の配列は、clientの評価ごとの表の、正誤を比べた行。
    correctness_compared = record_tables["client_cross_evaluation.both_correct_sample_count"] >= 0
    for legacy_suffix in (
        "candidate_model_id",
        "target_model_id",
        "n",
        "candidate_only_correct",
        "target_only_correct",
        "both_correct",
        "both_wrong",
    ):
        legacy_array_name = "model_pair_" + legacy_suffix
        legacy_array = legacy_saved_arrays[legacy_array_name]
        column = record_tables[
            "client_cross_evaluation."
            + CROSS_EVALUATION_COLUMN_BY_LEGACY_ARRAY_NAME_SUFFIX[legacy_suffix]
        ][correctness_compared]
        assert column.dtype == legacy_array.dtype, legacy_array_name
        assert np.array_equal(column, legacy_array), legacy_array_name
        compared_legacy_array_names.add(legacy_array_name)
    assert len(compared_legacy_array_names) == 11 + 10 + 7
    # 旧の配列のうち、この3つのまとまりの名前を持つものは、全部、比べた。
    assert compared_legacy_array_names == {
        array_name
        for array_name in legacy_saved_arrays
        if array_name.startswith(("cross_evaluation_", "model_pair_"))
    }
    # 正誤の4つの数は、比べなかった行では、4つとも−1。比べた行では、和が、標本数。
    correctness_columns = np.stack(
        [
            record_tables[f"client_cross_evaluation.{column_name}"]
            for column_name in (
                "candidate_only_correct_sample_count",
                "target_only_correct_sample_count",
                "both_correct_sample_count",
                "both_wrong_sample_count",
            )
        ]
    )
    assert (correctness_columns[:, ~correctness_compared] == -1).all()
    assert (
        correctness_columns[:, correctness_compared].sum(axis=0)
        == record_tables["client_cross_evaluation.evaluated_sample_count"][correctness_compared]
    ).all()
    evaluated_sample_counts = record_tables["client_cross_evaluation.evaluated_sample_count"]
    CROSS_EVALUATION_TABLE_OBSERVATIONS.append(
        dict(
            correctness_compared_evaluation=bool(correctness_compared.any()),
            correctness_not_compared_evaluation=bool((~correctness_compared).any()),
            class_row=bool(record_tables["client_cross_evaluation_class.class_id"].size),
            evaluation_without_samples=bool((evaluated_sample_counts == 0).any()),
            candidate_differs_from_target_without_comparison=bool(
                (
                    ~correctness_compared
                    & (
                        record_tables["client_cross_evaluation.candidate_model_id"]
                        != record_tables["client_cross_evaluation.target_model_id"]
                    )
                ).any()
            ),
        )
    )


# 旧の配列の名前から、予測の診断の、新の表の列への対応（値が同じもの）。
NEW_COLUMN_BY_LEGACY_PREDICTION_DIAGNOSTIC_ARRAY_NAME = {
    # 予測に使う重みでの値と、Fixed-Shareの重みでの値（提案構成では、同じ値）。
    "history_routing_effective_experts": "sample_prediction.effective_model_count",
    "history_routing_switching_effective_experts": (
        "sample_prediction.fixed_share_effective_model_count"
    ),
    "history_routing_max_weight": "sample_prediction.maximum_prediction_weight",
    "history_routing_oracle_correct": (
        "sample_prediction.any_model_or_combined_prediction_is_correct"
    ),
    "history_routing_leader_correct": "sample_prediction.maximum_weight_model_prediction_is_correct",
    "history_routing_oracle_concept_correct": (
        "sample_prediction.true_concept_diagnostic_prediction_is_correct"
    ),
    "routing_pool_reset_counts": "client_weight_update_counts.adahedge_weight_model_pool_reset_count",
    "routing_concept_restart_counts": (
        "client_weight_update_counts.adahedge_weight_concept_operation_restart_count"
    ),
    "routing_aggregation_recalibration_counts": (
        "client_weight_update_counts.adahedge_weight_aggregation_recalibration_count"
    ),
    "routing_aggregation_recalibration_sample_counts": (
        "client_weight_update_counts.adahedge_weight_aggregation_recalibration_replayed_sample_count"
    ),
}
# 条件ごとの、予測の診断の列が通った場合の観測（goldenの3ケースと、小さい条件）。
PREDICTION_DIAGNOSTIC_COLUMN_OBSERVATIONS = []
# どの条件も通らない場合（実測で決める）。
PREDICTION_DIAGNOSTIC_COLUMN_CASES_NOT_COVERED = set()


def derive_legacy_class_arrays_from_sample_prediction_table(record_tables):
    """標本ごとの表から、旧の、clientとクラスごとの配列（`routing_class_*`）を数える（clientの順、クラスの昇順）。"""
    observed_class_ids = record_tables["sample_prediction.observed_class_id"]
    combined_correct = record_tables["sample_prediction.combined_prediction_is_correct"]
    any_correct = record_tables["sample_prediction.any_model_or_combined_prediction_is_correct"]
    maximum_weight_correct = record_tables[
        "sample_prediction.maximum_weight_model_prediction_is_correct"
    ]
    highest_confidence_correct = record_tables[
        "sample_prediction.highest_confidence_model_prediction_is_correct"
    ]
    rows = []
    for client_id in range(observed_class_ids.shape[0]):
        for class_id in sorted(set(observed_class_ids[client_id].tolist())):
            class_samples = observed_class_ids[client_id] == class_id
            client_any_correct = any_correct[client_id][class_samples]
            rows.append(
                (
                    client_id,
                    class_id,
                    int(class_samples.sum()),
                    int(client_any_correct.sum()),
                    int(combined_correct[client_id][class_samples].sum()),
                    int(maximum_weight_correct[client_id][class_samples].sum()),
                    int(highest_confidence_correct[client_id][class_samples].sum()),
                    int((client_any_correct & ~combined_correct[client_id][class_samples]).sum()),
                    int(
                        (
                            client_any_correct
                            & ~highest_confidence_correct[client_id][class_samples]
                        ).sum()
                    ),
                )
            )
    return {
        legacy_array_name: np.asarray([row[position] for row in rows], dtype=np.int64)
        for position, legacy_array_name in enumerate(
            (
                "routing_class_client_ids",
                "routing_class_ids",
                "routing_class_sample_counts",
                "routing_class_oracle_correct_counts",
                "routing_class_mixture_correct_counts",
                "routing_class_leader_correct_counts",
                "routing_class_confidence_leader_correct_counts",
                "routing_class_missed_oracle_counts",
                "routing_class_confidence_leader_missed_oracle_counts",
            )
        )
    }


def assert_prediction_diagnostic_columns_match_legacy_saved_arrays(
    record_tables, legacy_saved_arrays, *, experiment_run
):
    """予測の診断の、足した列が、実旧が保存した配列と、形と値で一致する。"""
    for (
        legacy_array_name,
        table_column_name,
    ) in NEW_COLUMN_BY_LEGACY_PREDICTION_DIAGNOSTIC_ARRAY_NAME.items():
        legacy_array = legacy_saved_arrays[legacy_array_name]
        column = record_tables[table_column_name]
        assert column.shape == legacy_array.shape, legacy_array_name
        if legacy_array_name == "history_routing_oracle_concept_correct":
            # 新は、整数（正解1、不正解0、真の概念が分からなければ−1）。旧は、真偽。
            assert legacy_array.dtype == np.bool_ and column.dtype == np.int64
            assert np.array_equal(column, legacy_array.astype(np.int64)), legacy_array_name
        else:
            assert column.dtype.kind == legacy_array.dtype.kind, legacy_array_name
            assert np.array_equal(column, legacy_array), legacy_array_name
    # 旧の、clientとクラスごとの配列は、標本ごとの表から数えた値。
    class_arrays = derive_legacy_class_arrays_from_sample_prediction_table(record_tables)
    assert len(class_arrays) + len(NEW_COLUMN_BY_LEGACY_PREDICTION_DIAGNOSTIC_ARRAY_NAME) == 19
    for legacy_array_name, class_array in class_arrays.items():
        legacy_array = legacy_saved_arrays[legacy_array_name]
        assert class_array.shape == legacy_array.shape, legacy_array_name
        assert legacy_array.dtype.kind == "i", legacy_array_name
        assert np.array_equal(class_array, legacy_array), legacy_array_name
    # 観測した真の概念は、評価用の概念列の、処理した標本の部分（旧は、この列を保存しない）。
    observed_concept_ids = record_tables["sample_prediction.observed_concept_id"]
    for client_id, concept_trace in enumerate(
        experiment_run.measured_run.run_result.evaluation_concept_traces
    ):
        assert observed_concept_ids[client_id].tolist() == list(
            concept_trace.concept_ids_by_sample_index[: observed_concept_ids.shape[1]]
        )
    # clientごとの表は、clientの順に、1行ずつ。
    assert record_tables["client_weight_update_counts.client_id"].tolist() == list(
        range(observed_concept_ids.shape[0])
    )
    # 予測重みの側の回数は、指標（全clientの合計）と合う。
    run_metrics = experiment_run.run_metrics
    diagnostic_metrics = run_metrics.prediction_diagnostic_metrics
    assert (
        record_tables["client_weight_update_counts.fixed_share_weight_leader_switch_count"].sum()
        == diagnostic_metrics.fixed_share_weight_leader_switch_count
    )
    assert (
        record_tables["client_weight_update_counts.fixed_share_weight_model_pool_reset_count"].sum()
        == diagnostic_metrics.fixed_share_weight_model_pool_reset_count
    )
    assert (
        record_tables[
            "client_weight_update_counts.fixed_share_weight_aggregation_recalibration_replayed_sample_count"
        ].sum()
        == run_metrics.fixed_share_weight_recalibration_replayed_sample_count
    )
    # 正否の列の合計は、指標の数と合う（globalの診断重みでの結合と、確信度が最大のモデルを含む）。
    for column_name, correct_count in (
        (
            "adahedge_weighted_prediction_is_correct",
            diagnostic_metrics.adahedge_weighted_prediction_correct_count,
        ),
        (
            "highest_confidence_model_prediction_is_correct",
            diagnostic_metrics.prediction_correct_counts.highest_confidence_model_prediction_correct_count,
        ),
        (
            "any_model_or_combined_prediction_is_correct",
            diagnostic_metrics.prediction_correct_counts.any_model_or_combined_prediction_correct_count,
        ),
    ):
        assert record_tables[f"sample_prediction.{column_name}"].sum() == correct_count, column_name
    # 最大の重みと、実効モデル数の関係: モデルが1つなら、どちらも1。
    single_model_samples = record_tables["sample_prediction.held_model_count_at_prediction"] == 1
    assert (
        record_tables["sample_prediction.maximum_prediction_weight"][single_model_samples] == 1
    ).all()
    assert (
        record_tables["sample_prediction.effective_model_count"][single_model_samples] == 1
    ).all()
    PREDICTION_DIAGNOSTIC_COLUMN_OBSERVATIONS.append(
        dict(
            effective_model_count_above_one=bool(
                (record_tables["sample_prediction.effective_model_count"] > 1).any()
            ),
            correctness_columns_differ=bool(
                (
                    record_tables["sample_prediction.any_model_or_combined_prediction_is_correct"]
                    != record_tables["sample_prediction.combined_prediction_is_correct"]
                ).any()
                and (
                    record_tables["sample_prediction.maximum_weight_model_prediction_is_correct"]
                    != record_tables["sample_prediction.combined_prediction_is_correct"]
                ).any()
                and (
                    record_tables["sample_prediction.true_concept_diagnostic_prediction_is_correct"]
                    != record_tables["sample_prediction.combined_prediction_is_correct"]
                ).any()
            ),
            adahedge_weight_model_pool_reset=bool(
                record_tables[
                    "client_weight_update_counts.adahedge_weight_model_pool_reset_count"
                ].any()
            ),
            adahedge_weight_concept_operation_restart=bool(
                record_tables[
                    "client_weight_update_counts.adahedge_weight_concept_operation_restart_count"
                ].any()
            ),
            adahedge_weight_aggregation_recalibration=bool(
                record_tables[
                    "client_weight_update_counts.adahedge_weight_aggregation_recalibration_count"
                ].any()
            ),
            fixed_share_weight_leader_switch=bool(
                record_tables[
                    "client_weight_update_counts.fixed_share_weight_leader_switch_count"
                ].any()
            ),
            multiple_classes_per_client=bool(
                np.unique(record_tables["sample_prediction.observed_class_id"]).size > 1
            ),
        )
    )


def assert_model_diagnostic_columns_match_legacy_saved_arrays(record_tables, legacy_saved_arrays):
    """モデル別の診断・登録・クラスタリングの、足した列が、実旧が保存した配列と、形と値で一致する。"""
    for (
        legacy_array_name,
        table_column_name,
    ) in NEW_COLUMN_BY_LEGACY_MODEL_DIAGNOSTIC_ARRAY_NAME.items():
        legacy_array = legacy_saved_arrays[legacy_array_name]
        column = record_tables[table_column_name]
        assert column.shape == legacy_array.shape, legacy_array_name
        assert column.dtype.kind == legacy_array.dtype.kind, legacy_array_name
        # 値がないとき（距離を持つ対がない、最も近い相手がいない）は、新旧とも、距離はNaN、IDは−1。
        assert np.array_equal(column, legacy_array, equal_nan=column.dtype.kind == "f"), (
            legacy_array_name
        )


def observe_model_diagnostic_column_cases(record_tables):
    """足した列が通った場合を、観測へ積む。"""
    MODEL_DIAGNOSTIC_COLUMN_OBSERVATIONS.append(
        dict(
            absorbed_model=bool(
                record_tables["model_clustering.absorbed_into_representative"].any()
            ),
            model_without_nearest_model=bool(
                (record_tables["model_clustering.nearest_model_id"] == -1).any()
            ),
            true_concepts_match_values=set(
                record_tables["model_pair_clustering.true_concepts_match"].tolist()
            ),
            decision_score_differs_from_distance=bool(
                (
                    record_tables["model_pair_clustering.decision_score"]
                    != record_tables["model_pair_clustering.loss_increase_distance"]
                ).any()
            ),
            client_holds_unregistered_model=bool(
                (record_tables["client_held_model.model_id"] < 0).any()
            ),
            cluster_with_three_or_more_models=bool(
                (record_tables["model_clustering.cluster_model_count"] >= 3).any()
            ),
        )
    )


def test_golden_condition_model_diagnostic_columns_match_legacy_saved_arrays(
    golden_condition_runs, golden_experiment_runs
):
    """goldenの条件で、登録・クラスタリング・保有モデルの、足した列が、実旧が保存した配列と、値で一致する。"""
    record_tables = build_fedsda_run_record_tables(experiment_run=golden_experiment_runs)
    assert_model_diagnostic_columns_match_legacy_saved_arrays(
        record_tables, golden_condition_runs.legacy_saved_arrays
    )
    observe_model_diagnostic_column_cases(record_tables)
    # 予測の診断の列も、実旧が保存した配列と一致する。
    assert_prediction_diagnostic_columns_match_legacy_saved_arrays(
        record_tables,
        golden_condition_runs.legacy_saved_arrays,
        experiment_run=golden_experiment_runs,
    )
    # clientの評価ごとの表と、クラスごとの表も、実旧が保存した配列と一致する。
    assert_cross_evaluation_tables_match_legacy_saved_arrays(
        record_tables, golden_condition_runs.legacy_saved_arrays
    )
    # 保有モデルごとの表は、clientの順・モデルIDの昇順で、clientごとに、1つ以上の行を持つ。
    held_model_rows = list(
        zip(
            record_tables["client_held_model.client_id"].tolist(),
            record_tables["client_held_model.model_id"].tolist(),
            strict=True,
        )
    )
    assert held_model_rows == sorted(held_model_rows)
    assert len(set(held_model_rows)) == len(held_model_rows)
    assert set(record_tables["client_held_model.client_id"].tolist()) == set(
        range(record_tables["sample_prediction.maximum_weight_model_id"].shape[0])
    )
    # 最大重みになった標本の数は、標本ごとの列から数えた値。登録されたモデル以外（一時IDのモデル）が
    # 最大重みの標本は、どのモデルにも数えない。
    maximum_weight_model_ids = record_tables["sample_prediction.maximum_weight_model_id"]
    selection_counts = record_tables["model_registration.maximum_weight_prediction_sample_count"]
    assert (
        selection_counts.sum()
        == np.isin(maximum_weight_model_ids, record_tables["model_registration.model_id"]).sum()
    )
    assert selection_counts.sum() > 0
    expected_paths = GOLDEN_CONDITION_PATHS_BY_DATASET_NAME[golden_condition_runs.dataset_name]
    assert (record_tables["model_pair_clustering.loss_increase_distance"].size > 0) == (
        expected_paths["multiple_global_models"]
    )


# モデルの登録・統合を通る、小さい条件: (seed, client数, 標本数, 集約間隔, 変更までの最小の間隔, 変更の確率, 学習の間隔)。
# 最後の要素: 登録されたモデルが、同じラウンドの集約でグローバルモデルに残らない場合を含むか
# （含むとき、登録の数と、統合で吸収された数からは、グローバルモデル数を求められない）。
ROUND_MODEL_COUNT_CONDITIONS = [
    (7, 3, 300, 10, 30, 0.05, 2, False),
    (2, 5, 300, 10, 50, 0.03, 1, True),
    (4, 3, 500, 25, 60, 0.03, 1, False),
]


@pytest.mark.parametrize(
    "random_seed,client_count,per_client_sample_count,aggregation_interval,minimum_change_gap,concept_change_probability,update_interval,registered_model_is_dropped",
    ROUND_MODEL_COUNT_CONDITIONS,
)
@pytest.mark.slow
def test_round_model_counts_match_real_legacy_whole_run(
    random_seed,
    client_count,
    per_client_sample_count,
    aggregation_interval,
    minimum_change_gap,
    concept_change_probability,
    update_interval,
    registered_model_is_dropped,
    monkeypatch,
    tmp_path,
    valid_run_settings_mapping,
):
    """登録・統合を通る小さい条件で、ラウンドごとのモデル数が、実旧の全体runの、同期の後の値と一致する。"""
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
    )
    experiment_run = execute_fedsda_experiment_run(
        run_settings=FedsdaRunSettings(
            execution_settings=execution_settings,
            run_participant_settings=make_run_participant_settings(
                valid_run_settings_mapping, update_interval=update_interval
            ),
            model_consolidation_settings=valid_run_settings_mapping["model_consolidation_settings"],
            run_metric_settings=RunMetricSettings(
                maximum_detection_delay_sample_count=20, post_change_recovery_window_sample_count=5
            ),
        )
    )
    record_tables = build_fedsda_run_record_tables(experiment_run=experiment_run)
    # 実旧の保存処理を、実旧の全体runの結果へ呼んで、旧の保存形式の配列を得る（設定の差し替えが有効な間に）。
    legacy_raw_path = tmp_path / "legacy-small-run.npz"
    experiment._save_raw_run(
        str(legacy_raw_path),
        legacy_run["legacy_clients"],
        legacy_run["legacy_server"],
        extract_true_drift_events(legacy_run["legacy_concept_schedules"]),
        whole_run_test_module.LEGACY_MODE_NAME,
        whole_run_test_module.LEGACY_MODE_NAME,
        random_seed,
        experiment._new_round_telemetry(),
        client_test_module.MAXIMUM_TOLERATED_MEAN_LOSS_INCREASE,
    )
    with np.load(legacy_raw_path, allow_pickle=False) as legacy_arrays:
        legacy_saved_arrays = {
            array_name: legacy_arrays[array_name] for array_name in legacy_arrays.files
        }
    # 登録・クラスタリング・保有モデルの、足した列が、実旧が保存した配列と一致する。
    assert_model_diagnostic_columns_match_legacy_saved_arrays(record_tables, legacy_saved_arrays)
    observe_model_diagnostic_column_cases(record_tables)
    assert_cross_evaluation_tables_match_legacy_saved_arrays(record_tables, legacy_saved_arrays)
    assert_prediction_diagnostic_columns_match_legacy_saved_arrays(
        record_tables, legacy_saved_arrays, experiment_run=experiment_run
    )
    # 予測重みの側の、clientごとの回数が、実旧のclientの値と一致する。
    for column_name, legacy_attribute_name in (
        ("fixed_share_weight_model_pool_reset_count", "pool_reset_count"),
        ("fixed_share_weight_leader_switch_count", "leader_switch_count"),
        ("fixed_share_weight_aggregation_recalibration_count", "aggregation_recalibration_count"),
        (
            "fixed_share_weight_aggregation_recalibration_replayed_sample_count",
            "aggregation_recalibration_sample_count",
        ),
    ):
        assert record_tables[f"client_weight_update_counts.{column_name}"].tolist() == [
            getattr(legacy_client.switching_expert_router, legacy_attribute_name)
            for legacy_client in legacy_run["legacy_clients"]
        ], column_name
    legacy_round_model_counts = legacy_run["legacy_round_model_counts"]
    assert (
        record_tables["server_round.global_model_count_after_synchronization"].tolist()
        == legacy_round_model_counts["global_model_count"]
    )
    assert (
        record_tables["server_round.client_held_model_count_after_synchronization"].tolist()
        == legacy_round_model_counts["client_held_model_count"]
    )
    # 旧の「クラスタリング待ちのモデル数」は、最終構成のサーバでは、いつも0（表に入れない）。
    assert not any(legacy_round_model_counts["pending_clustering_count"])
    # 登録の数と、統合で吸収された数から数えた値（最初の案）と、実際のグローバルモデル数の比較。
    synchronizations = experiment_run.measured_run.participants.server_operations.snapshot_server_round_synchronizations()
    registered_minus_absorbed_counts = []
    registered_minus_absorbed_count = 1
    for synchronization in synchronizations:
        registered_minus_absorbed_count += len(synchronization.registered_client_models)
        if synchronization.model_consolidation is not None:
            registered_minus_absorbed_count -= len(
                synchronization.model_consolidation.absorbed_model_ids
            )
        registered_minus_absorbed_counts.append(registered_minus_absorbed_count)
    assert (
        registered_minus_absorbed_counts != legacy_round_model_counts["global_model_count"]
    ) == registered_model_is_dropped
    # 中身のある照合になっている: clientごとの保有モデル数が、グローバルモデル数と違うラウンドがある。
    assert any(
        held_model_count != global_model_count
        for global_model_count, held_model_counts in zip(
            legacy_round_model_counts["global_model_count"],
            legacy_round_model_counts["client_held_model_count"],
            strict=True,
        )
        for held_model_count in held_model_counts
    )


def test_pair_with_unknown_true_concepts_and_model_without_distance_are_written_as_missing():
    """真の概念の一致が分からない対は−1、距離を持つ対がないモデルは、最も近い相手が−1・距離がNaNになる。"""
    experiment_run = execute_fedsda_experiment_run(run_settings=make_small_run_settings())
    record_store = experiment_run.measured_run.participants.server_operations.owners.model_clustering_record_store
    row_counts_before = (
        len(record_store.snapshot_pair_clustering_observations()),
        len(record_store.snapshot_model_clustering_observations()),
    )
    record_store.append_clustering_observations(
        pair_observations=(
            ModelPairClusteringObservation(
                round_index=0,
                lower_model_id=0,
                higher_model_id=1,
                loss_increase_distance=0.25,
                decision_score=0.5,
                assigned_to_same_cluster=False,
                true_concepts_match=None,
                concept_specific_parameter_distance=1.5,
            ),
        ),
        model_observations=(
            ModelClusteringObservation(
                round_index=0,
                model_id=2,
                nearest_model_id=None,
                nearest_loss_increase_distance=None,
                representative_model_id=2,
                cluster_model_count=1,
                maximum_within_cluster_distance=None,
                evaluated_within_cluster_pair_count=0,
                possible_within_cluster_pair_count=0,
                merged_with_other_models=False,
                absorbed_into_representative=False,
            ),
        ),
    )
    record_tables = build_fedsda_run_record_tables(experiment_run=experiment_run)
    pair_row, model_row = row_counts_before
    assert record_tables["model_pair_clustering.true_concepts_match"][pair_row] == -1
    assert record_tables["model_pair_clustering.loss_increase_distance"][pair_row] == 0.25
    assert record_tables["model_pair_clustering.decision_score"][pair_row] == 0.5
    assert (
        record_tables["model_pair_clustering.concept_specific_parameter_distance"][pair_row] == 1.5
    )
    assert record_tables["model_clustering.nearest_model_id"][model_row] == -1
    assert np.isnan(record_tables["model_clustering.nearest_loss_increase_distance"][model_row])
    assert np.isnan(record_tables["model_clustering.maximum_within_cluster_distance"][model_row])
    assert record_tables["model_clustering.cluster_model_count"][model_row] == 1
    # 真の概念の一致が分からない対は、指標の比較に入らない。
    assert (
        derive_fedsda_model_diagnostic_metrics(
            run_clients=tuple(experiment_run.measured_run.participants.client_operations),
            run_server=experiment_run.measured_run.participants.server_operations,
        ).clustering_true_concept_agreement_metrics
        == experiment_run.run_metrics.model_diagnostic_metrics.clustering_true_concept_agreement_metrics
    )


def test_cross_evaluation_table_conditions_cover_required_cases():
    """goldenの3ケースと、小さい条件が、正誤を比べた評価・比べなかった評価・クラスごとの行を通っている。"""
    if len(CROSS_EVALUATION_TABLE_OBSERVATIONS) < len(ROUND_MODEL_COUNT_CONDITIONS) + len(
        legacy_regression.CASES
    ):
        pytest.skip("goldenの3ケースと、小さい条件の全部を実行したときだけ確かめる")
    assert CROSS_EVALUATION_TABLE_CASES_NOT_COVERED == {
        case_name
        for case_name in CROSS_EVALUATION_TABLE_OBSERVATIONS[0]
        if not any(observation[case_name] for observation in CROSS_EVALUATION_TABLE_OBSERVATIONS)
    }


def test_prediction_diagnostic_column_conditions_cover_required_cases():
    """goldenの3ケースと、小さい条件が、予測の診断の列の、値が変わる場合を通っている。"""
    if len(PREDICTION_DIAGNOSTIC_COLUMN_OBSERVATIONS) < len(ROUND_MODEL_COUNT_CONDITIONS) + len(
        legacy_regression.CASES
    ):
        pytest.skip("goldenの3ケースと、小さい条件の全部を実行したときだけ確かめる")
    assert PREDICTION_DIAGNOSTIC_COLUMN_CASES_NOT_COVERED == {
        case_name
        for case_name in PREDICTION_DIAGNOSTIC_COLUMN_OBSERVATIONS[0]
        if not any(
            observation[case_name] for observation in PREDICTION_DIAGNOSTIC_COLUMN_OBSERVATIONS
        )
    }


def test_model_diagnostic_column_conditions_cover_required_cases():
    """goldenの3ケースと、上の小さい条件が、統合で吸収されたモデル、真の概念の一致・不一致の対を通っている。"""
    if len(MODEL_DIAGNOSTIC_COLUMN_OBSERVATIONS) < len(ROUND_MODEL_COUNT_CONDITIONS) + len(
        legacy_regression.CASES
    ):
        pytest.skip("goldenの3ケースと、小さい条件の全部を実行したときだけ確かめる")
    assert any(
        observation["absorbed_model"] for observation in MODEL_DIAGNOSTIC_COLUMN_OBSERVATIONS
    )
    observed_true_concepts_match_values = set().union(
        *(
            observation["true_concepts_match_values"]
            for observation in MODEL_DIAGNOSTIC_COLUMN_OBSERVATIONS
        )
    )
    assert {0, 1} <= observed_true_concepts_match_values <= {-1, 0, 1}
    # 通っていない場合（照合が、その場合を確かめていないこと）を、ここに書く。
    cases_not_covered = {
        case_name
        for case_name in (
            "model_without_nearest_model",
            "decision_score_differs_from_distance",
            "client_holds_unregistered_model",
            "cluster_with_three_or_more_models",
        )
        if not any(observation[case_name] for observation in MODEL_DIAGNOSTIC_COLUMN_OBSERVATIONS)
    }
    if -1 not in observed_true_concepts_match_values:
        cases_not_covered.add("true_concepts_unknown_pair")
    assert cases_not_covered == MODEL_DIAGNOSTIC_COLUMN_CASES_NOT_COVERED


# ---- 保存と読込み ----


def test_saved_artifacts_round_trip(small_experiment_run, tmp_path):
    """保存した要約と表を読み込むと、保存したときと同じ内容が返る。"""
    artifact_paths = save_fedsda_experiment_run(
        experiment_run=small_experiment_run,
        output_directory=tmp_path,
        file_stem="run-a",
        run_provenance=TEST_RUN_PROVENANCE,
    )
    assert artifact_paths == FedsdaRunArtifactPaths(
        summary_file_path=tmp_path / "run-a.json", record_table_file_path=tmp_path / "run-a.npz"
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["run-a.json", "run-a.npz"]
    summary, record_tables = load_fedsda_run_artifacts(
        summary_file_path=artifact_paths.summary_file_path
    )
    assert list(summary) == [
        "format_version",
        "run_settings",
        "run_metrics",
        "environment",
        "run_provenance",
        "record_table_file_name",
    ]
    assert summary["format_version"] == RUN_ARTIFACT_FORMAT_VERSION == 2
    assert summary["run_provenance"] == convert_run_provenance_to_mapping(TEST_RUN_PROVENANCE)
    assert summary["run_settings"] == convert_settings_to_plain_mapping(
        small_experiment_run.run_settings
    )
    assert summary["run_metrics"] == convert_settings_to_plain_mapping(
        small_experiment_run.run_metrics, non_finite_float_as_none=True
    )
    assert summary["run_metrics"]["settings_type"] == "FedsdaRunMetrics"
    assert summary["run_metrics"]["prediction_accuracy"] == (
        small_experiment_run.run_metrics.prediction_accuracy
    )
    assert summary["record_table_file_name"] == "run-a.npz"
    assert set(summary["environment"]) == {"system", "machine", "python", "numpy", "torch"}
    assert summary["environment"]["numpy"] == np.__version__
    assert summary["environment"]["torch"] == torch.__version__
    expected_record_tables = build_fedsda_run_record_tables(experiment_run=small_experiment_run)
    assert set(record_tables) == set(expected_record_tables)
    for table_column_name, expected_column in expected_record_tables.items():
        column = record_tables[table_column_name]
        assert column.shape == expected_column.shape, table_column_name
        assert column.dtype == expected_column.dtype, table_column_name
        assert np.array_equal(column, expected_column, equal_nan=column.dtype.kind == "f"), (
            table_column_name
        )
    # 要約のファイルは、非有限の値を含まないJSON。
    json.loads(
        artifact_paths.summary_file_path.read_text(encoding="utf-8"),
        parse_constant=lambda constant: pytest.fail(f"non-finite value in summary: {constant}"),
    )


def test_saving_refuses_to_overwrite_and_writes_nothing(small_experiment_run, tmp_path):
    """どちらかのファイルがあれば、上書きせず、何も書かない。"""
    for existing_file_name in ("run-a.json", "run-a.npz"):
        output_directory = tmp_path / existing_file_name
        output_directory.mkdir()
        (output_directory / existing_file_name).write_text("existing", encoding="utf-8")
        with pytest.raises(FileExistsError, match="run-a"):
            save_fedsda_experiment_run(
                experiment_run=small_experiment_run,
                output_directory=output_directory,
                file_stem="run-a",
                run_provenance=TEST_RUN_PROVENANCE,
            )
        assert [path.name for path in output_directory.iterdir()] == [existing_file_name]
        assert (output_directory / existing_file_name).read_text(encoding="utf-8") == "existing"


def test_failed_save_leaves_no_partial_files(small_experiment_run, tmp_path, monkeypatch):
    """保存が途中で失敗したとき、書きかけのファイルも、片方だけのファイルも、残さない。"""

    def fail_to_replace_summary(source_path, destination_path):
        if str(destination_path).endswith(".json"):
            raise OSError("simulated failure while placing the summary")
        real_replace(source_path, destination_path)

    real_replace = fedsda_run_artifact_files.replace
    monkeypatch.setattr(fedsda_run_artifact_files, "replace", fail_to_replace_summary)
    with pytest.raises(OSError, match="simulated failure"):
        save_fedsda_experiment_run(
            experiment_run=small_experiment_run,
            output_directory=tmp_path,
            file_stem="run-a",
            run_provenance=TEST_RUN_PROVENANCE,
        )
    assert list(tmp_path.iterdir()) == []

    def fail_to_write_tables(*arguments, **keyword_arguments):
        raise OSError("simulated failure while writing the tables")

    monkeypatch.setattr(fedsda_run_artifact_files, "replace", real_replace)
    monkeypatch.setattr(fedsda_run_artifact_files, "savez_compressed", fail_to_write_tables)
    with pytest.raises(OSError, match="simulated failure"):
        save_fedsda_experiment_run(
            experiment_run=small_experiment_run,
            output_directory=tmp_path,
            file_stem="run-a",
            run_provenance=TEST_RUN_PROVENANCE,
        )
    assert list(tmp_path.iterdir()) == []
    monkeypatch.undo()

    # 表の一時のファイルを書いた後、要約の内容を作る・書く段で失敗する（表の一時のファイルも、残さない）。
    def fail_to_place_tables(source_path, destination_path):
        raise OSError("simulated failure while placing the tables")

    monkeypatch.setattr(fedsda_run_artifact_files, "replace", fail_to_place_tables)
    with pytest.raises(OSError, match="simulated failure"):
        save_fedsda_experiment_run(
            experiment_run=small_experiment_run,
            output_directory=tmp_path,
            file_stem="run-a",
            run_provenance=TEST_RUN_PROVENANCE,
        )
    assert list(tmp_path.iterdir()) == []
    monkeypatch.undo()
    # 失敗の後、同じ保存先へ、保存できる。
    save_fedsda_experiment_run(
        experiment_run=small_experiment_run,
        output_directory=tmp_path,
        file_stem="run-a",
        run_provenance=TEST_RUN_PROVENANCE,
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["run-a.json", "run-a.npz"]


@pytest.mark.parametrize(
    "invalid_file_stem", ["", " run", "run ", ".", "..", "a/b", "a\\b", "c:run", None, 1]
)
def test_saving_rejects_invalid_destinations(small_experiment_run, tmp_path, invalid_file_stem):
    with pytest.raises((TypeError, ValueError), match="file_stem"):
        save_fedsda_experiment_run(
            experiment_run=small_experiment_run,
            output_directory=tmp_path,
            file_stem=invalid_file_stem,
            run_provenance=TEST_RUN_PROVENANCE,
        )
    assert list(tmp_path.iterdir()) == []


def test_saving_rejects_invalid_arguments(small_experiment_run, tmp_path):
    with pytest.raises(FileNotFoundError):
        save_fedsda_experiment_run(
            experiment_run=small_experiment_run,
            output_directory=tmp_path / "missing",
            file_stem="run-a",
            run_provenance=TEST_RUN_PROVENANCE,
        )
    assert not (tmp_path / "missing").exists()
    with pytest.raises(TypeError, match="output_directory"):
        save_fedsda_experiment_run(
            experiment_run=small_experiment_run,
            output_directory=str(tmp_path),
            file_stem="run-a",
            run_provenance=TEST_RUN_PROVENANCE,
        )
    with pytest.raises(TypeError, match="experiment_run"):
        save_fedsda_experiment_run(
            experiment_run=small_experiment_run.run_metrics,
            output_directory=tmp_path,
            file_stem="run-a",
            run_provenance=TEST_RUN_PROVENANCE,
        )
    with pytest.raises(TypeError):
        save_fedsda_experiment_run(small_experiment_run, tmp_path, "run-a")  # type: ignore[misc]
    assert list(tmp_path.iterdir()) == []


def test_loading_accepts_previous_format_without_run_provenance(small_experiment_run, tmp_path):
    """版1の要約（来歴を持たない）も読める。"""
    artifact_paths = save_fedsda_experiment_run(
        experiment_run=small_experiment_run,
        output_directory=tmp_path,
        file_stem="run-a",
        run_provenance=TEST_RUN_PROVENANCE,
    )
    summary = json.loads(artifact_paths.summary_file_path.read_text(encoding="utf-8"))
    previous_summary = {
        key: value for key, value in summary.items() if key != "run_provenance"
    } | dict(format_version=1)
    artifact_paths.summary_file_path.write_text(json.dumps(previous_summary), encoding="utf-8")
    loaded_summary, record_tables = load_fedsda_run_artifacts(
        summary_file_path=artifact_paths.summary_file_path
    )
    assert loaded_summary == previous_summary
    assert record_tables


def test_loading_rejects_unsupported_or_damaged_summaries(small_experiment_run, tmp_path):
    artifact_paths = save_fedsda_experiment_run(
        experiment_run=small_experiment_run,
        output_directory=tmp_path,
        file_stem="run-a",
        run_provenance=TEST_RUN_PROVENANCE,
    )
    summary = json.loads(artifact_paths.summary_file_path.read_text(encoding="utf-8"))
    for damaged_summary in (
        summary | dict(format_version=RUN_ARTIFACT_FORMAT_VERSION + 1),
        # 版2で来歴が無い・壊れている。版1なのに来歴を持つ。
        {key: value for key, value in summary.items() if key != "run_provenance"},
        summary | dict(run_provenance=None),
        summary
        | dict(run_provenance=summary["run_provenance"] | dict(behavior_baseline_version=0)),
        summary | dict(run_provenance={**summary["run_provenance"], "extra": 1}),
        summary | dict(format_version=1),
        {key: value for key, value in summary.items() if key != "format_version"},
        summary | dict(format_version=True),
        summary | dict(format_version=float(RUN_ARTIFACT_FORMAT_VERSION)),
        summary | dict(record_table_file_name="../run-a.npz"),
        summary | dict(record_table_file_name=".."),
        summary | dict(record_table_file_name="."),
        summary | dict(record_table_file_name=""),
        summary | dict(record_table_file_name=None),
        [summary],
    ):
        damaged_summary_file_path = tmp_path / "damaged.json"
        damaged_summary_file_path.write_text(json.dumps(damaged_summary), encoding="utf-8")
        with pytest.raises(ValueError):
            load_fedsda_run_artifacts(summary_file_path=damaged_summary_file_path)
    with pytest.raises(TypeError, match="summary_file_path"):
        load_fedsda_run_artifacts(summary_file_path=str(artifact_paths.summary_file_path))
    with pytest.raises(FileNotFoundError):
        load_fedsda_run_artifacts(summary_file_path=tmp_path / "missing.json")


def test_golden_condition_saved_artifacts_match_real_legacy_and_golden(
    golden_condition_runs, golden_experiment_runs, tmp_path
):
    """goldenの条件を、設定から実行・保存・読込みした結果が、実旧の33指標と31の配列（と、実行環境のgolden）に一致する。"""
    artifact_paths = save_fedsda_experiment_run(
        experiment_run=golden_experiment_runs,
        output_directory=tmp_path,
        file_stem=golden_condition_runs.dataset_name,
        run_provenance=TEST_RUN_PROVENANCE,
    )
    summary, record_tables = load_fedsda_run_artifacts(
        summary_file_path=artifact_paths.summary_file_path
    )
    # 要約の指標: 保存した辞書が、導出した指標の辞書。そこから作った旧の33指標が、実旧の実行結果。
    assert summary["run_metrics"] == convert_settings_to_plain_mapping(
        golden_experiment_runs.run_metrics, non_finite_float_as_none=True
    )
    legacy_metric_values = derive_legacy_metric_values(golden_experiment_runs.run_metrics)
    assert set(legacy_metric_values) == set(legacy_regression.METRICS)
    for legacy_metric_name, metric_value in legacy_metric_values.items():
        assert metric_value == golden_condition_runs.legacy_result[legacy_metric_name], (
            legacy_metric_name
        )
    assert (
        summary["run_metrics"]["prediction_accuracy"]
        == (golden_condition_runs.legacy_result["accuracy"])
    )
    assert (
        summary["run_metrics"]["stable_period_prediction_accuracy"]
        == (golden_condition_runs.legacy_result["stable_accuracy"])
    )
    assert (
        summary["run_metrics"]["final_global_model_count"]
        == (golden_condition_runs.legacy_result["final_model_count"])
    )
    # 読み込んだ表から作った31の配列が、実旧が保存した配列。
    legacy_arrays = derive_legacy_trace_arrays_from_record_tables(record_tables)
    for trace_name, trace_values in legacy_arrays.items():
        legacy_trace_values = golden_condition_runs.legacy_trace_arrays[trace_name]
        assert trace_values.dtype == legacy_trace_values.dtype, trace_name
        assert np.array_equal(trace_values, legacy_trace_values), trace_name
    # 実行環境のgoldenがあれば、読み込んだ表から作った配列のSHA-256が、goldenの記録と一致する。
    from test_environment_proposed_regression import find_golden_path

    golden_path = find_golden_path(legacy_regression.environment())
    if golden_path is not None:
        golden_case = json.loads(golden_path.read_text(encoding="utf-8"))["cases"][
            golden_condition_runs.dataset_name
        ]
        for trace_name, trace_values in legacy_arrays.items():
            serialized_values = json.dumps(
                trace_values.tolist(), separators=(",", ":"), ensure_ascii=True
            )
            assert golden_case["traces"][trace_name] == dict(
                shape=list(trace_values.shape),
                sha256=hashlib.sha256(serialized_values.encode()).hexdigest(),
            ), trace_name
    # 完全なrun設定が、保存されている。
    assert summary["run_settings"] == convert_settings_to_plain_mapping(
        make_golden_condition_run_settings(golden_condition_runs.dataset_name)
    )
