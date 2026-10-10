"""全体runの指標の導出: 実旧の実行と保存結果、Windows用のgolden、小さい条件の実旧の指標との照合。

31の離散列は、sourceではなく、このtestが、新の記録の読取りから、旧の保存形式へ写して作る
（新の記録から、旧の列がすべて作れることを確かめる）。
"""

import hashlib
import io
import json
import platform
import random
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from math import isnan
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_fedsda_run_client as client_test_module
import test_fedsda_stream_protocol_run as whole_run_test_module
import test_proposed_regression as legacy_regression
from test_fedsda_run_client import (
    LEGACY_ACTION_BY_ADAPTATION_OUTCOME,
    LEGACY_REASON_BY_DECISION_REASON,
    snapshot_run_client_state,
)
from test_fedsda_stream_protocol_run import (
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

GOLDEN_DATASET_NAME = "sine2"
# goldenが比べる33指標のうち、導出していないもの（なし。計算量の7項目も、計測つきの全体runから導出する）。
UNDERIVED_LEGACY_METRIC_NAMES = ()


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
            run_metrics.prediction_weight_recalibration_replayed_sample_count
        ),
        routing_aggregation_recalibration_sample_count=(
            run_metrics.global_diagnostic_recalibration_replayed_sample_count
        ),
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
        # 最終構成では、Fixed-Shareの予測が、そのまま実際の予測（旧の2組の列は同じ値）。
        history_routing_switching_correct=np.asarray(combined_correctness, dtype=np.bool_),
        history_routing_switching_leader_id=np.asarray(maximum_weight_model_ids, dtype=np.int32),
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
    *, model_computation_counts, participants
):
    """積和演算の数が、標本数の計数×（部品の全結合層の、入力の次元×出力の次元の合計）と一致する。

    逆伝播の見積りは、共有部の最初の層だけ、入力へ勾配を戻さない（重みの勾配だけ）。
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
    assert len(shared_layer_sizes) >= 1
    assert concept_specific_size > 0
    counts = model_computation_counts
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


def make_golden_condition_settings(*, legacy_values, hidden_layer_widths, monkeypatch):
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


@pytest.fixture(scope="module")
def golden_condition_runs(tmp_path_factory):
    """goldenの条件（sine2）で、実旧の`run_random_drift_experiment`（保存つき）と、新の全体run＋導出を、1回ずつ実行する。

    旧の実行は、旧の回帰test（`run_case`）と同じ設定の文脈・同じ引数で行う。呼出し側の乱数は進めない。
    """
    configuration = ExperimentConfiguration(
        mode=legacy_regression.MODE,
        dataset=GOLDEN_DATASET_NAME,
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
    raw_path = tmp_path_factory.mktemp("legacy-raw") / f"{GOLDEN_DATASET_NAME}.npz"
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
                        **legacy_regression.CASES[GOLDEN_DATASET_NAME],
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
        with np.load(raw_path, allow_pickle=False) as legacy_arrays:
            legacy_trace_arrays = {
                trace_name: legacy_arrays[trace_name] for trace_name in legacy_regression.TRACES
            }
        # 定数の差し替えは、設定の束を作る間だけ（同じmoduleの、小さい条件のtestへ残さない）。
        with pytest.MonkeyPatch.context() as monkeypatch:
            execution_settings, run_participant_settings, run_metric_settings = (
                make_golden_condition_settings(
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
            measured_run=measured_run,
            legacy_result=legacy_result,
            legacy_trace_arrays=legacy_trace_arrays,
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
    # 旧の回帰testが求める経路（複数のモデル、候補の採用、統合、再較正）を通っている。
    assert run_metrics.final_global_model_count > 1
    assert run_metrics.candidate_validation_decision_count > 0
    assert run_metrics.prediction_weight_recalibration_replayed_sample_count > 0
    assert run_metrics.training_model_switch_detection_metrics.matched_detection_count > 0
    assert (
        0
        < run_metrics.training_model_switch_detection_metrics.matched_detection_count
        < run_metrics.training_model_switch_detection_metrics.detection_count
    )
    # 計算量: 積和演算の数が、標本数の計数と、層の形から計算した値と一致する。
    assert_multiply_accumulate_counts_match_example_counts(
        model_computation_counts=run_metrics.model_computation_counts,
        participants=golden_condition_runs.participants,
    )
    # 共有部の特徴を使い回すので、共有部を通った標本数は、概念固有部より少ない。
    assert (
        run_metrics.model_computation_counts.shared_part_inference_example_count
        < run_metrics.model_computation_counts.concept_specific_part_inference_example_count
    )
    # 準備の間の計算（事前学習）は、指標に含めない。
    preparation_counts = golden_condition_runs.measured_run.preparation_model_computation_counts
    assert preparation_counts.concept_specific_part_training_example_count == (
        legacy_regression.COMMON["PRETRAIN_SAMPLES"] * legacy_regression.COMMON["PRETRAIN_EPOCHS"]
    )
    assert preparation_counts.concept_specific_parameter_optimizer_step_count > 0
    # 定常精度は、精度と違う値（回復の窓が、実際に標本を除いている）。
    assert run_metrics.stable_period_prediction_accuracy != run_metrics.prediction_accuracy


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
    # 列が、中身のある比較になっている（旧の回帰testの経路の件数）。
    assert trace_arrays["model_registration_ids"].size > 1
    assert trace_arrays["provisional_accepted"].sum() > 0
    assert (~trace_arrays["provisional_accepted"]).sum() > 0
    assert trace_arrays["clustering_absorbed"].sum() > 0
    assert trace_arrays["switch_positions"].size > 0
    assert trace_arrays["drift_positions"].size > 0
    assert trace_arrays["clustering_pair_same_cluster"].size > 0
    assert len(set(trace_arrays["adaptation_actions"].tolist())) > 2
    assert len(set(trace_arrays["provisional_reasons"].tolist())) > 1


def test_golden_condition_metrics_and_traces_match_windows_golden(golden_condition_runs):
    """Windowsの基準環境で、導出した指標と列が、Windows用のgolden（sine2）と一致する。

    goldenは、実行環境に依存する（Linuxでは、旧実装の結果自体が、このgoldenと違う）。Windows以外では
    照合しない。同じprocessの中の実旧との照合（上の2つ）は、どの環境でも行う。
    """
    if platform.system() != "Windows":
        pytest.skip("Windows用のgoldenとの照合は、Windowsでだけ行う（Linux用のgoldenは後のspec）")
    golden = json.loads(legacy_regression.GOLDEN_PATH.read_text(encoding="utf-8"))
    golden_case = golden["cases"][GOLDEN_DATASET_NAME]
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
    assert observed_reasons == GOLDEN_CONDITION_DECISION_REASONS, sorted(observed_reasons)


# goldenの条件（sine2）の全体runが通る、候補の判定の理由（実旧の名前）。
# 別の保有モデルの再利用（alternative_reference_refit）を含む。
GOLDEN_CONDITION_DECISION_REASONS = {"accepted", "alternative_reference_refit", "first_interval"}


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
    assert unmeasured_run_metrics == replace(
        golden_condition_runs.run_metrics, model_computation_counts=None
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
    """小さい条件で、導出した指標が、実旧の全体runの後に、旧の指標の計算を呼んだ結果と一致する。"""
    condition = (
        random_seed,
        client_count,
        per_client_sample_count,
        aggregation_interval,
        minimum_change_gap,
        concept_change_probability,
        update_interval,
    )
    if condition not in SMALL_RUNS_BY_CONDITION:
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
        measured_run = execute_fedsda_stream_protocol_run_with_computation_measurement(
            execution_settings=execution_settings,
            run_participant_settings=make_run_participant_settings(
                valid_run_settings_mapping, update_interval=update_interval
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
            whole_run_test_module.LEGACY_MODE_NAME,
            whole_run_test_module.LEGACY_MODE_NAME,
            random_seed,
            experiment._new_round_telemetry(),
            client_test_module.MAXIMUM_TOLERATED_MEAN_LOSS_INCREASE,
        )
        with np.load(legacy_raw_path, allow_pickle=False) as legacy_arrays:
            legacy_trace_arrays = {
                trace_name: legacy_arrays[trace_name] for trace_name in legacy_regression.TRACES
            }
        SMALL_RUNS_BY_CONDITION[condition] = (
            legacy_run,
            run_result,
            participants,
            legacy_trace_arrays,
            measured_run.model_computation_counts,
        )
    legacy_run, run_result, participants, legacy_trace_arrays, model_computation_counts = (
        SMALL_RUNS_BY_CONDITION[condition]
    )
    # 新の記録から作った31の離散列が、実旧の保存処理が作った配列と、形・型・値で一致する。
    trace_arrays = derive_legacy_trace_arrays(run_result=run_result, participants=participants)
    assert tuple(trace_arrays) == legacy_regression.TRACES
    for trace_name, trace_values in trace_arrays.items():
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
    legacy_metric_values = derive_legacy_metric_values(run_metrics)
    for legacy_metric_name in legacy_regression.METRICS:
        if legacy_metric_name.startswith("compute_"):
            assert (
                legacy_metric_values[legacy_metric_name]
                == legacy_computation_results[legacy_metric_name]
            ), legacy_metric_name
    assert_multiply_accumulate_counts_match_example_counts(
        model_computation_counts=model_computation_counts, participants=participants
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
    assert run_metrics.mixed_prediction_sample_count == sum(
        legacy_client.soft_routing_activation.soft_sample_count for legacy_client in legacy_clients
    )
    assert run_metrics.prediction_weight_recalibration_replayed_sample_count == sum(
        legacy_client.switching_expert_router.aggregation_recalibration_sample_count
        for legacy_client in legacy_clients
    )
    assert run_metrics.global_diagnostic_recalibration_replayed_sample_count == sum(
        legacy_client.expert_router.aggregation_recalibration_sample_count
        for legacy_client in legacy_clients
    )
    processed_sample_count = run_result.processed_sample_count_per_client
    SMALL_RUN_OBSERVATIONS[
        (random_seed, per_client_sample_count, maximum_delay, recovery_window)
    ] = dict(
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
                client_operations=participants.client_operations[:2],
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
                    participants.client_operations[2],
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
    # 設定を変えると、検出の指標と定常精度だけが変わる。
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
    assert (
        strict_run_metrics.training_model_switch_detection_metrics
        != run_metrics.training_model_switch_detection_metrics
    )
    assert (
        replace(
            strict_run_metrics,
            stable_period_prediction_accuracy=run_metrics.stable_period_prediction_accuracy,
            training_model_switch_detection_metrics=run_metrics.training_model_switch_detection_metrics,
        )
        == run_metrics
    )
