"""標本1件の処理を、実旧の最終構成のclientの標本処理（process_one_step。予測を含む）と、標本ごとに照合する。"""

import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from test_adahedge_diagnostic_evidence import assert_adahedge_matches_legacy
from test_adopted_candidate_initial_local_registration import (
    assert_held_model_states_match_legacy,
)
from test_adopted_candidate_local_adoption import assert_training_samples_match_legacy
from test_alarm_adaptation_recording import RECORDING_ORACLE_CASES
from test_alarm_occurrence_handling import build_alarm_occurrence_oracle
from test_alarm_response_completion import (
    LEGACY_ACTION_BY_RESPONSE_OUTCOME,
    run_legacy_alarm_with_real_completion,
)
from test_candidate_validation_adaptation_recording import (
    LEGACY_ACTION_BY_VALIDATION_ADAPTATION_OUTCOME,
)
from test_fixed_share_prediction_weights import capture_fixed_share_controller_state
from test_held_adahedge_diagnostic_notification import get_diagnostic_collection_snapshot
from test_held_candidate_validation_progress import make_subclass_copy
from test_joint_model_parameter_update import assert_nested_state_equal
from test_loss_change_monitoring import assert_class_monitor_matches_reference
from test_loss_statistics_model_id_reassignment import assert_store_statistics_match_legacy
from test_model_training_and_assignment_counts import assert_model_counts_match_legacy
from test_observed_sample_prediction import (
    assert_prediction_state_matches_legacy,
    enable_real_legacy_final_configuration_prediction,
)
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

import federated_learning_experiments.runtime.observed_sample_processing as sample_processing_module
from federated_drift_experiment import config
from federated_drift_experiment.detection_episode import DetectionEpisodeController
from federated_learning_experiments.evaluation.loss_change_alarm_record_store import (
    LossChangeAlarmRecordStore,
)
from federated_learning_experiments.evaluation.sample_prediction_record_store import (
    SamplePredictionRecord,
    SamplePredictionRecordStore,
)
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
)
from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
    IndexedObservedTrainingSample,
)
from federated_learning_experiments.learning.training.local_training_request_schedule import (
    LocalTrainingRequestSchedule,
)
from federated_learning_experiments.learning.training.local_training_schedule_settings import (
    LocalTrainingScheduleSettings,
)
from federated_learning_experiments.learning.training.local_training_settings import (
    LocalTrainingSettings,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.temporary_model_id_allocation import (
    TemporaryModelIdAllocator,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import (
    PendingModelUploadState,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_prediction_weights import (
    FixedSharePredictionWeightController,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_sample_observation_store import (
    PendingSampleObservationStore,
)
from federated_learning_experiments.runtime.alarm_occurrence_handling import (
    handle_alarm_occurrence,
)
from federated_learning_experiments.runtime.observed_sample_processing import (
    ObservedSampleProcessing,
    process_observed_sample,
)

LEGACY_ACTION_BY_ADAPTATION_OUTCOME = (
    LEGACY_ACTION_BY_RESPONSE_OUTCOME | LEGACY_ACTION_BY_VALIDATION_ADAPTATION_OUTCOME
)
UPDATE_INTERVAL = 2
ITERATIONS_PER_REQUEST = 1
BATCH_SAMPLE_COUNT = 2
UPLOAD_DELAY_ROUND_COUNT = 2
VALIDATION_SAMPLE_COUNT = 4


def build_sample_processing_oracle(
    *,
    monkeypatch,
    valid_run_settings_mapping,
    class_count,
    recording_oracle_case,
    historical_mean_losses,
    minimum_candidate_mean_loss_improvement,
):
    """警報1回ぶんの処理のoracleで最初の警報を両実装に処理させ、その直後から標本処理を続けられる状態を作る。

    最初の警報（実旧`_resolve_drift`と新`handle_alarm_occurrence`の一致は上流のtestが確かめている）の後、
    実旧clientへ標本処理が読む属性と設定を与え、新側へ保留標本・警報の記録・学習要求・予測のownerを足す。
    実旧clientは最終構成のクラスにし、実旧の予測をそのまま実行させる。
    """
    (
        handling_arguments,
        response_arguments,
        _,
        _,
        _,
        shared_optimizer_owners,
        legacy_client,
    ) = build_alarm_occurrence_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=class_count,
        recording_oracle_case=recording_oracle_case,
    )
    pending_training_assignment_buffer = handling_arguments["pending_training_assignment_buffer"]
    pending_sample_observation_store = PendingSampleObservationStore()
    alarm_pending_observations = handling_arguments["pending_sample_observations"]
    for indexed_observation in alarm_pending_observations:
        pending_sample_observation_store.append_pending_sample_observation(
            indexed_observation=indexed_observation
        )
    initial_torch_random_state = torch.get_rng_state().clone()
    run_legacy_alarm_with_real_completion(
        response_arguments=response_arguments, legacy_client=legacy_client, monkeypatch=monkeypatch
    )
    # 新側は、実旧と同じ状態から始めた生成器とtorchの乱数で、同じ警報を処理する（一致は上流のtestが確かめている）。
    torch.set_rng_state(initial_torch_random_state)
    alarm_occurrence_handling = handle_alarm_occurrence(**handling_arguments)
    alarm_buffer_response = (
        alarm_occurrence_handling.alarm_response_completion.alarm_buffer_response
    )
    if alarm_buffer_response.response_outcome == "alarm_interval_candidate_validation_started":
        concept_ids_by_training_sample_identity = {
            id(indexed_observation.training_sample): indexed_observation.observed_concept_id
            for indexed_observation in alarm_pending_observations
        }
        pending_sample_observation_store.hold_validation_assignment_sample_concept_ids(
            sample_concept_ids=tuple(
                concept_ids_by_training_sample_identity[id(training_sample)]
                for training_sample in alarm_buffer_response.active_validation_session.pending_assignment_training_samples
            )
        )
    pending_sample_observation_store.retain_latest_pending_sample_observations(
        retained_sample_indices=pending_training_assignment_buffer.get_state_snapshot().pending_sample_indices
    )
    capacity = pending_training_assignment_buffer._training_data_assignment_settings.pending_assignment_buffer_capacity_samples
    # 実旧の標本処理が読む属性と設定。計算時間の記録は対象外なので、入れ物だけ与える。
    monkeypatch.setattr(config, "LOCAL_UPDATE_INTERVAL", UPDATE_INTERVAL)
    # 旧は、候補の学習の早期終了と、候補検証の採否の余裕に、同じ設定値を使う。
    monkeypatch.setattr(
        config, "NEW_MODEL_EARLY_STOPPING_MIN_DELTA", minimum_candidate_mean_loss_improvement
    )
    monkeypatch.setattr(config, "SHARED_BACKBONE_TRAINING", "joint")
    monkeypatch.setattr(config, "SHARED_BACKBONE_GRADIENT_STRATEGY", "mean")
    legacy_client.processed_samples = handling_arguments["alarm_sample_index"] + 1
    legacy_client.fifo_size = capacity
    legacy_client.phase_seconds = defaultdict(float)
    legacy_client.processing_times = defaultdict(list)
    legacy_client.history_drift_type = []
    legacy_client.detected_event_positions = []
    legacy_client.estimated_drift_start_positions = []
    legacy_client.detector_candidate_start_positions = []
    legacy_client.detection_episodes = DetectionEpisodeController(enabled=False, length=10)
    # Fixed-Shareの時間尺度は、旧と同じく、保留の容量にする。
    enable_real_legacy_final_configuration_prediction(
        legacy_client=legacy_client,
        monkeypatch=monkeypatch,
        fixed_share_time_scale_sample_count=capacity,
    )
    legacy_client._pending_updates = 0
    legacy_client.updates_per_sample = ITERATIONS_PER_REQUEST
    legacy_client.batch_size = BATCH_SAMPLE_COUNT
    legacy_client.backbone_gradient_diagnostics = defaultdict(float)
    legacy_client.forward_validation_samples = VALIDATION_SAMPLE_COUNT
    legacy_client.model_upload_delay_rounds = UPLOAD_DELAY_ROUND_COUNT
    legacy_client.__dict__.pop("_sample_training_batches", None)
    temporary_model_id_allocator = TemporaryModelIdAllocator(client_id=legacy_client.client_id)
    # 履歴統計を両実装で同じ値へ置き換え、監視を両実装の実物のresetで始め直す（基準が低いと警報が起きやすい）。
    loss_statistics_store = handling_arguments["loss_statistics_store"]
    for model_id, historical_mean_loss in zip(tuple(legacy_client.models), historical_mean_losses):
        loss_statistics_store.set_model_loss_statistics(
            model_id=model_id,
            loss_statistics=ModelAndClassLossStatistics(
                overall_loss_moments=BoundedLossMoments(
                    observed_loss_count=40,
                    mean_loss=historical_mean_loss,
                    sum_squared_loss_deviations=0.5,
                )
            ),
        )
        legacy_client.model_stats[model_id] = {
            "n": 40,
            "mean": historical_mean_loss,
            "M2": 0.5,
            "class_stats": {},
        }
    legacy_client._reset_drift_detectors()
    handling_arguments["loss_change_monitor"].reset(
        baseline_loss_mean=legacy_client._e_detector_baseline()
    )
    legacy_client.next_temp_id = temporary_model_id_allocator.next_temporary_model_id
    registry = handling_arguments["held_model_training_state_registry"]
    processing_arguments = {
        argument_name: argument
        for argument_name, argument in handling_arguments.items()
        if argument_name
        not in (
            "alarm_sample_index",
            "pending_sample_observations",
            "estimated_change_span_sample_count",
            "estimated_change_point_sample_index",
            "detection_episode_id",
        )
    }
    processing_arguments.update(
        fixed_share_prediction_weight_controller=FixedSharePredictionWeightController(
            prediction_combination_settings=replace(
                valid_run_settings_mapping["prediction_combination_settings"],
                fixed_share_weight_redistribution_time_scale_samples=capacity,
            )
        ),
        sample_prediction_record_store=SamplePredictionRecordStore(),
        loss_change_alarm_record_store=LossChangeAlarmRecordStore(),
        pending_sample_observation_store=pending_sample_observation_store,
        temporary_model_id_allocator=temporary_model_id_allocator,
        pending_model_upload_state=PendingModelUploadState(),
        local_training_request_schedule=LocalTrainingRequestSchedule(
            local_training_schedule_settings=LocalTrainingScheduleSettings(
                training_requests_per_update_interval=UPDATE_INTERVAL,
                joint_update_iterations_per_training_request=ITERATIONS_PER_REQUEST,
            )
        ),
        maximum_reference_mean_loss_increase=legacy_client.distance_threshold,
        minimum_candidate_mean_loss_improvement=minimum_candidate_mean_loss_improvement,
        candidate_epoch_training_settings=replace(
            handling_arguments["candidate_epoch_training_settings"],
            minimum_validation_loss_decrease=minimum_candidate_mean_loss_improvement,
        ),
        upload_delay_round_count=UPLOAD_DELAY_ROUND_COUNT,
        batch_sample_count=BATCH_SAMPLE_COUNT,
        local_training_settings=LocalTrainingSettings(
            local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
            shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
        ),
        shared_feature_extractor=registry.get_held_model_training_state(
            model_id=legacy_client.current_model_id
        ).classifier.feature_extractor,
        shared_parameter_optimizer=shared_optimizer_owners[0].parameter_optimizer,
    )
    return processing_arguments, shared_optimizer_owners, legacy_client


def make_stream_observation(*, sample_index, class_count, stream_name, legacy_client):
    """決まった規則の標本列。streamごとにラベルの付け方を変えて、損失の列を変える。

    損失の大小を作るstreamでは、その時点の実旧の現行モデルの損失が最小・最大になるラベルを選ぶ
    （入力を作るためだけに実旧のモデルを読む。期待値には使わない）。
    """
    input_features = torch.tensor([[(sample_index % 7) / 7.0, ((sample_index * 3) % 5) / 5.0]])
    if stream_name == "alternating_labels":
        class_id = sample_index % class_count
    else:
        current_model = legacy_client.models[legacy_client.current_model_id]
        losses_by_class_id = [
            current_model.get_absolute_error(
                input_features, torch.tensor([[float(candidate_class_id)]])
            )
            for candidate_class_id in range(class_count)
        ]
        block_length = {"short_loss_blocks": 4, "long_loss_blocks": 13, "high_loss_only": 1}[
            stream_name
        ]
        high_loss_block = stream_name == "high_loss_only" or (sample_index // block_length) % 2 == 1
        class_id = losses_by_class_id.index(
            max(losses_by_class_id) if high_loss_block else min(losses_by_class_id)
        )
    return IndexedObservedTrainingSample(
        sample_index=sample_index,
        training_sample=ObservedTrainingSample(
            input_features=input_features,
            observed_class_labels=torch.tensor([[float(class_id)]]),
        ),
        # 実旧の予測は、真の概念IDがない標本を処理できないので、整数だけを使う。
        observed_concept_id=sample_index % 3,
    )


def run_legacy_sample_processing(*, legacy_client, indexed_observation, legacy_random_state):
    """実旧の標本処理を、指定の乱数状態から実行し、実行後の乱数状態を返す。"""
    global_python_random_state = random.getstate()
    try:
        random.setstate(legacy_random_state)
        legacy_client.process_one_step(
            indexed_observation.training_sample.input_features,
            indexed_observation.training_sample.observed_class_labels,
            indexed_observation.observed_concept_id,
        )
        return random.getstate()
    finally:
        random.setstate(global_python_random_state)


def assert_sample_processing_state_matches_legacy(
    *, processing_arguments, shared_optimizer_owners, legacy_client, probe_input_features
):
    """全ownerの状態を、実旧clientの対応する属性と照合する。"""
    registry = processing_arguments["held_model_training_state_registry"]
    assert (
        processing_arguments["current_training_model_assignment"].current_training_model_id
        == legacy_client.current_model_id
    )
    assert_held_model_states_match_legacy(
        registry=registry,
        shared_optimizer_owners=shared_optimizer_owners,
        legacy_client=legacy_client,
        input_features=probe_input_features,
    )
    assert_training_samples_match_legacy(
        training_sample_store=processing_arguments["training_sample_store"],
        legacy_client=legacy_client,
    )
    assert_model_counts_match_legacy(
        counts_store=processing_arguments["model_training_and_assignment_counts_store"],
        legacy_client=legacy_client,
    )
    assert_store_statistics_match_legacy(
        loss_statistics_store=processing_arguments["loss_statistics_store"],
        legacy_client=legacy_client,
    )
    # 保留: 位置のownerと標本のownerが同じ並びで、実旧のFIFOと同じ標本を持つ。
    pending_assignment_state = processing_arguments[
        "pending_training_assignment_buffer"
    ].get_state_snapshot()
    pending_sample_observations = processing_arguments[
        "pending_sample_observation_store"
    ].snapshot_pending_sample_observations()
    assert pending_assignment_state.pending_sample_indices == tuple(
        indexed_observation.sample_index for indexed_observation in pending_sample_observations
    )
    assert len(pending_sample_observations) == len(legacy_client.buffer)
    for indexed_observation, legacy_pending_sample in zip(
        pending_sample_observations, legacy_client.buffer
    ):
        assert indexed_observation.training_sample.input_features is legacy_pending_sample[0]
        assert indexed_observation.training_sample.observed_class_labels is legacy_pending_sample[1]
        assert indexed_observation.observed_concept_id == legacy_pending_sample[2]
    assert (
        pending_assignment_state.last_observed_sample_index == legacy_client.processed_samples - 1
    )
    # 監視と警報の記録。
    assert_class_monitor_matches_reference(
        monitor=processing_arguments["loss_change_monitor"], reference_monitor=legacy_client
    )
    # 候補検証の保持と、学習要求の保留。
    validation_session_holder = processing_arguments["validation_session_holder"]
    assert (validation_session_holder.held_validation_session is None) == (
        legacy_client._forward_validation is None
    )
    # 候補検証へ渡した標本の概念IDは、実旧のsessionが持つ標本の概念IDと同じ。
    assert processing_arguments[
        "pending_sample_observation_store"
    ].validation_assignment_sample_concept_ids == (
        None
        if legacy_client._forward_validation is None
        else tuple(
            legacy_held_sample[2]
            for legacy_held_sample in legacy_client._forward_validation.held_data
        )
    )
    assert (
        processing_arguments["local_training_request_schedule"].pending_training_request_count
        == legacy_client._pending_updates
    )
    # 適応記録: 実旧のイベント列と1件ずつ対応する。
    adaptation_record_snapshot = processing_arguments[
        "adaptation_record_store"
    ].get_state_snapshot()
    assert [
        dict(
            position=adaptation_record.adaptation_sample_index,
            action=LEGACY_ACTION_BY_ADAPTATION_OUTCOME[adaptation_record.adaptation_outcome],
            old_model_id=adaptation_record.previous_training_model_id,
            new_model_id=adaptation_record.current_training_model_id,
            estimated_change_point=adaptation_record.estimated_change_point_sample_index,
        )
        for adaptation_record in adaptation_record_snapshot.adaptation_records
    ] == [
        dict(
            position=legacy_event.position,
            action=legacy_event.action,
            old_model_id=legacy_event.old_model_id,
            new_model_id=legacy_event.new_model_id,
            estimated_change_point=legacy_event.estimated_change_point,
        )
        for legacy_event in legacy_client.adaptation_events
    ]
    assert adaptation_record_snapshot.training_model_switch_sample_indices == tuple(
        legacy_client.local_switch_positions
    )
    assert_adahedge_matches_legacy(
        processing_arguments["diagnostic_evidence_collection"].global_diagnostic_evidence,
        legacy_client.expert_router,
    )
    # 予測: Fixed-Shareの重み、globalと真の概念別の診断証拠、標本ごとの記録と旧の列・集計の計数。
    assert_prediction_state_matches_legacy(
        fixed_share_prediction_weight_controller=processing_arguments[
            "fixed_share_prediction_weight_controller"
        ],
        diagnostic_evidence_collection=processing_arguments["diagnostic_evidence_collection"],
        sample_prediction_record_store=processing_arguments["sample_prediction_record_store"],
        legacy_client=legacy_client,
    )


STREAM_NAMES = ("alternating_labels", "short_loss_blocks", "long_loss_blocks", "high_loss_only")
# 保有モデル（保有順）の履歴統計の平均損失。
HISTORICAL_MEAN_LOSS_PROFILES = (
    (0.05, 0.05, 0.05),
    (0.1, 0.1, 0.1),
    (0.1, 0.6, 0.35),
    (0.45, 0.1, 0.6),
)
# 最初の警報の結果: 現行の維持（保持なしで始まる）と、候補検証の開始（保持ありで始まる）。
TRAJECTORY_ORACLE_CASES = (RECORDING_ORACLE_CASES[2], RECORDING_ORACLE_CASES[3])
PROCESSED_SAMPLE_COUNT = 60
# 対照のparametrizeの条件数（全条件を実行したかどうかの判定に使う）と、条件ごとに観測した適応結果。
EXPECTED_TRAJECTORY_CONDITION_COUNT = (
    2 * len(TRAJECTORY_ORACLE_CASES) * len(STREAM_NAMES) * len(HISTORICAL_MEAN_LOSS_PROFILES) * 2
)
OBSERVED_OUTCOMES_BY_CONDITION = {}
# 条件ごとの、Fixed-Shareの重みがモデル集合の変更で初期化された回数と、最大重みのモデルの種類数。
OBSERVED_PREDICTION_COVERAGE_BY_CONDITION = {}


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("recording_oracle_case", TRAJECTORY_ORACLE_CASES)
@pytest.mark.parametrize("stream_name", STREAM_NAMES)
@pytest.mark.parametrize("historical_mean_losses", HISTORICAL_MEAN_LOSS_PROFILES)
@pytest.mark.parametrize("minimum_candidate_mean_loss_improvement", (10.0, 0.0))
def test_sample_processing_matches_real_legacy_sample_processing(
    class_count,
    recording_oracle_case,
    stream_name,
    historical_mean_losses,
    minimum_candidate_mean_loss_improvement,
    monkeypatch,
    valid_run_settings_mapping,
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        processing_arguments, shared_optimizer_owners, legacy_client = (
            build_sample_processing_oracle(
                monkeypatch=monkeypatch,
                valid_run_settings_mapping=valid_run_settings_mapping,
                class_count=class_count,
                recording_oracle_case=recording_oracle_case,
                historical_mean_losses=historical_mean_losses,
                minimum_candidate_mean_loss_improvement=minimum_candidate_mean_loss_improvement,
            )
        )
        python_random_generator = processing_arguments["python_random_generator"]
        alarm_record_store = processing_arguments["loss_change_alarm_record_store"]
        first_sample_index = legacy_client.processed_samples
        monitored_log_e_value_count = len(legacy_client.history_detector_log_e)
        probe_input_features = torch.tensor([[0.25, 0.5], [0.75, 0.125]])
        legacy_random_state = python_random_generator.getstate()
        observed_outcomes = []
        for sample_index in range(first_sample_index, first_sample_index + PROCESSED_SAMPLE_COUNT):
            indexed_observation = make_stream_observation(
                sample_index=sample_index,
                class_count=class_count,
                stream_name=stream_name,
                legacy_client=legacy_client,
            )
            record_count = len(
                processing_arguments["adaptation_record_store"]
                .get_state_snapshot()
                .adaptation_records
            )
            torch_random_state = torch.get_rng_state().clone()
            legacy_random_state = run_legacy_sample_processing(
                legacy_client=legacy_client,
                indexed_observation=indexed_observation,
                legacy_random_state=legacy_random_state,
            )
            legacy_torch_random_state = torch.get_rng_state().clone()
            torch.set_rng_state(torch_random_state)
            sample_processing = process_observed_sample(
                indexed_observation=indexed_observation, **processing_arguments
            )
            assert type(sample_processing) is ObservedSampleProcessing
            assert (
                processing_arguments[
                    "sample_prediction_record_store"
                ].snapshot_sample_prediction_records()[-1]
                is sample_processing.observed_sample_prediction.sample_prediction_record
            )
            assert torch.equal(torch.get_rng_state(), legacy_torch_random_state)
            assert python_random_generator.getstate() == legacy_random_state
            assert_sample_processing_state_matches_legacy(
                processing_arguments=processing_arguments,
                shared_optimizer_owners=shared_optimizer_owners,
                legacy_client=legacy_client,
                probe_input_features=probe_input_features,
            )
            # 警報の記録と監視の値の列。
            alarm_record_snapshot = alarm_record_store.get_state_snapshot()
            assert alarm_record_snapshot.alarm_sample_indices == tuple(
                legacy_client.detected_event_positions
            )
            assert alarm_record_snapshot.estimated_change_point_sample_indices == tuple(
                legacy_client.estimated_drift_start_positions
            )
            assert alarm_record_snapshot.detector_candidate_start_sample_indices == tuple(
                legacy_client.detector_candidate_start_positions
            )
            assert alarm_record_snapshot.monitored_log_e_values == tuple(
                legacy_client.history_detector_log_e[monitored_log_e_value_count:]
            )
            # 結果の記録: 警報の有無と、警報がなければ空の警報処理。
            assert (sample_processing.alarm_occurrence_handling is not None) == (
                sample_processing.loss_monitoring_observation.drift_detected
            )
            observed_outcomes.extend(
                adaptation_record.adaptation_outcome
                for adaptation_record in processing_arguments["adaptation_record_store"]
                .get_state_snapshot()
                .adaptation_records[record_count:]
            )
    condition = (
        class_count,
        recording_oracle_case,
        stream_name,
        historical_mean_losses,
        minimum_candidate_mean_loss_improvement,
    )
    OBSERVED_OUTCOMES_BY_CONDITION[condition] = tuple(observed_outcomes)
    OBSERVED_PREDICTION_COVERAGE_BY_CONDITION[condition] = (
        processing_arguments["fixed_share_prediction_weight_controller"].model_pool_reset_count,
        len(
            {
                sample_prediction_record.maximum_weight_model_id
                for sample_prediction_record in processing_arguments[
                    "sample_prediction_record_store"
                ].snapshot_sample_prediction_records()
            }
        ),
    )


def test_sample_processing_trajectories_cover_every_adaptation_outcome():
    """上の対照が、警報の5結果と候補検証の確定の4結果をすべて通っていること（全条件を実行したときだけ確かめる）。"""
    if len(OBSERVED_OUTCOMES_BY_CONDITION) < EXPECTED_TRAJECTORY_CONDITION_COUNT:
        pytest.skip("対照の全条件を実行したときだけ確かめる")
    observed_outcomes = {
        adaptation_outcome
        for condition_outcomes in OBSERVED_OUTCOMES_BY_CONDITION.values()
        for adaptation_outcome in condition_outcomes
    }
    assert observed_outcomes == set(LEGACY_ACTION_BY_ADAPTATION_OUTCOME) - {
        "post_alarm_validation_incomplete_candidate_rejected"
    }
    # 予測の対照が、モデル集合の変更（重みの初期化）と、最大重みのモデルの交代を通っていること。
    assert any(
        model_pool_reset_count > 0
        for model_pool_reset_count, _ in OBSERVED_PREDICTION_COVERAGE_BY_CONDITION.values()
    )
    assert any(
        maximum_weight_model_count >= 3
        for _, maximum_weight_model_count in OBSERVED_PREDICTION_COVERAGE_BY_CONDITION.values()
    )


def build_default_sample_processing_oracle(
    *, monkeypatch, valid_run_settings_mapping, validation_is_held=True
):
    """最初の警報で候補検証を開始した（保持ありで始まる）状態。保持なしを指定すると、最初の警報が現行を維持した状態。

    保持なしの状態では、履歴統計の平均を高くして、続く標本で警報が起きないようにする。
    """
    return build_sample_processing_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=2,
        recording_oracle_case=RECORDING_ORACLE_CASES[3 if validation_is_held else 2],
        historical_mean_losses=(0.1, 0.1, 0.1) if validation_is_held else (0.9, 0.9, 0.9),
        minimum_candidate_mean_loss_improvement=10.0,
    )


def snapshot_sample_processing_state(*, processing_arguments):
    """標本1件の処理が受け取るownerと乱数の、比較できる読取り。

    保持中の候補検証のsessionと送信保留は、同じオブジェクトであることだけを比べる（sessionの中の候補の
    分類器や損失の記録は読まない）。
    """
    registry = processing_arguments["held_model_training_state_registry"]
    held_model_training_states = registry.snapshot_ordered_held_model_training_states()
    pending_sample_observation_store = processing_arguments["pending_sample_observation_store"]
    return dict(
        held_model_ids=tuple(state.model_id for state in held_model_training_states),
        parameters=tuple(
            parameter.detach().clone()
            for state in held_model_training_states
            for parameter in state.classifier.parameters()
        ),
        training_samples=tuple(
            (collection.model_id, collection.training_samples)
            for collection in processing_arguments[
                "training_sample_store"
            ].snapshot_ordered_model_training_samples()
        ),
        counts=processing_arguments[
            "model_training_and_assignment_counts_store"
        ].snapshot_model_training_and_assignment_counts(),
        loss_statistics=processing_arguments["loss_statistics_store"].get_state_snapshot(),
        current_training_model_id=processing_arguments[
            "current_training_model_assignment"
        ].current_training_model_id,
        pending_assignment=processing_arguments[
            "pending_training_assignment_buffer"
        ].get_state_snapshot(),
        pending_sample_observations=pending_sample_observation_store.snapshot_pending_sample_observations(),
        validation_assignment_sample_concept_ids=pending_sample_observation_store.validation_assignment_sample_concept_ids,
        monitoring=processing_arguments["loss_change_monitor"].get_state_snapshot(),
        alarm_records=processing_arguments["loss_change_alarm_record_store"].get_state_snapshot(),
        adaptation_records=processing_arguments["adaptation_record_store"].get_state_snapshot(),
        held_validation_session=processing_arguments[
            "validation_session_holder"
        ].held_validation_session,
        pending_training_request_count=processing_arguments[
            "local_training_request_schedule"
        ].pending_training_request_count,
        python_random_state=processing_arguments["python_random_generator"].getstate(),
        torch_random_state=(torch.get_rng_state().clone(),),
        diagnostics=get_diagnostic_collection_snapshot(
            processing_arguments["diagnostic_evidence_collection"]
        ),
        fixed_share_prediction_weights=capture_fixed_share_controller_state(
            controller=processing_arguments["fixed_share_prediction_weight_controller"]
        ),
        sample_prediction_records=processing_arguments[
            "sample_prediction_record_store"
        ].snapshot_sample_prediction_records(),
        next_temporary_model_id=processing_arguments[
            "temporary_model_id_allocator"
        ].next_temporary_model_id,
        pending_model_upload=processing_arguments[
            "pending_model_upload_state"
        ].get_pending_model_upload(),
        evaluation_samples=processing_arguments[
            "model_evaluation_sample_store"
        ].snapshot_ordered_model_evaluation_samples(),
        optimizer_states=tuple(
            deepcopy(parameter_optimizer.state_dict())
            for parameter_optimizer in (
                *(
                    state.concept_specific_parameter_optimizer_state.parameter_optimizer
                    for state in held_model_training_states
                ),
                processing_arguments["shared_parameter_optimizer"],
            )
        ),
    )


def assert_sample_processing_state_unchanged(*, state_snapshot, processing_arguments):
    current_snapshot = snapshot_sample_processing_state(processing_arguments=processing_arguments)
    assert current_snapshot.keys() == state_snapshot.keys()
    for state_name, previous_state in state_snapshot.items():
        current_state = current_snapshot[state_name]
        if state_name in ("parameters", "torch_random_state"):
            assert len(current_state) == len(previous_state), state_name
            assert all(
                torch.equal(current_tensor, previous_tensor)
                for current_tensor, previous_tensor in zip(current_state, previous_state)
            ), state_name
        elif state_name == "optimizer_states":
            assert len(current_state) == len(previous_state)
            for current_optimizer_state, previous_optimizer_state in zip(
                current_state, previous_state
            ):
                assert_nested_state_equal(current_optimizer_state, previous_optimizer_state)
        elif state_name in ("held_validation_session", "pending_model_upload"):
            # 保持中のsessionと送信保留は、同じオブジェクトのままであることを確かめる（中身は比べない）。
            assert current_state is previous_state
        else:
            assert current_state == previous_state, state_name


def make_next_observation(*, processing_arguments, class_id=0):
    """保留位置のownerの最終観測位置の次の位置の標本。"""
    last_observed_sample_index = (
        processing_arguments["pending_training_assignment_buffer"]
        .get_state_snapshot()
        .last_observed_sample_index
    )
    return IndexedObservedTrainingSample(
        sample_index=last_observed_sample_index + 1,
        training_sample=ObservedTrainingSample(
            input_features=torch.tensor([[0.25, 0.5]]),
            observed_class_labels=torch.tensor([[float(class_id)]]),
        ),
        observed_concept_id=1,
    )


def make_store_with_other_pending_observations(
    pending_sample_observation_store, *, pending_sample_observations
):
    """候補検証へ渡した標本の概念IDは同じで、保留標本だけが違うowner。"""
    replaced_store = PendingSampleObservationStore()
    for pending_observation in pending_sample_observations:
        replaced_store.append_pending_sample_observation(indexed_observation=pending_observation)
    validation_assignment_sample_concept_ids = (
        pending_sample_observation_store.validation_assignment_sample_concept_ids
    )
    if validation_assignment_sample_concept_ids is not None:
        replaced_store.hold_validation_assignment_sample_concept_ids(
            sample_concept_ids=validation_assignment_sample_concept_ids
        )
    return replaced_store


OWNER_ARGUMENT_NAMES_VALIDATED_FIRST = (
    "fixed_share_prediction_weight_controller",
    "sample_prediction_record_store",
    "validation_session_holder",
    "adaptation_record_store",
    "diagnostic_evidence_collection",
    "loss_change_monitor",
    "loss_change_alarm_record_store",
    "pending_training_assignment_buffer",
    "pending_sample_observation_store",
    "held_model_training_state_registry",
    "loss_statistics_store",
    "training_sample_store",
    "model_training_and_assignment_counts_store",
    "current_training_model_assignment",
    "model_evaluation_sample_store",
    "temporary_model_id_allocator",
    "pending_model_upload_state",
    "local_training_request_schedule",
    "python_random_generator",
)

# 条件名 -> (正常な引数と次の標本から、差し替える引数を作る操作, 期待する例外)。
INVALID_SAMPLE_PROCESSING_INPUT_CASES = {
    "observation_other_type": (
        lambda processing_arguments, indexed_observation: dict(indexed_observation=object()),
        TypeError,
    ),
    "observation_subclass": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=make_subclass_copy(indexed_observation)
        ),
        TypeError,
    ),
    "sample_index_bool": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=replace(indexed_observation, sample_index=True)
        ),
        TypeError,
    ),
    "sample_index_negative": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=replace(indexed_observation, sample_index=-1)
        ),
        ValueError,
    ),
    "sample_index_repeated": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=replace(
                indexed_observation, sample_index=indexed_observation.sample_index - 1
            )
        ),
        ValueError,
    ),
    "sample_index_skipped": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=replace(
                indexed_observation, sample_index=indexed_observation.sample_index + 1
            )
        ),
        ValueError,
    ),
    # 保留標本のownerが、保留位置のownerと違う並びを持つ。
    "pending_observation_store_not_aligned": (
        lambda processing_arguments, indexed_observation: dict(
            pending_sample_observation_store=make_store_with_other_pending_observations(
                processing_arguments["pending_sample_observation_store"],
                pending_sample_observations=(replace(indexed_observation, sample_index=0),),
            )
        ),
        ValueError,
    ),
    # 候補検証の保持の有無と、候補検証へ渡した標本の概念IDの保持の有無が合わない。
    "validation_concept_ids_do_not_match_held_session": (
        lambda processing_arguments, indexed_observation: dict(
            pending_sample_observation_store=make_store_with_inverted_concept_id_holding(
                processing_arguments["pending_sample_observation_store"]
            )
        ),
        ValueError,
    ),
    # 標本の型と形（1標本であること）は、本処理が最初に確かめる。
    "training_sample_other_type": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=replace(indexed_observation, training_sample=object())
        ),
        TypeError,
    ),
    "training_sample_subclass": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=replace(
                indexed_observation,
                training_sample=make_subclass_copy(indexed_observation.training_sample),
            )
        ),
        TypeError,
    ),
    "concept_id_bool": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=replace(indexed_observation, observed_concept_id=True)
        ),
        TypeError,
    ),
    "concept_id_text": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=replace(indexed_observation, observed_concept_id="1")
        ),
        TypeError,
    ),
    "input_features_not_tensor": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=replace(
                indexed_observation,
                training_sample=ObservedTrainingSample(
                    input_features=[[0.25, 0.5]],
                    observed_class_labels=indexed_observation.training_sample.observed_class_labels,
                ),
            )
        ),
        TypeError,
    ),
    "observed_class_labels_not_tensor": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=replace(
                indexed_observation,
                training_sample=ObservedTrainingSample(
                    input_features=indexed_observation.training_sample.input_features,
                    observed_class_labels=[[0.0]],
                ),
            )
        ),
        TypeError,
    ),
    # torch.nn.ParameterはTensorの派生型。exact torch.Tensorだけを受け入れる。
    "input_features_tensor_subclass": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=replace(
                indexed_observation,
                training_sample=ObservedTrainingSample(
                    input_features=torch.nn.Parameter(
                        indexed_observation.training_sample.input_features.clone(),
                        requires_grad=False,
                    ),
                    observed_class_labels=indexed_observation.training_sample.observed_class_labels,
                ),
            )
        ),
        TypeError,
    ),
    "observed_class_labels_tensor_subclass": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=replace(
                indexed_observation,
                training_sample=ObservedTrainingSample(
                    input_features=indexed_observation.training_sample.input_features,
                    observed_class_labels=torch.nn.Parameter(
                        indexed_observation.training_sample.observed_class_labels.clone(),
                        requires_grad=False,
                    ),
                ),
            )
        ),
        TypeError,
    ),
    # 1標本ぶんの特徴に、2標本ぶんのラベル。
    "two_labels_for_one_sample": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=replace(
                indexed_observation,
                training_sample=ObservedTrainingSample(
                    input_features=indexed_observation.training_sample.input_features,
                    observed_class_labels=torch.tensor([[0.0], [1.0]]),
                ),
            )
        ),
        ValueError,
    ),
    # 監視の最後の観測の位置が、この標本の直前でない（保留位置とは連続している）。
    "monitor_last_observation_is_not_previous_sample": (
        lambda processing_arguments, indexed_observation: dict(
            loss_change_monitor=make_monitor_observed_at(
                processing_arguments["loss_change_monitor"],
                sample_index=indexed_observation.sample_index + 3,
            )
        ),
        ValueError,
    ),
    # 2標本をまとめた入力。候補検証を保持していないと、損失の評価は通ってしまう。
    "two_samples_in_one_observation": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=replace(
                indexed_observation,
                training_sample=ObservedTrainingSample(
                    input_features=torch.tensor([[0.25, 0.5], [0.5, 0.25]]),
                    observed_class_labels=torch.tensor([[0.0], [1.0]]),
                ),
            )
        ),
        ValueError,
    ),
    "one_dimensional_input_features": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=replace(
                indexed_observation,
                training_sample=ObservedTrainingSample(
                    input_features=torch.tensor([0.25, 0.5]),
                    observed_class_labels=indexed_observation.training_sample.observed_class_labels,
                ),
            )
        ),
        ValueError,
    ),
    "one_dimensional_labels": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=replace(
                indexed_observation,
                training_sample=ObservedTrainingSample(
                    input_features=indexed_observation.training_sample.input_features,
                    observed_class_labels=torch.tensor([0.0]),
                ),
            )
        ),
        ValueError,
    ),
    # 予測の記録の最後の位置が、この標本の直前でない（保留位置と監視とは連続している）。予測の段が拒否する。
    "prediction_record_is_not_previous_sample": (
        lambda processing_arguments, indexed_observation: dict(
            sample_prediction_record_store=make_record_store_recorded_at(
                sample_index=indexed_observation.sample_index + 3
            )
        ),
        ValueError,
    ),
    # 標本の中身の不正は、最初にこの標本を評価する予測の段が、どの更新より前に拒否する。
    # 特徴の有限性とラベルのdtypeは、後の段（損失の監視、候補検証）だけが確かめる条件で、予測の段が先に確かめる。
    "input_features_negative_infinity": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=replace(
                indexed_observation,
                training_sample=ObservedTrainingSample(
                    input_features=torch.tensor([[float("-inf"), 0.5]]),
                    observed_class_labels=indexed_observation.training_sample.observed_class_labels,
                ),
            )
        ),
        ValueError,
    ),
    "labels_int64": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=replace(
                indexed_observation,
                training_sample=ObservedTrainingSample(
                    input_features=indexed_observation.training_sample.input_features,
                    observed_class_labels=torch.tensor([[0]]),
                ),
            )
        ),
        ValueError,
    ),
    "label_out_of_range": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=replace(
                indexed_observation,
                training_sample=ObservedTrainingSample(
                    input_features=indexed_observation.training_sample.input_features,
                    observed_class_labels=torch.tensor([[7.0]]),
                ),
            )
        ),
        ValueError,
    ),
    "feature_count_mismatch": (
        lambda processing_arguments, indexed_observation: dict(
            indexed_observation=replace(
                indexed_observation,
                training_sample=ObservedTrainingSample(
                    input_features=torch.tensor([[0.25, 0.5, 0.75]]),
                    observed_class_labels=indexed_observation.training_sample.observed_class_labels,
                ),
            )
        ),
        ValueError,
    ),
}


# 拒否条件ごとの、例外の文言の一部（どの検査が拒否したかを確かめる。別の検査や後の段が代わりに拒否していないこと）。
REJECTION_MESSAGES = {
    "observation_other_type": "indexed_observation must be exact",
    "observation_subclass": "indexed_observation must be exact",
    "sample_index_bool": "sample_index must be builtin int",
    "sample_index_negative": "sample_index must be nonnegative",
    "sample_index_repeated": "follow the last observed sample index",
    "sample_index_skipped": "follow the last observed sample index",
    "monitor_last_observation_is_not_previous_sample": "follow the last monitored sample index",
    "pending_observation_store_not_aligned": "must match pending sample indices",
    "validation_concept_ids_do_not_match_held_session": "exactly while a session is held",
    "training_sample_other_type": "training_sample must be exact",
    "training_sample_subclass": "training_sample must be exact",
    "concept_id_bool": "observed_concept_id must be builtin int or None",
    "concept_id_text": "observed_concept_id must be builtin int or None",
    "input_features_not_tensor": "must be exact torch.Tensor",
    "observed_class_labels_not_tensor": "must be exact torch.Tensor",
    "input_features_tensor_subclass": "must be exact torch.Tensor",
    "observed_class_labels_tensor_subclass": "must be exact torch.Tensor",
    "two_samples_in_one_observation": "exactly one sample",
    "one_dimensional_input_features": "exactly one sample",
    "one_dimensional_labels": "must have shape",
    "two_labels_for_one_sample": "must have shape",
    # 次の5つは、本処理の検査を通った後、予測の段（その検査、または最初にこの標本を評価する既存の部品）が拒否する。
    "prediction_record_is_not_previous_sample": "follow the last recorded sample index",
    "input_features_negative_infinity": "input_features must be finite",
    "labels_int64": "observed_class_labels must be a CPU float32",
    "label_out_of_range": "ラベルは0～K-1の整数クラス値",
    "feature_count_mismatch": "input_features.*特徴数",
}


def make_monitor_observed_at(loss_change_monitor, *, sample_index):
    """渡された監視の複製へ、指定の位置で1件観測させたもの（元の監視は変えない）。"""
    observed_monitor = deepcopy(loss_change_monitor)
    observed_monitor.reset(baseline_loss_mean=0.5)
    observed_monitor.observe_loss_after_label_observation(
        observed_loss=0.5,
        observed_class_id=0,
        sample_index=sample_index,
        current_model_baseline_loss_mean=0.5,
    )
    return observed_monitor


def make_record_store_recorded_at(*, sample_index):
    """指定の位置の記録を1件だけ持つ、予測の記録のowner。"""
    recorded_store = SamplePredictionRecordStore()
    recorded_store.append_sample_prediction_record(
        sample_prediction_record=SamplePredictionRecord(
            sample_index=sample_index,
            observed_concept_id=None,
            observed_class_id=0,
            combined_prediction_is_correct=True,
            maximum_weight_model_id=0,
            maximum_prediction_weight=1.0,
            effective_model_count=1.0,
            any_model_or_combined_prediction_is_correct=True,
            maximum_weight_model_prediction_is_correct=True,
            global_diagnostic_prediction_is_correct=True,
            true_concept_diagnostic_prediction_is_correct=None,
            highest_confidence_model_prediction_is_correct=True,
        )
    )
    return recorded_store


def make_store_with_inverted_concept_id_holding(pending_sample_observation_store):
    """保留標本は同じで、候補検証へ渡した標本の概念IDの保持の有無だけが逆のowner。"""
    replaced_store = PendingSampleObservationStore()
    for (
        pending_observation
    ) in pending_sample_observation_store.snapshot_pending_sample_observations():
        replaced_store.append_pending_sample_observation(indexed_observation=pending_observation)
    if pending_sample_observation_store.validation_assignment_sample_concept_ids is None:
        replaced_store.hold_validation_assignment_sample_concept_ids(sample_concept_ids=(0,))
    return replaced_store


@pytest.mark.parametrize("invalid_case", INVALID_SAMPLE_PROCESSING_INPUT_CASES)
@pytest.mark.parametrize("validation_is_held", (True, False))
def test_sample_processing_rejects_invalid_input_before_any_update(
    invalid_case, validation_is_held, monkeypatch, valid_run_settings_mapping
):
    make_invalid_arguments, expected_exception = INVALID_SAMPLE_PROCESSING_INPUT_CASES[invalid_case]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        processing_arguments, _, _ = build_default_sample_processing_oracle(
            monkeypatch=monkeypatch,
            valid_run_settings_mapping=valid_run_settings_mapping,
            validation_is_held=validation_is_held,
        )
        validation_session_holder = processing_arguments["validation_session_holder"]
        assert (validation_session_holder.held_validation_session is not None) == (
            validation_is_held
        )
        indexed_observation = make_next_observation(processing_arguments=processing_arguments)
        state_snapshot = snapshot_sample_processing_state(processing_arguments=processing_arguments)
        with pytest.raises(expected_exception, match=REJECTION_MESSAGES[invalid_case]):
            process_observed_sample(
                **dict(indexed_observation=indexed_observation, **processing_arguments)
                | make_invalid_arguments(processing_arguments, indexed_observation)
            )
        assert_sample_processing_state_unchanged(
            state_snapshot=state_snapshot, processing_arguments=processing_arguments
        )
        # 拒否の後も、正しい標本は処理できる。
        process_observed_sample(indexed_observation=indexed_observation, **processing_arguments)


@pytest.mark.parametrize("owner_argument_name", OWNER_ARGUMENT_NAMES_VALIDATED_FIRST)
@pytest.mark.parametrize("invalid_owner_kind", ("other_type", "subclass"))
def test_sample_processing_rejects_invalid_owner_before_any_update(
    owner_argument_name, invalid_owner_kind, monkeypatch, valid_run_settings_mapping
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        processing_arguments, _, _ = build_default_sample_processing_oracle(
            monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
        )
        indexed_observation = make_next_observation(processing_arguments=processing_arguments)
        state_snapshot = snapshot_sample_processing_state(processing_arguments=processing_arguments)
        invalid_owner = (
            object()
            if invalid_owner_kind == "other_type"
            else make_owner_subclass_instance(processing_arguments[owner_argument_name])
        )
        with pytest.raises(TypeError, match=owner_argument_name):
            process_observed_sample(
                **dict(indexed_observation=indexed_observation, **processing_arguments)
                | {owner_argument_name: invalid_owner}
            )
        assert_sample_processing_state_unchanged(
            state_snapshot=state_snapshot, processing_arguments=processing_arguments
        )


def make_owner_subclass_instance(owner):
    """exact型の検査が拒否するべき、派生型の値（乱数生成器は属性を写せないので、派生型を新しく作る）。"""
    if type(owner) is random.Random:
        return type("RandomSubclass", (random.Random,), {})(0)
    return make_subclass_copy(owner)


SAMPLE_PROCESSING_STEP_NAMES = (
    "predict_observed_sample_and_update_prediction_weights",
    "advance_held_candidate_validation",
    "evaluate_classifier_per_sample_bounded_losses",
    "train_held_models_for_pending_training_requests",
    "handle_alarm_occurrence",
    "assign_released_pending_samples_to_current_training_model",
    "record_training_request_and_train_held_models_when_due",
)


def record_sample_processing_steps(monkeypatch, *, processing_arguments):
    """標本1件の処理が呼ぶ段を、呼出しの順と、その時点の保留・警報の記録・監視つきで記録する（実物をそのまま実行する）。"""
    step_calls = []
    for step_name in SAMPLE_PROCESSING_STEP_NAMES:
        original_step = getattr(sample_processing_module, step_name)

        def record_step_call(*, _step_name=step_name, _original_step=original_step, **arguments):
            step_calls.append(
                dict(
                    step_name=_step_name,
                    arguments=arguments,
                    pending_sample_indices=processing_arguments[
                        "pending_training_assignment_buffer"
                    ]
                    .get_state_snapshot()
                    .pending_sample_indices,
                    pending_sample_observations=processing_arguments[
                        "pending_sample_observation_store"
                    ].snapshot_pending_sample_observations(),
                    alarm_records=processing_arguments[
                        "loss_change_alarm_record_store"
                    ].get_state_snapshot(),
                    last_monitoring_observation=processing_arguments[
                        "loss_change_monitor"
                    ].last_observation,
                    sample_prediction_records=processing_arguments[
                        "sample_prediction_record_store"
                    ].snapshot_sample_prediction_records(),
                )
            )
            return _original_step(**arguments)

        monkeypatch.setattr(sample_processing_module, step_name, record_step_call)
    return step_calls


def test_sample_without_alarm_runs_steps_in_legacy_order(monkeypatch, valid_run_settings_mapping):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        processing_arguments, _, _ = build_default_sample_processing_oracle(
            monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
        )
        indexed_observation = make_next_observation(processing_arguments=processing_arguments)
        pending_sample_indices_before = (
            processing_arguments["pending_training_assignment_buffer"]
            .get_state_snapshot()
            .pending_sample_indices
        )
        alarm_records_before = processing_arguments[
            "loss_change_alarm_record_store"
        ].get_state_snapshot()
        held_concept_ids = processing_arguments[
            "pending_sample_observation_store"
        ].validation_assignment_sample_concept_ids
        step_calls = record_sample_processing_steps(
            monkeypatch, processing_arguments=processing_arguments
        )
        sample_processing = process_observed_sample(
            indexed_observation=indexed_observation, **processing_arguments
        )
    assert not sample_processing.loss_monitoring_observation.drift_detected
    assert [step_call["step_name"] for step_call in step_calls] == [
        "predict_observed_sample_and_update_prediction_weights",
        "advance_held_candidate_validation",
        "evaluate_classifier_per_sample_bounded_losses",
        "assign_released_pending_samples_to_current_training_model",
        "record_training_request_and_train_held_models_when_due",
    ]
    prediction_call, advance_call, evaluation_call, assignment_call, training_call = step_calls
    # 予測は最初の段。その時点では、この標本の予測の記録はまだなく、候補検証の進行の時点では足されている。
    assert prediction_call["arguments"]["indexed_observation"] is indexed_observation
    assert prediction_call["pending_sample_indices"] == pending_sample_indices_before
    prediction_record = sample_processing.observed_sample_prediction.sample_prediction_record
    assert prediction_record.sample_index == indexed_observation.sample_index
    assert all(
        recorded_record is not prediction_record
        for recorded_record in prediction_call["sample_prediction_records"]
    )
    assert advance_call["sample_prediction_records"][-1] is prediction_record
    # 候補検証の観測と損失の評価の時点では、この標本はまだ保留にも監視の記録にも入っていない。
    for step_call in (advance_call, evaluation_call):
        assert step_call["pending_sample_indices"] == pending_sample_indices_before
        assert all(
            pending_observation is not indexed_observation
            for pending_observation in step_call["pending_sample_observations"]
        )
        assert step_call["alarm_records"] == alarm_records_before
    assert advance_call["arguments"]["sample_index"] == indexed_observation.sample_index
    assert (
        advance_call["arguments"]["input_features"]
        is indexed_observation.training_sample.input_features
    )
    assert held_concept_ids is not None
    assert advance_call["arguments"]["pending_assignment_sample_concept_ids"] == held_concept_ids
    assert (
        evaluation_call["arguments"]["input_features"]
        is indexed_observation.training_sample.input_features
    )
    # 帰属の確定の時点では、監視の観測と値の記録が済み、この標本が保留の末尾にある。
    monitoring = sample_processing.loss_monitoring_observation
    assert assignment_call["last_monitoring_observation"] is monitoring
    assert assignment_call["alarm_records"].monitored_log_e_values == (
        *alarm_records_before.monitored_log_e_values,
        monitoring.log_e_value,
    )
    assert assignment_call["pending_sample_indices"] == (
        *pending_sample_indices_before,
        indexed_observation.sample_index,
    )
    assert assignment_call["pending_sample_observations"][-1] is indexed_observation
    assert (
        assignment_call["arguments"]["pending_sample_observations"]
        == assignment_call["pending_sample_observations"]
    )
    # 学習要求の時点では、保留標本が保留位置のownerに残った位置へ合わせてある。
    assert (
        tuple(
            pending_observation.sample_index
            for pending_observation in training_call["pending_sample_observations"]
        )
        == training_call["pending_sample_indices"]
    )
    assert sample_processing.alarm_occurrence_handling is None


def test_sample_with_alarm_runs_steps_in_legacy_order(monkeypatch, valid_run_settings_mapping):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        processing_arguments, _, legacy_client = build_sample_processing_oracle(
            monkeypatch=monkeypatch,
            valid_run_settings_mapping=valid_run_settings_mapping,
            class_count=2,
            recording_oracle_case=RECORDING_ORACLE_CASES[2],
            historical_mean_losses=(0.1, 0.1, 0.1),
            minimum_candidate_mean_loss_improvement=10.0,
        )
        schedule = processing_arguments["local_training_request_schedule"]
        monitor = processing_arguments["loss_change_monitor"]
        alarm_record_store = processing_arguments["loss_change_alarm_record_store"]
        # 損失が大きくなるラベルの標本を処理し、次の標本で警報が起きる状態（かつ学習要求が保留中）まで進める。
        # 次の標本で警報が起きるかは、監視の複製へ同じ損失を観測させて確かめる。
        step_calls = None
        for _ in range(60):
            indexed_observation = make_stream_observation(
                sample_index=make_next_observation(
                    processing_arguments=processing_arguments
                ).sample_index,
                class_count=2,
                stream_name="high_loss_only",
                legacy_client=legacy_client,
            )
            run_legacy_sample_processing(
                legacy_client=legacy_client,
                indexed_observation=indexed_observation,
                legacy_random_state=random.getstate(),
            )
            alarm_occurs_at_this_sample = legacy_client.detected_event_positions[-1:] == [
                indexed_observation.sample_index
            ]
            alarm_records_before = alarm_record_store.get_state_snapshot()
            pending_training_request_count = schedule.pending_training_request_count
            if alarm_occurs_at_this_sample and pending_training_request_count:
                step_calls = record_sample_processing_steps(
                    monkeypatch, processing_arguments=processing_arguments
                )
            sample_processing = process_observed_sample(
                indexed_observation=indexed_observation, **processing_arguments
            )
            if step_calls is not None:
                break
    assert step_calls is not None, "保留中の学習要求がある状態での警報が起きなかった"
    assert monitor.last_observation is None
    assert sample_processing.loss_monitoring_observation.drift_detected
    assert [step_call["step_name"] for step_call in step_calls] == [
        "predict_observed_sample_and_update_prediction_weights",
        "advance_held_candidate_validation",
        "evaluate_classifier_per_sample_bounded_losses",
        "train_held_models_for_pending_training_requests",
        "handle_alarm_occurrence",
    ]
    _, _, _, training_call, alarm_call = step_calls
    # 保留中の学習要求の学習は、警報の位置の記録と警報の処理より前。
    assert training_call["alarm_records"].alarm_sample_indices == (
        alarm_records_before.alarm_sample_indices
    )
    assert schedule.pending_training_request_count == 0
    # 警報の処理の時点では、警報の位置が記録済みで、この標本を含む全保留標本が渡る。
    monitoring = sample_processing.loss_monitoring_observation
    alarm_arguments = alarm_call["arguments"]
    pending_observations = alarm_call["pending_sample_observations"]
    assert alarm_call["alarm_records"].alarm_sample_indices == (
        *alarm_records_before.alarm_sample_indices,
        indexed_observation.sample_index,
    )
    assert alarm_arguments["alarm_sample_index"] == indexed_observation.sample_index
    assert alarm_arguments["pending_sample_observations"] == pending_observations
    assert pending_observations[-1] is indexed_observation
    assert (
        alarm_arguments["estimated_change_span_sample_count"]
        == monitoring.estimated_change_span_sample_count
    )
    assert alarm_arguments["estimated_change_point_sample_index"] == max(
        0,
        indexed_observation.sample_index
        - min(len(pending_observations), monitoring.estimated_change_span_sample_count)
        + 1,
    )
    assert (
        alarm_arguments["estimated_change_point_sample_index"]
        == (alarm_call["alarm_records"].estimated_change_point_sample_indices[-1])
    )
    assert alarm_arguments["detection_episode_id"] is None
    assert sample_processing.alarm_occurrence_handling is not None
    assert sample_processing.released_sample_observations == ()
    # 警報の処理の後、保留標本は保留位置のownerに残った位置と一致する。
    assert tuple(
        pending_observation.sample_index
        for pending_observation in processing_arguments[
            "pending_sample_observation_store"
        ].snapshot_pending_sample_observations()
    ) == (
        processing_arguments["pending_training_assignment_buffer"]
        .get_state_snapshot()
        .pending_sample_indices
    )


def test_failed_step_stops_sample_processing_before_later_steps(
    monkeypatch, valid_run_settings_mapping
):
    """途中の段（帰属の確定）が失敗したら、後の段（学習要求の記録）を呼ばずに、例外をそのまま伝える。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(823)
        processing_arguments, _, _ = build_default_sample_processing_oracle(
            monkeypatch=monkeypatch,
            valid_run_settings_mapping=valid_run_settings_mapping,
            validation_is_held=False,
        )
        indexed_observation = make_next_observation(processing_arguments=processing_arguments)
        step_calls = record_sample_processing_steps(
            monkeypatch, processing_arguments=processing_arguments
        )
        record_assignment_step = (
            sample_processing_module.assign_released_pending_samples_to_current_training_model
        )

        def fail_after_recording_assignment_step(**assignment_arguments):
            record_assignment_step(**assignment_arguments)
            raise RuntimeError("injected failure")

        monkeypatch.setattr(
            sample_processing_module,
            "assign_released_pending_samples_to_current_training_model",
            fail_after_recording_assignment_step,
        )
        schedule = processing_arguments["local_training_request_schedule"]
        pending_training_request_count = schedule.pending_training_request_count
        with pytest.raises(RuntimeError, match="injected failure"):
            process_observed_sample(indexed_observation=indexed_observation, **processing_arguments)
    assert [step_call["step_name"] for step_call in step_calls] == [
        "predict_observed_sample_and_update_prediction_weights",
        "advance_held_candidate_validation",
        "evaluate_classifier_per_sample_bounded_losses",
        "assign_released_pending_samples_to_current_training_model",
    ]
    assert schedule.pending_training_request_count == pending_training_request_count
    # 失敗より前の段（監視、保留への追加）は済んだまま残る。
    assert (
        processing_arguments["pending_training_assignment_buffer"]
        .get_state_snapshot()
        .last_observed_sample_index
        == indexed_observation.sample_index
    )


def test_validation_concept_ids_are_selected_from_latest_pending_observations():
    """候補検証へ渡された標本は保留の末尾。同じ標本のオブジェクトが古い側にもあっても、末尾の概念IDを選ぶ。"""
    select_concept_ids = sample_processing_module._select_validation_assignment_sample_concept_ids
    shared_training_sample = ObservedTrainingSample(
        input_features=torch.tensor([[0.25, 0.5]]), observed_class_labels=torch.tensor([[0.0]])
    )
    other_training_sample = ObservedTrainingSample(
        input_features=torch.tensor([[0.5, 0.25]]), observed_class_labels=torch.tensor([[1.0]])
    )
    pending_sample_observations = tuple(
        IndexedObservedTrainingSample(
            sample_index=sample_index,
            training_sample=training_sample,
            observed_concept_id=observed_concept_id,
        )
        for sample_index, (training_sample, observed_concept_id) in enumerate(
            (
                (shared_training_sample, 1),
                (other_training_sample, None),
                (shared_training_sample, 2),
            )
        )
    )
    assert select_concept_ids(
        pending_sample_observations=pending_sample_observations,
        validation_assignment_training_samples=(shared_training_sample,),
    ) == (2,)
    assert select_concept_ids(
        pending_sample_observations=pending_sample_observations,
        validation_assignment_training_samples=(other_training_sample, shared_training_sample),
    ) == (None, 2)
    assert select_concept_ids(
        pending_sample_observations=pending_sample_observations,
        validation_assignment_training_samples=(
            shared_training_sample,
            other_training_sample,
            shared_training_sample,
        ),
    ) == (1, None, 2)
    assert (
        select_concept_ids(
            pending_sample_observations=pending_sample_observations,
            validation_assignment_training_samples=(),
        )
        == ()
    )
    # 末尾と同一でない標本（値が同じ別のオブジェクト、古い側だけにある並び、保留より多い）は拒否する。
    equal_training_sample = ObservedTrainingSample(
        input_features=shared_training_sample.input_features,
        observed_class_labels=shared_training_sample.observed_class_labels,
    )
    for invalid_training_samples in (
        (equal_training_sample,),
        (other_training_sample,),
        (shared_training_sample, other_training_sample),
        (shared_training_sample,) * 4,
    ):
        with pytest.raises(ValueError, match="latest pending sample observations"):
            select_concept_ids(
                pending_sample_observations=pending_sample_observations,
                validation_assignment_training_samples=invalid_training_samples,
            )
