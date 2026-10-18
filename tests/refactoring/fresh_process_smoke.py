"""新実装だけを別processで動かし、接続済みの流れが旧実装とtest moduleなしで実行できることを確かめる。

worktreeルートで `python tests/refactoring/fresh_process_smoke.py` として実行する（pytestからは
test_fresh_process_smoke.py が別processで実行する）。旧実装（federated_drift_experiment）とtest moduleを
importしない。新しい接続を移植したら、その接続を通る流れをここへ足す。新全体runを接続したら、全体runの
testへ置き換えて廃止する。

現在の流れ: 1つの保持・適応記録・診断証拠・損失監視・保留位置のownerで、
警報のない標本での帰属確定（容量を超えた最古の保留標本を現在のモデルへ確定）と学習要求の記録→
警報（不足／他モデルの再利用／現行の維持／候補検証の開始）→候補検証中の警報、または
標本の観測と確定（採用／棄却／現行の維持／他モデルの再利用）→次の警報→保留中の学習要求の学習→
標本1件の処理（予測→候補検証の進行→監視→保留→警報の処理｜帰属の確定と学習）を続けて呼ぶ。
別の流れとして、初期モデルを事前学習し、その結果と設定からclientを組み立て、実行の枠（参加者の検査と区間の進行。サーバは
ラウンドごとに新規モデルの登録と集約だけを行う代役。配布は未移植）で、標本列を最後まで進める。
"""

import importlib.util
import sys
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch
from numpy.random import RandomState

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.run_provenance import (
    BEHAVIOR_BASELINE_VERSION,
    RunProvenance,
)
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_settings import (
    RandomConceptScheduleSettings,
)
from federated_learning_experiments.data.dataset_definitions import get_dataset_definition
from federated_learning_experiments.data.observed_streams import (
    ClientConceptTrace,
    ClientObservedStream,
    ObservedSample,
)
from federated_learning_experiments.data.sine.sine_sample_generation import SineSampleGenerator
from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationRecordStore
from federated_learning_experiments.evaluation.communication_volume_record_store import (
    CommunicationVolumeRecordStore,
)
from federated_learning_experiments.evaluation.cross_evaluation_record_store import (
    CrossEvaluationRecordStore,
)
from federated_learning_experiments.evaluation.loss_change_alarm_record_store import (
    LossChangeAlarmRecordStore,
)
from federated_learning_experiments.evaluation.model_clustering_record_store import (
    ModelClusteringRecordStore,
)
from federated_learning_experiments.evaluation.model_evaluation_sample_store import (
    ModelEvaluationSampleStore,
)
from federated_learning_experiments.evaluation.run_metric_calculations import RunMetricSettings
from federated_learning_experiments.evaluation.sample_prediction_record_store import (
    SamplePredictionRecordStore,
)
from federated_learning_experiments.evaluation.true_concept_adahedge_diagnostic_evidence_collection import (
    TrueConceptAdaHedgeDiagnosticEvidenceCollection,
)
from federated_learning_experiments.execution.run_participant_contracts import RunParticipants
from federated_learning_experiments.execution.stream_protocol_execution_loop import (
    run_stream_protocol_intervals,
)
from federated_learning_experiments.execution.stream_protocol_execution_settings import (
    StreamProtocolExecutionSettings,
)
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
    ModelAndClassLossStatisticsStore,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.models.model_architecture_classifier import (
    ModelArchitectureClassifier,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.prediction.classifier_bounded_loss_evaluation import (
    evaluate_classifier_per_sample_bounded_losses,
)
from federated_learning_experiments.learning.training.candidate_epoch_training_settings import (
    CandidateEpochTrainingSettings,
)
from federated_learning_experiments.learning.training.current_training_model_assignment import (
    CurrentTrainingModelAssignment,
)
from federated_learning_experiments.learning.training.held_model_training_state_registry import (
    HeldModelTrainingStateRegistry,
)
from federated_learning_experiments.learning.training.indexed_observed_training_sample import (
    IndexedObservedTrainingSample,
)
from federated_learning_experiments.learning.training.initial_model_pretraining_settings import (
    InitialModelPretrainingSettings,
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
from federated_learning_experiments.learning.training.model_training_and_assignment_counts import (
    ModelTrainingAndAssignmentCountsStore,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.learning.training.model_training_sample_store import (
    ModelTrainingSampleStore,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.learning.training.temporary_model_id_allocation import (
    TemporaryModelIdAllocator,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_model_training_and_acceptance_settings import (
    CandidateModelTrainingAndAcceptanceSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import (
    CandidateParameterInitializationSettings,
)
from federated_learning_experiments.methods.fedsda.consolidation.model_clustering_calculations import (
    ModelClusteringCriteria,
)
from federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings import (
    RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
    LossChangeDetectionSettings,
)
from federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring import (
    OverallAndTrueClassLossMonitor,
)
from federated_learning_experiments.methods.fedsda.model_registration.global_model_repository import (
    GlobalModelRepository,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_registration_settings import (
    KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME,
    PendingModelRegistrationSettings,
)
from federated_learning_experiments.methods.fedsda.model_registration.pending_model_upload import (
    PendingModelUploadState,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.all_sample_adahedge_weight_evidence_holder import (
    AllSampleAdaHedgeWeightEvidenceHolder,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights import (
    FixedShareWeightController,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.prediction_combination_settings import (
    PredictionCombinationSettings,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_sample_observation_store import (
    PendingSampleObservationStore,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.pending_training_assignment_buffer import (
    PendingTrainingAssignmentBuffer,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings import (
    KEEP_ALL_PENDING_SAMPLES_POLICY_NAME,
    TrainingDataAssignmentSettings,
)
from federated_learning_experiments.runtime.alarm_occurrence_handling import (
    handle_alarm_occurrence,
)
from federated_learning_experiments.runtime.candidate_validation_session_holder import (
    CandidateValidationSessionHolder,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_final_configuration_run_settings import (
    build_final_configuration_fedsda_run_settings,
)
from federated_learning_experiments.runtime.fedsda_measured_run_execution import (
    execute_fedsda_stream_protocol_run_with_computation_measurement,
)
from federated_learning_experiments.runtime.fedsda_run_artifact_files import (
    load_fedsda_run_artifacts,
    save_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_run_client import assemble_fedsda_run_client
from federated_learning_experiments.runtime.fedsda_run_client_settings import (
    FedsdaRunClientScalarSettings,
    FedsdaRunClientSettings,
)
from federated_learning_experiments.runtime.fedsda_run_metric_derivation import (
    derive_fedsda_run_metrics,
)
from federated_learning_experiments.runtime.fedsda_run_participant_factory import (
    FedsdaRunParticipantSettings,
)
from federated_learning_experiments.runtime.held_candidate_validation_progress import (
    advance_held_candidate_validation,
)
from federated_learning_experiments.runtime.held_model_training_request_handling import (
    record_training_request_and_train_held_models_when_due,
    train_held_models_for_pending_training_requests,
)
from federated_learning_experiments.runtime.initial_model_pretraining import pretrain_initial_model
from federated_learning_experiments.runtime.observed_sample_processing import (
    process_observed_sample,
)
from federated_learning_experiments.runtime.released_pending_sample_assignment import (
    assign_released_pending_samples_to_current_training_model,
)
from federated_learning_experiments.runtime.server_round_synchronization import (
    synchronize_models_in_server_round,
)
from federated_learning_experiments.runtime.single_run_execution import (
    validate_prepared_run_participants,
)

HELD_MODEL_IDS = (7, -103, 2)
SAMPLE_COUNT = 11
# 各流れの最後に、標本1件の処理を続けて呼ぶ回数。
PROCESSED_SAMPLE_COUNT = 30
VALIDATION_SAMPLE_COUNT = 4
DETECTOR_NAME = "ClassESR"
CURRENT_MODEL_ID = 2
OTHER_MODEL_ID = -103
EPISODE_ID = 1
ALARM_OUTCOMES_WITHOUT_ACTIVE_VALIDATION = (
    "alarm_change_interval_too_short",
    "alarm_interval_held_model_reused",
    "alarm_interval_current_model_maintained",
    "alarm_interval_candidate_validation_started",
)
MODEL_SWITCHING_OUTCOMES = (
    "alarm_interval_held_model_reused",
    "post_alarm_validation_candidate_adopted",
    "post_alarm_validation_held_model_reused",
)
# (名前, 1回目の警報の推定区間長, 警報時に履歴基準へ適合させるモデル, 期待する1回目の結果, 候補検証の進め方)
# 候補検証の進め方: None（開始しない）、"alarm_during_validation"（未到達のまま次の警報）、
# または (参照の許容損失増加, 候補に要求する改善, 期待する確定の結果)。許容損失増加がNoneなら、
# 他の保有モデルだけが履歴基準に適合する値を実行時に決める。
SMOKE_SCENARIOS = (
    ("too_short", 2, None, "alarm_change_interval_too_short", None),
    ("reuse_at_alarm", 99, OTHER_MODEL_ID, "alarm_interval_held_model_reused", None),
    ("maintain_at_alarm", 99, CURRENT_MODEL_ID, "alarm_interval_current_model_maintained", None),
    (
        "alarm_during_validation",
        99,
        None,
        "alarm_interval_candidate_validation_started",
        "alarm_during_validation",
    ),
    (
        "adopt_candidate",
        99,
        None,
        "alarm_interval_candidate_validation_started",
        (0.0, 0.0, "post_alarm_validation_candidate_adopted"),
    ),
    (
        "reject_candidate",
        99,
        None,
        "alarm_interval_candidate_validation_started",
        (0.0, 1.0, "post_alarm_validation_candidate_rejected"),
    ),
    (
        "maintain_after_validation",
        99,
        None,
        "alarm_interval_candidate_validation_started",
        (1.0, 1.0, "post_alarm_validation_current_model_maintained"),
    ),
    (
        "reuse_other_after_validation",
        99,
        None,
        "alarm_interval_candidate_validation_started",
        (None, 1.0, "post_alarm_validation_held_model_reused"),
    ),
)


# 保存の関数へ渡すrunの来歴（test moduleをimportしないので、ここで作る）。
TEST_RUN_PROVENANCE = RunProvenance(
    code_commit="0123456789abcdef0123456789abcdef01234567",
    has_uncommitted_changes=False,
    code_revision_unavailable_reason=None,
    behavior_baseline_version=BEHAVIOR_BASELINE_VERSION,
)


def run_smoke_scenario(*, class_count, smoke_scenario):
    """1つの流れを実行し、観測した適応結果の列を返す。食い違いはAssertionError。"""
    (
        scenario_name,
        change_span_sample_count,
        fitting_model_id,
        expected_first_outcome,
        validation_plan,
    ) = smoke_scenario
    torch.manual_seed(881 + class_count)
    acceptance_settings = CandidateModelTrainingAndAcceptanceSettings(
        candidate_model_acceptance_policy="current_model_first_reuse_then_two_segment_candidate_validation",
        candidate_post_alarm_validation_sample_count=VALIDATION_SAMPLE_COUNT,
    )
    parameter_optimizer_settings = AdamParameterOptimizerSettings(
        learning_rate=0.05, weight_decay=0.0, adam_variant="standard"
    )
    registry = HeldModelTrainingStateRegistry()
    classifiers = {}
    for model_id in HELD_MODEL_IDS:
        classifier = ModelArchitectureClassifier(
            model_architecture_settings=ModelArchitectureSettings(
                model_architecture_name="shared_backbone_residual_adapter",
                residual_adapter_requested_rank=2,
            ),
            input_feature_count=3,
            hidden_layer_widths=(5,),
            class_count=class_count,
            shared_feature_extractor=classifiers[7].feature_extractor if classifiers else None,
        )
        classifiers[model_id] = classifier
        registry.register_held_model_training_state(
            model_id=model_id,
            classifier=classifier,
            concept_specific_parameter_optimizer_state=ParameterOptimizerState(
                parameters=tuple(classifier.model_owned_feature_transform.parameters())
                + tuple(classifier.classification_layer.parameters()),
                optimizer_settings=parameter_optimizer_settings,
            ),
        )
    input_features = (
        torch.arange(3 * SAMPLE_COUNT, dtype=torch.float32).reshape(SAMPLE_COUNT, 3) / 33
    )
    observed_class_labels = (
        (torch.arange(SAMPLE_COUNT) * class_count // SAMPLE_COUNT).float().reshape(SAMPLE_COUNT, 1)
    )

    def make_indexed_observation(*, sample_index, source_position):
        return IndexedObservedTrainingSample(
            sample_index=sample_index,
            training_sample=ObservedTrainingSample(
                input_features=input_features[source_position : source_position + 1],
                observed_class_labels=observed_class_labels[source_position : source_position + 1],
            ),
            observed_concept_id=None if sample_index % 4 == 2 else sample_index % 2,
        )

    def evaluate_mean_losses(*, sample_count):
        return {
            model_id: torch.mean(
                evaluate_classifier_per_sample_bounded_losses(
                    classifier=classifier,
                    input_features=input_features[:sample_count],
                    observed_class_labels=observed_class_labels[:sample_count],
                )
            ).item()
            for model_id, classifier in classifiers.items()
        }

    # 位置0は警報より前に確定する標本。警報のときに保留されているのは位置1からSAMPLE_COUNTまで。
    earliest_observation = make_indexed_observation(sample_index=0, source_position=0)
    first_alarm_observations = tuple(
        make_indexed_observation(sample_index=source_position + 1, source_position=source_position)
        for source_position in range(SAMPLE_COUNT)
    )
    interval_mean_losses = evaluate_mean_losses(sample_count=SAMPLE_COUNT)
    reuses_other_after_validation = (
        isinstance(validation_plan, tuple)
        and validation_plan[2] == "post_alarm_validation_held_model_reused"
    )
    # 履歴基準を区間平均より十分小さくすると、警報時にそのモデルは適合しない。同じにすると適合する。
    # 検証後に他モデルを再利用する流れだけ、他モデルの履歴基準を高めにして検証時に適合しやすくする。
    historical_mean_losses = {
        model_id: interval_mean_losses[model_id]
        * (
            1.0
            if model_id == fitting_model_id
            else 0.7
            if reuses_other_after_validation and model_id == OTHER_MODEL_ID
            else 0.25
        )
        for model_id in HELD_MODEL_IDS
    }
    loss_statistics_store = ModelAndClassLossStatisticsStore()
    for model_id in HELD_MODEL_IDS:
        loss_statistics_store.set_model_loss_statistics(
            model_id=model_id,
            loss_statistics=ModelAndClassLossStatistics(
                overall_loss_moments=BoundedLossMoments(
                    # 警報の前に確定する1標本で履歴基準がほとんど動かない件数にする。
                    observed_loss_count=3000,
                    mean_loss=historical_mean_losses[model_id],
                    sum_squared_loss_deviations=0.0,
                )
            ),
        )
    pending_training_assignment_buffer = PendingTrainingAssignmentBuffer(
        training_data_assignment_settings=TrainingDataAssignmentSettings(
            pending_sample_retention_after_too_short_change_interval_policy=KEEP_ALL_PENDING_SAMPLES_POLICY_NAME,
            pending_assignment_buffer_capacity_samples=10,
        )
    )
    current_training_model_assignment = CurrentTrainingModelAssignment(
        initial_model_id=CURRENT_MODEL_ID
    )
    training_sample_store = ModelTrainingSampleStore()
    counts_store = ModelTrainingAndAssignmentCountsStore()
    validation_session_holder = CandidateValidationSessionHolder()
    adaptation_record_store = AdaptationRecordStore()
    all_sample_adahedge_weight_evidence_holder = AllSampleAdaHedgeWeightEvidenceHolder()
    all_sample_weight_evidence = (
        all_sample_adahedge_weight_evidence_holder.all_sample_weight_evidence
    )
    # 再始動で消えることを確かめるため、診断証拠へ損失を1回与えておく。
    diagnostic_losses = {CURRENT_MODEL_ID: 0.8, OTHER_MODEL_ID: 0.1}
    all_sample_weight_evidence.update_evidence_after_loss_observation(
        observed_losses_by_model_id=diagnostic_losses,
        weights_by_model_id=all_sample_weight_evidence.get_weights_before_loss_observation(
            model_ids=tuple(diagnostic_losses)
        ),
    )
    loss_change_monitor = OverallAndTrueClassLossMonitor(
        loss_change_detection_settings=LossChangeDetectionSettings(
            short_change_interval_alarm_policy=RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
            drift_detector_name="e_sr",
            loss_monitoring_scope="overall_and_true_class_losses",
            e_sr_false_alarm_control_alpha=0.05,
            adwin_confidence_delta=0.05,
            adwin_minimum_test_window_sample_count=10,
        ),
        class_count=class_count,
        initial_baseline_loss_mean=0.5,
        maximum_retained_candidate_count=7,
        betting_fractions=(0.05, 0.1, 0.2, 0.4, 0.8),
    )
    training_owners = dict(
        held_model_training_state_registry=registry,
        loss_statistics_store=loss_statistics_store,
        training_sample_store=training_sample_store,
        model_training_and_assignment_counts_store=counts_store,
        current_training_model_assignment=current_training_model_assignment,
    )
    # 警報のない標本: 容量＋1件が保留された時点で、最古の1件（位置0）が現在のモデルへ確定する。
    observations_before_alarm = (earliest_observation, *first_alarm_observations[:-1])
    for indexed_observation in observations_before_alarm:
        pending_training_assignment_buffer.append_observed_sample_index(
            sample_index=indexed_observation.sample_index
        )
    released_sample_observations = assign_released_pending_samples_to_current_training_model(
        pending_training_assignment_buffer=pending_training_assignment_buffer,
        pending_sample_observations=observations_before_alarm,
        **training_owners,
    )
    assert released_sample_observations == (earliest_observation,)
    assert pending_training_assignment_buffer.get_state_snapshot().pending_sample_indices == (
        tuple(
            indexed_observation.sample_index
            for indexed_observation in first_alarm_observations[:-1]
        )
    )
    assert [
        (collection.model_id, len(collection.training_samples))
        for collection in training_sample_store.snapshot_ordered_model_training_samples()
    ] == [(CURRENT_MODEL_ID, 1)]
    # 同じ標本の学習要求: 間隔2の1件目なので保留されるだけで、学習は行われない。
    local_training_request_schedule = LocalTrainingRequestSchedule(
        local_training_schedule_settings=LocalTrainingScheduleSettings(
            training_requests_per_update_interval=2,
            joint_update_iterations_per_training_request=3,
        )
    )
    training_request_arguments = dict(
        local_training_request_schedule=local_training_request_schedule,
        held_model_training_state_registry=registry,
        training_sample_store=training_sample_store,
        model_training_and_assignment_counts_store=counts_store,
        batch_sample_count=1,
        python_random_generator=Random(431),
        local_training_settings=LocalTrainingSettings(
            local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
            shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
        ),
        shared_feature_extractor=classifiers[7].feature_extractor,
        shared_parameter_optimizer=ParameterOptimizerState(
            parameters=tuple(classifiers[7].feature_extractor.parameters()),
            optimizer_settings=parameter_optimizer_settings,
        ).parameter_optimizer,
    )
    assert (
        record_training_request_and_train_held_models_when_due(**training_request_arguments) == ()
    )
    assert local_training_request_schedule.pending_training_request_count == 1
    # 警報が起きる標本を保留へ足す（警報のときは、容量を超えた分を解放する前に応答する）。
    pending_training_assignment_buffer.append_observed_sample_index(
        sample_index=first_alarm_observations[-1].sample_index
    )
    # 候補の採用に使うowner（警報の処理と、候補検証の進行で、同じものを使う）。
    temporary_model_id_allocator = TemporaryModelIdAllocator(client_id=5)
    pending_model_upload_state = PendingModelUploadState(
        pending_model_registration_policy=KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME
    )
    alarm_handling_arguments = dict(
        temporary_model_id_allocator=temporary_model_id_allocator,
        pending_model_upload_state=pending_model_upload_state,
        upload_delay_round_count=1,
        validation_session_holder=validation_session_holder,
        adaptation_record_store=adaptation_record_store,
        all_sample_adahedge_weight_evidence_holder=all_sample_adahedge_weight_evidence_holder,
        loss_change_monitor=loss_change_monitor,
        pending_training_assignment_buffer=pending_training_assignment_buffer,
        minimum_change_interval_sample_count=3,
        model_evaluation_sample_store=ModelEvaluationSampleStore(
            maximum_stored_sample_count_per_model=3, added_batch_sample_count=2
        ),
        python_random_generator=Random(712),
        maximum_alarm_interval_mean_loss_increase=min(interval_mean_losses.values()) / 8,
        candidate_parameter_initialization_settings=CandidateParameterInitializationSettings(
            candidate_parameter_initialization_source="lowest_evaluated_mean_loss_model"
        ),
        architecture_reference_classifier=classifiers[7],
        parameter_optimizer_settings=parameter_optimizer_settings,
        candidate_epoch_training_settings=CandidateEpochTrainingSettings(
            candidate_training_strategy="validation_loss_early_stopping",
            maximum_epoch_count=400,
            maximum_batch_sample_count=3,
            validation_sample_fraction=0.2,
            consecutive_non_improving_epoch_limit=400,
            minimum_validation_loss_decrease=0.0,
        ),
        candidate_model_training_and_acceptance_settings=acceptance_settings,
        detection_episode_id=EPISODE_ID,
        detector_name=DETECTOR_NAME,
        **training_owners,
    )
    observed_outcomes = []

    def handle_alarm_and_check_owners(
        *, alarm_sample_index, pending_sample_observations, estimated_change_span_sample_count
    ):
        """警報を1回処理し、戻り値と記録・保持・監視・保留位置・診断の状態が互いに合うことを確かめる。"""
        model_id_before_alarm = current_training_model_assignment.current_training_model_id
        restart_count_before_alarm = all_sample_weight_evidence.concept_operation_restart_count
        alarm_occurrence_handling = handle_alarm_occurrence(
            alarm_sample_index=alarm_sample_index,
            pending_sample_observations=pending_sample_observations,
            estimated_change_span_sample_count=estimated_change_span_sample_count,
            estimated_change_point_sample_index=(
                pending_sample_observations[0].sample_index if pending_sample_observations else None
            ),
            **alarm_handling_arguments,
        )
        adaptation_record = alarm_occurrence_handling.adaptation_record
        alarm_buffer_response = (
            alarm_occurrence_handling.alarm_response_completion.alarm_buffer_response
        )
        response_outcome = alarm_buffer_response.response_outcome
        assert (
            adaptation_record_store.get_state_snapshot().adaptation_records[-1] == adaptation_record
        )
        assert (
            adaptation_record.adaptation_sample_index,
            adaptation_record.detector_name,
            adaptation_record.adaptation_outcome,
            adaptation_record.current_training_model_id,
            adaptation_record.detection_episode_id,
        ) == (
            alarm_sample_index,
            DETECTOR_NAME,
            response_outcome,
            current_training_model_assignment.current_training_model_id,
            EPISODE_ID,
        )
        assert (
            validation_session_holder.held_validation_session
            is alarm_buffer_response.active_validation_session
        )
        assert (validation_session_holder.held_validation_session is not None) == (
            response_outcome
            in ("alarm_interval_candidate_validation_started", "alarm_during_candidate_validation")
        )
        assert loss_change_monitor.last_observation is None
        pending_assignment_state = pending_training_assignment_buffer.get_state_snapshot()
        assert pending_assignment_state.last_observed_sample_index == alarm_sample_index
        assert (pending_assignment_state.pending_sample_indices != ()) == (
            response_outcome == "alarm_change_interval_too_short"
        )
        # 診断の再始動は、警報で他モデルを再利用したとき（学習帰属が変わったとき）だけ1回。
        model_switched = response_outcome == "alarm_interval_held_model_reused"
        assert (
            current_training_model_assignment.current_training_model_id != model_id_before_alarm
        ) == model_switched
        assert all_sample_weight_evidence.concept_operation_restart_count == (
            restart_count_before_alarm + int(model_switched)
        )
        observed_outcomes.append(response_outcome)
        return response_outcome

    first_alarm_sample_index = SAMPLE_COUNT
    first_outcome = handle_alarm_and_check_owners(
        alarm_sample_index=first_alarm_sample_index,
        pending_sample_observations=first_alarm_observations,
        estimated_change_span_sample_count=change_span_sample_count,
    )
    assert first_outcome == expected_first_outcome, (scenario_name, first_outcome)
    if validation_plan is not None:
        started_validation_session = validation_session_holder.held_validation_session
        if validation_plan == "alarm_during_validation":
            observed_validation_sample_count = 2
            maximum_increase, minimum_improvement, expected_completion_outcome = 0.0, 0.0, None
        else:
            observed_validation_sample_count = VALIDATION_SAMPLE_COUNT
            maximum_increase, minimum_improvement, expected_completion_outcome = validation_plan
        if maximum_increase is None:
            # 検証標本での平均損失と履歴基準の差が、他モデルだけ許容内になる値を選ぶ。
            validation_mean_losses = evaluate_mean_losses(sample_count=VALIDATION_SAMPLE_COUNT)
            validation_loss_increases = {
                model_id: validation_mean_losses[model_id] - historical_mean_losses[model_id]
                for model_id in HELD_MODEL_IDS
            }
            other_increase = validation_loss_increases[OTHER_MODEL_ID]
            smallest_remaining_increase = min(
                increase
                for model_id, increase in validation_loss_increases.items()
                if model_id != OTHER_MODEL_ID
            )
            assert 0.0 <= other_increase < smallest_remaining_increase, validation_loss_increases
            maximum_increase = (other_increase + smallest_remaining_increase) / 2
        later_observations = tuple(
            make_indexed_observation(
                sample_index=first_alarm_sample_index + 1 + sample_offset,
                source_position=sample_offset,
            )
            for sample_offset in range(observed_validation_sample_count)
        )
        model_id_before_validation = current_training_model_assignment.current_training_model_id
        restart_count_before_validation = all_sample_weight_evidence.concept_operation_restart_count
        held_validation_advance = None
        for indexed_observation in later_observations:
            # 未到達の間: 保持が続き、診断は再始動しない。
            assert validation_session_holder.held_validation_session is started_validation_session
            assert (
                all_sample_weight_evidence.concept_operation_restart_count
                == restart_count_before_validation
            )
            pending_training_assignment_buffer.append_observed_sample_index(
                sample_index=indexed_observation.sample_index
            )
            held_validation_advance = advance_held_candidate_validation(
                validation_session_holder=validation_session_holder,
                adaptation_record_store=adaptation_record_store,
                all_sample_adahedge_weight_evidence_holder=all_sample_adahedge_weight_evidence_holder,
                sample_index=indexed_observation.sample_index,
                input_features=indexed_observation.training_sample.input_features,
                observed_class_labels=indexed_observation.training_sample.observed_class_labels,
                candidate_model_training_and_acceptance_settings=acceptance_settings,
                maximum_reference_mean_loss_increase=maximum_increase,
                minimum_candidate_mean_loss_improvement=minimum_improvement,
                pending_assignment_sample_concept_ids=tuple(
                    indexed_observation.observed_concept_id
                    for indexed_observation in first_alarm_observations
                ),
                temporary_model_id_allocator=temporary_model_id_allocator,
                upload_delay_round_count=1,
                pending_model_upload_state=pending_model_upload_state,
                **training_owners,
            )
        assert held_validation_advance is not None
        if expected_completion_outcome is None:
            # 候補検証が未到達のまま次の警報: 保持中のsessionが応答へ渡り、保持は変わらない。
            assert held_validation_advance.adaptation_record is None
            assert validation_session_holder.held_validation_session is started_validation_session
        else:
            # 確定: 記録が増え、保持が空になり、学習帰属が変わったときだけ診断が1回再始動する。
            completion_record = held_validation_advance.adaptation_record
            assert completion_record is not None
            assert completion_record.adaptation_outcome == expected_completion_outcome, (
                scenario_name,
                completion_record.adaptation_outcome,
            )
            assert validation_session_holder.held_validation_session is None
            model_switched = expected_completion_outcome in MODEL_SWITCHING_OUTCOMES
            assert (
                current_training_model_assignment.current_training_model_id
                != model_id_before_validation
            ) == model_switched
            assert all_sample_weight_evidence.concept_operation_restart_count == (
                restart_count_before_validation + int(model_switched)
            )
            observed_outcomes.append(expected_completion_outcome)
        second_outcome = handle_alarm_and_check_owners(
            alarm_sample_index=later_observations[-1].sample_index,
            pending_sample_observations=later_observations,
            estimated_change_span_sample_count=99,
        )
        if expected_completion_outcome is None:
            assert second_outcome == "alarm_during_candidate_validation"
            assert validation_session_holder.held_validation_session is started_validation_session
        else:
            assert second_outcome in ALARM_OUTCOMES_WITHOUT_ACTIVE_VALIDATION, second_outcome
    # 保留中の学習要求の学習: 標本を持つ保有モデルだけが、要求1件ぶん（3回）共同学習され、計数へ反映される。
    held_model_ids_at_training = frozenset(
        training_state.model_id
        for training_state in registry.snapshot_ordered_held_model_training_states()
    )
    trained_model_ids = tuple(
        collection.model_id
        for collection in training_sample_store.snapshot_ordered_model_training_samples()
        if collection.training_samples and collection.model_id in held_model_ids_at_training
    )
    assert trained_model_ids
    counts_before_training = counts_store.snapshot_model_training_and_assignment_counts()
    completed_joint_update_losses = train_held_models_for_pending_training_requests(
        **training_request_arguments
    )
    assert len(completed_joint_update_losses) == 3
    assert local_training_request_schedule.pending_training_request_count == 0
    counts_after_training = counts_store.snapshot_model_training_and_assignment_counts()
    for model_id in trained_model_ids:
        assert counts_after_training.parameter_update_step_counts_by_model_id[model_id] == (
            counts_before_training.parameter_update_step_counts_by_model_id.get(model_id, 0) + 3
        )
        assert counts_after_training.trained_sample_counts_by_model_id[model_id] == (
            counts_before_training.trained_sample_counts_by_model_id.get(model_id, 0) + 3
        )
    assert train_held_models_for_pending_training_requests(**training_request_arguments) == ()
    state_snapshot = adaptation_record_store.get_state_snapshot()
    assert tuple(record.adaptation_outcome for record in state_snapshot.adaptation_records) == (
        tuple(observed_outcomes)
    )
    assert all_sample_weight_evidence.concept_operation_restart_count == sum(
        outcome in MODEL_SWITCHING_OUTCOMES for outcome in observed_outcomes
    )
    # 標本1件の処理: ここまでの流れの後の状態から、標本を続けて処理する。
    # 保留標本のownerは、保留位置のownerに残っている位置の標本と、保持中の候補検証へ渡した標本の概念IDから始める。
    pending_sample_observation_store = PendingSampleObservationStore()
    pending_assignment_state = pending_training_assignment_buffer.get_state_snapshot()
    for sample_index in pending_assignment_state.pending_sample_indices:
        pending_sample_observation_store.append_pending_sample_observation(
            indexed_observation=make_indexed_observation(
                sample_index=sample_index, source_position=sample_index % SAMPLE_COUNT
            )
        )
    held_validation_session = validation_session_holder.held_validation_session
    if held_validation_session is not None:
        pending_sample_observation_store.hold_validation_assignment_sample_concept_ids(
            sample_concept_ids=(None,)
            * len(held_validation_session.pending_assignment_training_samples)
        )
    loss_change_alarm_record_store = LossChangeAlarmRecordStore()
    # 予測のowner。Fixed-Shareの時間尺度は、保留の容量と同じにする。
    fixed_share_weight_controller = FixedShareWeightController(
        prediction_combination_settings=PredictionCombinationSettings(
            prediction_combination_strategy="fixed_share_weighted_prediction",
            prediction_mixture_activation_policy="always",
            prediction_weight_recalibration_after_aggregation_policy=(
                "recompute_buffer_losses_and_replay_weight_updates"
            ),
            prediction_state_reset_on_training_assignment_change_policy=(
                "restart_adahedge_preserve_fixed_share_prediction_state"
            ),
            fixed_share_weight_redistribution_time_scale_samples=10,
            prediction_model_parameter_source="current_local_parameters",
        )
    )
    sample_prediction_record_store = SamplePredictionRecordStore()
    sample_processing_arguments = {
        argument_name: argument
        for argument_name, argument in alarm_handling_arguments.items()
        if argument_name != "detection_episode_id"
    } | dict(
        training_request_arguments,
        loss_change_alarm_record_store=loss_change_alarm_record_store,
        fixed_share_weight_controller=fixed_share_weight_controller,
        sample_prediction_record_store=sample_prediction_record_store,
        true_concept_adahedge_diagnostic_evidence_collection=TrueConceptAdaHedgeDiagnosticEvidenceCollection(),
        prediction_combination_strategy="fixed_share_weighted_prediction",
        pending_sample_observation_store=pending_sample_observation_store,
        temporary_model_id_allocator=TemporaryModelIdAllocator(client_id=6),
        pending_model_upload_state=PendingModelUploadState(
            pending_model_registration_policy=KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME
        ),
        maximum_reference_mean_loss_increase=0.0,
        minimum_candidate_mean_loss_improvement=0.0,
        upload_delay_round_count=1,
        short_change_interval_alarm_policy=RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
    )
    record_count_before_processing = len(state_snapshot.adaptation_records)
    first_processed_sample_index = pending_assignment_state.last_observed_sample_index + 1
    processed_alarm_count = 0
    completed_validation_count = 0
    for sample_offset in range(PROCESSED_SAMPLE_COUNT):
        sample_index = first_processed_sample_index + sample_offset
        sample_processing = process_observed_sample(
            indexed_observation=make_indexed_observation(
                sample_index=sample_index, source_position=(sample_offset * 3) % SAMPLE_COUNT
            ),
            **sample_processing_arguments,
        )
        assert sample_processing.loss_monitoring_observation.sample_index == sample_index
        # 予測: 標本ごとに記録が1件増え、保有する全モデル（複数）の重みの総和が1で、最大重みのモデルは保有モデル。
        sample_prediction = sample_processing.observed_sample_prediction
        sample_prediction_records = (
            sample_prediction_record_store.snapshot_sample_prediction_records()
        )
        assert len(sample_prediction_records) == sample_offset + 1
        assert sample_prediction_records[-1] is sample_prediction.sample_prediction_record
        assert sample_prediction.sample_prediction_record.sample_index == sample_index
        prediction_weights_by_model_id = sample_prediction.prediction_weights_by_model_id
        assert len(prediction_weights_by_model_id) >= 2, prediction_weights_by_model_id
        assert abs(sum(prediction_weights_by_model_id.values()) - 1.0) <= 1e-12
        assert (
            sample_prediction.sample_prediction_record.maximum_weight_model_id
            in prediction_weights_by_model_id
        )
        assert sample_prediction.predicted_class_labels.shape == (1, 1)
        alarm_occurred = sample_processing.loss_monitoring_observation.drift_detected
        assert (sample_processing.alarm_occurrence_handling is not None) == alarm_occurred
        processed_alarm_count += int(alarm_occurred)
        completed_validation_count += int(
            sample_processing.held_validation_advance.adaptation_record is not None
        )
        # 標本ごとに: 保留標本と保留位置が同じ並び、概念IDの保持は候補検証の保持と同じ、位置は連続。
        pending_assignment_state = pending_training_assignment_buffer.get_state_snapshot()
        assert pending_assignment_state.last_observed_sample_index == sample_index
        assert pending_assignment_state.pending_sample_indices == tuple(
            pending_observation.sample_index
            for pending_observation in pending_sample_observation_store.snapshot_pending_sample_observations()
        )
        assert (
            pending_sample_observation_store.validation_assignment_sample_concept_ids is None
        ) == (validation_session_holder.held_validation_session is None)
    alarm_record_snapshot = loss_change_alarm_record_store.get_state_snapshot()
    assert len(alarm_record_snapshot.monitored_log_e_values) == PROCESSED_SAMPLE_COUNT
    assert len(alarm_record_snapshot.alarm_sample_indices) == processed_alarm_count
    processed_outcomes = tuple(
        record.adaptation_outcome
        for record in adaptation_record_store.get_state_snapshot().adaptation_records[
            record_count_before_processing:
        ]
    )
    # 警報1回につき記録が1件、候補検証の確定1回につき記録が1件。
    assert len(processed_outcomes) == processed_alarm_count + completed_validation_count
    PROCESSED_ALARM_COUNTS.append(processed_alarm_count)
    # 予測重みは、標本の処理の後、均等のままではない（ラベル観測後の更新が行われている）。
    updated_prediction_weights = fixed_share_weight_controller.weights_by_model_id
    assert len(set(updated_prediction_weights.values())) > 1, updated_prediction_weights
    return tuple(observed_outcomes)


# 流れごとの、標本1件の処理の中で起きた警報の回数（全体で1回以上あることを確かめる）。
PROCESSED_ALARM_COUNTS = []


class ServerOperationsSynchronizingModels:
    """実行の枠が求めるサーバの操作の代役。ラウンドごとに、サーバの1ラウンドの同期の関数を呼ぶ。

    同期: 新規モデルの登録→集約→（新規モデルがあり、モデルが2つ以上なら）クロス評価→クラスタリングと統合→
    配布→全clientの再較正。呼ぶたびに、各段の結果と、サーバとclientの状態の対応を確かめる。
    """

    def __init__(
        self,
        *,
        run_clients,
        global_model_repository,
        communication_volume_record_store,
        python_random_generator,
    ):
        self.run_clients = run_clients
        self.global_model_repository = global_model_repository
        self.communication_volume_record_store = communication_volume_record_store
        self.python_random_generator = python_random_generator
        self.cross_evaluation_record_store = CrossEvaluationRecordStore()
        self.model_clustering_record_store = ModelClusteringRecordStore()
        self.synchronizations = []
        self.registered_model_count = 0
        self.distributed_model_count = 0
        self.consolidation_message_count = 0
        self.absorbed_model_count = 0
        self.replayed_recalibration_sample_count = 0

    def record_client_states_before_synchronization(self, *, round_index):
        self.communication_volume_record_store.record_messages(
            transfer_direction="upload", message_count=len(self.run_clients)
        )

    def synchronize_models(self, *, round_index, new_model_registration_available):
        cross_evaluation_record_count = len(
            self.cross_evaluation_record_store.snapshot_client_cross_evaluation_records()
        )
        model_observation_count = len(
            self.model_clustering_record_store.snapshot_model_clustering_observations()
        )
        global_model_ids_before = self.global_model_repository.global_model_ids
        synchronization = synchronize_models_in_server_round(
            run_clients=self.run_clients,
            global_model_repository=self.global_model_repository,
            communication_volume_record_store=self.communication_volume_record_store,
            cross_evaluation_record_store=self.cross_evaluation_record_store,
            model_clustering_record_store=self.model_clustering_record_store,
            model_clustering_criteria=ModelClusteringCriteria(
                maximum_same_cluster_decision_score=0.1,
                minimum_pair_evaluation_sample_count=5,
                clustering_confidence_level=0.95,
                model_pair_comparison_strategy="classwise_unique_correctness_lower_confidence_bound",
            ),
            maximum_evaluating_client_count_per_model=len(self.run_clients),
            python_random_generator=self.python_random_generator,
            round_index=round_index,
            model_clustering_enabled=new_model_registration_available,
        )
        self.synchronizations.append(synchronization)
        # 登録: 送信できるモデルを持つclientがいたラウンドだけ。
        assert bool(synchronization.registered_client_models) == new_model_registration_available
        self.registered_model_count += len(synchronization.registered_client_models)
        aggregated_model_ids = synchronization.client_model_aggregation.aggregated_global_model_ids
        assert set(global_model_ids_before) <= set(aggregated_model_ids)
        # クロス評価とクラスタリング: 新規モデルがあり、モデルが2つ以上のラウンドだけ。
        clustered = new_model_registration_available and len(aggregated_model_ids) > 1
        assert (synchronization.model_cross_evaluation is not None) == clustered
        assert (synchronization.model_consolidation is not None) == clustered
        added_cross_evaluation_records = (
            self.cross_evaluation_record_store.snapshot_client_cross_evaluation_records()[
                cross_evaluation_record_count:
            ]
        )
        added_model_observations = (
            self.model_clustering_record_store.snapshot_model_clustering_observations()[
                model_observation_count:
            ]
        )
        absorbed_model_ids = ()
        if clustered:
            model_cross_evaluation = synchronization.model_cross_evaluation
            model_consolidation = synchronization.model_consolidation
            assert model_cross_evaluation.cross_evaluated_model_ids == aggregated_model_ids
            assert list(model_cross_evaluation.loss_sums_by_candidate_and_target_model_id) == [
                (candidate_model_id, target_model_id)
                for candidate_model_id in aggregated_model_ids
                for target_model_id in aggregated_model_ids
            ]
            # 表の件数は、clientの評価の記録を足し合わせたもの。
            for (
                model_pair,
                loss_sums,
            ) in model_cross_evaluation.loss_sums_by_candidate_and_target_model_id.items():
                assert loss_sums.evaluated_sample_count == sum(
                    record.evaluated_sample_count
                    for record in added_cross_evaluation_records
                    if (record.candidate_model_id, record.target_model_id) == model_pair
                )
                assert 0.0 <= loss_sums.bounded_loss_sum <= loss_sums.evaluated_sample_count
            assert added_cross_evaluation_records
            # クラスタリングの観測は、モデルごとに1件。クラスタは、全モデルをちょうど1回ずつ持つ。
            assert [observation.model_id for observation in added_model_observations] == list(
                aggregated_model_ids
            )
            assert sorted(
                model_id
                for model_cluster in model_consolidation.model_clusters
                for model_id in model_cluster
            ) == sorted(aggregated_model_ids)
            absorbed_model_ids = model_consolidation.absorbed_model_ids
            assert absorbed_model_ids == tuple(
                observation.model_id
                for observation in added_model_observations
                if observation.absorbed_into_representative
            )
            if model_consolidation.model_id_mapping:
                # 統合した: ID対応は全モデルを持ち、配布で、clientの数だけ軽量メッセージを数える。
                assert sorted(model_consolidation.model_id_mapping) == sorted(aggregated_model_ids)
                assert absorbed_model_ids
                self.consolidation_message_count += len(self.run_clients)
            else:
                assert not absorbed_model_ids
            self.absorbed_model_count += len(absorbed_model_ids)
        else:
            assert not added_cross_evaluation_records
            assert not added_model_observations
        # 配布の後、全clientが、全グローバルモデルを同じ値で保有し、client内では1つの共有部につながっている。
        global_model_ids = self.global_model_repository.global_model_ids
        assert global_model_ids == tuple(
            model_id for model_id in aggregated_model_ids if model_id not in absorbed_model_ids
        )
        self.distributed_model_count += len(global_model_ids) * len(self.run_clients)
        shared_source_parameters = self.global_model_repository.get_global_model_parameters(
            model_id=min(global_model_ids)
        )
        for run_client, distribution_application, prediction_recalibration in zip(
            self.run_clients,
            synchronization.distribution_applications,
            synchronization.prediction_recalibrations,
            strict=True,
        ):
            owners = run_client.owners
            held_model_training_states = owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
            assert (
                tuple(
                    held_state.model_id
                    for held_state in held_model_training_states
                    if held_state.model_id >= 0
                )
                == global_model_ids
            )
            assert distribution_application.held_model_ids == tuple(
                held_state.model_id for held_state in held_model_training_states
            )
            # 現在の学習帰属は、保有しているモデル。吸収されたモデルが現行だったclientは、代表へ付け替わっている。
            current_training_model_id = (
                owners.current_training_model_assignment.current_training_model_id
            )
            assert current_training_model_id in distribution_application.held_model_ids
            if distribution_application.training_assignment_change is not None:
                assert (
                    distribution_application.training_assignment_change.previous_model_id
                    in absorbed_model_ids
                )
            assert (
                len(
                    {
                        id(held_state.classifier.feature_extractor)
                        for held_state in held_model_training_states
                    }
                )
                == 1
            )
            for held_state in held_model_training_states:
                if held_state.model_id < 0:
                    continue
                global_parameters = self.global_model_repository.get_global_model_parameters(
                    model_id=held_state.model_id
                )
                # 共有部は、つなぎ先（非負のIDが最小のモデル）のもの。概念固有部は、そのモデルのもの。
                for (
                    parameter_name,
                    held_parameter_values,
                ) in held_state.classifier.state_dict().items():
                    expected_parameters = (
                        shared_source_parameters
                        if parameter_name.startswith("feature_extractor.")
                        else global_parameters
                    )
                    assert torch.equal(held_parameter_values, expected_parameters[parameter_name])
            # 再較正: 再生した列は、保有する全モデルの損失を持ち、重みは、その上の分布になる。
            if prediction_recalibration.replayed_sample_count > 0:
                weights_by_model_id = owners.fixed_share_weight_controller.weights_by_model_id
                assert (
                    tuple(sorted(weights_by_model_id))
                    == prediction_recalibration.replayed_model_ids
                    == tuple(sorted(distribution_application.held_model_ids))
                )
                assert abs(sum(weights_by_model_id.values()) - 1.0) < 1e-9
            self.replayed_recalibration_sample_count += (
                prediction_recalibration.replayed_sample_count
            )

    def finalize_started_communications(self, *, completed_round_count):
        assert len(self.synchronizations) == completed_round_count


ASSEMBLED_CLIENT_COUNT = 2
ASSEMBLED_CLIENT_STREAM_SAMPLE_COUNT = 97
ASSEMBLED_CLIENT_INTERVAL_SAMPLE_COUNT = 10


def run_assembled_client_flow(*, class_count):
    """初期モデルを事前学習し、clientを組み立て、実行の枠（参加者の検査と区間の進行）で、標本列を最後まで進める。"""
    torch.manual_seed(907 + class_count)
    parameter_optimizer_settings = AdamParameterOptimizerSettings(
        learning_rate=0.05, weight_decay=0.0, adam_variant="standard"
    )
    # 初期モデルは、事前学習で作る（runの乱数源と、SINEの標本生成器を使う）。
    python_random_generator = Random(31 + class_count)
    pretrained_initial_model = pretrain_initial_model(
        initial_model_pretraining_settings=InitialModelPretrainingSettings(
            pretraining_sample_count=40, pretraining_epoch_count=3, pretraining_batch_sample_count=8
        ),
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=2,
        ),
        input_feature_count=2,
        hidden_layer_widths=(5,),
        class_count=class_count,
        parameter_optimizer_settings=parameter_optimizer_settings,
        sample_generator=SineSampleGenerator(numpy_random_generator=RandomState(31 + class_count)),
        python_random_generator=python_random_generator,
    )
    initial_classifier = pretrained_initial_model.classifier
    assert pretrained_initial_model.loss_statistics.overall_loss_moments.observed_loss_count == 40
    pending_capacity = 6
    run_client_settings = FedsdaRunClientSettings(
        loss_change_detection_settings=LossChangeDetectionSettings(
            short_change_interval_alarm_policy=RAISE_ALARM_IMMEDIATELY_POLICY_NAME,
            drift_detector_name="e_sr",
            loss_monitoring_scope="overall_and_true_class_losses",
            e_sr_false_alarm_control_alpha=0.05,
            adwin_confidence_delta=0.05,
            adwin_minimum_test_window_sample_count=10,
        ),
        prediction_combination_settings=PredictionCombinationSettings(
            prediction_combination_strategy="fixed_share_weighted_prediction",
            prediction_mixture_activation_policy="always",
            prediction_weight_recalibration_after_aggregation_policy=(
                "recompute_buffer_losses_and_replay_weight_updates"
            ),
            prediction_state_reset_on_training_assignment_change_policy=(
                "restart_adahedge_preserve_fixed_share_prediction_state"
            ),
            fixed_share_weight_redistribution_time_scale_samples=pending_capacity,
            prediction_model_parameter_source="current_local_parameters",
        ),
        local_training_settings=LocalTrainingSettings(
            local_model_parameter_update_strategy="joint_backbone_adapter_and_head_training",
            shared_backbone_gradient_combination_strategy="sample_weighted_mean_per_concept_gradients",
        ),
        local_training_schedule_settings=LocalTrainingScheduleSettings(
            training_requests_per_update_interval=2,
            joint_update_iterations_per_training_request=1,
        ),
        training_data_assignment_settings=TrainingDataAssignmentSettings(
            pending_sample_retention_after_too_short_change_interval_policy=KEEP_ALL_PENDING_SAMPLES_POLICY_NAME,
            pending_assignment_buffer_capacity_samples=pending_capacity,
        ),
        candidate_model_training_and_acceptance_settings=CandidateModelTrainingAndAcceptanceSettings(
            candidate_model_acceptance_policy="current_model_first_reuse_then_two_segment_candidate_validation",
            candidate_post_alarm_validation_sample_count=VALIDATION_SAMPLE_COUNT,
        ),
        candidate_parameter_initialization_settings=CandidateParameterInitializationSettings(
            candidate_parameter_initialization_source="lowest_evaluated_mean_loss_model"
        ),
        candidate_epoch_training_settings=CandidateEpochTrainingSettings(
            candidate_training_strategy="validation_loss_early_stopping",
            maximum_epoch_count=5,
            maximum_batch_sample_count=4,
            validation_sample_fraction=0.2,
            consecutive_non_improving_epoch_limit=3,
            minimum_validation_loss_decrease=0.0001,
        ),
        parameter_optimizer_settings=parameter_optimizer_settings,
        rebuilt_model_parameter_optimizer_settings=parameter_optimizer_settings,
        pending_model_registration_settings=PendingModelRegistrationSettings(
            pending_model_registration_policy=KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME
        ),
        scalar_settings=FedsdaRunClientScalarSettings(
            maximum_tolerated_mean_loss_increase=0.1,
            minimum_candidate_mean_loss_improvement=0.0001,
            new_model_upload_delay_round_count=2,
            minimum_change_interval_sample_count=3,
            local_training_batch_sample_count=4,
            maximum_stored_evaluation_sample_count_per_model=12,
            added_evaluation_batch_sample_count=3,
            maximum_cross_evaluation_sample_count=8,
            loss_monitor_maximum_retained_candidate_count=50,
            detector_name="overall + class-conditional e-SR mixture",
        ),
        loss_monitor_betting_fractions=(0.05, 0.1, 0.2, 0.4, 0.8),
    )
    initial_model_arguments = dict(
        initial_model_id=0,
        initial_classifier=initial_classifier,
        initial_concept_specific_parameter_optimizer_state=pretrained_initial_model.concept_specific_parameter_optimizer_state,
        initial_shared_parameter_optimizer_state=pretrained_initial_model.shared_parameter_optimizer_state,
        initial_loss_statistics=pretrained_initial_model.loss_statistics,
        run_client_settings=run_client_settings,
        python_random_generator=python_random_generator,
    )
    pretrained_parameters = [
        parameter.detach().clone() for parameter in initial_classifier.parameters()
    ]
    run_clients = tuple(
        assemble_fedsda_run_client(client_id=client_id, **initial_model_arguments)
        for client_id in range(ASSEMBLED_CLIENT_COUNT)
    )
    initial_global_parameters = snapshot_classifier_parameters(classifier=initial_classifier)
    global_model_repository = GlobalModelRepository(
        initial_model_id=0,
        initial_parameter_snapshot=initial_global_parameters,
        initial_loss_statistics=pretrained_initial_model.loss_statistics,
    )
    communication_volume_record_store = CommunicationVolumeRecordStore()
    server_operations = ServerOperationsSynchronizingModels(
        run_clients=run_clients,
        global_model_repository=global_model_repository,
        communication_volume_record_store=communication_volume_record_store,
        python_random_generator=python_random_generator,
    )
    participants = RunParticipants(
        client_operations=run_clients, server_operations=server_operations
    )
    validate_prepared_run_participants(
        participants=participants, client_count=ASSEMBLED_CLIENT_COUNT
    )
    # 標本列: 特徴は決まった規則の値（float32で表せる値）、ラベルは区間ごとに反転する境界で決める。
    observed_client_streams = tuple(
        ClientObservedStream(
            client_id=client_id,
            observed_samples=tuple(
                ObservedSample(
                    feature_values=(
                        ((sample_index * 5 + client_id) % 16) / 16.0,
                        ((sample_index * 3) % 8) / 8.0,
                    ),
                    class_label=int(
                        (((sample_index * 5 + client_id) % 16) >= 8)
                        == ((sample_index // 24) % 2 == 0)
                    ),
                )
                for sample_index in range(ASSEMBLED_CLIENT_STREAM_SAMPLE_COUNT)
            ),
        )
        for client_id in range(ASSEMBLED_CLIENT_COUNT)
    )
    # 概念列: ラベルの境界が反転する区間（24件ごと）を、真の概念とする。
    evaluation_concept_traces = tuple(
        ClientConceptTrace(
            client_id=client_id,
            concept_ids_by_sample_index=tuple(
                (sample_index // 24) % 2
                for sample_index in range(ASSEMBLED_CLIENT_STREAM_SAMPLE_COUNT)
            ),
        )
        for client_id in range(ASSEMBLED_CLIENT_COUNT)
    )
    execution_events = run_stream_protocol_intervals(
        participants=participants,
        observed_client_streams=observed_client_streams,
        evaluation_concept_traces=evaluation_concept_traces,
        server_aggregation_interval_per_client_samples=ASSEMBLED_CLIENT_INTERVAL_SAMPLE_COUNT,
    )
    round_count = ASSEMBLED_CLIENT_STREAM_SAMPLE_COUNT // ASSEMBLED_CLIENT_INTERVAL_SAMPLE_COUNT
    processed_sample_count = round_count * ASSEMBLED_CLIENT_INTERVAL_SAMPLE_COUNT
    assert (
        sum(
            execution_event.stage_name == "sample_processing"
            for execution_event in execution_events
        )
        == processed_sample_count * ASSEMBLED_CLIENT_COUNT
    )
    assert execution_events[-1].stage_name == "started_communication_finalization"
    alarm_count = 0
    for run_client in run_clients:
        owners = run_client.owners
        prediction_records = (
            owners.sample_prediction_record_store.snapshot_sample_prediction_records()
        )
        assert [record.sample_index for record in prediction_records] == list(
            range(processed_sample_count)
        )
        # 実行の枠が、標本位置の真の概念を渡している。
        assert [record.observed_concept_id for record in prediction_records] == list(
            evaluation_concept_traces[run_client.client_id].concept_ids_by_sample_index[
                :processed_sample_count
            ]
        )
        assert owners.local_training_request_schedule.pending_training_request_count == 0
        assert owners.validation_session_holder.held_validation_session is None
        assert (
            owners.pending_sample_observation_store.validation_assignment_sample_concept_ids is None
        )
        alarm_count += len(
            owners.loss_change_alarm_record_store.get_state_snapshot().alarm_sample_indices
        )
    # 渡した初期モデルは、clientが写しを持つので、clientの学習の後も、事前学習の直後の値のまま。
    assert all(
        torch.equal(parameter, pretrained_parameter)
        for parameter, pretrained_parameter in zip(
            initial_classifier.parameters(), pretrained_parameters, strict=True
        )
    )
    assert alarm_count > 0, alarm_count
    # サーバ: 毎ラウンド集約して配布し、グローバルモデル0が、clientの学習を反映して初期の値から変わっている。
    aggregations = [
        synchronization.client_model_aggregation
        for synchronization in server_operations.synchronizations
    ]
    assert len(aggregations) == round_count
    assert all(0 in aggregation.aggregated_global_model_ids for aggregation in aggregations)
    assert aggregations[-1].aggregated_training_sample_counts_by_model_id[0] > 0
    aggregated_global_parameters = global_model_repository.get_global_model_parameters(model_id=0)
    assert list(aggregated_global_parameters) == list(initial_global_parameters)
    assert any(
        not torch.equal(aggregated_global_parameters[parameter_name], initial_parameter_values)
        for parameter_name, initial_parameter_values in initial_global_parameters.items()
    )
    communication_volume = communication_volume_record_store.get_state_snapshot()
    cross_evaluation_records = (
        server_operations.cross_evaluation_record_store.snapshot_client_cross_evaluation_records()
    )
    # クロス評価とクラスタリングが、1回以上行われた（新規モデルを登録したラウンド）。
    model_cross_evaluations = [
        synchronization.model_cross_evaluation
        for synchronization in server_operations.synchronizations
        if synchronization.model_cross_evaluation is not None
    ]
    # クラスタリングが起きたかどうかは、標本列による（どちらかの流れで起きたことを、mainが確かめる）。
    assert bool(model_cross_evaluations) == bool(cross_evaluation_records)
    model_clustering_observations = (
        server_operations.model_clustering_record_store.snapshot_model_clustering_observations()
    )
    assert bool(model_clustering_observations) == bool(model_cross_evaluations)
    assert len({observation.round_index for observation in model_clustering_observations}) == len(
        model_cross_evaluations
    )
    # 上りの軽量メッセージは、ラウンドごとの状態の報告と、クロス評価の統計の返信。
    assert communication_volume.uploaded_message_count == (
        round_count * ASSEMBLED_CLIENT_COUNT + len(cross_evaluation_records)
    )
    assert communication_volume.uploaded_model_count > 0
    assert communication_volume.uploaded_byte_count == 4 * (
        communication_volume.uploaded_parameter_value_count
    )
    # 配布: 下りの通信量が、ラウンドごとの（グローバルモデルの数×clientの数）だけ足されている。
    # 下りのモデル転送数は、配布の分と、クロス評価の分（（評価する側のモデル、client）ごとに1回）。
    assert communication_volume.downloaded_model_count == (
        server_operations.distributed_model_count
        + len(
            {
                (record.round_index, record.candidate_model_id, record.client_id)
                for record in cross_evaluation_records
            }
        )
    )
    assert len(model_cross_evaluations) == len(
        {record.round_index for record in cross_evaluation_records}
    )
    assert server_operations.distributed_model_count >= round_count * ASSEMBLED_CLIENT_COUNT
    # 再較正: 空でない損失の列での再生が、1回以上あった。
    assert server_operations.replayed_recalibration_sample_count > 0
    # 下りの軽量メッセージは、クロス評価の依頼と、統合したラウンドの配布のID対応。
    assert communication_volume.downloaded_message_count == (
        len(cross_evaluation_records) + server_operations.consolidation_message_count
    )
    assert communication_volume.downloaded_byte_count == 4 * (
        communication_volume.downloaded_parameter_value_count
    )
    assert communication_volume.downloaded_byte_count > 0
    # 登録した数だけ、次の正式IDが進み、来歴が増えている。
    assert (
        global_model_repository.next_global_model_id == 1 + server_operations.registered_model_count
    )
    assert (
        len(global_model_repository.snapshot_model_registration_records())
        == 1 + server_operations.registered_model_count
    )
    # グローバルモデルの数は、初期モデルと、登録したモデルから、統合で吸収されたモデルを除いた数を超えない
    # （採番だけが行われて、モデルが登録されないことがある）。
    assert len(global_model_repository.global_model_ids) <= (
        1 + server_operations.registered_model_count - server_operations.absorbed_model_count
    )
    print(
        "  synchronized rounds:",
        round_count,
        "registered:",
        server_operations.registered_model_count,
        "clustered rounds:",
        len(model_cross_evaluations),
        "absorbed:",
        server_operations.absorbed_model_count,
        "global models:",
        global_model_repository.global_model_ids,
    )
    return alarm_count, len(model_cross_evaluations), run_client_settings


WHOLE_RUN_CLIENT_COUNT = 3
WHOLE_RUN_SAMPLE_COUNT = 305
WHOLE_RUN_INTERVAL_SAMPLE_COUNT = 10


def run_whole_stream_protocol_run(*, run_client_settings, random_seed, dataset_name="sine2"):
    """factoryと実行の枠で、最終構成のFedSDAの全体run（事前学習→datasetの標本の供給→区間の進行→終端）を実行する。

    戻り値: 比較できる結果の要約（同じ条件の2回の実行が、同じ結果になることを、mainが確かめる）。
    """
    run_participant_settings = FedsdaRunParticipantSettings(
        run_client_settings=run_client_settings,
        initial_model_pretraining_settings=InitialModelPretrainingSettings(
            pretraining_sample_count=40,
            pretraining_epoch_count=3,
            pretraining_batch_sample_count=8,
        ),
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=2,
        ),
        hidden_layer_widths=(5,),
        model_clustering_criteria=ModelClusteringCriteria(
            maximum_same_cluster_decision_score=0.1,
            minimum_pair_evaluation_sample_count=5,
            clustering_confidence_level=0.95,
            model_pair_comparison_strategy="classwise_unique_correctness_lower_confidence_bound",
        ),
        maximum_evaluating_client_count_per_model=WHOLE_RUN_CLIENT_COUNT,
    )
    torch_random_state = torch.get_rng_state().clone()
    # モデルの計算の計測つきで実行する（計測は、この呼出しの間だけ）。
    measured_run = execute_fedsda_stream_protocol_run_with_computation_measurement(
        execution_settings=StreamProtocolExecutionSettings(
            experiment_run_conditions=ExperimentRunConditions(
                dataset_name=dataset_name,
                random_seed=random_seed,
                client_count=WHOLE_RUN_CLIENT_COUNT,
                per_client_sample_count=WHOLE_RUN_SAMPLE_COUNT,
                server_aggregation_interval_per_client_samples=WHOLE_RUN_INTERVAL_SAMPLE_COUNT,
            ),
            concept_schedule_settings=RandomConceptScheduleSettings(
                concept_schedule_strategy="random_changes_after_minimum_index_gap",
                minimum_sample_index_gap_before_change_trial=30,
                per_eligible_sample_concept_change_probability=0.05,
            ),
            execution_strategy="sample_index_then_client_order_with_interval_synchronization",
        ),
        run_participant_settings=run_participant_settings,
    )
    run_result = measured_run.run_result
    # 全体runは、呼出し側のtorchの乱数を進めない。
    assert torch.equal(torch.get_rng_state(), torch_random_state)
    round_count = WHOLE_RUN_SAMPLE_COUNT // WHOLE_RUN_INTERVAL_SAMPLE_COUNT
    processed_sample_count = round_count * WHOLE_RUN_INTERVAL_SAMPLE_COUNT
    assert run_result.synchronization_interval_count == round_count
    assert run_result.processed_sample_count_per_client == processed_sample_count
    assert run_result.unprocessed_tail_sample_count_per_client == 5
    stage_counts = {}
    for execution_event in run_result.execution_events:
        stage_counts[execution_event.stage_name] = (
            stage_counts.get(execution_event.stage_name, 0) + 1
        )
    assert stage_counts["sample_processing"] == processed_sample_count * WHOLE_RUN_CLIENT_COUNT
    assert stage_counts["server_synchronization"] == round_count
    assert stage_counts["incomplete_candidate_validation_finalization"] == WHOLE_RUN_CLIENT_COUNT
    assert run_result.execution_events[-1].stage_name == "started_communication_finalization"
    participants = measured_run.participants
    run_server = participants.server_operations
    server_owners = run_server.owners
    synchronizations = run_server.snapshot_server_round_synchronizations()
    assert len(synchronizations) == round_count
    global_model_ids = server_owners.global_model_repository.global_model_ids
    communication_volume = server_owners.communication_volume_record_store.get_state_snapshot()
    cross_evaluation_records = (
        server_owners.cross_evaluation_record_store.snapshot_client_cross_evaluation_records()
    )
    # 上りの軽量メッセージは、ラウンドごとの状態の報告と、クロス評価の統計の返信。
    assert communication_volume.uploaded_message_count == (
        round_count * WHOLE_RUN_CLIENT_COUNT + len(cross_evaluation_records)
    )
    assert communication_volume.uploaded_model_count > 0
    assert communication_volume.downloaded_model_count >= round_count * WHOLE_RUN_CLIENT_COUNT
    alarm_count = 0
    whole_run_decision_record_count = 0
    client_summaries = []
    for run_client in participants.client_operations:
        owners = run_client.owners
        prediction_records = (
            owners.sample_prediction_record_store.snapshot_sample_prediction_records()
        )
        assert [record.sample_index for record in prediction_records] == list(
            range(processed_sample_count)
        )
        # 実行の枠は、標本位置の真の概念を、診断専用の引数として、clientへ渡す。
        assert [record.observed_concept_id for record in prediction_records] == list(
            run_result.evaluation_concept_traces[run_client.client_id].concept_ids_by_sample_index[
                :processed_sample_count
            ]
        )
        assert owners.true_concept_adahedge_diagnostic_evidence_collection.created_true_concept_ids
        # 候補検証の判定記録は、適応記録の、候補検証の確定と終端回収の結果と、同じ数だけ保持されている。
        decision_record_count = len(
            owners.candidate_validation_decision_record_store.snapshot_candidate_validation_decision_records()
        )
        assert decision_record_count == sum(
            adaptation_record.adaptation_outcome.startswith("post_alarm_validation_")
            for adaptation_record in owners.adaptation_record_store.get_state_snapshot().adaptation_records
        )
        whole_run_decision_record_count += decision_record_count
        held_model_training_states = (
            owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        )
        # 最後の配布の後、全clientが、全グローバルモデルを保有し、1つの共有部につながっている。
        assert (
            tuple(
                held_state.model_id
                for held_state in held_model_training_states
                if held_state.model_id >= 0
            )
            == global_model_ids
        )
        assert (
            len(
                {
                    id(held_state.classifier.feature_extractor)
                    for held_state in held_model_training_states
                }
            )
            == 1
        )
        assert owners.validation_session_holder.held_validation_session is None
        alarm_sample_indices = (
            owners.loss_change_alarm_record_store.get_state_snapshot().alarm_sample_indices
        )
        alarm_count += len(alarm_sample_indices)
        client_summaries.append(
            (
                tuple(record.combined_prediction_is_correct for record in prediction_records),
                alarm_sample_indices,
                owners.current_training_model_assignment.current_training_model_id,
                tuple(held_state.model_id for held_state in held_model_training_states),
            )
        )
    # 指標の導出: 参加者の記録から計算する。値が、記録と整合している。
    run_metrics = derive_fedsda_run_metrics(
        run_result=run_result,
        participants=participants,
        run_metric_settings=RunMetricSettings(
            maximum_detection_delay_sample_count=20, post_change_recovery_window_sample_count=5
        ),
        model_computation_counts=measured_run.model_computation_counts,
    )
    # 計算量: 準備の間（事前学習: 40件×3 epoch）は別に数え、指標には、その後の計算だけを含める。
    preparation_counts = measured_run.preparation_model_computation_counts
    assert preparation_counts.concept_specific_part_training_example_count == 40 * 3
    model_computation_counts = run_metrics.model_computation_counts
    assert model_computation_counts == measured_run.model_computation_counts
    assert model_computation_counts.concept_specific_part_training_example_count > 0
    # 予測は、標本ごとに、保有する全モデルの概念固有部を通す（共有部は1回）。
    assert model_computation_counts.concept_specific_part_inference_example_count > (
        processed_sample_count * WHOLE_RUN_CLIENT_COUNT
    )
    assert (
        model_computation_counts.shared_part_inference_example_count
        <= model_computation_counts.concept_specific_part_inference_example_count
    )
    # 積和演算: 共有部は、全結合層が1つ（datasetの特徴数→幅5）。
    input_feature_count = get_dataset_definition(dataset_name=dataset_name).input_feature_count
    assert all(
        len(observed_sample.feature_values) == input_feature_count
        for observed_client_stream in run_result.observed_client_streams
        for observed_sample in observed_client_stream.observed_samples
    )
    assert model_computation_counts.shared_part_inference_forward_multiply_accumulate_count == (
        model_computation_counts.shared_part_inference_example_count * input_feature_count * 5
    )
    assert model_computation_counts.shared_part_estimated_backward_multiply_accumulate_count == (
        model_computation_counts.shared_part_training_example_count * input_feature_count * 5
    )
    # 検出器: 標本1件につき、全体と正解クラスの2つを更新する。
    assert (
        run_metrics.loss_monitoring_computation_counts.detector_component_update_count
        == 2 * processed_sample_count * WHOLE_RUN_CLIENT_COUNT
    )
    assert run_metrics.loss_monitoring_computation_counts.evaluated_candidate_bet_count > 0
    # 比較用の計算量: サーバの集約の積和演算は、上りのパラメータの値の数。
    server_computation_counts = run_metrics.server_computation_counts
    assert (
        server_computation_counts.aggregation_parameter_multiply_accumulate_count
        == communication_volume.uploaded_parameter_value_count
    )
    # 保有モデル×標本の延べ数は、標本ごとの保有モデル数の合計（どの標本でも、1つ以上）。
    assert run_metrics.processed_sample_count == processed_sample_count * WHOLE_RUN_CLIENT_COUNT
    assert run_metrics.held_model_sample_count >= run_metrics.processed_sample_count
    assert run_metrics.held_model_sample_count == sum(
        sum(run_client.owners.held_model_count_record_store.snapshot_held_model_counts())
        for run_client in participants.client_operations
    )
    computation_cost_summary = run_metrics.computation_cost_summary
    assert computation_cost_summary.client_forward_multiply_accumulate_count == (
        computation_cost_summary.client_shared_part_forward_multiply_accumulate_count
        + computation_cost_summary.client_concept_specific_part_forward_multiply_accumulate_count
    )
    assert computation_cost_summary.mean_held_model_count >= 1.0
    assert (
        computation_cost_summary.client_forward_multiply_accumulate_count_per_held_model_sample
        <= computation_cost_summary.client_forward_multiply_accumulate_count_per_processed_sample
    )
    # ラウンドごとのモデルの計算と、終端の計算の合計は、全体と一致する。
    assert len(measured_run.round_model_computation_counts) == round_count
    assert (
        sum(
            each_round.local_processing_model_computation_counts.concept_specific_part_inference_example_count
            + each_round.synchronization_model_computation_counts.concept_specific_part_inference_example_count
            for each_round in measured_run.round_model_computation_counts
        )
        + measured_run.finalization_model_computation_counts.concept_specific_part_inference_example_count
        == model_computation_counts.concept_specific_part_inference_example_count
    )
    # 偶然の正解率（1／クラス数）より高い。
    chance_accuracy = 1.0 / get_dataset_definition(dataset_name=dataset_name).class_count
    assert chance_accuracy < run_metrics.prediction_accuracy <= 1.0
    assert chance_accuracy < run_metrics.stable_period_prediction_accuracy <= 1.0
    detection_metrics = run_metrics.training_model_switch_detection_metrics
    assert detection_metrics.detection_count == sum(
        len(
            run_client.owners.adaptation_record_store.get_state_snapshot().training_model_switch_sample_indices
        )
        for run_client in participants.client_operations
    )
    assert detection_metrics.concept_change_count > 0
    assert 0 <= detection_metrics.matched_detection_count <= detection_metrics.detection_count
    assert run_metrics.final_global_model_count == len(global_model_ids)
    assert run_metrics.communication_volume == communication_volume
    assert run_metrics.final_parameter_value_count > 0
    assert run_metrics.final_parameter_byte_count == 4 * run_metrics.final_parameter_value_count
    assert run_metrics.candidate_validation_decision_count == whole_run_decision_record_count
    assert (
        run_metrics.mixed_prediction_sample_count == processed_sample_count * WHOLE_RUN_CLIENT_COUNT
    )
    return dict(
        run_metrics=run_metrics,
        observed_client_streams=run_result.observed_client_streams,
        evaluation_concept_traces=run_result.evaluation_concept_traces,
        global_model_ids=global_model_ids,
        next_global_model_id=server_owners.global_model_repository.next_global_model_id,
        communication_volume=communication_volume,
        cross_evaluation_record_count=len(cross_evaluation_records),
        clustered_round_count=sum(
            synchronization.model_consolidation is not None for synchronization in synchronizations
        ),
        absorbed_model_count=sum(
            len(synchronization.model_consolidation.absorbed_model_ids)
            for synchronization in synchronizations
            if synchronization.model_consolidation is not None
        ),
        alarm_count=alarm_count,
        decision_record_count=whole_run_decision_record_count,
        client_summaries=tuple(client_summaries),
    )


def run_feddrift_detection_batch_smoke():
    """FedDriftの数値判定を、新実装の独立モデルだけで呼び出す。"""
    from federated_learning_experiments.methods.feddrift.detection_batch_assessment import (
        assess_feddrift_detection_batch,
    )
    from federated_learning_experiments.runtime.feddrift_client_round_training import (
        train_feddrift_held_models_for_round,
    )
    from federated_learning_experiments.runtime.feddrift_client_stream_processing import (
        FeddriftClientStreamProcessor,
    )
    from federated_learning_experiments.runtime.feddrift_detection_batch_processing import (
        FeddriftDetectionBatchProcessor,
    )
    from federated_learning_experiments.runtime.feddrift_run_client import (
        assemble_feddrift_run_client,
    )
    from federated_learning_experiments.runtime.feddrift_run_client_settings import (
        FeddriftRunClientSettings,
    )

    with torch.random.fork_rng():
        for class_count in (2, 4):
            torch.manual_seed(61)
            held_models = tuple(
                (
                    model_id,
                    ModelArchitectureClassifier(
                        model_architecture_settings=ModelArchitectureSettings(
                            model_architecture_name="independent_models_without_shared_backbone",
                            residual_adapter_requested_rank=8,
                        ),
                        input_feature_count=3,
                        hidden_layer_widths=(6, 4),
                        class_count=class_count,
                    ),
                )
                for model_id in (7, -104)
            )
            held_models[1][1].load_state_dict(held_models[0][1].state_dict())
            arguments = dict(
                held_models=held_models,
                input_features=torch.ones(3, 3),
                observed_class_labels=torch.zeros(3, 1),
                loss_increase_threshold=0.0,
            )
            first_batch = assess_feddrift_detection_batch(
                **arguments,
                previous_minimum_mean_bounded_loss=None,
            )
            assert first_batch.minimum_loss_model_id == 7
            assert not first_batch.drift_detected
            assert first_batch.batch_sample_count == 3
            changed_batch = assess_feddrift_detection_batch(
                **arguments,
                previous_minimum_mean_bounded_loss=0.0,
            )
            assert changed_batch.drift_detected
            assert changed_batch.model_mean_bounded_losses == first_batch.model_mean_bounded_losses
            registry = HeldModelTrainingStateRegistry()
            optimizer_settings = AdamParameterOptimizerSettings(
                learning_rate=0.01, weight_decay=0.0, adam_variant="standard"
            )
            for model_id, classifier in held_models:
                registry.register_held_model_training_state(
                    model_id=model_id,
                    classifier=classifier,
                    concept_specific_parameter_optimizer_state=ParameterOptimizerState(
                        parameters=tuple(classifier.parameters()),
                        optimizer_settings=optimizer_settings,
                    ),
                )
            uploads = PendingModelUploadState(
                pending_model_registration_policy=KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME
            )
            assignment = CurrentTrainingModelAssignment(initial_model_id=7)
            training_store = ModelTrainingSampleStore()
            processor = FeddriftDetectionBatchProcessor(
                held_model_training_state_registry=registry,
                loss_statistics_store=ModelAndClassLossStatisticsStore(),
                current_training_model_assignment=assignment,
                training_sample_store=training_store,
                evaluation_sample_store=ModelEvaluationSampleStore(
                    maximum_stored_sample_count_per_model=5, added_batch_sample_count=2
                ),
                temporary_model_id_allocator=TemporaryModelIdAllocator(client_id=6),
                pending_model_upload_state=uploads,
                python_random_generator=Random(87),
                new_model_optimizer_settings=optimizer_settings,
                loss_increase_threshold=0.0,
                previous_minimum_mean_bounded_loss=0.0,
            )
            samples = tuple(
                ObservedTrainingSample(
                    input_features=torch.ones(1, 3), observed_class_labels=torch.zeros(1, 1)
                )
                for _ in range(3)
            )
            created = processor.process_detection_batch(
                training_samples=samples, processed_sample_count=3
            )
            assert created.adaptation_kind == "new_model_created"
            assert assignment.current_training_model_id == -106
            assert uploads.has_ready_model_upload()
            retained = processor.process_detection_batch(
                training_samples=samples, processed_sample_count=6
            )
            assert retained.adaptation_kind == "held_model_selected"
            assert retained.assigned_model_id == 7
            stream = FeddriftClientStreamProcessor(
                detection_batch_processor=processor, detection_batch_sample_count=3
            )
            observations = tuple(
                IndexedObservedTrainingSample(
                    sample_index=index, training_sample=sample, observed_concept_id=None
                )
                for index, sample in enumerate(samples + samples + samples[:1])
            )
            assert not stream.process_observations(indexed_observations=observations[:2])
            assert stream.process_observations(indexed_observations=observations[2:])
            assert stream.processed_sample_count == 7
            assert len(stream.completed_detection_batch_transitions) == 2
            predictions_before_flush = stream.sample_predictions
            assert stream.flush_detection_buffer() is not None
            assert stream.flush_detection_buffer() is None
            assert stream.sample_predictions == predictions_before_flush
            assert not stream.buffered_training_samples
            counts = ModelTrainingAndAssignmentCountsStore()
            classifier = registry.get_held_model_training_state(model_id=7).classifier
            before = tuple(parameter.detach().clone() for parameter in classifier.parameters())
            losses = train_feddrift_held_models_for_round(
                held_model_training_state_registry=registry,
                training_sample_store=training_store,
                model_training_and_assignment_counts_store=counts,
                python_random_generator=Random(41),
                batch_sample_count=3,
                updates_per_sample=2,
                k_steps=2,
            )
            assert len(losses) == 8
            assert (
                counts.snapshot_model_training_and_assignment_counts().parameter_update_step_counts_by_model_id
                == {-106: 4, 7: 4}
            )
            assert any(
                not torch.equal(actual, expected)
                for actual, expected in zip(classifier.parameters(), before, strict=True)
            )
            initial_state = registry.get_held_model_training_state(model_id=7)
            client = assemble_feddrift_run_client(
                client_id=6,
                initial_classifier=classifier,
                initial_parameter_optimizer_state=initial_state.concept_specific_parameter_optimizer_state,
                initial_loss_statistics=ModelAndClassLossStatistics(
                    overall_loss_moments=BoundedLossMoments(
                        observed_loss_count=0, mean_loss=0.0, sum_squared_loss_deviations=0.0
                    )
                ),
                run_client_settings=FeddriftRunClientSettings(
                    loss_increase_threshold=1.0,
                    detection_batch_sample_count=3,
                    local_training_batch_sample_count=3,
                    updates_per_sample=2,
                    maximum_stored_evaluation_sample_count_per_model=5,
                    added_evaluation_batch_sample_count=2,
                    new_model_optimizer_settings=optimizer_settings,
                ),
                python_random_generator=Random(41),
            )
            block = tuple(
                ObservedSample(feature_values=(1.0, 1.0, 1.0), class_label=0) for _ in range(7)
            )
            assert client.process_observed_block(
                observed_samples=block, evaluation_concept_ids=(None,) * 7
            )
            assert len(client.train_local_round()) == 6
            assert client.flush_detection_buffer() is not None
            assert client.flush_detection_buffer() is None
            assert not client.has_model_ready_for_server_registration()
            assert (
                client.owners.model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts().parameter_update_step_counts_by_model_id
                == {0: 6}
            )
            from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
                snapshot_classifier_parameters,
            )
            from federated_learning_experiments.runtime.feddrift_run_server import FeddriftRunServer
            from federated_learning_experiments.runtime.feddrift_server_settings import (
                FeddriftServerSettings,
            )

            server = FeddriftRunServer(
                run_clients=(client,),
                server_settings=FeddriftServerSettings(
                    maximum_same_cluster_loss_increase=1.0,
                    cluster_linkage="complete",
                    minimum_cross_evaluation_sample_count=5,
                    maximum_cross_evaluation_client_count=1,
                    maximum_client_evaluation_sample_count=7,
                    new_model_isolation_timestep_count=2,
                    local_training_round_count=2,
                    initial_model_aggregation_weight=0,
                    rebuilt_model_optimizer_settings=optimizer_settings,
                ),
                initial_parameter_snapshot=snapshot_classifier_parameters(
                    classifier=client.owners.held_model_training_state_registry.get_held_model_training_state(
                        model_id=0
                    ).classifier
                ),
                initial_loss_statistics=client.owners.loss_statistics_store.get_model_loss_statistics(
                    model_id=0
                ),
                python_random_generator=client.python_random_generator,
            )
            server.synchronize_timestep(timestep_index=0)
            assert server.snapshot_model_weights() == {0: 7}
            assert (
                server.communication_volume_record_store.get_state_snapshot().uploaded_model_count
                == 2
            )
    from federated_learning_experiments.data.observed_sample_generation import (
        create_observed_sample_generator,
    )
    from federated_learning_experiments.execution.run_random_sources import (
        create_run_random_sources,
    )
    from federated_learning_experiments.learning.training.initial_model_pretraining_settings import (
        InitialModelPretrainingSettings,
    )
    from federated_learning_experiments.runtime.feddrift_run_participant_factory import (
        FeddriftRunParticipantFactory,
        FeddriftRunParticipantSettings,
    )

    sources = create_run_random_sources(random_seed=71)
    generator = create_observed_sample_generator(
        dataset_name="sine2", numpy_random_generator=sources.numpy_random_generator
    )
    settings = FeddriftRunParticipantSettings(
        run_client_settings=FeddriftRunClientSettings(
            loss_increase_threshold=1.0,
            detection_batch_sample_count=3,
            local_training_batch_sample_count=3,
            updates_per_sample=1,
            maximum_stored_evaluation_sample_count_per_model=5,
            added_evaluation_batch_sample_count=2,
            new_model_optimizer_settings=optimizer_settings,
        ),
        server_settings=FeddriftServerSettings(
            maximum_same_cluster_loss_increase=1.0,
            cluster_linkage="complete",
            minimum_cross_evaluation_sample_count=5,
            maximum_cross_evaluation_client_count=1,
            maximum_client_evaluation_sample_count=7,
            new_model_isolation_timestep_count=2,
            local_training_round_count=2,
            initial_model_aggregation_weight=7,
            rebuilt_model_optimizer_settings=optimizer_settings,
        ),
        initial_model_pretraining_settings=InitialModelPretrainingSettings(
            pretraining_sample_count=7,
            pretraining_epoch_count=1,
            pretraining_batch_sample_count=3,
        ),
        hidden_layer_widths=(7, 5),
    )
    participants = FeddriftRunParticipantFactory(run_participant_settings=settings).prepare_run(
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name="sine2",
            random_seed=71,
            client_count=2,
            per_client_sample_count=3,
            server_aggregation_interval_per_client_samples=3,
        ),
        run_random_sources=sources,
        sample_generator=generator,
    )
    for client in participants.run_clients:
        client.process_observed_block(
            observed_samples=tuple(generator.generate_sample(concept_id=0) for _ in range(3)),
            evaluation_concept_ids=(0, 0, 0),
        )
    participants.run_server.synchronize_timestep(timestep_index=0)
    assert participants.run_server.snapshot_model_weights() == {0: 6}
    assert (
        participants.run_server.communication_volume_record_store.get_state_snapshot().uploaded_model_count
        == 4
    )
    trained_before = tuple(
        client.owners.model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts().trained_sample_counts_by_model_id
        for client in participants.run_clients
    )
    for client in participants.run_clients:
        client.detection_batch_processor._previous_loss = 0.0
        client.detection_batch_processor._threshold = 0.0
        client.process_observed_block(
            observed_samples=tuple(generator.generate_sample(concept_id=1) for _ in range(2)),
            evaluation_concept_ids=(1, 1),
        )
    with torch.no_grad():
        participants.run_server.finalize_observed_stream(timestep_index=0)
    assert participants.run_server.snapshot_model_weights() == {0: 6, 1: 7, 2: 7}
    assert participants.run_server.snapshot_isolation_until() == {}
    assert all(
        client.stream_processor.processed_sample_count == 5 for client in participants.run_clients
    )
    assert all(
        not client.stream_processor.buffered_training_samples for client in participants.run_clients
    )
    assert trained_before == tuple(
        client.owners.model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts().trained_sample_counts_by_model_id
        for client in participants.run_clients
    )
    assert (
        participants.run_server.communication_volume_record_store.get_state_snapshot().uploaded_model_count
        == 10
    )
    print("PASS FedDrift: participant factory observation/sync/stream finalization")
    from federated_learning_experiments.runtime.feddrift_experiment_run import (
        execute_feddrift_experiment_run,
    )
    from federated_learning_experiments.runtime.feddrift_run_artifact_files import (
        load_feddrift_run_artifacts,
        save_feddrift_experiment_run,
    )
    from federated_learning_experiments.runtime.feddrift_run_settings import FeddriftRunSettings
    from federated_learning_experiments.runtime.feddrift_stream_run_execution import (
        FeddriftStreamProtocolExecutionSettings,
    )

    experiment_run = execute_feddrift_experiment_run(
        run_settings=FeddriftRunSettings(
            execution_settings=FeddriftStreamProtocolExecutionSettings(
                experiment_run_conditions=ExperimentRunConditions(
                    dataset_name="sine2",
                    random_seed=71,
                    client_count=2,
                    per_client_sample_count=11,
                    server_aggregation_interval_per_client_samples=3,
                ),
                concept_schedule_settings=RandomConceptScheduleSettings(
                    concept_schedule_strategy="random_changes_after_minimum_index_gap",
                    minimum_sample_index_gap_before_change_trial=1,
                    per_eligible_sample_concept_change_probability=0.5,
                ),
            ),
            run_participant_settings=settings,
            run_metric_settings=RunMetricSettings(
                maximum_detection_delay_sample_count=3, post_change_recovery_window_sample_count=3
            ),
        )
    )
    measured = experiment_run.measured_run

    result = measured.run_result
    assert measured.model_computation_counts.concept_specific_parameter_optimizer_step_count > 0
    assert len(measured.timestep_model_computation_counts) == 3
    metrics = experiment_run.run_metrics
    assert result.processed_sample_count_per_client == 9
    assert result.unprocessed_tail_sample_count_per_client == 2
    assert metrics.final_global_model_count >= 1
    assert metrics.communication_volume.uploaded_model_count >= 12
    assert 0.0 <= metrics.prediction_accuracy <= 1.0
    assert metrics.processed_sample_count == 18
    assert metrics.held_model_sample_count >= 18
    assert metrics.computation_cost_summary is not None
    assert metrics.server_computation_counts.aggregation_parameter_multiply_accumulate_count > 0
    assert metrics.client_consolidation_parameter_multiply_accumulate_count >= 0
    assert len(metrics.model_diagnostic_metrics.synchronized_timestep_model_counts) == 3
    assert metrics.model_diagnostic_metrics.mean_global_model_count_after_synchronization >= 1
    assert metrics.model_diagnostic_metrics.client_held_model_training_amounts
    with TemporaryDirectory() as artifact_directory:
        paths = save_feddrift_experiment_run(
            experiment_run=experiment_run,
            output_directory=Path(artifact_directory),
            file_stem="feddrift",
            run_provenance=TEST_RUN_PROVENANCE,
        )
        summary, tables = load_feddrift_run_artifacts(summary_file_path=paths.summary_file_path)
        assert summary["method_name"] == "feddrift"
        assert tables["sample_prediction.prediction_is_correct"].shape == (2, 9)
        assert tables["server_timestep.timestep_index"].tolist() == [0, 1, 2]
    print("PASS FedDrift measured whole run and common core/cost/model metrics and artifacts")


def main():
    from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
        BoundedLossMoments,
    )
    from federated_learning_experiments.methods.feddrift.cached_model_cluster_consolidation import (
        average_feddrift_cached_cluster_parameters,
        consolidate_feddrift_cached_model_clusters,
    )

    parameters = {3: {"weight": torch.tensor([2.0])}, 0: {"weight": torch.tensor([8.0])}}
    consolidation = consolidate_feddrift_cached_model_clusters(
        timestep_index=2,
        model_clusters=((3, 0),),
        model_parameters_by_model_id=parameters,
        historical_model_weights_by_model_id={3: 1, 0: 3},
        loss_moments_by_model_id={
            3: BoundedLossMoments(
                observed_loss_count=2, mean_loss=0.25, sum_squared_loss_deviations=0.125
            )
        },
        isolation_until_by_model_id={3: 4},
    )
    assert torch.equal(
        consolidation.model_parameters_by_representative_id[0]["weight"], torch.tensor([6.5])
    )
    assert consolidation.model_weights_by_representative_id == {0: 4}
    assert consolidation.isolation_until_by_representative_id == {0: 4}
    assert consolidation.loss_moments_by_representative_id[0].sum_squared_loss_deviations == 0.0
    own_parameters = {
        mid: {name: value + 1 for name, value in p.items()} for mid, p in parameters.items()
    }
    own_result = average_feddrift_cached_cluster_parameters(
        model_clusters=((3, 0),),
        model_parameters_by_model_id=own_parameters,
        cluster_weights_by_representative_id=consolidation.cluster_weights_by_representative_id,
    )
    assert torch.equal(own_result[0]["weight"], torch.tensor([7.5]))
    print("PASS FedDrift cached cluster consolidation: server and client own parameters")
    run_feddrift_detection_batch_smoke()
    all_observed_outcomes = set()
    for class_count in (2, 4):
        for smoke_scenario in SMOKE_SCENARIOS:
            observed_outcomes = run_smoke_scenario(
                class_count=class_count, smoke_scenario=smoke_scenario
            )
            all_observed_outcomes.update(observed_outcomes)
            print(
                "PASS",
                class_count,
                "classes:",
                smoke_scenario[0],
                "->",
                " / ".join(observed_outcomes),
            )
    assembled_client_flow_results = [
        run_assembled_client_flow(class_count=class_count) for class_count in (2, 4)
    ]
    assembled_client_alarm_counts = [
        alarm_count for alarm_count, _, _ in assembled_client_flow_results
    ]
    clustered_round_counts = [
        clustered_round_count for _, clustered_round_count, _ in assembled_client_flow_results
    ]
    # サーバの同期の中で、クロス評価とクラスタリングが、1回以上行われた。
    assert sum(clustered_round_counts) > 0, clustered_round_counts
    print(
        "PASS assembled clients in the stream protocol loop: alarms",
        assembled_client_alarm_counts,
        "clustered rounds",
        clustered_round_counts,
    )
    # 全体run: factoryと実行の枠で、同じ条件を2回実行して、同じ結果になることを確かめる。
    whole_run_client_settings = assembled_client_flow_results[0][2]
    whole_run_summaries = [
        run_whole_stream_protocol_run(run_client_settings=whole_run_client_settings, random_seed=7)
        for _ in range(2)
    ]
    assert whole_run_summaries[0] == whole_run_summaries[1]
    assert whole_run_summaries[0]["alarm_count"] > 0
    assert whole_run_summaries[0]["decision_record_count"] > 0
    print(
        "PASS whole stream protocol run through the participant factory:",
        "alarms",
        whole_run_summaries[0]["alarm_count"],
        "global models",
        whole_run_summaries[0]["global_model_ids"],
        "clustered rounds",
        whole_run_summaries[0]["clustered_round_count"],
        "absorbed",
        whole_run_summaries[0]["absorbed_model_count"],
    )
    # sine2以外のdataset（MNISTは、`FDE_MNIST_DATA_DIR`か、リポジトリ直下の`data/mnist`のファイルを読む）でも、同じ条件を2回実行して、同じ結果になる（特徴数と概念数は、datasetの定義から）。
    for dataset_name in ("sea2", "sea4", "circle2", "mnist2", "mnist4"):
        dataset_run_summaries = [
            run_whole_stream_protocol_run(
                run_client_settings=whole_run_client_settings,
                random_seed=7,
                dataset_name=dataset_name,
            )
            for _ in range(2)
        ]
        assert dataset_run_summaries[0] == dataset_run_summaries[1]
        observed_concept_ids = sorted(
            {
                concept_id
                for concept_trace in dataset_run_summaries[0]["evaluation_concept_traces"]
                for concept_id in concept_trace.concept_ids_by_sample_index
            }
        )
        assert observed_concept_ids == list(
            range(get_dataset_definition(dataset_name=dataset_name).concept_count)
        ), observed_concept_ids
        print(
            "PASS whole stream protocol run of",
            dataset_name,
            "concepts",
            observed_concept_ids,
            "alarms",
            dataset_run_summaries[0]["alarm_count"],
            "global models",
            dataset_run_summaries[0]["global_model_ids"],
        )
    # 完全なrun設定（最終構成の既定値）から、実行・保存・読込みを行う。
    experiment_run = execute_fedsda_experiment_run(
        run_settings=build_final_configuration_fedsda_run_settings(
            experiment_run_conditions=ExperimentRunConditions(
                dataset_name="circle2",
                random_seed=11,
                client_count=2,
                per_client_sample_count=65,
                server_aggregation_interval_per_client_samples=10,
            )
        )
    )
    # 最終構成の既定は、送信保留のモデルを、現行モデルとして待った後に、それ自身のIDで登録する。runの終わりに、
    # 送信保留でない一時IDの保有モデル（登録されないままの採用）は、残らない（現行でない送信保留は、送信保留のまま
    # 残る）。
    for run_client in experiment_run.measured_run.participants.client_operations:
        client_owners = run_client.owners
        assert (
            client_owners.pending_model_upload_state.pending_model_registration_policy
            == "register_pending_model_after_waiting_as_current_model"
        )
        pending_model_ids = {
            pending_model_upload.model_id
            for pending_model_upload in client_owners.pending_model_upload_state.get_all_pending_model_uploads()
        }
        assert all(
            held_state.model_id >= 0 or held_state.model_id in pending_model_ids
            for held_state in client_owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        )
        # 既定は、短い警報のあと、取り込んだ前区間を保留から外す。同じ標本が、2回取り込まれない。
        assert client_owners.pending_training_assignment_buffer.releases_earlier_samples_after_too_short_change_interval
        training_sample_object_ids = [
            id(training_sample)
            for training_sample_collection in client_owners.training_sample_store.snapshot_ordered_model_training_samples()
            for training_sample in training_sample_collection.training_samples
        ]
        assert len(training_sample_object_ids) == len(set(training_sample_object_ids))
    with TemporaryDirectory() as artifact_directory:
        artifact_paths = save_fedsda_experiment_run(
            experiment_run=experiment_run,
            output_directory=Path(artifact_directory),
            file_stem="smoke-run",
            run_provenance=TEST_RUN_PROVENANCE,
        )
        run_summary, record_tables = load_fedsda_run_artifacts(
            summary_file_path=artifact_paths.summary_file_path
        )
    assert run_summary["run_metrics"]["processed_sample_count"] == 2 * 60
    assert run_summary["run_settings"]["execution_settings"]["experiment_run_conditions"][
        "dataset_name"
    ] == ("circle2")
    assert record_tables["sample_prediction.combined_prediction_is_correct"].shape == (2, 60)
    assert record_tables["server_round.client_held_model_count_after_synchronization"].shape == (
        6,
        2,
    )
    assert record_tables["server_round.global_model_count_after_synchronization"].tolist()[0] >= 1
    print(
        "PASS experiment run from final configuration settings, saved and loaded:",
        len(record_tables),
        "table columns, accuracy",
        run_summary["run_metrics"]["prediction_accuracy"],
    )
    required_outcomes = {
        "alarm_change_interval_too_short",
        "alarm_interval_held_model_reused",
        "alarm_interval_current_model_maintained",
        "alarm_interval_candidate_validation_started",
        "alarm_during_candidate_validation",
        "post_alarm_validation_candidate_adopted",
        "post_alarm_validation_candidate_rejected",
        "post_alarm_validation_current_model_maintained",
        "post_alarm_validation_held_model_reused",
    }
    assert required_outcomes <= all_observed_outcomes, required_outcomes - all_observed_outcomes
    assert len(PROCESSED_ALARM_COUNTS) == len(SMOKE_SCENARIOS) * 2
    assert sum(PROCESSED_ALARM_COUNTS) > 0, PROCESSED_ALARM_COUNTS
    # 新CLIから、既定FedDrift条件・専用軸・保存・再開を旧importなしで通す。
    from federated_learning_experiments.sweep.experiment_sweep_command import main as sweep_main

    with TemporaryDirectory() as directory:
        arguments = [
            "--configurations",
            "feddrift",
            "--datasets",
            "sine2",
            "--seeds",
            "0",
            "--client-count",
            "2",
            "--per-client-sample-count",
            "5",
            "--aggregation-intervals",
            "3",
            "--feddrift-loss-increase-thresholds",
            "0.0",
            "--workers",
            "1",
            "--purpose",
            "fresh process smoke",
            "--output-directory",
            directory,
        ]
        assert sweep_main(arguments + ["--print-plan"]) == 0
        assert sweep_main(arguments) == 0
        assert sweep_main(arguments) == 0
    print("PASS FedDrift default configuration and sweep command, artifacts and resume")
    # 独立モデルのクロス評価を含む新実装だけの全体runと保存を通す。
    from federated_learning_experiments.sweep.experiment_configurations import (
        build_experiment_run_settings,
    )

    independent_run = execute_fedsda_experiment_run(
        run_settings=build_experiment_run_settings(
            configuration_name="fedsda-independent-models",
            experiment_run_conditions=ExperimentRunConditions(
                dataset_name="sine2",
                random_seed=0,
                client_count=2,
                per_client_sample_count=30,
                server_aggregation_interval_per_client_samples=10,
            ),
        )
    )
    with TemporaryDirectory() as directory:
        paths = save_fedsda_experiment_run(
            experiment_run=independent_run,
            output_directory=Path(directory),
            file_stem="independent",
            run_provenance=TEST_RUN_PROVENANCE,
        )
        summary, _ = load_fedsda_run_artifacts(summary_file_path=paths.summary_file_path)
    assert (
        summary["run_metrics"]["communication_volume"]["downloaded_byte_count"]
        == independent_run.run_metrics.communication_volume.downloaded_byte_count
    )
    print("PASS independent-model cross evaluation, run metrics and saved communication volume")
    from federated_learning_experiments.runtime.fedavg_experiment_run import (
        execute_fedavg_experiment_run,
    )
    from federated_learning_experiments.runtime.fedavg_run_artifact_files import (
        load_fedavg_run_artifacts,
        save_fedavg_experiment_run,
    )
    from federated_learning_experiments.runtime.fedavg_run_settings import (
        build_default_fedavg_run_settings,
    )

    fedavg_run = execute_fedavg_experiment_run(
        run_settings=build_default_fedavg_run_settings(
            experiment_run_conditions=ExperimentRunConditions(
                dataset_name="sine2",
                random_seed=0,
                client_count=2,
                per_client_sample_count=35,
                server_aggregation_interval_per_client_samples=10,
            )
        )
    )
    with TemporaryDirectory() as directory:
        paths = save_fedavg_experiment_run(
            experiment_run=fedavg_run,
            output_directory=Path(directory),
            file_stem="fedavg",
            run_provenance=TEST_RUN_PROVENANCE,
        )
        summary, tables = load_fedavg_run_artifacts(summary_file_path=paths.summary_file_path)
    assert summary["method_name"] == "fedavg"
    assert fedavg_run.run_metrics.final_global_model_count == 1
    assert tables["sample_prediction.sample_index"].shape == (2, 30)
    print("PASS FedAvg full run, measurement and artifact roundtrip")
    diagnostic_spec = importlib.util.spec_from_file_location(
        "_waiting_diagnostic_smoke",
        Path(__file__).resolve().parents[2] / "run_candidate_validation_waiting_diagnostic.py",
    )
    diagnostic = importlib.util.module_from_spec(diagnostic_spec)
    sys.modules[diagnostic_spec.name] = diagnostic
    diagnostic_spec.loader.exec_module(diagnostic)
    diagnostic_run, diagnostic_events = diagnostic._run_with_policy(
        build_experiment_run_settings(
            configuration_name="fedsda",
            experiment_run_conditions=ExperimentRunConditions(
                dataset_name="sea2",
                random_seed=0,
                client_count=2,
                per_client_sample_count=30,
                server_aggregation_interval_per_client_samples=10,
            ),
        ),
        "delayed-forced",
    )
    assert diagnostic_run.run_metrics.processed_sample_count == 60
    assert diagnostic_events == []
    print("PASS candidate validation waiting diagnostic, fresh full run")
    loaded_forbidden_modules = sorted(
        name
        for name in sys.modules
        if name.startswith(("federated_drift_experiment", "test_", "tests."))
    )
    assert not loaded_forbidden_modules, loaded_forbidden_modules
    print(
        "FRESH PROCESS SMOKE PASSED:",
        len(SMOKE_SCENARIOS) * 2,
        "flows;",
        "legacy and test modules not imported",
    )


if __name__ == "__main__":
    main()
