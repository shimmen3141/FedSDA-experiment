"""組み立てたclientを、実__init__で作った実旧の最終構成のclient（サーバなし）と、生成直後から終端まで照合する。"""

import random
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from math import isnan

import numpy as np
import pytest
import torch
from test_adopted_candidate_initial_local_registration import (
    assert_held_model_states_match_legacy,
    convert_legacy_parameter_name,
)
from test_classifier_bounded_loss_evaluation import assert_initial_loss_statistics_match_legacy
from test_fedsda_run_client_settings import VALID_SCALAR_VALUES
from test_held_candidate_validation_progress import make_subclass_copy
from test_joint_model_parameter_update import assert_nested_state_equal
from test_loss_change_monitoring import assert_class_monitor_matches_reference
from test_model_training_and_assignment_counts import assert_model_counts_match_legacy
from test_observed_sample_prediction import assert_prediction_state_matches_legacy
from test_observed_sample_processing import (
    LEGACY_ACTION_BY_ADAPTATION_OUTCOME,
    assert_sample_processing_state_unchanged,
    snapshot_sample_processing_state,
)
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

from federated_drift_experiment import config, experiment
from federated_drift_experiment.clients.shared_backbone import (
    ResidualAdapterRestartingSoftRoutingFedSDAClient,
)
from federated_drift_experiment.data.specs import DATASET_SPECS, DatasetSpec
from federated_drift_experiment.models import ResidualAdapterMLP
from federated_drift_experiment.provisional_model import ProvisionalModelDecision
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.data.observed_streams import (
    ClientConceptTrace,
    ClientObservedStream,
    ObservedSample,
)
from federated_learning_experiments.execution.run_participant_contracts import RunParticipants
from federated_learning_experiments.execution.stream_protocol_execution_loop import (
    run_stream_protocol_intervals,
)
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.models.model_computation_measurement import (
    measure_model_computation,
)
from federated_learning_experiments.learning.models.residual_adapter_classifier import (
    ResidualAdapterClassifier,
)
from federated_learning_experiments.learning.training.candidate_epoch_training_settings import (
    CandidateEpochTrainingSettings,
)
from federated_learning_experiments.learning.training.local_training_schedule_settings import (
    LocalTrainingScheduleSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_state import (
    ParameterOptimizerState,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.candidate_parameter_initialization_settings import (
    CandidateParameterInitializationSettings,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.incomplete_post_alarm_candidate_validation_decision_record import (
    IncompletePostAlarmCandidateValidationDecisionRecord,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_validation_decision_record import (
    PostAlarmCandidateValidationDecisionRecord,
)
from federated_learning_experiments.methods.fedsda.training_data_assignment.training_data_assignment_settings import (
    TrainingDataAssignmentSettings,
)
from federated_learning_experiments.runtime.fedsda_run_client import (
    FedsdaRunClient,
    FedsdaRunClientOwners,
    assemble_fedsda_run_client,
)
from federated_learning_experiments.runtime.fedsda_run_client_settings import (
    FedsdaRunClientScalarSettings,
    FedsdaRunClientSettings,
)
from federated_learning_experiments.runtime.observed_sample_processing import (
    ObservedSampleProcessing,
)
from federated_learning_experiments.runtime.single_run_execution import (
    validate_prepared_run_participants,
)

# 両実装へ与える条件（旧の設定名で書く。新の束は、ここから作る）。
PENDING_CAPACITY = 6
VALIDATION_SAMPLE_COUNT = 4
UPLOAD_DELAY_ROUND_COUNT = 2
BATCH_SAMPLE_COUNT = 4
STORED_EVALUATION_SAMPLE_LIMIT = 12
ADDED_EVALUATION_SAMPLE_COUNT = 3
CROSS_EVALUATION_SAMPLE_LIMIT = 50
CROSS_EVALUATION_CLIENT_LIMIT = 3
MINIMUM_CHANGE_INTERVAL_SAMPLE_COUNT = 3
FALSE_ALARM_CONTROL_ALPHA = 0.05
MAXIMUM_RETAINED_CANDIDATE_COUNT = 50
MAXIMUM_TOLERATED_MEAN_LOSS_INCREASE = 0.1
MINIMUM_CANDIDATE_MEAN_LOSS_IMPROVEMENT = 0.0001
CANDIDATE_EPOCH_COUNT = 5
EARLY_STOPPING_PATIENCE = 3
VALIDATION_SAMPLE_FRACTION = 0.2
ADAPTER_RANK = 2
HIDDEN_LAYER_WIDTHS = (5, 4)
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.001


def set_legacy_configuration(
    monkeypatch,
    *,
    class_count,
    update_interval,
    validation_sample_count=VALIDATION_SAMPLE_COUNT,
    base_learning_rate=LEARNING_RATE,
    new_model_learning_rate=LEARNING_RATE,
    stored_evaluation_sample_limit=STORED_EVALUATION_SAMPLE_LIMIT,
    routing_recalibration="none",
    added_evaluation_sample_count=ADDED_EVALUATION_SAMPLE_COUNT,
    cross_evaluation_sample_limit=CROSS_EVALUATION_SAMPLE_LIMIT,
    cross_evaluation_client_limit=CROSS_EVALUATION_CLIENT_LIMIT,
    dataset_name="sine2",
):
    """実旧clientが生成時と実行時に読む設定を、最終構成の値と、上の小さい条件へ差し替える。

    特徴数と概念数は、実旧の`dataset_name`の定義から取る（隠れ層の幅とクラス数だけ、小さい条件へ差し替える）。

    学習率は2つある（事前学習と配布での作り直しに使う`BASE_LR`、候補とつなぎ直しに使う`NEW_MODEL_LR`）。
    """
    legacy_dataset_spec = DatasetSpec(
        input_dim=DATASET_SPECS[dataset_name].input_dim,
        num_concepts=DATASET_SPECS[dataset_name].num_concepts,
        num_classes=class_count,
        hidden_dims=HIDDEN_LAYER_WIDTHS,
    )
    monkeypatch.setattr(config, "dataset_spec", lambda dataset=None: legacy_dataset_spec)
    monkeypatch.setattr(config, "num_classes", lambda dataset=None: class_count)
    for legacy_setting_name, legacy_setting_value in dict(
        # 最終構成の方式。
        SOFT_ROUTING_CONTEXT="switching",
        SOFT_ROUTING_ACTIVATION_POLICY="always",
        ROUTING_ACTIVE_SET_POLICY="all",
        ROUTING_ARCHIVE_SHADOW_DIAGNOSTICS=False,
        NEW_MODEL_CREATION_POLICY="forward_persistent",
        NEW_MODEL_TRAINING="early_stopping",
        NEW_MODEL_INITIALIZATION="best_candidate",
        SHARED_BACKBONE_TRAINING="joint",
        SHARED_BACKBONE_GRADIENT_STRATEGY="mean",
        FEDSDA_DETECTION_EPISODES_ENABLED=False,
        # サーバのクラスタリング（サーバの生成時に読まれる）。
        FEDSDA_CLUSTERING_POLICY="on_new_model",
        FEDSDA_CLUSTERING_DECISION="class_functional_confidence",
        FEDSDA_CLUSTER_LINKAGE="average",
        FEDSDA_CLUSTERING_CONSOLIDATION="merge",
        FEDSDA_CLUSTERING_CONFIDENCE=0.95,
        OPTIMIZER="adam",
        AMSGRAD=True,
        # 条件。
        SHARED_ADAPTER_RANK=ADAPTER_RANK,
        BASE_LR=base_learning_rate,
        NEW_MODEL_LR=new_model_learning_rate,
        WEIGHT_DECAY=WEIGHT_DECAY,
        PRETRAIN_SAMPLES=24,
        PRETRAIN_EPOCHS=2,
        PRETRAIN_BATCH_SIZE=8,
        FIFO_BUFFER_SIZE=PENDING_CAPACITY,
        NEW_MODEL_FORWARD_VALIDATION_SAMPLES=validation_sample_count,
        FEDSDA_MODEL_UPLOAD_DELAY_ROUNDS=UPLOAD_DELAY_ROUND_COUNT,
        LOCAL_UPDATE_INTERVAL=update_interval,
        UPDATES_PER_SAMPLE=1,
        CLIENT_BATCH_SIZE=BATCH_SAMPLE_COUNT,
        STORED_DATA_LIMIT=stored_evaluation_sample_limit,
        # 集約後の再較正の方式（サーバのrun_roundが読む。最終構成はfifo_replay）。
        SHARED_BACKBONE_ROUTING_RECALIBRATION=routing_recalibration,
        EVAL_STORE_SAMPLE_SIZE=added_evaluation_sample_count,
        # サーバのクロス評価: clientが1回の評価に使う標本の上限と、1つのモデルを評価するclientの上限。
        EVAL_MAX_SAMPLES=cross_evaluation_sample_limit,
        CROSS_EVAL_MAX_CLIENTS=cross_evaluation_client_limit,
        MIN_DRIFT_DATA=MINIMUM_CHANGE_INTERVAL_SAMPLE_COUNT,
        E_DETECTOR_ALPHA=FALSE_ALARM_CONTROL_ALPHA,
        ADWIN_MAX_WINDOW=MAXIMUM_RETAINED_CANDIDATE_COUNT,
        NEW_MODEL_EPOCHS=CANDIDATE_EPOCH_COUNT,
        NEW_MODEL_EARLY_STOPPING_PATIENCE=EARLY_STOPPING_PATIENCE,
        NEW_MODEL_EARLY_STOPPING_MIN_DELTA=MINIMUM_CANDIDATE_MEAN_LOSS_IMPROVEMENT,
        NEW_MODEL_VALIDATION_FRACTION=VALIDATION_SAMPLE_FRACTION,
    ).items():
        assert hasattr(config, legacy_setting_name), legacy_setting_name
        monkeypatch.setattr(config, legacy_setting_name, legacy_setting_value)


def make_run_client_settings(
    valid_run_settings_mapping,
    *,
    update_interval,
    validation_sample_count=VALIDATION_SAMPLE_COUNT,
    base_learning_rate=LEARNING_RATE,
    new_model_learning_rate=LEARNING_RATE,
    stored_evaluation_sample_limit=STORED_EVALUATION_SAMPLE_LIMIT,
    routing_recalibration="none",
    added_evaluation_sample_count=ADDED_EVALUATION_SAMPLE_COUNT,
    cross_evaluation_sample_limit=CROSS_EVALUATION_SAMPLE_LIMIT,
    cross_evaluation_client_limit=CROSS_EVALUATION_CLIENT_LIMIT,
):
    """上の条件と同じ値の、新の束。"""
    return FedsdaRunClientSettings(
        loss_change_detection_settings=replace(
            valid_run_settings_mapping["loss_change_detection_settings"],
            e_sr_false_alarm_control_alpha=FALSE_ALARM_CONTROL_ALPHA,
        ),
        prediction_combination_settings=replace(
            valid_run_settings_mapping["prediction_combination_settings"],
            fixed_share_weight_redistribution_time_scale_samples=PENDING_CAPACITY,
        ),
        local_training_settings=valid_run_settings_mapping["local_training_settings"],
        local_training_schedule_settings=LocalTrainingScheduleSettings(
            training_requests_per_update_interval=update_interval,
            joint_update_iterations_per_training_request=1,
        ),
        training_data_assignment_settings=TrainingDataAssignmentSettings(
            pending_assignment_buffer_capacity_samples=PENDING_CAPACITY
        ),
        candidate_model_training_and_acceptance_settings=replace(
            valid_run_settings_mapping["candidate_model_training_and_acceptance_settings"],
            candidate_post_alarm_validation_sample_count=validation_sample_count,
        ),
        candidate_parameter_initialization_settings=CandidateParameterInitializationSettings(
            candidate_parameter_initialization_source="lowest_evaluated_mean_loss_model"
        ),
        candidate_epoch_training_settings=CandidateEpochTrainingSettings(
            candidate_training_strategy="validation_loss_early_stopping",
            maximum_epoch_count=CANDIDATE_EPOCH_COUNT,
            maximum_batch_sample_count=BATCH_SAMPLE_COUNT,
            validation_sample_fraction=VALIDATION_SAMPLE_FRACTION,
            consecutive_non_improving_epoch_limit=EARLY_STOPPING_PATIENCE,
            minimum_validation_loss_decrease=MINIMUM_CANDIDATE_MEAN_LOSS_IMPROVEMENT,
        ),
        parameter_optimizer_settings=AdamParameterOptimizerSettings(
            learning_rate=new_model_learning_rate, weight_decay=WEIGHT_DECAY, adam_variant="amsgrad"
        ),
        rebuilt_model_parameter_optimizer_settings=AdamParameterOptimizerSettings(
            learning_rate=base_learning_rate, weight_decay=WEIGHT_DECAY, adam_variant="amsgrad"
        ),
        scalar_settings=FedsdaRunClientScalarSettings(
            **VALID_SCALAR_VALUES
            | dict(
                maximum_tolerated_mean_loss_increase=MAXIMUM_TOLERATED_MEAN_LOSS_INCREASE,
                minimum_candidate_mean_loss_improvement=MINIMUM_CANDIDATE_MEAN_LOSS_IMPROVEMENT,
                new_model_upload_delay_round_count=UPLOAD_DELAY_ROUND_COUNT,
                minimum_change_interval_sample_count=MINIMUM_CHANGE_INTERVAL_SAMPLE_COUNT,
                local_training_batch_sample_count=BATCH_SAMPLE_COUNT,
                maximum_stored_evaluation_sample_count_per_model=stored_evaluation_sample_limit,
                added_evaluation_batch_sample_count=added_evaluation_sample_count,
                maximum_cross_evaluation_sample_count=cross_evaluation_sample_limit,
                loss_monitor_maximum_retained_candidate_count=MAXIMUM_RETAINED_CANDIDATE_COUNT,
            )
        ),
        loss_monitor_betting_fractions=(0.05, 0.1, 0.2, 0.4, 0.8),
    )


def convert_legacy_loss_statistics(legacy_statistics):
    """実旧の統計の辞書を、同じ値の新の統計へ写す。"""

    def convert_moments(legacy_moments):
        return BoundedLossMoments(
            observed_loss_count=legacy_moments["n"],
            mean_loss=legacy_moments["mean"],
            sum_squared_loss_deviations=legacy_moments["M2"],
        )

    return ModelAndClassLossStatistics(
        overall_loss_moments=convert_moments(legacy_statistics),
        class_loss_moments_by_class_id=tuple(
            (class_id, convert_moments(legacy_class_moments))
            for class_id, legacy_class_moments in legacy_statistics["class_stats"].items()
        ),
    )


def build_initial_model_from_legacy(*, legacy_model, class_count, run_client_settings):
    """実旧の初期モデルと同じ値の分類器と、同じ蓄積状態の2つのoptimizerの状態を作る。"""
    initial_classifier = ResidualAdapterClassifier(
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=ADAPTER_RANK,
        ),
        input_feature_count=2,
        hidden_layer_widths=HIDDEN_LAYER_WIDTHS,
        class_count=class_count,
    )
    initial_classifier.load_state_dict(
        {
            convert_legacy_parameter_name(parameter_name): parameter_value.clone()
            for parameter_name, parameter_value in legacy_model.state_dict().items()
        }
    )
    concept_specific_parameter_optimizer_state = ParameterOptimizerState(
        parameters=tuple(initial_classifier.residual_adapter.parameters())
        + tuple(initial_classifier.classification_layer.parameters()),
        optimizer_settings=run_client_settings.rebuilt_model_parameter_optimizer_settings,
    )
    concept_specific_parameter_optimizer_state.parameter_optimizer.load_state_dict(
        deepcopy(legacy_model.head_optimizer.state_dict())
    )
    shared_parameter_optimizer_state = ParameterOptimizerState(
        parameters=tuple(initial_classifier.feature_extractor.parameters()),
        optimizer_settings=run_client_settings.rebuilt_model_parameter_optimizer_settings,
    )
    shared_parameter_optimizer_state.parameter_optimizer.load_state_dict(
        deepcopy(legacy_model.backbone.optimizer.state_dict())
    )
    return (
        initial_classifier,
        concept_specific_parameter_optimizer_state,
        shared_parameter_optimizer_state,
    )


def build_run_client_oracle(
    *,
    monkeypatch,
    valid_run_settings_mapping,
    class_count,
    update_interval=2,
    validation_sample_count=VALIDATION_SAMPLE_COUNT,
    client_id=1,
    base_learning_rate=LEARNING_RATE,
    new_model_learning_rate=LEARNING_RATE,
    stored_evaluation_sample_limit=STORED_EVALUATION_SAMPLE_LIMIT,
    routing_recalibration="none",
    added_evaluation_sample_count=ADDED_EVALUATION_SAMPLE_COUNT,
    cross_evaluation_sample_limit=CROSS_EVALUATION_SAMPLE_LIMIT,
    cross_evaluation_client_limit=CROSS_EVALUATION_CLIENT_LIMIT,
):
    """実旧の事前学習と実__init__で実旧clientを作り、同じ初期モデル・統計・条件から新clientを組み立てる。

    戻り値: (新client, 実旧client, 組立ての引数)。呼出し側の乱数は進めない。
    """
    set_legacy_configuration(
        monkeypatch,
        class_count=class_count,
        update_interval=update_interval,
        validation_sample_count=validation_sample_count,
        base_learning_rate=base_learning_rate,
        new_model_learning_rate=new_model_learning_rate,
        stored_evaluation_sample_limit=stored_evaluation_sample_limit,
        routing_recalibration=routing_recalibration,
        added_evaluation_sample_count=added_evaluation_sample_count,
        cross_evaluation_sample_limit=cross_evaluation_sample_limit,
        cross_evaluation_client_limit=cross_evaluation_client_limit,
    )
    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            random.seed(17)
            np.random.seed(17)
            legacy_initial_model, legacy_initial_statistics = experiment._pretrain_initial_model(
                ResidualAdapterMLP
            )
            legacy_client = ResidualAdapterRestartingSoftRoutingFedSDAClient(
                client_id=client_id,
                initial_models={0: legacy_initial_model},
                initial_stats={0: legacy_initial_statistics},
                distance_threshold=MAXIMUM_TOLERATED_MEAN_LOSS_INCREASE,
                verbose=False,
            )
    finally:
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)
    run_client_settings = make_run_client_settings(
        valid_run_settings_mapping,
        update_interval=update_interval,
        validation_sample_count=validation_sample_count,
        base_learning_rate=base_learning_rate,
        new_model_learning_rate=new_model_learning_rate,
        stored_evaluation_sample_limit=stored_evaluation_sample_limit,
        added_evaluation_sample_count=added_evaluation_sample_count,
        cross_evaluation_sample_limit=cross_evaluation_sample_limit,
    )
    (
        initial_classifier,
        initial_concept_specific_parameter_optimizer_state,
        initial_shared_parameter_optimizer_state,
    ) = build_initial_model_from_legacy(
        legacy_model=legacy_initial_model,
        class_count=class_count,
        run_client_settings=run_client_settings,
    )
    assembly_arguments = dict(
        client_id=client_id,
        initial_model_id=0,
        initial_classifier=initial_classifier,
        initial_concept_specific_parameter_optimizer_state=initial_concept_specific_parameter_optimizer_state,
        initial_shared_parameter_optimizer_state=initial_shared_parameter_optimizer_state,
        initial_loss_statistics=convert_legacy_loss_statistics(legacy_initial_statistics),
        run_client_settings=run_client_settings,
        python_random_generator=random.Random(29),
    )
    return assemble_fedsda_run_client(**assembly_arguments), legacy_client, assembly_arguments


def assert_samples_equal_legacy(samples, legacy_samples):
    """標本の列を、実旧の（特徴、ラベル、…）の列と、値で照合する（新clientは自分でtensorを作る）。"""
    assert len(samples) == len(legacy_samples)
    for sample, legacy_sample in zip(samples, legacy_samples):
        assert sample.input_features.dtype == legacy_sample[0].dtype
        assert sample.observed_class_labels.dtype == legacy_sample[1].dtype
        assert torch.equal(sample.input_features, legacy_sample[0])
        assert torch.equal(sample.observed_class_labels, legacy_sample[1])


# 判定記録の理由の名前から、実旧の候補の判定の理由への対応。
LEGACY_REASON_BY_DECISION_REASON = {
    "current_reference_within_historical_loss_tolerance": "current_reference_refit",
    "alternative_reference_within_historical_loss_tolerance": "alternative_reference_refit",
    "first_segment_margin_failed": "first_interval",
    "second_segment_margin_failed": "second_interval",
    "both_segment_margins_failed": "first_and_second",
    "both_segment_margins_passed": "accepted",
}


def assert_candidate_validation_decision_records_match_legacy(
    *, decision_records, legacy_decisions
):
    """保持した判定記録の一覧を、実旧の候補の判定の一覧（`provisional_model_decisions`）と、全項目で照合する。"""
    assert len(decision_records) == len(legacy_decisions)
    for decision_record, legacy_decision in zip(decision_records, legacy_decisions, strict=True):
        assert type(legacy_decision) is ProvisionalModelDecision
        assert decision_record.proposal_sample_index == legacy_decision.position
        assert decision_record.detector_name == legacy_decision.detector
        assert (
            decision_record.candidate_training_interval_sample_count
            == legacy_decision.interval_count
            == legacy_decision.training_count
        )
        # 最終構成の判定は、すべて、警報後の標本での検証。
        assert legacy_decision.validation_source == "forward"
        if type(decision_record) is IncompletePostAlarmCandidateValidationDecisionRecord:
            # 終端での、未完了の候補検証の回収。比較は成立していない。
            assert decision_record.finalization_sample_index == legacy_decision.resolution_position
            assert decision_record.validation_sample_count == legacy_decision.validation_count
            assert legacy_decision.accepted is False
            assert legacy_decision.reason == "insufficient_forward_data"
            assert legacy_decision.reference_model_id is None
            assert all(
                isnan(legacy_loss)
                for legacy_loss in (
                    legacy_decision.candidate_mean_loss,
                    legacy_decision.reference_mean_loss,
                    legacy_decision.candidate_recent_loss,
                    legacy_decision.reference_recent_loss,
                    legacy_decision.reference_historical_mean,
                )
            )
            continue
        assert type(decision_record) is PostAlarmCandidateValidationDecisionRecord
        loss_evaluation = decision_record.post_alarm_candidate_loss_evaluation
        assert decision_record.resolution_sample_index == legacy_decision.resolution_position
        assert loss_evaluation.validation_sample_count == legacy_decision.validation_count
        assert loss_evaluation.candidate_accepted is legacy_decision.accepted
        assert (
            LEGACY_REASON_BY_DECISION_REASON[loss_evaluation.decision_reason]
            == legacy_decision.reason
        )
        assert loss_evaluation.comparison_reference_model_id == legacy_decision.reference_model_id
        assert (
            loss_evaluation.candidate_full_interval_mean_loss == legacy_decision.candidate_mean_loss
        )
        assert (
            loss_evaluation.reference_full_interval_mean_loss == legacy_decision.reference_mean_loss
        )
        assert (
            loss_evaluation.candidate_second_segment_mean_loss
            == legacy_decision.candidate_recent_loss
        )
        assert (
            loss_evaluation.reference_second_segment_mean_loss
            == legacy_decision.reference_recent_loss
        )
        if loss_evaluation.reference_historical_mean_loss is None:
            assert isnan(legacy_decision.reference_historical_mean)
        else:
            assert (
                loss_evaluation.reference_historical_mean_loss
                == legacy_decision.reference_historical_mean
            )


def assert_run_client_matches_legacy(*, run_client, legacy_client, python_random_generator=None):
    """clientの全ownerの状態を、実旧clientの対応する属性と照合する。"""
    owners = run_client.owners
    assert run_client.client_id == legacy_client.client_id
    assert (
        owners.current_training_model_assignment.current_training_model_id
        == legacy_client.current_model_id
    )
    assert_held_model_states_match_legacy(
        registry=owners.held_model_training_state_registry,
        shared_optimizer_owners=[
            owners.shared_parameter_optimizer_state_holder.held_shared_parameter_optimizer_state
        ],
        legacy_client=legacy_client,
        # 出力を比べる入力。特徴数は、実旧の（差し替えた）datasetの定義に合わせる。
        input_features=torch.tensor([[0.25, 0.5, 0.625], [0.75, 0.125, 0.875]])[
            :, : config.dataset_spec().input_dim
        ],
    )
    # 学習データ、評価標本、保留（値で照合する）。
    training_collections = owners.training_sample_store.snapshot_ordered_model_training_samples()
    assert tuple(collection.model_id for collection in training_collections) == tuple(
        legacy_client.train_data_store
    )
    for collection in training_collections:
        assert_samples_equal_legacy(
            collection.training_samples, legacy_client.train_data_store[collection.model_id]
        )
    evaluation_collections = (
        owners.model_evaluation_sample_store.snapshot_ordered_model_evaluation_samples()
    )
    assert tuple(collection.model_id for collection in evaluation_collections) == tuple(
        legacy_client.stored_data
    )
    for collection in evaluation_collections:
        assert_samples_equal_legacy(
            collection.evaluation_samples, legacy_client.stored_data[collection.model_id]
        )
    pending_assignment_state = owners.pending_training_assignment_buffer.get_state_snapshot()
    pending_sample_observations = (
        owners.pending_sample_observation_store.snapshot_pending_sample_observations()
    )
    assert pending_assignment_state.pending_sample_indices == tuple(
        pending_observation.sample_index for pending_observation in pending_sample_observations
    )
    assert_samples_equal_legacy(
        [
            pending_observation.training_sample
            for pending_observation in pending_sample_observations
        ],
        legacy_client.buffer,
    )
    assert [
        pending_observation.observed_concept_id
        for pending_observation in pending_sample_observations
    ] == [legacy_pending_sample[2] for legacy_pending_sample in legacy_client.buffer]
    assert (
        -1
        if pending_assignment_state.last_observed_sample_index is None
        else pending_assignment_state.last_observed_sample_index
    ) == legacy_client.processed_samples - 1
    assert_model_counts_match_legacy(
        counts_store=owners.model_training_and_assignment_counts_store, legacy_client=legacy_client
    )
    loss_statistics_snapshot = owners.loss_statistics_store.get_state_snapshot()
    assert tuple(model_id for model_id, _ in loss_statistics_snapshot) == tuple(
        legacy_client.model_stats
    )
    for model_id, loss_statistics in loss_statistics_snapshot:
        # サーバの集約で作られた統計は、クラス別の統計を持たない（配布で受け取ると、そのまま置かれる）。
        assert_initial_loss_statistics_match_legacy(
            loss_statistics, {"class_stats": {}} | legacy_client.model_stats[model_id]
        )
    # 監視、警報の記録。
    assert_class_monitor_matches_reference(
        monitor=owners.loss_change_monitor, reference_monitor=legacy_client
    )
    alarm_record_snapshot = owners.loss_change_alarm_record_store.get_state_snapshot()
    assert alarm_record_snapshot.monitored_log_e_values == tuple(
        legacy_client.history_detector_log_e
    )
    assert alarm_record_snapshot.alarm_sample_indices == tuple(
        legacy_client.detected_event_positions
    )
    assert alarm_record_snapshot.estimated_change_point_sample_indices == tuple(
        legacy_client.estimated_drift_start_positions
    )
    assert alarm_record_snapshot.detector_candidate_start_sample_indices == tuple(
        legacy_client.detector_candidate_start_positions
    )
    # 候補検証の保持、学習要求、一時ID、送信保留。
    assert (owners.validation_session_holder.held_validation_session is None) == (
        legacy_client._forward_validation is None
    )
    assert owners.pending_sample_observation_store.validation_assignment_sample_concept_ids == (
        None
        if legacy_client._forward_validation is None
        else tuple(
            legacy_held_sample[2]
            for legacy_held_sample in legacy_client._forward_validation.held_data
        )
    )
    assert (
        owners.local_training_request_schedule.pending_training_request_count
        == legacy_client._pending_updates
    )
    assert owners.temporary_model_id_allocator.next_temporary_model_id == legacy_client.next_temp_id
    pending_model_upload = owners.pending_model_upload_state.get_pending_model_upload()
    assert run_client.has_model_ready_for_server_registration() is legacy_client.has_pending_model()
    assert (pending_model_upload is None) == (legacy_client.pending_model_params is None)
    if pending_model_upload is not None:
        # 旧は送信保留のモデルのIDを持たない（採用の後に現行モデルが別のモデルへ戻ることがある）。
        # 新の送信保留は、保有している一時IDのモデルを指す。
        assert pending_model_upload.model_id < 0
        assert pending_model_upload.model_id in legacy_client.models
        assert owners.pending_model_upload_state.remaining_upload_delay_round_count == max(
            0, legacy_client._pending_upload_rounds
        )
        assert legacy_client.pending_model_ready is (
            owners.pending_model_upload_state.remaining_upload_delay_round_count == 0
        )
    # 適応記録: 実旧のイベント列と1件ずつ対応する。
    adaptation_record_snapshot = owners.adaptation_record_store.get_state_snapshot()
    assert [
        (
            adaptation_record.adaptation_sample_index,
            LEGACY_ACTION_BY_ADAPTATION_OUTCOME[adaptation_record.adaptation_outcome],
            adaptation_record.previous_training_model_id,
            adaptation_record.current_training_model_id,
            adaptation_record.estimated_change_point_sample_index,
        )
        for adaptation_record in adaptation_record_snapshot.adaptation_records
    ] == [
        (
            legacy_event.position,
            legacy_event.action,
            legacy_event.old_model_id,
            legacy_event.new_model_id,
            legacy_event.estimated_change_point,
        )
        for legacy_event in legacy_client.adaptation_events
    ]
    assert adaptation_record_snapshot.training_model_switch_sample_indices == tuple(
        legacy_client.local_switch_positions
    )
    # 標本ごとの保有モデル数: 処理した標本ごとに1件。合計は、実旧の、予測でモデルへ入力した標本の数
    # （旧は、標本ごとに、保有する全モデルを通す）。
    held_model_counts = owners.held_model_count_record_store.snapshot_held_model_counts()
    assert len(held_model_counts) == len(alarm_record_snapshot.monitored_log_e_values)
    assert sum(held_model_counts) == legacy_client.compute_counters["prediction_examples"]
    assert all(held_model_count >= 1 for held_model_count in held_model_counts)
    # 検出器の計算の計数: 実旧の、検出器の更新回数と、評価した候補×賭け率の数。
    loss_monitoring_computation_counts = (
        owners.loss_monitoring_computation_count_store.get_loss_monitoring_computation_counts()
    )
    assert (
        loss_monitoring_computation_counts.detector_component_update_count
        == legacy_client.compute_counters["drift_detector_updates"]
    )
    assert (
        loss_monitoring_computation_counts.evaluated_candidate_bet_count
        == legacy_client.compute_counters["drift_detector_hypotheses"]
    )
    # 更新は、処理した標本1件につき、全体と正解クラスの2つ。
    assert loss_monitoring_computation_counts.detector_component_update_count == 2 * len(
        alarm_record_snapshot.monitored_log_e_values
    )
    # 候補検証の判定記録: 実旧の候補の判定の一覧と1件ずつ対応する。
    decision_records = owners.candidate_validation_decision_record_store.snapshot_candidate_validation_decision_records()
    assert_candidate_validation_decision_records_match_legacy(
        decision_records=decision_records,
        legacy_decisions=legacy_client.provisional_model_decisions,
    )
    # 判定記録は、適応記録の、候補検証の確定と終端回収の結果と、同じ数・同じ位置にある。
    assert [
        (
            decision_record.finalization_sample_index
            if type(decision_record) is IncompletePostAlarmCandidateValidationDecisionRecord
            else decision_record.resolution_sample_index
        )
        for decision_record in decision_records
    ] == [
        adaptation_record.adaptation_sample_index
        for adaptation_record in adaptation_record_snapshot.adaptation_records
        if adaptation_record.adaptation_outcome.startswith("post_alarm_validation_")
    ]
    # 予測: Fixed-Shareの重み、診断証拠、標本ごとの記録と旧の列・集計の計数。
    assert_prediction_state_matches_legacy(
        fixed_share_prediction_weight_controller=owners.fixed_share_prediction_weight_controller,
        diagnostic_evidence_collection=owners.diagnostic_evidence_collection,
        sample_prediction_record_store=owners.sample_prediction_record_store,
        legacy_client=legacy_client,
    )


def make_concept_stream(*, sample_count, concept_block_length, stream_seed):
    """実旧のデータ生成（事前学習と同じdataset。概念は4つ）で作る標本列。

    概念は、決まった長さの区間ごとに 0→1→0→2→1→3… と切り替わる（同じ概念へ戻るので、保有モデルの
    再利用が起きうる）。呼出し側の乱数は進めない。値は入力を作るためだけに使い、期待値には使わない。
    """
    concept_cycle = (0, 1, 0, 2, 1, 3)
    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    stream = []
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(stream_seed)
            random.seed(stream_seed)
            np.random.seed(stream_seed)
            for sample_index in range(sample_count):
                concept_id = concept_cycle[
                    (sample_index // concept_block_length) % len(concept_cycle)
                ]
                legacy_features, legacy_labels = experiment.generate_data(concept_id)
                stream.append(
                    (
                        ObservedSample(
                            feature_values=(
                                float(legacy_features[0].item()),
                                float(legacy_features[1].item()),
                            ),
                            class_label=int(legacy_labels[0].item()),
                        ),
                        concept_id,
                    )
                )
    finally:
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)
    return stream


def sum_legacy_computation_counts(legacy_clients):
    """実旧のclientの計数の、全clientの合計。"""
    legacy_computation_counts = Counter()
    for legacy_client in legacy_clients:
        legacy_computation_counts.update(legacy_client.compute_counters)
    return legacy_computation_counts


def assert_model_computation_matches_legacy_counter_increase(
    *, model_computation_counts, legacy_clients, legacy_computation_counts_before
):
    """新の操作の、モデルの計算（外側から数えた値）が、実旧のclientの計数の増分（全clientの合計）と一致する。

    共有部・概念固有部を通った標本数、学習の標本数、optimizerの更新回数を比べる。
    """

    def legacy_increase(counter_name):
        return (
            sum_legacy_computation_counts(legacy_clients)[counter_name]
            - legacy_computation_counts_before[counter_name]
        )

    assert (
        model_computation_counts.shared_part_training_example_count
        + model_computation_counts.shared_part_inference_example_count
    ) == legacy_increase("backbone_examples")
    assert (
        model_computation_counts.concept_specific_part_training_example_count
        + model_computation_counts.concept_specific_part_inference_example_count
    ) == legacy_increase("head_examples")
    assert model_computation_counts.concept_specific_part_training_example_count == (
        legacy_increase("training_examples")
    )
    assert model_computation_counts.shared_part_training_example_count == (
        legacy_increase("training_examples")
    )
    assert model_computation_counts.concept_specific_parameter_optimizer_step_count == (
        legacy_increase("optimizer_steps")
    )
    assert model_computation_counts.shared_parameter_optimizer_step_count == (
        legacy_increase("backbone_optimizer_steps")
    )


def run_in_both(
    *,
    run_client,
    legacy_client,
    python_random_generator,
    legacy_operation,
    operation,
    legacy_clients=None,
):
    """同じ乱数の状態から、実旧の操作と新の操作を実行し、実行後の乱数の状態が同じであることを確かめる。

    新の操作の、モデルの計算（外側から数えた値）が、実旧のclientの計数の増分と一致することも確かめる。
    複数のclientにまたがる操作では、`legacy_clients`へ、実旧の全clientを渡す。
    """
    if legacy_clients is None:
        assert legacy_client is not None
        legacy_clients = [legacy_client]
    global_python_random_state = random.getstate()
    torch_random_state = torch.get_rng_state().clone()
    legacy_computation_counts_before = sum_legacy_computation_counts(legacy_clients)
    try:
        random.setstate(python_random_generator.getstate())
        legacy_result = legacy_operation()
        legacy_python_random_state = random.getstate()
    finally:
        random.setstate(global_python_random_state)
    legacy_torch_random_state = torch.get_rng_state().clone()
    torch.set_rng_state(torch_random_state)
    with measure_model_computation() as model_computation_meter:
        result = operation()
    assert torch.equal(torch.get_rng_state(), legacy_torch_random_state)
    assert python_random_generator.getstate() == legacy_python_random_state
    assert_model_computation_matches_legacy_counter_increase(
        model_computation_counts=model_computation_meter.get_model_computation_counts(),
        legacy_clients=legacy_clients,
        legacy_computation_counts_before=legacy_computation_counts_before,
    )
    return result, legacy_result


ROUND_SAMPLE_COUNT = 10
# 条件ごとの、通った適応結果・送信待ちの進行の有無・終端での回収の有無。
OBSERVED_COVERAGE_BY_CONDITION = {}
# (クラス数, 概念の区間長, 標本列のseed, 学習の間隔, 候補検証の標本数, 標本数)。
# 候補検証を長くした条件は候補検証中の警報を、標本数を短くした条件は終端での未完了の回収を通す。
TRAJECTORY_CONDITIONS = [
    (2, 30, 3, 1, 4, 137),
    (2, 30, 3, 2, 4, 137),
    (2, 13, 5, 1, 4, 137),
    (2, 13, 5, 2, 4, 137),
    (2, 22, 11, 2, 4, 137),
    (2, 13, 5, 1, 12, 137),
    (2, 16, 23, 1, 12, 137),
    (2, 30, 3, 1, 12, 116),
    (4, 30, 3, 1, 4, 137),
    (4, 30, 3, 2, 4, 137),
    (4, 22, 11, 1, 4, 137),
    (4, 22, 11, 2, 8, 78),
]


@pytest.mark.parametrize(
    "class_count,concept_block_length,stream_seed,update_interval,validation_sample_count,sample_count",
    TRAJECTORY_CONDITIONS,
)
def test_run_client_matches_real_legacy_client_from_construction_to_run_end(
    class_count,
    concept_block_length,
    stream_seed,
    update_interval,
    validation_sample_count,
    sample_count,
    monkeypatch,
    valid_run_settings_mapping,
):
    run_client, legacy_client, assembly_arguments = build_run_client_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=class_count,
        update_interval=update_interval,
        validation_sample_count=validation_sample_count,
    )
    python_random_generator = assembly_arguments["python_random_generator"]
    assert type(run_client) is FedsdaRunClient
    assert type(run_client.owners) is FedsdaRunClientOwners
    # 生成直後。
    assert_run_client_matches_legacy(run_client=run_client, legacy_client=legacy_client)
    upload_wait_advanced = False
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        for sample_index, (observed_sample, concept_id) in enumerate(
            make_concept_stream(
                sample_count=sample_count,
                concept_block_length=concept_block_length,
                stream_seed=stream_seed,
            )
        ):
            run_in_both(
                run_client=run_client,
                legacy_client=legacy_client,
                python_random_generator=python_random_generator,
                legacy_operation=lambda observed_sample=observed_sample, concept_id=concept_id: (
                    legacy_client.process_one_step(
                        torch.tensor(observed_sample.feature_values, dtype=torch.float32),
                        torch.tensor([float(observed_sample.class_label)]),
                        concept_id,
                    )
                ),
                operation=lambda observed_sample=observed_sample, sample_index=sample_index, concept_id=concept_id: (
                    run_client.process_observed_sample(
                        observed_sample=observed_sample,
                        sample_index=sample_index,
                        evaluation_concept_id=concept_id,
                    )
                ),
            )
            assert_run_client_matches_legacy(run_client=run_client, legacy_client=legacy_client)
            if (sample_index + 1) % ROUND_SAMPLE_COUNT == 0:
                # ラウンド境界: 保留中の学習→登録できるモデルの有無→（同期の後の）送信待ちの進行。
                round_index = sample_index // ROUND_SAMPLE_COUNT
                run_in_both(
                    run_client=run_client,
                    legacy_client=legacy_client,
                    python_random_generator=python_random_generator,
                    legacy_operation=legacy_client.flush_pending_updates,
                    operation=lambda round_index=round_index: (
                        run_client.flush_pending_local_updates(round_index=round_index)
                    ),
                )
                assert_run_client_matches_legacy(run_client=run_client, legacy_client=legacy_client)
                remaining_wait = (
                    run_client.owners.pending_model_upload_state.remaining_upload_delay_round_count
                )
                run_in_both(
                    run_client=run_client,
                    legacy_client=legacy_client,
                    python_random_generator=python_random_generator,
                    legacy_operation=legacy_client.promote_pending_to_ready,
                    operation=lambda round_index=round_index: (
                        run_client.advance_new_model_upload_wait_after_synchronization(
                            round_index=round_index
                        )
                    ),
                )
                upload_wait_advanced = upload_wait_advanced or (
                    run_client.owners.pending_model_upload_state.remaining_upload_delay_round_count
                    < remaining_wait
                )
                assert_run_client_matches_legacy(run_client=run_client, legacy_client=legacy_client)
        # 終端: 未完了の候補検証の回収。
        validation_was_held = (
            run_client.owners.validation_session_holder.held_validation_session is not None
        )
        finalization, _ = run_in_both(
            run_client=run_client,
            legacy_client=legacy_client,
            python_random_generator=python_random_generator,
            legacy_operation=legacy_client.finalize_incomplete_forward_validation,
            operation=run_client.finalize_incomplete_candidate_validation,
        )
        assert (finalization is not None) == validation_was_held
        assert run_client.owners.validation_session_holder.held_validation_session is None
        assert_run_client_matches_legacy(run_client=run_client, legacy_client=legacy_client)
        # 2回目の回収は何もしない。
        assert run_client.finalize_incomplete_candidate_validation() is None
        assert_run_client_matches_legacy(run_client=run_client, legacy_client=legacy_client)
    OBSERVED_COVERAGE_BY_CONDITION[
        (
            class_count,
            concept_block_length,
            stream_seed,
            update_interval,
            validation_sample_count,
            sample_count,
        )
    ] = (
        tuple(
            adaptation_record.adaptation_outcome
            for adaptation_record in run_client.owners.adaptation_record_store.get_state_snapshot().adaptation_records
        ),
        upload_wait_advanced,
        validation_was_held,
    )


def test_run_client_trajectories_cover_required_paths():
    """上の対照が、要求の経路をすべて通っていること（全条件を実行したときだけ確かめる）。"""
    if len(OBSERVED_COVERAGE_BY_CONDITION) < len(TRAJECTORY_CONDITIONS):
        pytest.skip("対照の全条件を実行したときだけ確かめる")
    observed_outcomes = {
        adaptation_outcome
        for condition_outcomes, _, _ in OBSERVED_COVERAGE_BY_CONDITION.values()
        for adaptation_outcome in condition_outcomes
    }
    # 候補検証の確定での他モデルの再利用は、標本1件の処理の対照（上流）が通している。
    # サーバの統合による付け替えは、サーバなしの対照では起きない。
    assert observed_outcomes >= set(LEGACY_ACTION_BY_ADAPTATION_OUTCOME) - {
        "post_alarm_validation_held_model_reused",
        "server_consolidation_training_model_remapped",
    }
    assert any(
        upload_wait_advanced
        for _, upload_wait_advanced, _ in OBSERVED_COVERAGE_BY_CONDITION.values()
    )
    assert any(
        validation_was_held for _, _, validation_was_held in OBSERVED_COVERAGE_BY_CONDITION.values()
    )


def collect_processing_arguments(*, run_client, python_random_generator):
    """clientのownerを、標本1件の処理のtestの読取りが受け取る形（引数名→owner）にする。"""
    owners = run_client.owners
    return {
        owner_name: owner
        for owner_name, owner in vars(owners).items()
        if owner_name != "shared_parameter_optimizer_state_holder"
    } | dict(
        python_random_generator=python_random_generator,
        shared_parameter_optimizer=owners.shared_parameter_optimizer_state_holder.held_shared_parameter_optimizer_state.parameter_optimizer,
    )


def snapshot_run_client_state(*, run_client, python_random_generator):
    """clientの全ownerと乱数の、比較できる読取り（標本1件の処理のtestの読取りを使う）。"""
    return snapshot_sample_processing_state(
        processing_arguments=collect_processing_arguments(
            run_client=run_client, python_random_generator=python_random_generator
        )
    )


def assert_run_client_state_unchanged(*, state_snapshot, run_client, python_random_generator):
    assert_sample_processing_state_unchanged(
        state_snapshot=state_snapshot,
        processing_arguments=collect_processing_arguments(
            run_client=run_client, python_random_generator=python_random_generator
        ),
    )


INITIAL_OPTIMIZER_STATE_ARGUMENT_NAMES = (
    "initial_concept_specific_parameter_optimizer_state",
    "initial_shared_parameter_optimizer_state",
)


def snapshot_initial_model(assembly_arguments):
    """組立てへ渡した初期モデル（分類器の値と訓練の別、2つのoptimizerの状態）と乱数の読取り。"""
    return dict(
        parameters={
            parameter_name: parameter.detach().clone()
            for parameter_name, parameter in assembly_arguments[
                "initial_classifier"
            ].named_parameters()
        },
        training_mode=assembly_arguments["initial_classifier"].training,
        optimizer_states=tuple(
            deepcopy(assembly_arguments[optimizer_state_name].parameter_optimizer.state_dict())
            for optimizer_state_name in INITIAL_OPTIMIZER_STATE_ARGUMENT_NAMES
        ),
        python_random_state=assembly_arguments["python_random_generator"].getstate(),
        global_python_random_state=random.getstate(),
        torch_random_state=torch.get_rng_state().clone(),
    )


def assert_initial_model_unchanged(*, initial_model_snapshot, assembly_arguments):
    current_snapshot = snapshot_initial_model(assembly_arguments)
    assert current_snapshot["parameters"].keys() == initial_model_snapshot["parameters"].keys()
    for parameter_name, parameter in initial_model_snapshot["parameters"].items():
        assert torch.equal(current_snapshot["parameters"][parameter_name], parameter)
    assert current_snapshot["training_mode"] == initial_model_snapshot["training_mode"]
    for current_optimizer_state, optimizer_state in zip(
        current_snapshot["optimizer_states"],
        initial_model_snapshot["optimizer_states"],
        strict=True,
    ):
        assert_nested_state_equal(current_optimizer_state, optimizer_state)
    for random_state_name in ("python_random_state", "global_python_random_state"):
        assert current_snapshot[random_state_name] == initial_model_snapshot[random_state_name]
    assert torch.equal(
        current_snapshot["torch_random_state"], initial_model_snapshot["torch_random_state"]
    )


def test_assembly_copies_initial_model_and_keeps_clients_independent(
    monkeypatch, valid_run_settings_mapping
):
    first_client, _, assembly_arguments = build_run_client_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=2,
    )
    initial_model_snapshot = snapshot_initial_model(assembly_arguments)
    second_random_generator = random.Random(5)
    second_client = assemble_fedsda_run_client(
        **assembly_arguments | dict(client_id=2, python_random_generator=second_random_generator)
    )
    # 組立ては、渡された初期モデルと乱数を変えない（乱数を使わない）。
    assert_initial_model_unchanged(
        initial_model_snapshot=initial_model_snapshot, assembly_arguments=assembly_arguments
    )
    # 写しの中で、optimizerは写しの分類器のパラメータを指す。渡された分類器・optimizerとは別のオブジェクト。
    initial_parameter_ids = {
        id(parameter) for parameter in assembly_arguments["initial_classifier"].parameters()
    }
    for run_client in (first_client, second_client):
        owners = run_client.owners
        (held_state,) = (
            owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        )
        assert held_state.model_id == 0
        held_classifier = held_state.classifier
        assert held_classifier is not assembly_arguments["initial_classifier"]
        assert all(
            id(parameter) not in initial_parameter_ids for parameter in held_classifier.parameters()
        )
        shared_optimizer = owners.shared_parameter_optimizer_state_holder.held_shared_parameter_optimizer_state.parameter_optimizer
        assert (
            shared_optimizer
            is not assembly_arguments[
                "initial_shared_parameter_optimizer_state"
            ].parameter_optimizer
        )
        assert [
            id(parameter)
            for parameter_group in shared_optimizer.param_groups
            for parameter in parameter_group["params"]
        ] == [id(parameter) for parameter in held_classifier.feature_extractor.parameters()]
        assert_nested_state_equal(
            held_state.concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict(),
            initial_model_snapshot["optimizer_states"][0],
        )
        assert_nested_state_equal(
            shared_optimizer.state_dict(), initial_model_snapshot["optimizer_states"][1]
        )
    assert first_client.client_id == 1
    assert second_client.client_id == 2
    assert second_client.owners.temporary_model_id_allocator.next_temporary_model_id == -102
    # 判定記録の保持は、clientごとに別々で、組立ての直後は空。
    first_decision_record_store = first_client.owners.candidate_validation_decision_record_store
    second_decision_record_store = second_client.owners.candidate_validation_decision_record_store
    assert first_decision_record_store is not second_decision_record_store
    assert (
        first_client.owners.loss_monitoring_computation_count_store
        is not second_client.owners.loss_monitoring_computation_count_store
    )
    assert not any(
        vars(
            second_client.owners.loss_monitoring_computation_count_store.get_loss_monitoring_computation_counts()
        ).values()
    )
    assert (
        first_client.owners.held_model_count_record_store
        is not second_client.owners.held_model_count_record_store
    )
    assert second_client.owners.held_model_count_record_store.snapshot_held_model_counts() == ()
    assert first_decision_record_store.snapshot_candidate_validation_decision_records() == ()
    assert second_decision_record_store.snapshot_candidate_validation_decision_records() == ()
    # 片方のclientだけ標本を処理しても、もう片方と、渡した初期モデルは変わらない。
    second_client_snapshot = snapshot_run_client_state(
        run_client=second_client, python_random_generator=second_random_generator
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        for sample_index, (observed_sample, concept_id) in enumerate(
            make_concept_stream(sample_count=40, concept_block_length=13, stream_seed=5)
        ):
            first_client.process_observed_sample(
                observed_sample=observed_sample,
                sample_index=sample_index,
                evaluation_concept_id=concept_id,
            )
        first_client.flush_pending_local_updates(round_index=0)
    first_counts_store = first_client.owners.model_training_and_assignment_counts_store
    second_counts_store = second_client.owners.model_training_and_assignment_counts_store
    assert (
        first_counts_store.snapshot_model_training_and_assignment_counts()
        != second_counts_store.snapshot_model_training_and_assignment_counts()
    )
    assert_run_client_state_unchanged(
        state_snapshot=second_client_snapshot,
        run_client=second_client,
        python_random_generator=second_random_generator,
    )
    initial_parameters = dict(assembly_arguments["initial_classifier"].named_parameters())
    for parameter_name, parameter in initial_model_snapshot["parameters"].items():
        assert torch.equal(initial_parameters[parameter_name], parameter)
    for optimizer_state_name, optimizer_state in zip(
        INITIAL_OPTIMIZER_STATE_ARGUMENT_NAMES,
        initial_model_snapshot["optimizer_states"],
        strict=True,
    ):
        assert_nested_state_equal(
            assembly_arguments[optimizer_state_name].parameter_optimizer.state_dict(),
            optimizer_state,
        )


def make_other_initial_model(assembly_arguments, *, input_feature_count=2):
    """同じ構造（または特徴数だけ違う構造）の、別の分類器と、それに対応する2つのoptimizerの状態。"""
    reference_classifier = assembly_arguments["initial_classifier"]
    other_classifier = ResidualAdapterClassifier(
        model_architecture_settings=reference_classifier.model_architecture_settings,
        input_feature_count=input_feature_count,
        hidden_layer_widths=HIDDEN_LAYER_WIDTHS,
        class_count=reference_classifier.class_count,
    )
    optimizer_settings = assembly_arguments["run_client_settings"].parameter_optimizer_settings
    return dict(
        initial_classifier=other_classifier,
        initial_concept_specific_parameter_optimizer_state=ParameterOptimizerState(
            parameters=tuple(other_classifier.residual_adapter.parameters())
            + tuple(other_classifier.classification_layer.parameters()),
            optimizer_settings=optimizer_settings,
        ),
        initial_shared_parameter_optimizer_state=ParameterOptimizerState(
            parameters=tuple(other_classifier.feature_extractor.parameters()),
            optimizer_settings=optimizer_settings,
        ),
    )


def make_float64_initial_model(assembly_arguments):
    """パラメータがfloat64の分類器と、それに対応する2つのoptimizerの状態。"""
    other_initial_model = make_other_initial_model(assembly_arguments)
    other_initial_model["initial_classifier"].double()
    return other_initial_model


def make_settings_mutated_around_frozen(run_client_settings):
    mutated_settings = replace(run_client_settings)
    object.__setattr__(mutated_settings, "loss_monitor_betting_fractions", ())
    return mutated_settings


# 条件名 -> (正常な引数から、差し替える引数を作る操作, 期待する例外)。
INVALID_ASSEMBLY_ARGUMENT_CASES = {
    "client_id_bool": (lambda arguments: dict(client_id=True), TypeError),
    "client_id_float": (lambda arguments: dict(client_id=1.0), TypeError),
    "client_id_negative": (lambda arguments: dict(client_id=-1), ValueError),
    "initial_model_id_bool": (lambda arguments: dict(initial_model_id=False), TypeError),
    "initial_model_id_text": (lambda arguments: dict(initial_model_id="0"), TypeError),
    # 負のIDは一時IDの領域。初期モデルのIDには使えない。
    "initial_model_id_negative": (lambda arguments: dict(initial_model_id=-1), ValueError),
    # 分類器のパラメータがfloat64（保有モデルの登録が、写しに対して拒否する）。
    "classifier_with_float64_parameters": (
        lambda arguments: make_float64_initial_model(arguments),
        ValueError,
    ),
    "random_generator_other_type": (
        lambda arguments: dict(python_random_generator=object()),
        TypeError,
    ),
    "random_generator_subclass": (
        lambda arguments: dict(
            python_random_generator=type("RandomSubclass", (random.Random,), {})(0)
        ),
        TypeError,
    ),
    "classifier_other_type": (lambda arguments: dict(initial_classifier=object()), TypeError),
    "classifier_subclass": (
        lambda arguments: dict(
            initial_classifier=make_subclass_copy(arguments["initial_classifier"])
        ),
        TypeError,
    ),
    "concept_optimizer_state_other_type": (
        lambda arguments: dict(
            initial_concept_specific_parameter_optimizer_state=arguments[
                "initial_concept_specific_parameter_optimizer_state"
            ].parameter_optimizer
        ),
        TypeError,
    ),
    "shared_optimizer_state_none": (
        lambda arguments: dict(initial_shared_parameter_optimizer_state=None),
        TypeError,
    ),
    "loss_statistics_other_type": (
        lambda arguments: dict(initial_loss_statistics={"n": 0, "mean": 0.0, "M2": 0.0}),
        TypeError,
    ),
    "settings_other_type": (lambda arguments: dict(run_client_settings=object()), TypeError),
    "settings_subclass": (
        lambda arguments: dict(
            run_client_settings=make_subclass_copy(arguments["run_client_settings"])
        ),
        TypeError,
    ),
    "settings_mutated_around_frozen": (
        lambda arguments: dict(
            run_client_settings=make_settings_mutated_around_frozen(
                arguments["run_client_settings"]
            )
        ),
        RunSettingsValidationError,
    ),
    # 別の分類器のパラメータを指すoptimizerの状態。
    "concept_optimizer_state_of_other_classifier": (
        lambda arguments: dict(
            initial_concept_specific_parameter_optimizer_state=make_other_initial_model(arguments)[
                "initial_concept_specific_parameter_optimizer_state"
            ]
        ),
        ValueError,
    ),
    "shared_optimizer_state_of_other_classifier": (
        lambda arguments: dict(
            initial_shared_parameter_optimizer_state=make_other_initial_model(arguments)[
                "initial_shared_parameter_optimizer_state"
            ]
        ),
        ValueError,
    ),
    # 概念固有部と共有部の入替え。
    "optimizer_states_swapped": (
        lambda arguments: dict(
            initial_concept_specific_parameter_optimizer_state=arguments[
                "initial_shared_parameter_optimizer_state"
            ],
            initial_shared_parameter_optimizer_state=arguments[
                "initial_concept_specific_parameter_optimizer_state"
            ],
        ),
        ValueError,
    ),
    # 分類層だけを最適化する（アダプタが抜けた）optimizerの状態。
    "concept_optimizer_state_missing_adapter": (
        lambda arguments: dict(
            initial_concept_specific_parameter_optimizer_state=ParameterOptimizerState(
                parameters=tuple(arguments["initial_classifier"].classification_layer.parameters()),
                optimizer_settings=arguments["run_client_settings"].parameter_optimizer_settings,
            )
        ),
        ValueError,
    ),
    # 順が逆の共有部のパラメータ。
    "shared_optimizer_state_in_reversed_order": (
        lambda arguments: dict(
            initial_shared_parameter_optimizer_state=ParameterOptimizerState(
                parameters=tuple(
                    reversed(tuple(arguments["initial_classifier"].feature_extractor.parameters()))
                ),
                optimizer_settings=arguments["run_client_settings"].parameter_optimizer_settings,
            )
        ),
        ValueError,
    ),
}


@pytest.mark.parametrize("invalid_case_name", INVALID_ASSEMBLY_ARGUMENT_CASES)
def test_assembly_rejects_invalid_arguments_without_changing_inputs(
    invalid_case_name, monkeypatch, valid_run_settings_mapping
):
    make_invalid_arguments, expected_exception = INVALID_ASSEMBLY_ARGUMENT_CASES[invalid_case_name]
    _, _, assembly_arguments = build_run_client_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=2,
    )
    invalid_arguments = make_invalid_arguments(assembly_arguments)
    initial_model_snapshot = snapshot_initial_model(assembly_arguments)
    with pytest.raises(expected_exception):
        assemble_fedsda_run_client(**assembly_arguments | invalid_arguments)
    assert_initial_model_unchanged(
        initial_model_snapshot=initial_model_snapshot, assembly_arguments=assembly_arguments
    )
    # 拒否の後に、正しい引数で組み立てられる。
    assert type(assemble_fedsda_run_client(**assembly_arguments)) is FedsdaRunClient


PROCESSED_SAMPLE_COUNT = 25


def build_processed_run_client(
    *, monkeypatch, valid_run_settings_mapping, sample_count=PROCESSED_SAMPLE_COUNT
):
    """標本を処理した後のclient（保留・学習データ・記録がある状態）と、乱数生成器、次の観測標本。"""
    run_client, _, assembly_arguments = build_run_client_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=2,
    )
    stream = make_concept_stream(
        sample_count=sample_count + 1, concept_block_length=13, stream_seed=5
    )
    for sample_index, (observed_sample, concept_id) in enumerate(stream[:sample_count]):
        run_client.process_observed_sample(
            observed_sample=observed_sample,
            sample_index=sample_index,
            evaluation_concept_id=concept_id,
        )
    return run_client, assembly_arguments["python_random_generator"], stream[sample_count][0]


class ObservedSampleSubclass(ObservedSample):
    pass


# 条件名 -> (clientと次の観測標本から、不正な操作を行う関数, 期待する例外)。
INVALID_OPERATION_CASES = {
    "observed_sample_tuple": (
        lambda run_client, observed_sample: run_client.process_observed_sample(
            observed_sample=(observed_sample.feature_values, observed_sample.class_label),
            sample_index=PROCESSED_SAMPLE_COUNT,
        ),
        TypeError,
    ),
    "observed_sample_subclass": (
        lambda run_client, observed_sample: run_client.process_observed_sample(
            observed_sample=ObservedSampleSubclass(**vars(observed_sample)),
            sample_index=PROCESSED_SAMPLE_COUNT,
        ),
        TypeError,
    ),
    "sample_index_bool": (
        lambda run_client, observed_sample: run_client.process_observed_sample(
            observed_sample=observed_sample, sample_index=True
        ),
        TypeError,
    ),
    "sample_index_float": (
        lambda run_client, observed_sample: run_client.process_observed_sample(
            observed_sample=observed_sample, sample_index=float(PROCESSED_SAMPLE_COUNT)
        ),
        TypeError,
    ),
    "sample_index_repeated": (
        lambda run_client, observed_sample: run_client.process_observed_sample(
            observed_sample=observed_sample, sample_index=PROCESSED_SAMPLE_COUNT - 1
        ),
        ValueError,
    ),
    "sample_index_skipped": (
        lambda run_client, observed_sample: run_client.process_observed_sample(
            observed_sample=observed_sample, sample_index=PROCESSED_SAMPLE_COUNT + 1
        ),
        ValueError,
    ),
    "evaluation_concept_id_bool": (
        lambda run_client, observed_sample: run_client.process_observed_sample(
            observed_sample=observed_sample,
            sample_index=PROCESSED_SAMPLE_COUNT,
            evaluation_concept_id=True,
        ),
        TypeError,
    ),
    "evaluation_concept_id_text": (
        lambda run_client, observed_sample: run_client.process_observed_sample(
            observed_sample=observed_sample,
            sample_index=PROCESSED_SAMPLE_COUNT,
            evaluation_concept_id="0",
        ),
        TypeError,
    ),
    "flush_round_index_bool": (
        lambda run_client, observed_sample: run_client.flush_pending_local_updates(
            round_index=True
        ),
        TypeError,
    ),
    "flush_round_index_float": (
        lambda run_client, observed_sample: run_client.flush_pending_local_updates(round_index=2.0),
        TypeError,
    ),
    "flush_round_index_negative": (
        lambda run_client, observed_sample: run_client.flush_pending_local_updates(round_index=-1),
        ValueError,
    ),
    "advance_round_index_none": (
        lambda run_client, observed_sample: (
            run_client.advance_new_model_upload_wait_after_synchronization(round_index=None)
        ),
        TypeError,
    ),
    "advance_round_index_negative": (
        lambda run_client, observed_sample: (
            run_client.advance_new_model_upload_wait_after_synchronization(round_index=-1)
        ),
        ValueError,
    ),
}


@pytest.mark.parametrize("invalid_case_name", INVALID_OPERATION_CASES)
def test_run_client_rejects_invalid_operation_input_before_any_update(
    invalid_case_name, monkeypatch, valid_run_settings_mapping
):
    run_invalid_operation, expected_exception = INVALID_OPERATION_CASES[invalid_case_name]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        run_client, python_random_generator, next_observed_sample = build_processed_run_client(
            monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
        )
        # 学習要求を保留している状態で確かめる（境界の学習が、拒否のときに行われないこと）。
        assert run_client.owners.local_training_request_schedule.pending_training_request_count > 0
        state_snapshot = snapshot_run_client_state(
            run_client=run_client, python_random_generator=python_random_generator
        )
        with pytest.raises(expected_exception):
            run_invalid_operation(run_client, next_observed_sample)
        assert_run_client_state_unchanged(
            state_snapshot=state_snapshot,
            run_client=run_client,
            python_random_generator=python_random_generator,
        )
        # 拒否の後に、正しい標本を処理できる。
        run_client.process_observed_sample(
            observed_sample=next_observed_sample, sample_index=PROCESSED_SAMPLE_COUNT
        )


def test_run_client_processes_samples_without_concept_id_as_the_execution_contract_calls_it(
    monkeypatch, valid_run_settings_mapping
):
    """実行の枠の契約どおり、観測標本と位置だけで呼ぶ。概念別の診断は行われない。"""
    run_client, _, _ = build_run_client_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=2,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        for sample_index, (observed_sample, _) in enumerate(
            make_concept_stream(sample_count=30, concept_block_length=13, stream_seed=5)
        ):
            sample_processing = run_client.process_observed_sample(
                observed_sample=observed_sample, sample_index=sample_index
            )
            assert type(sample_processing) is ObservedSampleProcessing
            prediction_record = (
                sample_processing.observed_sample_prediction.sample_prediction_record
            )
            assert prediction_record.sample_index == sample_index
            assert prediction_record.observed_concept_id is None
            assert prediction_record.observed_class_id == observed_sample.class_label
    owners = run_client.owners
    assert owners.diagnostic_evidence_collection.created_true_concept_ids == ()
    assert len(owners.sample_prediction_record_store.snapshot_sample_prediction_records()) == 30
    # 保留へ入った標本は、観測標本と同じ値の、1行のfloat32のtensor。
    last_pending_observation = (
        owners.pending_sample_observation_store.snapshot_pending_sample_observations()[-1]
    )
    last_training_sample = last_pending_observation.training_sample
    assert last_pending_observation.observed_concept_id is None
    assert last_training_sample.input_features.dtype == torch.float32
    assert last_training_sample.input_features.tolist() == [list(observed_sample.feature_values)]
    assert last_training_sample.observed_class_labels.dtype == torch.float32
    assert last_training_sample.observed_class_labels.tolist() == [
        [float(observed_sample.class_label)]
    ]


def test_finalization_rejects_held_validation_without_held_concept_ids(
    monkeypatch, valid_run_settings_mapping
):
    """候補検証を保持しているのに、概念IDを保持していない状態では、何も変えずに拒否する。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        run_client, python_random_generator, _ = build_processed_run_client(
            monkeypatch=monkeypatch,
            valid_run_settings_mapping=valid_run_settings_mapping,
            sample_count=19,
        )
        owners = run_client.owners
        pending_sample_observation_store = owners.pending_sample_observation_store
        assert owners.validation_session_holder.held_validation_session is not None
        held_concept_ids = (
            pending_sample_observation_store.release_validation_assignment_sample_concept_ids()
        )
        state_snapshot = snapshot_run_client_state(
            run_client=run_client, python_random_generator=python_random_generator
        )
        with pytest.raises(ValueError, match="concept IDs must be held"):
            run_client.finalize_incomplete_candidate_validation()
        assert_run_client_state_unchanged(
            state_snapshot=state_snapshot,
            run_client=run_client,
            python_random_generator=python_random_generator,
        )
        # 概念IDの保持を戻せば、回収できる。
        pending_sample_observation_store.hold_validation_assignment_sample_concept_ids(
            sample_concept_ids=held_concept_ids
        )
        assert run_client.finalize_incomplete_candidate_validation() is not None
        assert owners.validation_session_holder.held_validation_session is None
        assert pending_sample_observation_store.validation_assignment_sample_concept_ids is None


class ServerOperationsDoingNothing:
    """実行の枠が求めるサーバの操作の、何もしない代役（呼出しだけを記録する）。"""

    def __init__(self):
        self.calls = []

    def record_client_states_before_synchronization(self, *, round_index):
        self.calls.append(("record", round_index))

    def synchronize_models(self, *, round_index, new_model_registration_available):
        self.calls.append(("synchronize", round_index, new_model_registration_available))

    def finalize_started_communications(self, *, completed_round_count):
        self.calls.append(("finalize", completed_round_count))


def test_run_clients_run_as_participants_of_the_stream_protocol_loop(
    monkeypatch, valid_run_settings_mapping
):
    """組み立てたclientを、既存の実行の枠（参加者の検査と区間の進行）で、そのまま動かす。"""
    _, _, assembly_arguments = build_run_client_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=2,
    )
    client_count = 2
    interval_sample_count = 10
    stream_sample_count = 57
    run_clients = tuple(
        assemble_fedsda_run_client(**assembly_arguments | dict(client_id=client_id))
        for client_id in range(client_count)
    )
    server_operations = ServerOperationsDoingNothing()
    participants = RunParticipants(
        client_operations=run_clients, server_operations=server_operations
    )
    validate_prepared_run_participants(participants=participants, client_count=client_count)
    concept_streams = tuple(
        tuple(
            make_concept_stream(
                sample_count=stream_sample_count,
                concept_block_length=13,
                stream_seed=5 + client_id,
            )
        )
        for client_id in range(client_count)
    )
    observed_client_streams = tuple(
        ClientObservedStream(
            client_id=client_id,
            observed_samples=tuple(observed_sample for observed_sample, _ in concept_stream),
        )
        for client_id, concept_stream in enumerate(concept_streams)
    )
    # 概念列の型は0/1だけを受理するので、標本列の4つの概念を2値へ畳む（受渡しの確認にだけ使う）。
    evaluation_concept_traces = tuple(
        ClientConceptTrace(
            client_id=client_id,
            concept_ids_by_sample_index=tuple(concept_id % 2 for _, concept_id in concept_stream),
        )
        for client_id, concept_stream in enumerate(concept_streams)
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        execution_events = run_stream_protocol_intervals(
            participants=participants,
            observed_client_streams=observed_client_streams,
            evaluation_concept_traces=evaluation_concept_traces,
            server_aggregation_interval_per_client_samples=interval_sample_count,
        )
    round_count = stream_sample_count // interval_sample_count
    processed_sample_count = round_count * interval_sample_count
    # 実行の記録の段の列が、契約の順である。
    expected_stage_names = []
    for _ in range(round_count):
        expected_stage_names += ["sample_processing"] * (interval_sample_count * client_count)
        expected_stage_names += ["pending_update_flush"] * client_count
        expected_stage_names += ["pre_sync_recording"]
        expected_stage_names += ["registration_readiness_check"] * client_count
        expected_stage_names += ["server_synchronization"]
        expected_stage_names += ["upload_wait_advance"] * client_count
    expected_stage_names += ["incomplete_candidate_validation_finalization"] * client_count
    expected_stage_names += ["started_communication_finalization"]
    assert [
        execution_event.stage_name for execution_event in execution_events
    ] == expected_stage_names
    assert [call[0] for call in server_operations.calls] == [
        "record",
        "synchronize",
    ] * round_count + ["finalize"]
    prediction_records_by_client = []
    for run_client in run_clients:
        owners = run_client.owners
        prediction_records = (
            owners.sample_prediction_record_store.snapshot_sample_prediction_records()
        )
        prediction_records_by_client.append(prediction_records)
        assert [record.sample_index for record in prediction_records] == list(
            range(processed_sample_count)
        )
        # 実行の枠が、標本位置の真の概念を渡している。
        assert [record.observed_concept_id for record in prediction_records] == list(
            evaluation_concept_traces[run_client.client_id].concept_ids_by_sample_index[
                :processed_sample_count
            ]
        )
        # 境界で学習要求が消化され、終端で候補検証の保持が外れている。
        assert owners.local_training_request_schedule.pending_training_request_count == 0
        assert owners.validation_session_holder.held_validation_session is None
        assert (
            len(owners.loss_change_alarm_record_store.get_state_snapshot().monitored_log_e_values)
            == processed_sample_count
        )
    # 2つのclientは、別の標本列を処理して、別の状態になっている。警報は1回以上起きている。
    assert prediction_records_by_client[0] != prediction_records_by_client[1]
    assert any(
        run_client.owners.loss_change_alarm_record_store.get_state_snapshot().alarm_sample_indices
        for run_client in run_clients
    )
