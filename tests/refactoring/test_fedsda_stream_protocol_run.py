"""最終構成のFedSDAの全体run（factoryと実行の枠）を、実旧の全体runの最終状態と照合する。"""

import platform
import random
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_fedsda_run_client import (
    ADAPTER_RANK,
    CROSS_EVALUATION_CLIENT_LIMIT,
    HIDDEN_LAYER_WIDTHS,
    MAXIMUM_TOLERATED_MEAN_LOSS_INCREASE,
    assert_run_client_matches_legacy,
    make_run_client_settings,
    set_legacy_configuration,
    snapshot_run_client_state,
)
from test_held_candidate_validation_progress import make_subclass_copy
from test_initial_model_pretraining import assert_numpy_random_states_equal
from test_model_clustering_and_consolidation import (
    MODEL_CLUSTERING_CRITERIA,
    assert_clustering_records_match_legacy,
)
from test_model_cross_evaluation import derive_legacy_cross_evaluation_diagnostics
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping
from test_server_model_registration_and_aggregation import assert_server_state_matches_legacy

from federated_drift_experiment import config, experiment
from federated_drift_experiment.data.specs import DATASET_SPECS
from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_settings import (
    RandomConceptScheduleSettings,
)
from federated_learning_experiments.data.dataset_definitions import get_dataset_definition
from federated_learning_experiments.data.sea.sea_sample_generation import SeaSampleGenerator
from federated_learning_experiments.data.sine.sine_sample_generation import SineSampleGenerator
from federated_learning_experiments.execution.run_participant_contracts import RunParticipants
from federated_learning_experiments.execution.run_random_sources import create_run_random_sources
from federated_learning_experiments.execution.stream_protocol_execution_settings import (
    StreamProtocolExecutionSettings,
)
from federated_learning_experiments.learning.models.model_architecture_settings import (
    ModelArchitectureSettings,
)
from federated_learning_experiments.learning.training.initial_model_pretraining_settings import (
    InitialModelPretrainingSettings,
)
from federated_learning_experiments.runtime.fedsda_run_client import FedsdaRunClient
from federated_learning_experiments.runtime.fedsda_run_participant_factory import (
    FedsdaRunParticipantFactory,
    FedsdaRunParticipantSettings,
)
from federated_learning_experiments.runtime.fedsda_run_server import FedsdaRunServer
from federated_learning_experiments.runtime.server_model_registration_and_aggregation import (
    split_shared_and_concept_specific_parameters,
)
from federated_learning_experiments.runtime.single_run_execution import (
    execute_stream_protocol_run,
)

LEGACY_MODE_NAME = "FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting"
# 旧の設定の差し替え（set_legacy_configuration）が置く、事前学習の条件。
PRETRAINING_VALUES = dict(
    pretraining_sample_count=24, pretraining_epoch_count=2, pretraining_batch_sample_count=8
)


def make_execution_settings(
    *,
    random_seed,
    client_count,
    per_client_sample_count,
    aggregation_interval,
    minimum_change_gap,
    concept_change_probability,
    dataset_name="sine2",
):
    return StreamProtocolExecutionSettings(
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name=dataset_name,
            random_seed=random_seed,
            client_count=client_count,
            per_client_sample_count=per_client_sample_count,
            server_aggregation_interval_per_client_samples=aggregation_interval,
        ),
        concept_schedule_settings=RandomConceptScheduleSettings(
            concept_schedule_strategy="random_changes_after_minimum_index_gap",
            minimum_sample_index_gap_before_change_trial=minimum_change_gap,
            per_eligible_sample_concept_change_probability=concept_change_probability,
        ),
        execution_strategy="sample_index_then_client_order_with_interval_synchronization",
    )


def make_run_participant_settings(valid_run_settings_mapping, *, update_interval, **overrides):
    """旧の設定の差し替えと同じ値の、新の参加者の設定の束。"""
    return FedsdaRunParticipantSettings(
        run_client_settings=make_run_client_settings(
            valid_run_settings_mapping, update_interval=update_interval, **overrides
        ),
        initial_model_pretraining_settings=InitialModelPretrainingSettings(**PRETRAINING_VALUES),
        model_architecture_settings=ModelArchitectureSettings(
            model_architecture_name="shared_backbone_residual_adapter",
            residual_adapter_requested_rank=ADAPTER_RANK,
        ),
        hidden_layer_widths=HIDDEN_LAYER_WIDTHS,
        model_clustering_criteria=MODEL_CLUSTERING_CRITERIA,
        maximum_evaluating_client_count_per_model=CROSS_EVALUATION_CLIENT_LIMIT,
    )


def run_real_legacy_whole_run(*, monkeypatch, execution_settings, update_interval, **overrides):
    """旧の全体の流れ（指標の計算より前）を、旧の部品を、旧のrun_random_drift_experimentと同じ順に呼んで実行する。

    戻り値: 辞書（サーバ、clientのlist、概念列、観測列、終了時のPythonとNumPyの乱数の状態）。呼出し側の乱数は進めない。
    """
    experiment_run_conditions = execution_settings.experiment_run_conditions
    concept_schedule_settings = execution_settings.concept_schedule_settings
    set_legacy_configuration(
        monkeypatch,
        class_count=DATASET_SPECS[experiment_run_conditions.dataset_name].num_classes,
        update_interval=update_interval,
        routing_recalibration="fifo_replay",
        dataset_name=experiment_run_conditions.dataset_name,
        **overrides,
    )
    for legacy_setting_name, legacy_setting_value in dict(
        DATASET=experiment_run_conditions.dataset_name,
        CONCEPT_SCHEDULE="random",
        N_CLIENTS=experiment_run_conditions.client_count,
        TOTAL_DATA_POINTS=experiment_run_conditions.per_client_sample_count,
        AGGREGATION_INTERVAL=(
            experiment_run_conditions.server_aggregation_interval_per_client_samples
        ),
        MIN_STABLE_PERIOD=concept_schedule_settings.minimum_sample_index_gap_before_change_trial,
        DRIFT_PROB=concept_schedule_settings.per_eligible_sample_concept_change_probability,
        FEDSDA_DISTANCE_THRESHOLD=MAXIMUM_TOLERATED_MEAN_LOSS_INCREASE,
    ).items():
        assert hasattr(config, legacy_setting_name), legacy_setting_name
        monkeypatch.setattr(config, legacy_setting_name, legacy_setting_value)
    legacy_mode_specification = experiment.MODE_SPECS[LEGACY_MODE_NAME]
    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(experiment_run_conditions.random_seed)
            np.random.seed(experiment_run_conditions.random_seed)
            torch.manual_seed(experiment_run_conditions.random_seed)
            legacy_server, legacy_clients = experiment._setup_server_and_clients(
                legacy_mode_specification, MAXIMUM_TOLERATED_MEAN_LOSS_INCREASE, False
            )
            chunk_sample_count = getattr(config, legacy_mode_specification.chunk_attr)
            round_count = config.TOTAL_DATA_POINTS // chunk_sample_count
            legacy_concept_schedules = experiment.make_concept_schedules(
                config.N_CLIENTS, config.TOTAL_DATA_POINTS
            )
            legacy_data_streams = experiment.build_data_streams(legacy_concept_schedules)
            for round_index in range(round_count):
                start_index = round_index * chunk_sample_count
                end_index = start_index + chunk_sample_count
                legacy_mode_specification.run_timestep(
                    legacy_clients,
                    legacy_server,
                    [stream[start_index:end_index] for stream in legacy_data_streams],
                    [schedule[start_index:end_index] for schedule in legacy_concept_schedules],
                    round_index,
                    legacy_mode_specification.use_server,
                    False,
                )
            for legacy_client in legacy_clients:
                legacy_client.finalize_incomplete_forward_validation()
            assert not any(hasattr(legacy_client, "flush") for legacy_client in legacy_clients)
            legacy_server.finalize_protocol(round_count)
            return dict(
                legacy_server=legacy_server,
                legacy_clients=legacy_clients,
                legacy_concept_schedules=legacy_concept_schedules,
                legacy_data_streams=legacy_data_streams,
                python_random_state=random.getstate(),
                numpy_random_state=np.random.get_state(),
            )
    finally:
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)


class ConceptWithholdingClientOperations:
    """test専用の中継: 実行の枠が渡す真の概念を捨てて、真の概念なしでclientを呼ぶ。

    真の概念が、診断以外の状態へ影響しないことを確かめるときに、これで包む。
    ほかの操作は、そのままclientへ渡す。
    """

    def __init__(self, *, run_client):
        self.run_client = run_client
        self.client_id = run_client.client_id
        self.withheld_evaluation_concept_ids = []
        self.flush_pending_local_updates = run_client.flush_pending_local_updates
        self.has_model_ready_for_server_registration = (
            run_client.has_model_ready_for_server_registration
        )
        self.advance_new_model_upload_wait_after_synchronization = (
            run_client.advance_new_model_upload_wait_after_synchronization
        )
        self.finalize_incomplete_candidate_validation = (
            run_client.finalize_incomplete_candidate_validation
        )

    def process_observed_sample(self, *, observed_sample, sample_index, evaluation_concept_id):
        self.withheld_evaluation_concept_ids.append(evaluation_concept_id)
        return self.run_client.process_observed_sample(
            observed_sample=observed_sample, sample_index=sample_index
        )


class ObservingParticipantFactory:
    """test専用の中継: factoryの準備の引数（乱数源）を控える。求められたら、clientを、真の概念を捨てる中継で包む。"""

    def __init__(self, *, participant_factory, withhold_evaluation_concepts=False):
        self.participant_factory = participant_factory
        self.withhold_evaluation_concepts = withhold_evaluation_concepts
        self.withholding_client_operations = ()
        self.run_random_sources = None
        self.validate_configuration = participant_factory.validate_configuration

    def prepare_run(self, *, experiment_run_conditions, run_random_sources, sample_generator):
        self.run_random_sources = run_random_sources
        participants = self.participant_factory.prepare_run(
            experiment_run_conditions=experiment_run_conditions,
            run_random_sources=run_random_sources,
            sample_generator=sample_generator,
        )
        if not self.withhold_evaluation_concepts:
            return participants
        self.withholding_client_operations = tuple(
            ConceptWithholdingClientOperations(run_client=run_client)
            for run_client in participants.client_operations
        )
        return RunParticipants(
            client_operations=self.withholding_client_operations,
            server_operations=participants.server_operations,
        )


def execute_stream_protocol_run_with_factory(
    *, execution_settings, run_participant_settings, withhold_evaluation_concepts=False
):
    """新の全体runを実行する。戻り値: (実行の枠の結果, 準備された参加者, 準備に渡された乱数源)。

    `withhold_evaluation_concepts`が真のときだけ、clientを、真の概念を捨てる中継で包む。
    偽のとき、factoryを包む中継は、準備の引数を控えるだけで、参加者をそのまま返す（実行の枠の本来の経路）。
    """
    participant_factory = FedsdaRunParticipantFactory(
        run_participant_settings=run_participant_settings
    )
    observing_factory = ObservingParticipantFactory(
        participant_factory=participant_factory,
        withhold_evaluation_concepts=withhold_evaluation_concepts,
    )
    run_result = execute_stream_protocol_run(
        execution_settings=execution_settings, participant_factory=observing_factory
    )
    return (
        run_result,
        participant_factory.prepared_run_participants,
        observing_factory.run_random_sources,
    )


def assert_server_parameter_computation_counts_match_records(*, run_server, legacy_server):
    """サーバのパラメータの積和演算の数を、通信量（実旧と一致する値）と、診断の記録・層の形から計算した値と照合する。

    戻り値: (集約の合計, 統合の合計, 診断のパラメータ距離の合計)。
    """
    server_owners = run_server.owners
    synchronizations = run_server.snapshot_server_round_synchronizations()
    # 集約: 重み付きの和へ足したのは、上りとして数えたパラメータ（上りを数えるのは、集約だけ）。
    aggregation_count = sum(
        synchronization.client_model_aggregation.weighted_parameter_multiply_accumulate_count
        for synchronization in synchronizations
    )
    assert aggregation_count == legacy_server.comm_parameter_values_up
    assert (
        aggregation_count
        == server_owners.communication_volume_record_store.get_state_snapshot().uploaded_parameter_value_count
    )
    # 層の形: グローバルモデルの、完全なパラメータと、概念固有部の、値の数。
    global_model_repository = server_owners.global_model_repository
    parameter_snapshot = global_model_repository.get_global_model_parameters(
        model_id=global_model_repository.global_model_ids[0]
    )
    full_parameter_value_count = sum(
        parameter_values.numel() for parameter_values in parameter_snapshot.values()
    )
    _, concept_specific_parameters = split_shared_and_concept_specific_parameters(
        parameter_snapshot=parameter_snapshot
    )
    concept_specific_parameter_value_count = sum(
        parameter_values.numel() for parameter_values in concept_specific_parameters.values()
    )
    pair_observations = (
        server_owners.model_clustering_record_store.snapshot_pair_clustering_observations()
    )
    consolidation_count = 0
    diagnostic_distance_count = 0
    for round_index, synchronization in enumerate(synchronizations):
        model_consolidation = synchronization.model_consolidation
        if model_consolidation is None:
            continue
        # 診断のパラメータ距離: クラスタリングの対象のモデルの、全部の対ごとに、概念固有部の値の数×3
        # （距離は、診断の観測に残る対だけでなく、全部の対で計算される）。
        clustered_model_count = sum(
            len(model_cluster) for model_cluster in model_consolidation.model_clusters
        )
        expected_diagnostic_distance_count = (
            3
            * concept_specific_parameter_value_count
            * (clustered_model_count * (clustered_model_count - 1) // 2)
        )
        # 観測に残る対は、その一部（または全部）。
        assert (
            sum(
                pair_observation.round_index == round_index
                for pair_observation in pair_observations
            )
            <= clustered_model_count * (clustered_model_count - 1) // 2
        )
        assert (
            model_consolidation.diagnostic_parameter_distance_multiply_accumulate_count
            == expected_diagnostic_distance_count
        )
        # 統合: 統合したクラスタの、集約の件数が正のメンバーごとに、完全なパラメータの値の数。
        sample_counts = (
            synchronization.client_model_aggregation.aggregated_training_sample_counts_by_model_id
        )
        expected_consolidation_count = 0
        if model_consolidation.model_id_mapping:
            for model_cluster in model_consolidation.model_clusters:
                if len(model_cluster) > 1:
                    expected_consolidation_count += full_parameter_value_count * sum(
                        sample_counts[model_id] > 0 for model_id in model_cluster
                    )
        assert (
            model_consolidation.consolidation_parameter_multiply_accumulate_count
            == expected_consolidation_count
        )
        consolidation_count += expected_consolidation_count
        diagnostic_distance_count += expected_diagnostic_distance_count
    return aggregation_count, consolidation_count, diagnostic_distance_count


def assert_whole_run_matches_legacy(*, run_result, participants, run_random_sources, legacy_run):
    """概念列、観測列、サーバの全状態、診断の記録、各clientの全状態、runの乱数の最終状態を、実旧と照合する。"""
    legacy_server = legacy_run["legacy_server"]
    assert [
        list(concept_trace.concept_ids_by_sample_index)
        for concept_trace in run_result.evaluation_concept_traces
    ] == legacy_run["legacy_concept_schedules"]
    for observed_client_stream, legacy_data_stream in zip(
        run_result.observed_client_streams, legacy_run["legacy_data_streams"], strict=True
    ):
        assert len(observed_client_stream.observed_samples) == len(legacy_data_stream)
        for observed_sample, (legacy_features, legacy_labels) in zip(
            observed_client_stream.observed_samples, legacy_data_stream, strict=True
        ):
            assert observed_sample.feature_values == tuple(legacy_features.tolist())
            assert observed_sample.class_label == int(legacy_labels[0].item())
    run_server = participants.server_operations
    assert type(run_server) is FedsdaRunServer
    server_owners = run_server.owners
    assert_server_state_matches_legacy(
        global_model_repository=server_owners.global_model_repository,
        communication_volume_record_store=server_owners.communication_volume_record_store,
        legacy_server=legacy_server,
    )
    assert_clustering_records_match_legacy(
        server_owners.model_clustering_record_store, legacy_server
    )
    all_evaluations, compared_evaluations, class_evaluations = (
        derive_legacy_cross_evaluation_diagnostics(
            server_owners.cross_evaluation_record_store.snapshot_client_cross_evaluation_records()
        )
    )
    assert all_evaluations == legacy_server.cross_evaluation_diagnostics
    assert compared_evaluations == legacy_server.pair_prediction_diagnostics
    assert class_evaluations == legacy_server.cross_evaluation_class_diagnostics
    for run_client, legacy_client in zip(
        participants.client_operations, legacy_run["legacy_clients"], strict=True
    ):
        assert type(run_client) is FedsdaRunClient
        assert_run_client_matches_legacy(run_client=run_client, legacy_client=legacy_client)
        assert (
            run_client.owners.adaptation_record_store.get_state_snapshot().server_remapped_sample_indices
            == tuple(legacy_client.mapping_change_positions)
        )
    assert (
        run_random_sources.python_random_generator.getstate() == legacy_run["python_random_state"]
    )
    assert_server_parameter_computation_counts_match_records(
        run_server=run_server, legacy_server=legacy_server
    )
    assert_numpy_random_states_equal(
        run_random_sources.numpy_random_generator.get_state(), legacy_run["numpy_random_state"]
    )


WHOLE_RUN_OBSERVED_COVERAGE_BY_CONDITION = {}
# (seed, client数, clientごとの標本数, 集約間隔, 概念の変更までの最小の間隔, 変更の確率, 学習の間隔)。
# 統合が起きる条件は、実旧だけの全体runを64条件進めて選んだ（seed 0・2・4・7の4条件）。
# 最後の条件（seed 17、250件）は、終端で、未完了の候補検証が回収される（実旧だけの76条件から選んだ）。
WHOLE_RUN_CONDITIONS = [
    (0, 3, 300, 10, 30, 0.05, 2),
    (17, 3, 300, 10, 30, 0.05, 1),
    (23, 2, 300, 10, 25, 0.06, 2),
    (1, 3, 330, 50, 30, 0.05, 1),
    (0, 3, 500, 25, 60, 0.03, 2),
    (2, 5, 300, 10, 50, 0.03, 1),
    (4, 3, 500, 25, 60, 0.03, 1),
    (7, 3, 300, 10, 30, 0.05, 2),
    (17, 3, 250, 10, 30, 0.05, 2),
]


@pytest.mark.parametrize(
    "random_seed,client_count,per_client_sample_count,aggregation_interval,minimum_change_gap,concept_change_probability,update_interval",
    WHOLE_RUN_CONDITIONS,
)
def test_whole_stream_protocol_run_matches_real_legacy_whole_run(
    random_seed,
    client_count,
    per_client_sample_count,
    aggregation_interval,
    minimum_change_gap,
    concept_change_probability,
    update_interval,
    monkeypatch,
    valid_run_settings_mapping,
):
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
    python_random_state = random.getstate()
    torch_random_state = torch.get_rng_state().clone()
    run_result, participants, run_random_sources = execute_stream_protocol_run_with_factory(
        execution_settings=execution_settings,
        run_participant_settings=make_run_participant_settings(
            valid_run_settings_mapping, update_interval=update_interval
        ),
    )
    # 全体runは、呼出し側の乱数を進めない。
    assert random.getstate() == python_random_state
    assert torch.equal(torch.get_rng_state(), torch_random_state)
    assert_whole_run_matches_legacy(
        run_result=run_result,
        participants=participants,
        run_random_sources=run_random_sources,
        legacy_run=legacy_run,
    )
    # 通った経路。
    legacy_server = legacy_run["legacy_server"]
    legacy_clients = legacy_run["legacy_clients"]
    synchronizations = participants.server_operations.snapshot_server_round_synchronizations()
    assert len(synchronizations) == run_result.synchronization_interval_count
    observed_paths = set()
    if any(legacy_client.detected_event_positions for legacy_client in legacy_clients):
        observed_paths.add("alarm_raised")
    legacy_actions = {
        legacy_event.action
        for legacy_client in legacy_clients
        for legacy_event in legacy_client.adaptation_events
    }
    observed_paths |= {f"adaptation_{legacy_action}" for legacy_action in legacy_actions}
    # 新の適応記録の結果種別（候補の採用、終端での未完了の候補検証の回収、を含む）。
    observed_paths |= {
        adaptation_record.adaptation_outcome
        for run_client in participants.client_operations
        for adaptation_record in run_client.owners.adaptation_record_store.get_state_snapshot().adaptation_records
    }
    # サーバの計数: 統合の加重平均と、診断のパラメータ距離を、実際に通った条件。
    _, consolidation_count, diagnostic_distance_count = (
        assert_server_parameter_computation_counts_match_records(
            run_server=participants.server_operations, legacy_server=legacy_server
        )
    )
    if consolidation_count > 0:
        observed_paths.add("server_consolidation_parameters_accumulated")
    if diagnostic_distance_count > 0:
        observed_paths.add("server_parameter_distance_computed")
    # 候補の判定の種類（実旧の理由）。保持した判定記録との照合は、clientの全状態の照合が行う。
    observed_paths |= {
        f"decision_{legacy_decision.reason}"
        for legacy_client in legacy_clients
        for legacy_decision in legacy_client.provisional_model_decisions
    }
    assert sum(
        len(
            run_client.owners.candidate_validation_decision_record_store.snapshot_candidate_validation_decision_records()
        )
        for run_client in participants.client_operations
    ) == sum(len(legacy_client.provisional_model_decisions) for legacy_client in legacy_clients)
    # 終端での回収は、実旧では、理由が「前向きの標本が足りない」の、候補の判定として残る。
    assert ("post_alarm_validation_incomplete_candidate_rejected" in observed_paths) == any(
        legacy_decision.reason == "insufficient_forward_data"
        for legacy_client in legacy_clients
        for legacy_decision in legacy_client.provisional_model_decisions
    )
    # 終端の後、どのclientも、候補検証を保持していない。
    assert all(legacy_client._forward_validation is None for legacy_client in legacy_clients)
    assert all(
        run_client.owners.validation_session_holder.held_validation_session is None
        for run_client in participants.client_operations
    )
    if any(synchronization.registered_client_models for synchronization in synchronizations):
        observed_paths.add("new_model_registered")
    if any(synchronization.model_consolidation is not None for synchronization in synchronizations):
        observed_paths.add("models_clustered")
    if any(
        synchronization.model_consolidation is not None
        and synchronization.model_consolidation.absorbed_model_ids
        for synchronization in synchronizations
    ):
        observed_paths.add("models_consolidated")
    if len(legacy_server.global_models) >= 2:
        observed_paths.add("multiple_global_models_at_run_end")
    if run_result.unprocessed_tail_sample_count_per_client:
        observed_paths.add("unprocessed_tail_samples")
    WHOLE_RUN_OBSERVED_COVERAGE_BY_CONDITION[
        (random_seed, client_count, per_client_sample_count, update_interval)
    ] = observed_paths


def test_whole_run_conditions_cover_required_paths():
    """上の対照が、要求の経路をすべて通っていること（全条件を実行したときだけ確かめる）。"""
    if len(WHOLE_RUN_OBSERVED_COVERAGE_BY_CONDITION) < len(WHOLE_RUN_CONDITIONS):
        pytest.skip("対照の全条件を実行したときだけ確かめる")
    observed_paths = set().union(*WHOLE_RUN_OBSERVED_COVERAGE_BY_CONDITION.values())
    assert observed_paths >= {
        "alarm_raised",
        # 候補の採用と、終端での未完了の候補検証の回収。
        "post_alarm_validation_candidate_adopted",
        "post_alarm_validation_incomplete_candidate_rejected",
        "new_model_registered",
        "models_clustered",
        "models_consolidated",
        "multiple_global_models_at_run_end",
        "unprocessed_tail_samples",
        "adaptation_server_merge",
        "server_consolidation_parameters_accumulated",
        "server_parameter_distance_computed",
        # 候補の判定の種類: 採用、区間の判定での棄却（前半、後半、両方）、現行モデルの維持、終端の回収。
        # 別の保有モデルの再利用（alternative_reference_refit）は、この条件では通らない。
        "decision_accepted",
        "decision_first_interval",
        "decision_second_interval",
        "decision_first_and_second",
        "decision_current_reference_refit",
        "decision_insufficient_forward_data",
    }, sorted(observed_paths)


# sine2以外のdataset: (dataset, seed, client数, clientごとの標本数, 集約間隔, 変更までの最小の間隔, 変更の確率, 学習の間隔)。
# 実旧だけでなく新旧の全体runを、3 dataset×6 seed×3条件（54条件。全部一致）進めて、新しいモデルの登録を通る条件を選んだ。
# 最後の要素は、その条件が通ることを確かめる経路（sine2の条件が通らない、別の保有モデルの再利用を含む）。Windowsでの値で、
# ほかの環境では、通る経路が違うことがある（Windows以外では、経路は確かめない）。
OTHER_DATASET_WHOLE_RUN_CONDITIONS = [
    ("sea2", 1, 3, 500, 25, 60, 0.03, 2, {"decision_alternative_reference_refit"}),
    ("sea2", 17, 3, 300, 10, 30, 0.05, 1, set()),
    ("sea4", 23, 5, 300, 10, 50, 0.03, 2, {"multiple_global_models_at_run_end"}),
    ("sea4", 2, 5, 300, 10, 50, 0.03, 2, {"models_consolidated"}),
    ("circle2", 7, 3, 500, 25, 60, 0.03, 2, {"models_consolidated"}),
    ("circle2", 17, 3, 300, 10, 30, 0.05, 1, set()),
    # MNIST（784特徴・10クラス）: 新旧の全体runを、2 dataset×6 seed×4条件（48条件。全部一致）進めて選んだ。
    ("mnist2", 17, 3, 400, 10, 30, 0.05, 2, {"multiple_global_models_at_run_end"}),
    ("mnist2", 2, 3, 500, 25, 60, 0.03, 2, {"models_consolidated"}),
    (
        "mnist4",
        17,
        3,
        400,
        10,
        30,
        0.05,
        2,
        {"multiple_global_models_at_run_end", "decision_alternative_reference_refit"},
    ),
    ("mnist4", 7, 3, 400, 10, 30, 0.05, 2, {"decision_alternative_reference_refit"}),
]
# MNISTの小さい条件の、隠れ層の幅（合成データの小さい条件の幅では、概念の変化が損失に現れず、モデルが増えない）。
MNIST_HIDDEN_LAYER_WIDTHS = (48,)


@pytest.mark.parametrize(
    "dataset_name,random_seed,client_count,per_client_sample_count,aggregation_interval,minimum_change_gap,concept_change_probability,update_interval,required_paths",
    OTHER_DATASET_WHOLE_RUN_CONDITIONS,
)
def test_whole_run_of_other_datasets_matches_real_legacy_whole_run(
    dataset_name,
    random_seed,
    client_count,
    per_client_sample_count,
    aggregation_interval,
    minimum_change_gap,
    concept_change_probability,
    update_interval,
    required_paths,
    monkeypatch,
    valid_run_settings_mapping,
):
    """sea2・sea4・circle2・mnist2・mnist4でも、全体runの全状態が実旧と一致する（特徴数・概念数・クラス数は、datasetの定義から）。"""
    dataset_definition = get_dataset_definition(dataset_name=dataset_name)
    if dataset_name.startswith("mnist"):
        # 新旧の組立てが読む、testの定数（旧の設定の差し替えと、新の参加者の設定の束）。
        for test_module_name in ("test_fedsda_run_client", __name__):
            monkeypatch.setattr(
                sys.modules[test_module_name], "HIDDEN_LAYER_WIDTHS", MNIST_HIDDEN_LAYER_WIDTHS
            )
    execution_settings = make_execution_settings(
        random_seed=random_seed,
        client_count=client_count,
        per_client_sample_count=per_client_sample_count,
        aggregation_interval=aggregation_interval,
        minimum_change_gap=minimum_change_gap,
        concept_change_probability=concept_change_probability,
        dataset_name=dataset_name,
    )
    legacy_run = run_real_legacy_whole_run(
        monkeypatch=monkeypatch,
        execution_settings=execution_settings,
        update_interval=update_interval,
    )
    run_result, participants, run_random_sources = execute_stream_protocol_run_with_factory(
        execution_settings=execution_settings,
        run_participant_settings=make_run_participant_settings(
            valid_run_settings_mapping, update_interval=update_interval
        ),
    )
    assert_whole_run_matches_legacy(
        run_result=run_result,
        participants=participants,
        run_random_sources=run_random_sources,
        legacy_run=legacy_run,
    )
    # datasetの定義どおりの特徴数・概念で、実行している。
    assert all(
        len(observed_sample.feature_values) == dataset_definition.input_feature_count
        for observed_client_stream in run_result.observed_client_streams
        for observed_sample in observed_client_stream.observed_samples
    )
    assert all(
        legacy_features.shape == (dataset_definition.input_feature_count,)
        for legacy_data_stream in legacy_run["legacy_data_streams"]
        for legacy_features, _ in legacy_data_stream
    )
    visited_concept_ids = {
        concept_id
        for concept_trace in run_result.evaluation_concept_traces
        for concept_id in concept_trace.concept_ids_by_sample_index
    }
    # 全概念を通る（sea4は4概念）。
    assert visited_concept_ids == set(range(dataset_definition.concept_count))
    legacy_server = legacy_run["legacy_server"]
    legacy_clients = legacy_run["legacy_clients"]
    synchronizations = participants.server_operations.snapshot_server_round_synchronizations()
    # 通る経路の期待は、条件を選んだ環境（Windows）でだけ確かめる。ほかの環境では、浮動小数点の差で、
    # 警報やモデルの登録の起き方が変わる（WSL Ubuntuで確認。新旧の全状態の一致は、上で、どの環境でも確かめている）。
    if platform.system() != "Windows":
        return
    assert any(legacy_client.detected_event_positions for legacy_client in legacy_clients)
    observed_paths = {
        f"decision_{legacy_decision.reason}"
        for legacy_client in legacy_clients
        for legacy_decision in legacy_client.provisional_model_decisions
    }
    if any(synchronization.registered_client_models for synchronization in synchronizations):
        observed_paths.add("new_model_registered")
    if any(
        synchronization.model_consolidation is not None
        and synchronization.model_consolidation.absorbed_model_ids
        for synchronization in synchronizations
    ):
        observed_paths.add("models_consolidated")
    if len(legacy_server.global_models) >= 2:
        observed_paths.add("multiple_global_models_at_run_end")
    if required_paths:
        assert observed_paths >= required_paths | {"new_model_registered", "decision_accepted"}


def owners_decision_records(run_client):
    """clientが保持している、候補検証の判定記録の一覧。"""
    return run_client.owners.candidate_validation_decision_record_store.snapshot_candidate_validation_decision_records()


# 真の概念に依存する診断（読取りの項目名）。真の概念を受け取らない全体runでは、これらだけが違ってよい。
CONCEPT_DEPENDENT_STATE_NAMES = (
    "diagnostics",
    "counts",
    "sample_prediction_records",
    # 保留中の標本と、候補検証へ渡した標本は、診断用の概念IDを一緒に持つ。
    "pending_sample_observations",
    "validation_assignment_sample_concept_ids",
)


def test_true_concepts_affect_only_concept_dependent_diagnostics_of_whole_run(
    valid_run_settings_mapping,
):
    """真の概念は、診断にだけ使われる: 真の概念を捨ててclientを呼んだ全体runは、真の概念に依存する診断だけが、本来の全体runと違う。"""
    # クラスタリングと統合が起きる条件（上の対照の条件の1つ）。
    execution_settings = make_execution_settings(
        random_seed=7,
        client_count=3,
        per_client_sample_count=300,
        aggregation_interval=10,
        minimum_change_gap=30,
        concept_change_probability=0.05,
    )
    run_participant_settings = make_run_participant_settings(
        valid_run_settings_mapping, update_interval=2
    )
    # 真の概念を捨てる全体run（clientは、真の概念を受け取らない）。
    participant_factory = FedsdaRunParticipantFactory(
        run_participant_settings=run_participant_settings
    )
    withholding_factory = ObservingParticipantFactory(
        participant_factory=participant_factory, withhold_evaluation_concepts=True
    )
    run_result = execute_stream_protocol_run(
        execution_settings=execution_settings, participant_factory=withholding_factory
    )
    participants = participant_factory.prepared_run_participants
    run_random_sources = withholding_factory.run_random_sources
    # 中継は、実行の枠から、概念列の値を受け取って、捨てている（渡されなかったのではない）。
    assert len(withholding_factory.withholding_client_operations) == 3
    for withholding_client_operations, concept_trace in zip(
        withholding_factory.withholding_client_operations,
        run_result.evaluation_concept_traces,
        strict=True,
    ):
        assert withholding_client_operations.withheld_evaluation_concept_ids == list(
            concept_trace.concept_ids_by_sample_index[
                : run_result.processed_sample_count_per_client
            ]
        )
    # 本来の全体run（実行の枠が、真の概念をclientへ渡す）。
    delivered_run_result, delivered_participants, delivered_random_sources = (
        execute_stream_protocol_run_with_factory(
            execution_settings=execution_settings,
            run_participant_settings=run_participant_settings,
        )
    )
    assert delivered_run_result == run_result
    assert (
        run_random_sources.python_random_generator.getstate()
        == delivered_random_sources.python_random_generator.getstate()
    )
    client_decision_record_counts = []
    for run_client, delivered_run_client in zip(
        participants.client_operations, delivered_participants.client_operations, strict=True
    ):
        assert run_client is not delivered_run_client
        state_snapshot = snapshot_run_client_state(
            run_client=run_client,
            python_random_generator=run_random_sources.python_random_generator,
        )
        delivered_state_snapshot = snapshot_run_client_state(
            run_client=delivered_run_client,
            python_random_generator=delivered_random_sources.python_random_generator,
        )
        # 許す項目の名前は、読取りの実際の項目名である。
        assert set(CONCEPT_DEPENDENT_STATE_NAMES) <= set(state_snapshot)
        differing_state_names = set()
        for state_name, state in state_snapshot.items():
            delivered_state = delivered_state_snapshot[state_name]
            if state_name in ("parameters", "torch_random_state"):
                states_equal = len(state) == len(delivered_state) and all(
                    torch.equal(tensor, delivered_tensor)
                    for tensor, delivered_tensor in zip(state, delivered_state, strict=True)
                )
            elif state_name == "optimizer_states":
                states_equal = repr(state) == repr(delivered_state)
            elif state_name in ("held_validation_session", "pending_model_upload"):
                # 別のrunの別のオブジェクトなので、有無だけを比べる。
                states_equal = (state is None) == (delivered_state is None)
            elif state_name in (
                "training_samples",
                "evaluation_samples",
                "pending_sample_observations",
            ):
                states_equal = repr(state) == repr(delivered_state)
            else:
                states_equal = state == delivered_state
            if not states_equal:
                differing_state_names.add(state_name)
        assert differing_state_names <= set(CONCEPT_DEPENDENT_STATE_NAMES), differing_state_names
        # 少なくとも、診断証拠・割当概念の計数・標本ごとの記録は、実際に違っている（比較が働いている）。
        assert differing_state_names >= {"diagnostics", "counts", "sample_prediction_records"}
        # 違ってよい項目も、違うのは、真の概念の部分だけである（ほかの部分は、同じ）。
        # 標本ごとの記録: 概念の欄と、真の概念別の診断証拠の予測の正誤を除くと同じ（予測、正誤、重みほか）。
        assert [
            replace(
                prediction_record,
                observed_concept_id=None,
                true_concept_diagnostic_prediction_is_correct=None,
            )
            for prediction_record in delivered_state_snapshot["sample_prediction_records"]
        ] == list(state_snapshot["sample_prediction_records"])
        assert all(
            prediction_record.true_concept_diagnostic_prediction_is_correct is not None
            for prediction_record in delivered_state_snapshot["sample_prediction_records"]
        )
        # 計数: 割当概念の計数を除くと同じ（学習した標本数、更新の回数）。
        assert replace(
            delivered_state_snapshot["counts"], assigned_sample_counts_by_model_and_concept_id={}
        ) == replace(state_snapshot["counts"], assigned_sample_counts_by_model_and_concept_id={})
        # 診断証拠: globalの診断証拠は同じ。違うのは、真の概念別の診断証拠だけ。
        global_evidence, created_true_concept_ids, _ = state_snapshot["diagnostics"]
        delivered_global_evidence, delivered_true_concept_ids, _ = delivered_state_snapshot[
            "diagnostics"
        ]
        assert global_evidence == delivered_global_evidence
        assert created_true_concept_ids == ()
        assert delivered_true_concept_ids
        # 保留中の標本: 概念の欄を除くと同じ（位置と標本）。
        assert repr(
            tuple(
                replace(pending_observation, observed_concept_id=None)
                for pending_observation in delivered_state_snapshot["pending_sample_observations"]
            )
        ) == repr(state_snapshot["pending_sample_observations"])
        # 候補検証の判定記録は、真の概念に依存しない。
        decision_records = owners_decision_records(run_client)
        assert decision_records == owners_decision_records(delivered_run_client)
        client_decision_record_counts.append(len(decision_records))
        # 真の概念を受け取らないと、割当概念の計数と、真の概念別の診断証拠は、作られない。
        owners = run_client.owners
        assert owners.diagnostic_evidence_collection.created_true_concept_ids == ()
        assert all(
            record.observed_concept_id is None
            for record in owners.sample_prediction_record_store.snapshot_sample_prediction_records()
        )
        assert delivered_run_client.owners.diagnostic_evidence_collection.created_true_concept_ids
    # 判定記録の比較は、空どうしの比較ではない。
    assert sum(client_decision_record_counts) > 0
    # サーバ側は、クラスタリングの真の概念の一致の診断だけが違う。
    server_owners = participants.server_operations.owners
    delivered_server_owners = delivered_participants.server_operations.owners
    assert (
        server_owners.communication_volume_record_store.get_state_snapshot()
        == delivered_server_owners.communication_volume_record_store.get_state_snapshot()
    )
    assert (
        server_owners.cross_evaluation_record_store.snapshot_client_cross_evaluation_records()
        == delivered_server_owners.cross_evaluation_record_store.snapshot_client_cross_evaluation_records()
    )
    assert (
        server_owners.model_clustering_record_store.snapshot_model_clustering_observations()
        == delivered_server_owners.model_clustering_record_store.snapshot_model_clustering_observations()
    )
    pair_observations = (
        server_owners.model_clustering_record_store.snapshot_pair_clustering_observations()
    )
    assert pair_observations
    assert all(
        pair_observation.true_concepts_match is None for pair_observation in pair_observations
    )
    assert [
        replace(pair_observation, true_concepts_match=None)
        for pair_observation in delivered_server_owners.model_clustering_record_store.snapshot_pair_clustering_observations()
    ] == list(pair_observations)


def test_factory_prepares_fresh_participants_for_each_run(valid_run_settings_mapping):
    """同じfactoryで2回実行すると、同じ結果になり、2回めの参加者は、1回めの参加者と状態を共有しない。"""
    execution_settings = make_execution_settings(
        random_seed=17,
        client_count=2,
        per_client_sample_count=120,
        aggregation_interval=10,
        minimum_change_gap=20,
        concept_change_probability=0.08,
    )
    participant_factory = FedsdaRunParticipantFactory(
        run_participant_settings=make_run_participant_settings(
            valid_run_settings_mapping, update_interval=2
        )
    )
    assert participant_factory.prepared_run_participants is None
    first_run_result = execute_stream_protocol_run(
        execution_settings=execution_settings, participant_factory=participant_factory
    )
    first_participants = participant_factory.prepared_run_participants
    assert type(first_participants) is RunParticipants
    first_server_owners = first_participants.server_operations.owners
    first_volume = first_server_owners.communication_volume_record_store.get_state_snapshot()
    first_prediction_records = [
        run_client.owners.sample_prediction_record_store.snapshot_sample_prediction_records()
        for run_client in first_participants.client_operations
    ]
    second_run_result = execute_stream_protocol_run(
        execution_settings=execution_settings, participant_factory=participant_factory
    )
    second_participants = participant_factory.prepared_run_participants
    assert second_run_result == first_run_result
    assert second_participants is not first_participants
    second_server_owners = second_participants.server_operations.owners
    for owner_name in vars(first_server_owners):
        assert getattr(second_server_owners, owner_name) is not getattr(
            first_server_owners, owner_name
        )
    assert (
        second_server_owners.communication_volume_record_store.get_state_snapshot() == first_volume
    )
    # 1回めの参加者は、2回めの実行で変わっていない。
    assert (
        first_server_owners.communication_volume_record_store.get_state_snapshot() == first_volume
    )
    for first_run_client, second_run_client, first_records in zip(
        first_participants.client_operations,
        second_participants.client_operations,
        first_prediction_records,
        strict=True,
    ):
        assert second_run_client is not first_run_client
        for owner_name in vars(first_run_client.owners):
            assert getattr(second_run_client.owners, owner_name) is not getattr(
                first_run_client.owners, owner_name
            )
        assert (
            first_run_client.owners.sample_prediction_record_store.snapshot_sample_prediction_records()
            == first_records
        )
        assert (
            second_run_client.owners.sample_prediction_record_store.snapshot_sample_prediction_records()
            == first_records
        )
    assert len(first_prediction_records[0]) == first_run_result.processed_sample_count_per_client


def test_participant_settings_reject_invalid_values(valid_run_settings_mapping):
    valid_settings = make_run_participant_settings(valid_run_settings_mapping, update_interval=2)
    for settings_field_name, invalid_value in (
        ("run_client_settings", None),
        ("run_client_settings", make_subclass_copy(valid_settings.run_client_settings)),
        ("initial_model_pretraining_settings", PRETRAINING_VALUES),
        ("model_architecture_settings", valid_settings.model_clustering_criteria),
        ("model_clustering_criteria", make_subclass_copy(valid_settings.model_clustering_criteria)),
        ("hidden_layer_widths", [5, 4]),
        ("hidden_layer_widths", ()),
        ("hidden_layer_widths", (5, 0)),
        ("hidden_layer_widths", (5, True)),
        ("maximum_evaluating_client_count_per_model", 0),
        ("maximum_evaluating_client_count_per_model", 3.0),
        ("maximum_evaluating_client_count_per_model", True),
    ):
        with pytest.raises(RunSettingsValidationError) as raised_error:
            replace(valid_settings, **{settings_field_name: invalid_value})
        assert raised_error.value.configuration_parameter_name == settings_field_name
    # クラスタリングの閾値は、clientの許容する平均損失の増加と同じ値にする（旧は、1つの値を両方に使う）。
    with pytest.raises(RunSettingsValidationError) as raised_error:
        replace(
            valid_settings,
            model_clustering_criteria=replace(
                valid_settings.model_clustering_criteria, maximum_same_cluster_decision_score=0.2
            ),
        )
    assert raised_error.value.configuration_parameter_name == "maximum_same_cluster_decision_score"
    # frozenを回避して書き換えた基準も、束の生成時に拒否する。
    mutated_criteria = replace(valid_settings.model_clustering_criteria)
    object.__setattr__(mutated_criteria, "clustering_confidence_level", 1.5)
    with pytest.raises(RunSettingsValidationError) as raised_error:
        replace(valid_settings, model_clustering_criteria=mutated_criteria)
    assert raised_error.value.configuration_parameter_name == "model_clustering_criteria"


def test_factory_rejects_invalid_preparation_before_consuming_random_numbers(
    valid_run_settings_mapping,
):
    valid_settings = make_run_participant_settings(valid_run_settings_mapping, update_interval=2)
    execution_settings = make_execution_settings(
        random_seed=3,
        client_count=2,
        per_client_sample_count=40,
        aggregation_interval=10,
        minimum_change_gap=20,
        concept_change_probability=0.05,
    )
    experiment_run_conditions = execution_settings.experiment_run_conditions
    run_random_sources = create_run_random_sources(random_seed=3)
    preparation_arguments = dict(
        experiment_run_conditions=experiment_run_conditions,
        run_random_sources=run_random_sources,
        sample_generator=SineSampleGenerator(
            numpy_random_generator=run_random_sources.numpy_random_generator
        ),
    )

    def get_random_states():
        return (
            run_random_sources.python_random_generator.getstate(),
            repr(run_random_sources.numpy_random_generator.get_state()),
            torch.get_rng_state().clone(),
            random.getstate(),
        )

    random_states = get_random_states()
    participant_factory = FedsdaRunParticipantFactory(run_participant_settings=valid_settings)
    # 事前検査は、乱数を消費せず、参加者を作らない。
    participant_factory.validate_configuration()
    assert participant_factory.prepared_run_participants is None
    for invalid_arguments in (
        dict(experiment_run_conditions=None),
        dict(experiment_run_conditions=make_subclass_copy(experiment_run_conditions)),
        # 生成器と違うdataset（sea2、mnist2）。
        dict(experiment_run_conditions=replace(experiment_run_conditions, dataset_name="sea2")),
        dict(experiment_run_conditions=replace(experiment_run_conditions, dataset_name="mnist2")),
        # sea2の実行条件へ、sea4の生成器（同じ型で、概念数が違う）。
        dict(
            experiment_run_conditions=replace(experiment_run_conditions, dataset_name="sea2"),
            sample_generator=SeaSampleGenerator(
                numpy_random_generator=run_random_sources.numpy_random_generator,
                concept_count=4,
            ),
        ),
        dict(run_random_sources=None),
        dict(run_random_sources=random.Random(3)),
        dict(sample_generator=None),
        dict(sample_generator=run_random_sources.numpy_random_generator),
    ):
        with pytest.raises((TypeError, ValueError)):
            participant_factory.prepare_run(**preparation_arguments | invalid_arguments)
        assert participant_factory.prepared_run_participants is None
    # 束が不正なfactoryは、事前検査でも、準備でも拒否する。
    mutated_settings = replace(valid_settings)
    object.__setattr__(mutated_settings, "hidden_layer_widths", ())
    for invalid_settings in (None, make_subclass_copy(valid_settings), mutated_settings):
        invalid_factory = FedsdaRunParticipantFactory(run_participant_settings=invalid_settings)
        with pytest.raises(RunSettingsValidationError):
            invalid_factory.validate_configuration()
        with pytest.raises(RunSettingsValidationError):
            invalid_factory.prepare_run(**preparation_arguments)
        assert invalid_factory.prepared_run_participants is None
    current_random_states = get_random_states()
    assert current_random_states[0] == random_states[0]
    assert current_random_states[1] == random_states[1]
    assert torch.equal(current_random_states[2], random_states[2])
    assert current_random_states[3] == random_states[3]
    # 正常な準備: clientは0から順、サーバは、同じclientの列と、初期モデルだけを持つ。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        participants = participant_factory.prepare_run(**preparation_arguments)
    assert participant_factory.prepared_run_participants is participants
    assert [run_client.client_id for run_client in participants.client_operations] == [0, 1]
    assert all(type(run_client) is FedsdaRunClient for run_client in participants.client_operations)
    server_owners = participants.server_operations.owners
    assert server_owners.global_model_repository.global_model_ids == (0,)
    assert server_owners.global_model_repository.next_global_model_id == 1
    assert participants.server_operations.snapshot_server_round_synchronizations() == ()
