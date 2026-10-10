"""最終構成のFedSDAの全体run（factoryと実行の枠）を、実旧の全体runの最終状態と照合する。"""

import random
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
from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.data.concept_schedules.random_concept_schedule_settings import (
    RandomConceptScheduleSettings,
)
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
):
    return StreamProtocolExecutionSettings(
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name="sine2",
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
        class_count=2,
        update_interval=update_interval,
        routing_recalibration="fifo_replay",
        **overrides,
    )
    for legacy_setting_name, legacy_setting_value in dict(
        DATASET="sine2",
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


class ConceptInjectingClientOperations:
    """test専用の中継: 実行の枠からの標本処理へ、標本位置の真の概念を足して、clientへ渡す。

    実行の枠の契約は、clientへ真の概念を渡さない。実旧のclientは、標本ごとに真の概念を受け取るので、
    診断まで照合するときに、これで包む。ほかの操作は、そのままclientへ渡す。
    """

    def __init__(self, *, run_client, concept_ids_by_sample_index):
        self.run_client = run_client
        self.client_id = run_client.client_id
        self.concept_ids_by_sample_index = concept_ids_by_sample_index
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

    def process_observed_sample(self, *, observed_sample, sample_index):
        return self.run_client.process_observed_sample(
            observed_sample=observed_sample,
            sample_index=sample_index,
            evaluation_concept_id=self.concept_ids_by_sample_index[sample_index],
        )


class ObservingParticipantFactory:
    """test専用の中継: factoryの準備の引数（乱数源）を控える。求められたら、clientを真の概念の中継で包む。"""

    def __init__(self, *, participant_factory, concept_schedules=None):
        self.participant_factory = participant_factory
        self.concept_schedules = concept_schedules
        self.run_random_sources = None
        self.validate_configuration = participant_factory.validate_configuration

    def prepare_run(self, *, experiment_run_conditions, run_random_sources, sample_generator):
        self.run_random_sources = run_random_sources
        participants = self.participant_factory.prepare_run(
            experiment_run_conditions=experiment_run_conditions,
            run_random_sources=run_random_sources,
            sample_generator=sample_generator,
        )
        if self.concept_schedules is None:
            return participants
        return RunParticipants(
            client_operations=tuple(
                ConceptInjectingClientOperations(
                    run_client=run_client, concept_ids_by_sample_index=concept_schedule
                )
                for run_client, concept_schedule in zip(
                    participants.client_operations, self.concept_schedules, strict=True
                )
            ),
            server_operations=participants.server_operations,
        )


def execute_stream_protocol_run_with_factory(
    *, execution_settings, run_participant_settings, concept_schedules=None
):
    """新の全体runを実行する。戻り値: (実行の枠の結果, 準備された参加者, 準備に渡された乱数源)。"""
    participant_factory = FedsdaRunParticipantFactory(
        run_participant_settings=run_participant_settings
    )
    observing_factory = ObservingParticipantFactory(
        participant_factory=participant_factory, concept_schedules=concept_schedules
    )
    run_result = execute_stream_protocol_run(
        execution_settings=execution_settings, participant_factory=observing_factory
    )
    return (
        run_result,
        participant_factory.prepared_run_participants,
        observing_factory.run_random_sources,
    )


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
    assert_numpy_random_states_equal(
        run_random_sources.numpy_random_generator.get_state(), legacy_run["numpy_random_state"]
    )


WHOLE_RUN_OBSERVED_COVERAGE_BY_CONDITION = {}
# (seed, client数, clientごとの標本数, 集約間隔, 概念の変更までの最小の間隔, 変更の確率, 学習の間隔)。
# 統合が起きる条件は、実旧だけの全体runを64条件進めて選んだ（seed 0・2・4・7の4条件）。
WHOLE_RUN_CONDITIONS = [
    (0, 3, 300, 10, 30, 0.05, 2),
    (17, 3, 300, 10, 30, 0.05, 1),
    (23, 2, 300, 10, 25, 0.06, 2),
    (1, 3, 330, 50, 30, 0.05, 1),
    (0, 3, 500, 25, 60, 0.03, 2),
    (2, 5, 300, 10, 50, 0.03, 1),
    (4, 3, 500, 25, 60, 0.03, 1),
    (7, 3, 300, 10, 30, 0.05, 2),
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
        concept_schedules=legacy_run["legacy_concept_schedules"],
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
        "new_model_registered",
        "models_clustered",
        "models_consolidated",
        "multiple_global_models_at_run_end",
        "unprocessed_tail_samples",
        "adaptation_server_merge",
    }, sorted(observed_paths)


# 真の概念に依存する診断（読取りの項目名）。真の概念を渡さない全体runでは、これらだけが違ってよい。
CONCEPT_DEPENDENT_STATE_NAMES = (
    "diagnostics",
    "counts",
    "sample_prediction_records",
    # 保留中の標本と、候補検証へ渡した標本は、診断用の概念IDを一緒に持つ。
    "pending_sample_observations",
    "validation_assignment_sample_concept_ids",
)


def test_whole_run_without_true_concepts_differs_only_in_concept_dependent_diagnostics(
    valid_run_settings_mapping,
):
    """実行の枠は、clientへ真の概念を渡さない。そのときの全体runは、真の概念に依存する診断だけが、渡した全体runと違う。"""
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
    run_result, participants, run_random_sources = execute_stream_protocol_run_with_factory(
        execution_settings=execution_settings, run_participant_settings=run_participant_settings
    )
    concept_schedules = [
        list(concept_trace.concept_ids_by_sample_index)
        for concept_trace in run_result.evaluation_concept_traces
    ]
    injected_run_result, injected_participants, injected_random_sources = (
        execute_stream_protocol_run_with_factory(
            execution_settings=execution_settings,
            run_participant_settings=run_participant_settings,
            concept_schedules=concept_schedules,
        )
    )
    assert injected_run_result == run_result
    assert (
        run_random_sources.python_random_generator.getstate()
        == injected_random_sources.python_random_generator.getstate()
    )
    for run_client, injected_run_client in zip(
        participants.client_operations, injected_participants.client_operations, strict=True
    ):
        assert run_client is not injected_run_client
        state_snapshot = snapshot_run_client_state(
            run_client=run_client,
            python_random_generator=run_random_sources.python_random_generator,
        )
        injected_state_snapshot = snapshot_run_client_state(
            run_client=injected_run_client,
            python_random_generator=injected_random_sources.python_random_generator,
        )
        differing_state_names = set()
        for state_name, state in state_snapshot.items():
            injected_state = injected_state_snapshot[state_name]
            if state_name in ("parameters", "torch_random_state"):
                states_equal = len(state) == len(injected_state) and all(
                    torch.equal(tensor, injected_tensor)
                    for tensor, injected_tensor in zip(state, injected_state, strict=True)
                )
            elif state_name == "optimizer_states":
                states_equal = repr(state) == repr(injected_state)
            elif state_name in ("held_validation_session", "pending_model_upload"):
                # 別のrunの別のオブジェクトなので、有無だけを比べる。
                states_equal = (state is None) == (injected_state is None)
            elif state_name in (
                "training_samples",
                "evaluation_samples",
                "pending_sample_observations",
            ):
                states_equal = repr(state) == repr(injected_state)
            else:
                states_equal = state == injected_state
            if not states_equal:
                differing_state_names.add(state_name)
        assert differing_state_names <= set(CONCEPT_DEPENDENT_STATE_NAMES), differing_state_names
        # 真の概念を渡さないと、割当概念の計数と、真の概念別の診断証拠は、作られない。
        owners = run_client.owners
        assert owners.diagnostic_evidence_collection.created_true_concept_ids == ()
        assert all(
            record.observed_concept_id is None
            for record in owners.sample_prediction_record_store.snapshot_sample_prediction_records()
        )
        assert injected_run_client.owners.diagnostic_evidence_collection.created_true_concept_ids
    # サーバ側は、クラスタリングの真の概念の一致の診断だけが違う。
    server_owners = participants.server_operations.owners
    injected_server_owners = injected_participants.server_operations.owners
    assert (
        server_owners.communication_volume_record_store.get_state_snapshot()
        == injected_server_owners.communication_volume_record_store.get_state_snapshot()
    )
    assert (
        server_owners.cross_evaluation_record_store.snapshot_client_cross_evaluation_records()
        == injected_server_owners.cross_evaluation_record_store.snapshot_client_cross_evaluation_records()
    )
    assert (
        server_owners.model_clustering_record_store.snapshot_model_clustering_observations()
        == injected_server_owners.model_clustering_record_store.snapshot_model_clustering_observations()
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
        for pair_observation in injected_server_owners.model_clustering_record_store.snapshot_pair_clustering_observations()
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
    # 全clientとサーバは、runの同じ乱数生成器を借りている（clientごとの乱数の列に分かれていない）。
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
        dict(experiment_run_conditions=replace(experiment_run_conditions, dataset_name="sea2")),
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
