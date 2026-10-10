"""最終構成のFedSDAのサーバの組立てと、実行の枠が求める3つの操作。"""

import random
from dataclasses import replace

import pytest
import torch
from test_fedsda_stream_protocol_run import make_execution_settings, make_run_participant_settings
from test_held_candidate_validation_progress import make_subclass_copy
from test_model_clustering_and_consolidation import MODEL_CLUSTERING_CRITERIA
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

import federated_learning_experiments.runtime.fedsda_run_server as run_server_module
from federated_learning_experiments.data.sine.sine_sample_generation import SineSampleGenerator
from federated_learning_experiments.evaluation.communication_volume_record_store import (
    CommunicationVolumeRecordStore,
)
from federated_learning_experiments.evaluation.cross_evaluation_record_store import (
    CrossEvaluationRecordStore,
)
from federated_learning_experiments.evaluation.model_clustering_record_store import (
    ModelClusteringRecordStore,
)
from federated_learning_experiments.execution.run_random_sources import create_run_random_sources
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.methods.fedsda.model_registration.global_model_repository import (
    GlobalModelRepository,
)
from federated_learning_experiments.runtime.fedsda_run_participant_factory import (
    FedsdaRunParticipantFactory,
)
from federated_learning_experiments.runtime.fedsda_run_server import (
    FedsdaRunServer,
    FedsdaRunServerOwners,
    assemble_fedsda_run_server,
)
from federated_learning_experiments.runtime.server_round_synchronization import (
    ServerRoundSynchronization,
)

CLIENT_COUNT = 3


def prepare_run_participants(valid_run_settings_mapping):
    """factoryで準備した、標本をまだ処理していない参加者（client 3つとサーバ）と、runの乱数源。"""
    execution_settings = make_execution_settings(
        random_seed=3,
        client_count=CLIENT_COUNT,
        per_client_sample_count=40,
        aggregation_interval=10,
        minimum_change_gap=20,
        concept_change_probability=0.05,
    )
    run_random_sources = create_run_random_sources(random_seed=3)
    participant_factory = FedsdaRunParticipantFactory(
        run_participant_settings=make_run_participant_settings(
            valid_run_settings_mapping, update_interval=2
        )
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        participants = participant_factory.prepare_run(
            experiment_run_conditions=execution_settings.experiment_run_conditions,
            run_random_sources=run_random_sources,
            sample_generator=SineSampleGenerator(
                numpy_random_generator=run_random_sources.numpy_random_generator
            ),
        )
    return participants, run_random_sources


def snapshot_server_state(run_server):
    owners = run_server.owners
    repository = owners.global_model_repository
    return dict(
        global_model_ids=repository.global_model_ids,
        next_global_model_id=repository.next_global_model_id,
        loss_statistics=repository.snapshot_global_model_loss_statistics(),
        registration_records=repository.snapshot_model_registration_records(),
        communication_volume=owners.communication_volume_record_store.get_state_snapshot(),
        cross_evaluation_records=owners.cross_evaluation_record_store.snapshot_client_cross_evaluation_records(),
        pair_observations=owners.model_clustering_record_store.snapshot_pair_clustering_observations(),
        model_observations=owners.model_clustering_record_store.snapshot_model_clustering_observations(),
        synchronizations=run_server.snapshot_server_round_synchronizations(),
    )


def test_assembled_server_holds_initial_model_and_empty_records(valid_run_settings_mapping):
    participants, _ = prepare_run_participants(valid_run_settings_mapping)
    run_server = participants.server_operations
    assert type(run_server) is FedsdaRunServer
    owners = run_server.owners
    assert type(owners) is FedsdaRunServerOwners
    assert type(owners.global_model_repository) is GlobalModelRepository
    assert type(owners.communication_volume_record_store) is CommunicationVolumeRecordStore
    assert type(owners.cross_evaluation_record_store) is CrossEvaluationRecordStore
    assert type(owners.model_clustering_record_store) is ModelClusteringRecordStore
    # グローバルモデルは、初期モデル（ID 0）だけ。値と統計は、clientの初期モデルと同じ。
    assert owners.global_model_repository.global_model_ids == (0,)
    global_parameters = owners.global_model_repository.get_global_model_parameters(model_id=0)
    for run_client in participants.client_operations:
        held_state = (
            run_client.owners.held_model_training_state_registry.get_held_model_training_state(
                model_id=0
            )
        )
        held_parameters = snapshot_classifier_parameters(classifier=held_state.classifier)
        assert list(held_parameters) == list(global_parameters)
        assert all(
            torch.equal(held_parameters[parameter_name], parameter_values)
            for parameter_name, parameter_values in global_parameters.items()
        )
        assert run_client.owners.loss_statistics_store.get_model_loss_statistics(
            model_id=0
        ) == owners.global_model_repository.get_global_model_loss_statistics(model_id=0)
    volume = owners.communication_volume_record_store.get_state_snapshot()
    assert volume == CommunicationVolumeRecordStore().get_state_snapshot()
    assert snapshot_server_state(run_server)["synchronizations"] == ()
    with pytest.raises(AttributeError):
        owners.global_model_repository = None


def test_server_counts_client_state_reports_before_synchronization(valid_run_settings_mapping):
    participants, _ = prepare_run_participants(valid_run_settings_mapping)
    run_server = participants.server_operations
    state_snapshot = snapshot_server_state(run_server)
    for invalid_round_index in (None, True, 1.0, -1):
        with pytest.raises((TypeError, ValueError)):
            run_server.record_client_states_before_synchronization(round_index=invalid_round_index)
        assert snapshot_server_state(run_server) == state_snapshot
    for report_count in (1, 2):
        run_server.record_client_states_before_synchronization(round_index=report_count - 1)
        volume = run_server.owners.communication_volume_record_store.get_state_snapshot()
        assert volume.uploaded_message_count == report_count * CLIENT_COUNT
        # ほかの通信量と、ほかの状態は変わらない。
        assert replace(volume, uploaded_message_count=0) == state_snapshot["communication_volume"]
        assert (
            snapshot_server_state(run_server)
            | dict(communication_volume=state_snapshot["communication_volume"])
            == state_snapshot
        )


def test_server_synchronizes_models_through_round_synchronization(
    monkeypatch, valid_run_settings_mapping
):
    """同期は、1ラウンドの同期の関数へ、自分のowner・基準・乱数生成器と、渡されたラウンドとフラグを渡す。"""
    participants, run_random_sources = prepare_run_participants(valid_run_settings_mapping)
    run_server = participants.server_operations
    owners = run_server.owners
    real_synchronization = run_server_module.synchronize_models_in_server_round
    received_arguments = []

    def observe_synchronization(**synchronization_arguments):
        received_arguments.append(synchronization_arguments)
        return real_synchronization(**synchronization_arguments)

    monkeypatch.setattr(
        run_server_module, "synchronize_models_in_server_round", observe_synchronization
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        for round_index, new_model_registration_available in enumerate((False, True)):
            assert (
                run_server.synchronize_models(
                    round_index=round_index,
                    new_model_registration_available=new_model_registration_available,
                )
                is None
            )
    assert [
        (arguments["round_index"], arguments["model_clustering_enabled"])
        for arguments in received_arguments
    ] == [(0, False), (1, True)]
    for arguments in received_arguments:
        assert arguments["run_clients"] == participants.client_operations
        assert arguments["global_model_repository"] is owners.global_model_repository
        assert (
            arguments["communication_volume_record_store"]
            is owners.communication_volume_record_store
        )
        assert arguments["cross_evaluation_record_store"] is owners.cross_evaluation_record_store
        assert arguments["model_clustering_record_store"] is owners.model_clustering_record_store
        assert arguments["model_clustering_criteria"] == MODEL_CLUSTERING_CRITERIA
        assert arguments["maximum_evaluating_client_count_per_model"] == 3
        # サーバは、clientと同じ、runの乱数生成器を使う。
        assert arguments["python_random_generator"] is run_random_sources.python_random_generator
    synchronizations = run_server.snapshot_server_round_synchronizations()
    assert len(synchronizations) == 2
    assert all(
        type(synchronization) is ServerRoundSynchronization for synchronization in synchronizations
    )
    # モデルが1つなので、クラスタリングは行われない。集約と配布は、毎ラウンド行われる。
    assert all(
        synchronization.client_model_aggregation.aggregated_global_model_ids == (0,)
        and synchronization.model_consolidation is None
        and len(synchronization.distribution_applications) == CLIENT_COUNT
        for synchronization in synchronizations
    )
    volume = owners.communication_volume_record_store.get_state_snapshot()
    # 標本をまだ処理していないclientは、学習データがなく、集約へ参加しない（上りの転送はない）。配布は全clientへ行う。
    assert volume.uploaded_model_count == 0
    assert volume.downloaded_model_count == 2 * CLIENT_COUNT
    # 不正な引数は、同期の関数が、何も変える前に拒否する。保持している結果は増えない。
    state_snapshot = snapshot_server_state(run_server)
    for invalid_arguments in (
        dict(round_index=-1, new_model_registration_available=False),
        dict(round_index=True, new_model_registration_available=False),
        dict(round_index=2, new_model_registration_available=None),
        dict(round_index=2, new_model_registration_available=1),
    ):
        with pytest.raises((TypeError, ValueError)):
            run_server.synchronize_models(**invalid_arguments)
        assert snapshot_server_state(run_server) == state_snapshot


def test_server_finalization_changes_nothing(valid_run_settings_mapping):
    participants, _ = prepare_run_participants(valid_run_settings_mapping)
    run_server = participants.server_operations
    state_snapshot = snapshot_server_state(run_server)
    for completed_round_count in (0, 4):
        assert (
            run_server.finalize_started_communications(completed_round_count=completed_round_count)
            is None
        )
        assert snapshot_server_state(run_server) == state_snapshot
    for invalid_round_count in (None, False, 4.0, -1):
        with pytest.raises((TypeError, ValueError)):
            run_server.finalize_started_communications(completed_round_count=invalid_round_count)
        assert snapshot_server_state(run_server) == state_snapshot


def test_server_assembly_rejects_invalid_arguments(valid_run_settings_mapping):
    participants, run_random_sources = prepare_run_participants(valid_run_settings_mapping)
    run_clients = participants.client_operations
    initial_classifier = (
        run_clients[0]
        .owners.held_model_training_state_registry.get_held_model_training_state(model_id=0)
        .classifier
    )
    assembly_arguments = dict(
        run_clients=run_clients,
        initial_model_id=0,
        initial_parameter_snapshot=snapshot_classifier_parameters(classifier=initial_classifier),
        initial_loss_statistics=run_clients[
            0
        ].owners.loss_statistics_store.get_model_loss_statistics(model_id=0),
        model_clustering_criteria=MODEL_CLUSTERING_CRITERIA,
        maximum_evaluating_client_count_per_model=3,
        python_random_generator=run_random_sources.python_random_generator,
    )
    python_random_state = run_random_sources.python_random_generator.getstate()
    mutated_criteria = replace(MODEL_CLUSTERING_CRITERIA)
    object.__setattr__(mutated_criteria, "minimum_pair_evaluation_sample_count", 0)
    for invalid_arguments in (
        dict(run_clients=list(run_clients)),
        dict(run_clients=(*run_clients, None)),
        dict(run_clients=(*run_clients[:2], make_subclass_copy(run_clients[2]))),
        dict(run_clients=(*run_clients, run_clients[0])),
        dict(initial_model_id=-1),
        dict(initial_model_id=True),
        dict(initial_parameter_snapshot={}),
        dict(initial_parameter_snapshot=None),
        dict(initial_loss_statistics=None),
        dict(model_clustering_criteria=None),
        dict(model_clustering_criteria=make_subclass_copy(MODEL_CLUSTERING_CRITERIA)),
        dict(model_clustering_criteria=mutated_criteria),
        dict(maximum_evaluating_client_count_per_model=0),
        dict(maximum_evaluating_client_count_per_model=3.0),
        dict(maximum_evaluating_client_count_per_model=True),
        dict(python_random_generator=random),
        dict(python_random_generator=type("RandomSubclass", (random.Random,), {})(3)),
    ):
        with pytest.raises((TypeError, ValueError)):
            assemble_fedsda_run_server(**assembly_arguments | invalid_arguments)
    # 組立ては、乱数を使わない。組み立てたサーバは、渡したclientの列を、そのまま使う。
    run_server = assemble_fedsda_run_server(**assembly_arguments)
    assert run_random_sources.python_random_generator.getstate() == python_random_state
    assert run_server.owners.global_model_repository.global_model_ids == (0,)
    assert run_server.owners is not participants.server_operations.owners
    # clientのいないサーバも組み立てられる（状態の報告は0件）。
    empty_server = assemble_fedsda_run_server(**assembly_arguments | dict(run_clients=()))
    empty_server.record_client_states_before_synchronization(round_index=0)
    assert (
        empty_server.owners.communication_volume_record_store.get_state_snapshot().uploaded_message_count
        == 0
    )
