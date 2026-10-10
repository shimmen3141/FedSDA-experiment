"""サーバの新規モデルの登録と集約を、実旧のサーバと、実__init__で作った実旧のclient複数に対して、ラウンドごとに照合する。"""

import random
from copy import deepcopy

import numpy as np
import pytest
import torch
from test_adopted_candidate_initial_local_registration import convert_legacy_parameter_name
from test_fedsda_run_client import (
    MAXIMUM_TOLERATED_MEAN_LOSS_INCREASE,
    assert_run_client_matches_legacy,
    assert_run_client_state_unchanged,
    build_initial_model_from_legacy,
    convert_legacy_loss_statistics,
    make_concept_stream,
    make_run_client_settings,
    run_in_both,
    set_legacy_configuration,
    snapshot_run_client_state,
)
from test_held_candidate_validation_progress import make_subclass_copy
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

import federated_learning_experiments.runtime.server_model_registration_and_aggregation as server_round_module
from federated_drift_experiment import experiment
from federated_drift_experiment.clients.shared_backbone import (
    ResidualAdapterRestartingSoftRoutingFedSDAClient,
)
from federated_drift_experiment.models import ResidualAdapterMLP
from federated_drift_experiment.servers import SharedBackboneFedSDANoCachedServer
from federated_learning_experiments.evaluation.communication_volume_record_store import (
    CommunicationVolumeRecordStore,
    CommunicationVolumeSnapshot,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.methods.fedsda.model_registration.global_model_repository import (
    GlobalModelRegistrationRecord,
    GlobalModelRepository,
)
from federated_learning_experiments.runtime.fedsda_run_client import assemble_fedsda_run_client
from federated_learning_experiments.runtime.server_model_registration_and_aggregation import (
    ClientModelAggregation,
    RegisteredClientModel,
    aggregate_client_models_into_global_models,
    register_ready_client_models,
)

CLIENT_COUNT = 3
ROUND_SAMPLE_COUNT = 10


def build_server_round_oracle(
    *,
    monkeypatch,
    valid_run_settings_mapping,
    class_count,
    client_count=CLIENT_COUNT,
    **condition_overrides,
):
    """実旧の事前学習・サーバ・client複数と、同じ初期モデルから作った新のclient複数・サーバのownerを作る。

    戻り値: 辞書（新のclientのtuple、グローバルモデルのowner、通信量のowner、共有する乱数生成器、
    実旧のサーバ、実旧のclientのlist）。呼出し側の乱数は進めない。
    """
    set_legacy_configuration(
        monkeypatch, class_count=class_count, update_interval=2, **condition_overrides
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
            # 旧の準備と同じ順: サーバへ初期モデルと統計を登録し、clientを作って登録する。
            legacy_server = SharedBackboneFedSDANoCachedServer(
                distance_threshold=MAXIMUM_TOLERATED_MEAN_LOSS_INCREASE, verbose=False
            )
            legacy_server.register_model_params(0, legacy_initial_model.get_params())
            legacy_server.register_model_stats(0, legacy_initial_statistics)
            legacy_clients = []
            for client_id in range(client_count):
                legacy_client = ResidualAdapterRestartingSoftRoutingFedSDAClient(
                    client_id=client_id,
                    initial_models={0: legacy_initial_model},
                    initial_stats={0: legacy_initial_statistics},
                    distance_threshold=MAXIMUM_TOLERATED_MEAN_LOSS_INCREASE,
                    verbose=False,
                )
                legacy_server.register_client(legacy_client)
                legacy_clients.append(legacy_client)
    finally:
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)
    run_client_settings = make_run_client_settings(
        valid_run_settings_mapping, update_interval=2, **condition_overrides
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
    initial_loss_statistics = convert_legacy_loss_statistics(legacy_initial_statistics)
    # 旧は、全clientがmodule全体の乱数を共有する。新は、1つの乱数生成器を全clientへ渡す。
    python_random_generator = random.Random(29)
    run_clients = tuple(
        assemble_fedsda_run_client(
            client_id=client_id,
            initial_model_id=0,
            initial_classifier=initial_classifier,
            initial_concept_specific_parameter_optimizer_state=initial_concept_specific_parameter_optimizer_state,
            initial_shared_parameter_optimizer_state=initial_shared_parameter_optimizer_state,
            initial_loss_statistics=initial_loss_statistics,
            run_client_settings=run_client_settings,
            python_random_generator=python_random_generator,
        )
        for client_id in range(client_count)
    )
    return dict(
        run_clients=run_clients,
        global_model_repository=GlobalModelRepository(
            initial_model_id=0,
            initial_parameter_snapshot=snapshot_classifier_parameters(
                classifier=initial_classifier
            ),
            initial_loss_statistics=initial_loss_statistics,
        ),
        communication_volume_record_store=CommunicationVolumeRecordStore(),
        python_random_generator=python_random_generator,
        legacy_server=legacy_server,
        legacy_clients=legacy_clients,
    )


def assert_server_state_matches_legacy(
    *, global_model_repository, communication_volume_record_store, legacy_server
):
    """グローバルモデル（IDの順、全パラメータ、損失統計）、次の正式ID、来歴、通信量を、実旧のサーバと照合する。"""
    assert global_model_repository.global_model_ids == tuple(legacy_server.global_models)
    for model_id, legacy_parameters in legacy_server.global_models.items():
        parameter_snapshot = global_model_repository.get_global_model_parameters(model_id=model_id)
        assert list(parameter_snapshot) == [
            convert_legacy_parameter_name(parameter_name) for parameter_name in legacy_parameters
        ]
        for parameter_name, legacy_parameter_values in legacy_parameters.items():
            assert torch.equal(
                parameter_snapshot[convert_legacy_parameter_name(parameter_name)],
                legacy_parameter_values,
            ), (model_id, parameter_name)
    # 旧の統計は、未設定のIDを読むと既定値を作る辞書。持っているIDだけを比べる。
    assert set(legacy_server.global_stats) == {
        model_id
        for model_id in range(global_model_repository.next_global_model_id)
        if global_model_repository.get_global_model_loss_statistics(model_id=model_id) is not None
    }
    for model_id, legacy_statistics in legacy_server.global_stats.items():
        assert global_model_repository.get_global_model_loss_statistics(
            model_id=model_id
        ) == convert_legacy_loss_statistics({"class_stats": {}} | legacy_statistics), model_id
    assert global_model_repository.next_global_model_id == legacy_server.next_model_id
    assert global_model_repository.snapshot_model_registration_records() == tuple(
        GlobalModelRegistrationRecord(
            model_id=legacy_registration.model_id,
            registered_round_index=(
                None if legacy_registration.round_index == -1 else legacy_registration.round_index
            ),
            registering_client_id=(
                None if legacy_registration.client_id == -1 else legacy_registration.client_id
            ),
        )
        for legacy_registration in legacy_server.model_lineage.registrations
    )
    assert communication_volume_record_store.get_state_snapshot() == CommunicationVolumeSnapshot(
        uploaded_model_count=legacy_server.comm_models_up,
        downloaded_model_count=legacy_server.comm_models_down,
        uploaded_message_count=legacy_server.comm_messages_up,
        downloaded_message_count=legacy_server.comm_messages_down,
        uploaded_parameter_value_count=legacy_server.comm_parameter_values_up,
        downloaded_parameter_value_count=legacy_server.comm_parameter_values_down,
        uploaded_byte_count=legacy_server.comm_bytes_up,
        downloaded_byte_count=legacy_server.comm_bytes_down,
    )


def assert_all_states_match_legacy(server_round_oracle):
    assert_server_state_matches_legacy(
        global_model_repository=server_round_oracle["global_model_repository"],
        communication_volume_record_store=server_round_oracle["communication_volume_record_store"],
        legacy_server=server_round_oracle["legacy_server"],
    )
    for run_client, legacy_client in zip(
        server_round_oracle["run_clients"], server_round_oracle["legacy_clients"], strict=True
    ):
        assert_run_client_matches_legacy(run_client=run_client, legacy_client=legacy_client)


def process_round_samples_in_both(*, server_round_oracle, client_streams, round_index):
    """1ラウンドぶんの標本を、標本ごとにclient順で、両実装へ処理させ、ラウンド境界の保留中の学習を行う。"""
    run_clients = server_round_oracle["run_clients"]
    legacy_clients = server_round_oracle["legacy_clients"]
    python_random_generator = server_round_oracle["python_random_generator"]
    for sample_index in range(
        round_index * ROUND_SAMPLE_COUNT, (round_index + 1) * ROUND_SAMPLE_COUNT
    ):
        for run_client, legacy_client, client_stream in zip(
            run_clients, legacy_clients, client_streams, strict=True
        ):
            observed_sample, concept_id = client_stream[sample_index]
            run_in_both(
                run_client=run_client,
                legacy_client=legacy_client,
                python_random_generator=python_random_generator,
                legacy_operation=lambda legacy_client=legacy_client, observed_sample=observed_sample, concept_id=concept_id: (
                    legacy_client.process_one_step(
                        torch.tensor(observed_sample.feature_values, dtype=torch.float32),
                        torch.tensor([float(observed_sample.class_label)]),
                        concept_id,
                    )
                ),
                operation=lambda run_client=run_client, observed_sample=observed_sample, sample_index=sample_index, concept_id=concept_id: (
                    run_client.process_observed_sample(
                        observed_sample=observed_sample,
                        sample_index=sample_index,
                        evaluation_concept_id=concept_id,
                    )
                ),
            )
    for run_client, legacy_client in zip(run_clients, legacy_clients, strict=True):
        run_in_both(
            run_client=run_client,
            legacy_client=legacy_client,
            python_random_generator=python_random_generator,
            legacy_operation=legacy_client.flush_pending_updates,
            operation=lambda run_client=run_client: run_client.flush_pending_local_updates(
                round_index=round_index
            ),
        )


def get_random_states(python_random_generator):
    return (
        python_random_generator.getstate(),
        random.getstate(),
        torch.get_rng_state().clone(),
    )


def assert_random_states_equal(random_states, expected_random_states):
    assert random_states[0] == expected_random_states[0]
    assert random_states[1] == expected_random_states[1]
    assert torch.equal(random_states[2], expected_random_states[2])


ROUND_COUNT = 13
# 条件ごとの、観測した経路。
OBSERVED_COVERAGE_BY_CONDITION = {}
# (クラス数, 概念の区間長, 標本列のseed)。(2, 9, 7)と(2, 10, 50)は、採用の後に現行モデルが非負のIDへ戻ってから
# 送信できるようになる（採番だけが行われる）条件。
SERVER_ROUND_CONDITIONS = [
    (2, 13, 5),
    (2, 22, 11),
    (2, 9, 7),
    (2, 10, 50),
    (4, 22, 11),
    (4, 30, 3),
]


@pytest.mark.parametrize("class_count,concept_block_length,stream_seed", SERVER_ROUND_CONDITIONS)
def test_registration_and_aggregation_match_real_legacy_server_for_each_round(
    class_count, concept_block_length, stream_seed, monkeypatch, valid_run_settings_mapping
):
    # 標本列は、旧の設定を差し替える前に作る。clientごとに別の標本列。
    client_streams = [
        make_concept_stream(
            sample_count=ROUND_COUNT * ROUND_SAMPLE_COUNT,
            concept_block_length=concept_block_length,
            stream_seed=stream_seed + client_id,
        )
        for client_id in range(CLIENT_COUNT)
    ]
    server_round_oracle = build_server_round_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=class_count,
    )
    run_clients = server_round_oracle["run_clients"]
    legacy_clients = server_round_oracle["legacy_clients"]
    legacy_server = server_round_oracle["legacy_server"]
    global_model_repository = server_round_oracle["global_model_repository"]
    communication_volume_record_store = server_round_oracle["communication_volume_record_store"]
    python_random_generator = server_round_oracle["python_random_generator"]
    assert_all_states_match_legacy(server_round_oracle)
    observed_paths = set()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        for round_index in range(ROUND_COUNT):
            process_round_samples_in_both(
                server_round_oracle=server_round_oracle,
                client_streams=client_streams,
                round_index=round_index,
            )
            # 状態の報告（clientの数だけの軽量メッセージ）。
            legacy_server.record_client_state_summaries()
            communication_volume_record_store.record_messages(
                transfer_direction="upload", message_count=len(run_clients)
            )
            random_states = get_random_states(python_random_generator)
            # 登録。
            ready_client_ids = [
                run_client.client_id
                for run_client in run_clients
                if run_client.has_model_ready_for_server_registration()
            ]
            current_model_ids_before = [
                run_client.owners.current_training_model_assignment.current_training_model_id
                for run_client in run_clients
            ]
            next_global_model_id = global_model_repository.next_global_model_id
            legacy_server._register_new_models(round_index)
            registered_client_models = register_ready_client_models(
                run_clients=run_clients,
                global_model_repository=global_model_repository,
                round_index=round_index,
            )
            assert all(
                type(registered_client_model) is RegisteredClientModel
                for registered_client_model in registered_client_models
            )
            assert [
                registered_client_model.client_id
                for registered_client_model in registered_client_models
            ] == ready_client_ids
            assert [
                registered_client_model.registered_global_model_id
                for registered_client_model in registered_client_models
            ] == list(range(next_global_model_id, next_global_model_id + len(ready_client_ids)))
            for registered_client_model in registered_client_models:
                client_id = registered_client_model.client_id
                current_model_id = run_clients[
                    client_id
                ].owners.current_training_model_assignment.current_training_model_id
                if current_model_ids_before[client_id] < 0:
                    observed_paths.add("temporary_model_registered")
                    assert registered_client_model.training_assignment_change is not None
                    assert current_model_id == registered_client_model.registered_global_model_id
                else:
                    # 採番だけが行われ、clientの学習帰属は変わらない（LEGACY-016）。
                    observed_paths.add("identifier_allocated_without_model")
                    assert registered_client_model.training_assignment_change is None
                    assert current_model_id == current_model_ids_before[client_id]
                assert not run_clients[client_id].has_model_ready_for_server_registration()
            assert_all_states_match_legacy(server_round_oracle)
            # 集約。
            legacy_active_model_ids = sorted(
                {
                    model_id
                    for legacy_client in legacy_server.clients
                    for model_id in legacy_client.models
                    if model_id >= 0
                }
            )
            legacy_aggregation_weights = legacy_server.update_global_models(legacy_active_model_ids)
            client_state_snapshots = [
                snapshot_run_client_state(
                    run_client=run_client, python_random_generator=python_random_generator
                )
                for run_client in run_clients
            ]
            uploaded_parameter_value_count_before = communication_volume_record_store.get_state_snapshot().uploaded_parameter_value_count
            client_model_aggregation = aggregate_client_models_into_global_models(
                run_clients=run_clients,
                global_model_repository=global_model_repository,
                communication_volume_record_store=communication_volume_record_store,
            )
            assert type(client_model_aggregation) is ClientModelAggregation
            # 重み付きの和へ足したパラメータの値の数は、この集約で上りとして数えた、パラメータの値の数。
            assert client_model_aggregation.weighted_parameter_multiply_accumulate_count == (
                communication_volume_record_store.get_state_snapshot().uploaded_parameter_value_count
                - uploaded_parameter_value_count_before
            )
            assert (
                type(client_model_aggregation.weighted_parameter_multiply_accumulate_count) is int
            )
            assert client_model_aggregation.aggregated_global_model_ids == tuple(
                legacy_active_model_ids
            )
            assert (
                client_model_aggregation.aggregated_training_sample_counts_by_model_id
                == legacy_aggregation_weights
            )
            assert_all_states_match_legacy(server_round_oracle)
            # 登録と集約は、乱数を使わない。
            assert_random_states_equal(get_random_states(python_random_generator), random_states)
            holders_by_model_id = {
                model_id: sum(model_id in legacy_client.models for legacy_client in legacy_clients)
                for model_id in legacy_active_model_ids
            }
            if any(holder_count > 1 for holder_count in holders_by_model_id.values()):
                observed_paths.add("model_held_by_multiple_clients")
            if len(legacy_active_model_ids) > 1:
                observed_paths.add("multiple_global_models")
            # 集約は、clientの状態を変えない。
            for run_client, client_state_snapshot in zip(
                run_clients, client_state_snapshots, strict=True
            ):
                assert_run_client_state_unchanged(
                    state_snapshot=client_state_snapshot,
                    run_client=run_client,
                    python_random_generator=python_random_generator,
                )
            # 同期の後の、送信待ちの進行。
            for run_client, legacy_client in zip(run_clients, legacy_clients, strict=True):
                legacy_client.promote_pending_to_ready()
                run_client.advance_new_model_upload_wait_after_synchronization(
                    round_index=round_index
                )
            assert_all_states_match_legacy(server_round_oracle)
    OBSERVED_COVERAGE_BY_CONDITION[(class_count, concept_block_length, stream_seed)] = (
        observed_paths
    )


def test_server_round_conditions_cover_required_paths():
    """上の対照が、要求の経路をすべて通っていること（全条件を実行したときだけ確かめる）。"""
    if len(OBSERVED_COVERAGE_BY_CONDITION) < len(SERVER_ROUND_CONDITIONS):
        pytest.skip("対照の全条件を実行したときだけ確かめる")
    observed_paths = set().union(*OBSERVED_COVERAGE_BY_CONDITION.values())
    assert observed_paths >= {
        "temporary_model_registered",
        "identifier_allocated_without_model",
        "model_held_by_multiple_clients",
        "multiple_global_models",
    }, observed_paths


def build_processed_server_round_oracle(*, monkeypatch, valid_run_settings_mapping, round_count=8):
    """標本を何ラウンドか処理して、登録できるモデルを持つclientがいる状態（登録と集約は、まだ行っていない）。

    この標本列では、8ラウンドめの後で、client 2が送信できるモデルを持つ（下で確かめる）。
    途中のラウンドでは、送信待ちの進行だけを行う。
    """
    client_streams = [
        make_concept_stream(
            sample_count=round_count * ROUND_SAMPLE_COUNT,
            concept_block_length=13,
            stream_seed=5 + client_id,
        )
        for client_id in range(CLIENT_COUNT)
    ]
    server_round_oracle = build_server_round_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=2,
    )
    for round_index in range(round_count):
        process_round_samples_in_both(
            server_round_oracle=server_round_oracle,
            client_streams=client_streams,
            round_index=round_index,
        )
        if round_index < round_count - 1:
            for run_client, legacy_client in zip(
                server_round_oracle["run_clients"],
                server_round_oracle["legacy_clients"],
                strict=True,
            ):
                legacy_client.promote_pending_to_ready()
                run_client.advance_new_model_upload_wait_after_synchronization(
                    round_index=round_index
                )
    return server_round_oracle


def snapshot_server_and_client_states(server_round_oracle):
    global_model_repository = server_round_oracle["global_model_repository"]
    return dict(
        next_global_model_id=global_model_repository.next_global_model_id,
        global_model_ids=global_model_repository.global_model_ids,
        global_parameters=tuple(
            deepcopy(global_model_repository.get_global_model_parameters(model_id=model_id))
            for model_id in global_model_repository.global_model_ids
        ),
        global_loss_statistics=tuple(
            global_model_repository.get_global_model_loss_statistics(model_id=model_id)
            for model_id in global_model_repository.global_model_ids
        ),
        registration_records=global_model_repository.snapshot_model_registration_records(),
        communication_volume=server_round_oracle[
            "communication_volume_record_store"
        ].get_state_snapshot(),
        clients=[
            snapshot_run_client_state(
                run_client=run_client,
                python_random_generator=server_round_oracle["python_random_generator"],
            )
            for run_client in server_round_oracle["run_clients"]
        ],
    )


def assert_server_and_client_states_unchanged(*, state_snapshot, server_round_oracle):
    current_snapshot = snapshot_server_and_client_states(server_round_oracle)
    for state_name in (
        "next_global_model_id",
        "global_model_ids",
        "global_loss_statistics",
        "registration_records",
        "communication_volume",
    ):
        assert current_snapshot[state_name] == state_snapshot[state_name], state_name
    for parameters, current_parameters in zip(
        state_snapshot["global_parameters"], current_snapshot["global_parameters"], strict=True
    ):
        assert list(parameters) == list(current_parameters)
        for parameter_name, parameter_values in parameters.items():
            assert torch.equal(current_parameters[parameter_name], parameter_values)
    for run_client, client_state_snapshot in zip(
        server_round_oracle["run_clients"], state_snapshot["clients"], strict=True
    ):
        assert_run_client_state_unchanged(
            state_snapshot=client_state_snapshot,
            run_client=run_client,
            python_random_generator=server_round_oracle["python_random_generator"],
        )


def get_registration_arguments(server_round_oracle):
    return dict(
        run_clients=server_round_oracle["run_clients"],
        global_model_repository=server_round_oracle["global_model_repository"],
        round_index=7,
    )


def get_aggregation_arguments(server_round_oracle):
    return dict(
        run_clients=server_round_oracle["run_clients"],
        global_model_repository=server_round_oracle["global_model_repository"],
        communication_volume_record_store=server_round_oracle["communication_volume_record_store"],
    )


# 条件名 -> (正常な引数から、差し替える引数を作る操作, 期待する例外)。登録と集約に共通の引数。
INVALID_COMMON_ARGUMENT_CASES = {
    "run_clients_list": (
        lambda arguments: dict(run_clients=list(arguments["run_clients"])),
        TypeError,
    ),
    "run_clients_holding_other_type": (
        lambda arguments: dict(run_clients=(*arguments["run_clients"][:2], object())),
        TypeError,
    ),
    "run_clients_holding_subclass": (
        lambda arguments: dict(
            run_clients=(
                *arguments["run_clients"][:2],
                make_subclass_copy(arguments["run_clients"][2]),
            )
        ),
        TypeError,
    ),
    "run_clients_with_duplicate_client": (
        lambda arguments: dict(
            run_clients=(*arguments["run_clients"], arguments["run_clients"][0])
        ),
        ValueError,
    ),
    "repository_other_type": (lambda arguments: dict(global_model_repository=object()), TypeError),
    "repository_subclass": (
        lambda arguments: dict(
            global_model_repository=make_subclass_copy(arguments["global_model_repository"])
        ),
        TypeError,
    ),
}
INVALID_REGISTRATION_ARGUMENT_CASES = INVALID_COMMON_ARGUMENT_CASES | {
    "round_index_bool": (lambda arguments: dict(round_index=True), TypeError),
    "round_index_float": (lambda arguments: dict(round_index=7.0), TypeError),
    "round_index_negative": (lambda arguments: dict(round_index=-1), ValueError),
}
INVALID_AGGREGATION_ARGUMENT_CASES = INVALID_COMMON_ARGUMENT_CASES | {
    "communication_store_other_type": (
        lambda arguments: dict(communication_volume_record_store=None),
        TypeError,
    ),
    "communication_store_subclass": (
        lambda arguments: dict(
            communication_volume_record_store=make_subclass_copy(
                arguments["communication_volume_record_store"]
            )
        ),
        TypeError,
    ),
}


@pytest.mark.parametrize("invalid_case_name", INVALID_REGISTRATION_ARGUMENT_CASES)
def test_registration_rejects_invalid_arguments_before_any_update(
    invalid_case_name, monkeypatch, valid_run_settings_mapping
):
    make_invalid_arguments, expected_exception = INVALID_REGISTRATION_ARGUMENT_CASES[
        invalid_case_name
    ]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        server_round_oracle = build_processed_server_round_oracle(
            monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
        )
        registration_arguments = get_registration_arguments(server_round_oracle)
        assert any(
            run_client.has_model_ready_for_server_registration()
            for run_client in server_round_oracle["run_clients"]
        )
        state_snapshot = snapshot_server_and_client_states(server_round_oracle)
        with pytest.raises(expected_exception):
            register_ready_client_models(
                **registration_arguments | make_invalid_arguments(registration_arguments)
            )
        assert_server_and_client_states_unchanged(
            state_snapshot=state_snapshot, server_round_oracle=server_round_oracle
        )
        # 拒否の後に、正しい引数で登録できる。
        assert register_ready_client_models(**registration_arguments)


@pytest.mark.parametrize("invalid_case_name", INVALID_AGGREGATION_ARGUMENT_CASES)
def test_aggregation_rejects_invalid_arguments_before_any_update(
    invalid_case_name, monkeypatch, valid_run_settings_mapping
):
    make_invalid_arguments, expected_exception = INVALID_AGGREGATION_ARGUMENT_CASES[
        invalid_case_name
    ]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        server_round_oracle = build_processed_server_round_oracle(
            monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
        )
        aggregation_arguments = get_aggregation_arguments(server_round_oracle)
        state_snapshot = snapshot_server_and_client_states(server_round_oracle)
        with pytest.raises(expected_exception):
            aggregate_client_models_into_global_models(
                **aggregation_arguments | make_invalid_arguments(aggregation_arguments)
            )
        assert_server_and_client_states_unchanged(
            state_snapshot=state_snapshot, server_round_oracle=server_round_oracle
        )
        aggregate_client_models_into_global_models(**aggregation_arguments)


def test_registration_changes_neither_communication_volume_nor_global_model_values(
    monkeypatch, valid_run_settings_mapping
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        server_round_oracle = build_processed_server_round_oracle(
            monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
        )
        state_snapshot = snapshot_server_and_client_states(server_round_oracle)
        registered_client_models = register_ready_client_models(
            **get_registration_arguments(server_round_oracle)
        )
    assert registered_client_models
    current_snapshot = snapshot_server_and_client_states(server_round_oracle)
    for unchanged_state_name in (
        "global_model_ids",
        "global_loss_statistics",
        "communication_volume",
    ):
        assert current_snapshot[unchanged_state_name] == state_snapshot[unchanged_state_name]
    for parameters, current_parameters in zip(
        state_snapshot["global_parameters"], current_snapshot["global_parameters"], strict=True
    ):
        for parameter_name, parameter_values in parameters.items():
            assert torch.equal(current_parameters[parameter_name], parameter_values)
    # 採番と来歴は進む。
    assert current_snapshot["next_global_model_id"] == state_snapshot["next_global_model_id"] + len(
        registered_client_models
    )
    assert len(current_snapshot["registration_records"]) == len(
        state_snapshot["registration_records"]
    ) + len(registered_client_models)
    assert current_snapshot["registration_records"][-1] == GlobalModelRegistrationRecord(
        model_id=registered_client_models[-1].registered_global_model_id,
        registered_round_index=7,
        registering_client_id=registered_client_models[-1].client_id,
    )


def test_registration_stops_at_failed_client_without_rollback(
    monkeypatch, valid_run_settings_mapping
):
    """あるclientへの確認が失敗したら、後のclientへ進まない。失敗したclientへの採番と来歴は残る。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        server_round_oracle = build_processed_server_round_oracle(
            monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
        )
        run_clients = server_round_oracle["run_clients"]
        global_model_repository = server_round_oracle["global_model_repository"]
        # 全clientを「送信できる」と答えさせ、2つめのclientへの確認で失敗させる。
        monkeypatch.setattr(
            type(run_clients[0]), "has_model_ready_for_server_registration", lambda self: True
        )
        confirmed_registry_ids = []
        real_confirmation = server_round_module.confirm_held_model_registration

        def fail_at_second_confirmation(**confirmation_arguments):
            confirmed_registry_ids.append(
                id(confirmation_arguments["held_model_training_state_registry"])
            )
            if len(confirmed_registry_ids) == 2:
                raise RuntimeError("injected failure")
            return real_confirmation(**confirmation_arguments)

        monkeypatch.setattr(
            server_round_module, "confirm_held_model_registration", fail_at_second_confirmation
        )
        next_global_model_id = global_model_repository.next_global_model_id
        with pytest.raises(RuntimeError, match="injected failure"):
            register_ready_client_models(**get_registration_arguments(server_round_oracle))
    assert confirmed_registry_ids == [
        id(run_client.owners.held_model_training_state_registry) for run_client in run_clients[:2]
    ]
    assert global_model_repository.next_global_model_id == next_global_model_id + 2
    assert [
        (registration_record.model_id, registration_record.registering_client_id)
        for registration_record in global_model_repository.snapshot_model_registration_records()[
            -2:
        ]
    ] == [(next_global_model_id, 0), (next_global_model_id + 1, 1)]


def test_aggregation_changes_nothing_when_calculation_fails(
    monkeypatch, valid_run_settings_mapping
):
    """計算の途中の失敗（最後のclientのパラメータの写しが失敗する）では、通信量もグローバルモデルも変わらない。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        server_round_oracle = build_processed_server_round_oracle(
            monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
        )
        last_client_classifiers = {
            id(held_model_training_state.classifier)
            for held_model_training_state in server_round_oracle["run_clients"][
                -1
            ].owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        }
        real_snapshot = server_round_module.snapshot_classifier_parameters
        snapshot_call_count = 0

        def fail_for_last_client(*, classifier):
            nonlocal snapshot_call_count
            snapshot_call_count += 1
            if id(classifier) in last_client_classifiers:
                raise RuntimeError("injected failure")
            return real_snapshot(classifier=classifier)

        state_snapshot = snapshot_server_and_client_states(server_round_oracle)
        with monkeypatch.context() as patch_context:
            patch_context.setattr(
                server_round_module, "snapshot_classifier_parameters", fail_for_last_client
            )
            with pytest.raises(RuntimeError, match="injected failure"):
                aggregate_client_models_into_global_models(
                    **get_aggregation_arguments(server_round_oracle)
                )
        # 先のclientの写しは済んでいた（計算は途中まで進んでいた）。
        assert snapshot_call_count > 2
        assert_server_and_client_states_unchanged(
            state_snapshot=state_snapshot, server_round_oracle=server_round_oracle
        )
        aggregate_client_models_into_global_models(**get_aggregation_arguments(server_round_oracle))
    assert (
        snapshot_server_and_client_states(server_round_oracle)["communication_volume"]
        != state_snapshot["communication_volume"]
    )


def test_aggregation_without_participating_clients_keeps_global_models(
    monkeypatch, valid_run_settings_mapping
):
    """どのclientも学習データを持たないとき（生成直後）は、グローバルモデルの値も通信量も変わらない。

    保有しているが学習データを持たないモデルを含む集約を、実旧と照合する（複数ラウンドの対照では、
    配布がないので、この状態にならない）。
    """
    server_round_oracle = build_server_round_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=2,
    )
    legacy_server = server_round_oracle["legacy_server"]
    state_snapshot = snapshot_server_and_client_states(server_round_oracle)
    legacy_aggregation_weights = legacy_server.update_global_models([0])
    client_model_aggregation = aggregate_client_models_into_global_models(
        **get_aggregation_arguments(server_round_oracle)
    )
    assert client_model_aggregation.aggregated_global_model_ids == (0,)
    assert client_model_aggregation.aggregated_training_sample_counts_by_model_id == {0: 0}
    assert legacy_aggregation_weights == {0: 0}
    assert_server_and_client_states_unchanged(
        state_snapshot=state_snapshot, server_round_oracle=server_round_oracle
    )
    assert_all_states_match_legacy(server_round_oracle)
