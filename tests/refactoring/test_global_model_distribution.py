"""グローバルモデルの配布と、clientでの受取りを、実旧のサーバと、実__init__で作った実旧のclient複数に対して照合する。"""

import random
from dataclasses import replace

import pytest
import torch
from test_adopted_candidate_initial_local_registration import convert_legacy_parameter_name
from test_fedsda_run_client import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    assert_run_client_state_unchanged,
    convert_legacy_loss_statistics,
    make_concept_stream,
    run_in_both,
    snapshot_run_client_state,
)
from test_held_candidate_validation_progress import make_subclass_copy
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping
from test_server_model_registration_and_aggregation import (
    CLIENT_COUNT,
    ROUND_SAMPLE_COUNT,
    assert_all_states_match_legacy,
    build_server_round_oracle,
    process_round_samples_in_both,
)

from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
)
from federated_learning_experiments.runtime.global_model_distribution_application import (
    GlobalModelDistributionApplication,
    apply_global_model_distribution,
)
from federated_learning_experiments.runtime.server_model_registration_and_aggregation import (
    aggregate_client_models_into_global_models,
    register_ready_client_models,
)


def convert_legacy_global_models(legacy_global_models):
    """実旧のグローバルモデルの辞書を、同じ順・同じ値の、新の（ID、パラメータ）の列へ写す。"""
    return tuple(
        (
            model_id,
            {
                convert_legacy_parameter_name(parameter_name): parameter_values.clone()
                for parameter_name, parameter_values in legacy_parameters.items()
            },
        )
        for model_id, legacy_parameters in legacy_global_models.items()
    )


def convert_legacy_global_statistics(legacy_global_statistics):
    """実旧のグローバルの統計の辞書を、同じ順・同じ値の、新の（ID、統計）の列へ写す。"""
    return tuple(
        (model_id, convert_legacy_loss_statistics({"class_stats": {}} | legacy_statistics))
        for model_id, legacy_statistics in legacy_global_statistics.items()
    )


def assert_all_states_match_legacy_after_distribution(server_round_oracle):
    """サーバと全clientの全状態に加えて、サーバによる付け替えの位置の列を、実旧と照合する。"""
    assert_all_states_match_legacy(server_round_oracle)
    for run_client, legacy_client in zip(
        server_round_oracle["run_clients"], server_round_oracle["legacy_clients"], strict=True
    ):
        assert (
            run_client.owners.adaptation_record_store.get_state_snapshot().server_remapped_sample_indices
            == tuple(legacy_client.mapping_change_positions)
        )


def apply_distribution_to_each_client_in_both(
    *, server_round_oracle, model_id_mapping, legacy_global_models, legacy_global_statistics
):
    """全clientへ、client順に、実旧の受取りと新の受取りを、同じ乱数の状態から行わせる。

    戻り値: 新の受取りの結果のlist。
    """
    distribution_applications = []
    for run_client, legacy_client in zip(
        server_round_oracle["run_clients"], server_round_oracle["legacy_clients"], strict=True
    ):
        distribution_application, _ = run_in_both(
            run_client=run_client,
            legacy_client=legacy_client,
            python_random_generator=server_round_oracle["python_random_generator"],
            legacy_operation=lambda legacy_client=legacy_client: legacy_client.apply_server_mapping(
                dict(model_id_mapping), legacy_global_models, legacy_global_statistics
            ),
            operation=lambda run_client=run_client: run_client.apply_global_model_distribution(
                model_id_mapping=dict(model_id_mapping),
                distributed_parameter_snapshots=convert_legacy_global_models(legacy_global_models),
                distributed_loss_statistics=convert_legacy_global_statistics(
                    legacy_global_statistics
                ),
            ),
        )
        assert type(distribution_application) is GlobalModelDistributionApplication
        distribution_applications.append(distribution_application)
    return distribution_applications


def distribute_to_each_client_in_both(server_round_oracle):
    """ID対応なしで、実旧のサーバが持つグローバルモデルと統計を、全clientへ受け取らせる（通信量は数えない）。"""
    legacy_server = server_round_oracle["legacy_server"]
    return apply_distribution_to_each_client_in_both(
        server_round_oracle=server_round_oracle,
        model_id_mapping={},
        legacy_global_models=legacy_server.global_models,
        legacy_global_statistics=legacy_server.global_stats,
    )


def run_synchronization_round_in_both(
    *, server_round_oracle, client_streams, round_index, distribute_in_both
):
    """1ラウンド: 標本処理→保留中の学習→状態の報告→登録→集約→配布→送信待ちの進行。配布の後に全状態を照合する。

    戻り値: 配布の前に各clientが保有していたモデルIDのlistと、配布の結果。
    """
    run_clients = server_round_oracle["run_clients"]
    legacy_clients = server_round_oracle["legacy_clients"]
    legacy_server = server_round_oracle["legacy_server"]
    process_round_samples_in_both(
        server_round_oracle=server_round_oracle,
        client_streams=client_streams,
        round_index=round_index,
    )
    legacy_server.record_client_state_summaries()
    server_round_oracle["communication_volume_record_store"].record_messages(
        transfer_direction="upload", message_count=len(run_clients)
    )
    legacy_server._register_new_models(round_index)
    register_ready_client_models(
        run_clients=run_clients,
        global_model_repository=server_round_oracle["global_model_repository"],
        round_index=round_index,
    )
    legacy_server.update_global_models(
        sorted(
            {
                model_id
                for legacy_client in legacy_clients
                for model_id in legacy_client.models
                if model_id >= 0
            }
        )
    )
    aggregate_client_models_into_global_models(
        run_clients=run_clients,
        global_model_repository=server_round_oracle["global_model_repository"],
        communication_volume_record_store=server_round_oracle["communication_volume_record_store"],
    )
    assert_all_states_match_legacy(server_round_oracle)
    held_model_ids_before_distribution = [
        list(legacy_client.models) for legacy_client in legacy_clients
    ]
    distribution_result = distribute_in_both(server_round_oracle)
    assert_all_states_match_legacy_after_distribution(server_round_oracle)
    for run_client, legacy_client in zip(run_clients, legacy_clients, strict=True):
        legacy_client.promote_pending_to_ready()
        run_client.advance_new_model_upload_wait_after_synchronization(round_index=round_index)
    assert_all_states_match_legacy_after_distribution(server_round_oracle)
    return held_model_ids_before_distribution, distribution_result


def make_client_streams(*, round_count, concept_block_length, stream_seed):
    """clientごとに別の標本列（旧の設定を差し替える前に作る）。"""
    return [
        make_concept_stream(
            sample_count=round_count * ROUND_SAMPLE_COUNT,
            concept_block_length=concept_block_length,
            stream_seed=stream_seed + client_id,
        )
        for client_id in range(CLIENT_COUNT)
    ]


# 学習率の2つの設定（作り直すモデル、候補とつなぎ直し）を違う値にした条件と、評価標本の上限を小さくした条件を含める。
DIFFERENT_LEARNING_RATES = dict(base_learning_rate=0.01, new_model_learning_rate=0.004)
SMALL_EVALUATION_SAMPLE_LIMIT = dict(stored_evaluation_sample_limit=4)


def exclude_remapped_models(legacy_values_by_model_id, model_id_mapping):
    return {
        model_id: legacy_value
        for model_id, legacy_value in legacy_values_by_model_id.items()
        if model_id not in model_id_mapping
    }


MAPPING_OBSERVED_COVERAGE_BY_CONDITION = {}
# (クラス数, 概念の区間長, 標本列のseed, ID対応を与える前のラウンド数, 条件の上書き, ID対応の作り方)。
# ID対応の作り方: "largest_into_smallest"は、最大の正式ID（3つ以上あれば、大きいほうから2つ）を最小の正式IDへ集める。
# "smallest_into_largest"は、最小の正式ID（つなぎ先だったモデル）を最大の正式IDへ集める。
MODEL_ID_MAPPING_CONDITIONS = [
    (2, 13, 5, 10, {}, "largest_into_smallest"),
    (2, 13, 5, 13, SMALL_EVALUATION_SAMPLE_LIMIT, "largest_into_smallest"),
    (2, 22, 11, 10, DIFFERENT_LEARNING_RATES, "largest_into_smallest"),
    (2, 10, 50, 10, {}, "largest_into_smallest"),
    (
        2,
        9,
        7,
        13,
        SMALL_EVALUATION_SAMPLE_LIMIT | DIFFERENT_LEARNING_RATES,
        "smallest_into_largest",
    ),
    (2, 16, 23, 11, {}, "smallest_into_largest"),
    (4, 22, 11, 11, DIFFERENT_LEARNING_RATES, "largest_into_smallest"),
    (4, 30, 3, 13, {}, "largest_into_smallest"),
]


@pytest.mark.parametrize(
    "class_count,concept_block_length,stream_seed,round_count_before_mapping,condition_overrides,mapping_kind",
    MODEL_ID_MAPPING_CONDITIONS,
)
def test_distribution_application_with_model_id_mapping_matches_real_legacy_client(
    class_count,
    concept_block_length,
    stream_seed,
    round_count_before_mapping,
    condition_overrides,
    mapping_kind,
    monkeypatch,
    valid_run_settings_mapping,
):
    """何ラウンドか同期した状態へ、正式IDのモデルを1つへ集めるID対応を与えて、実旧の受取りと照合する。

    その後、もう1ラウンドを両実装で進めて、受取りの後の状態で学習と予測が続けられることを確かめる。
    """
    client_streams = make_client_streams(
        round_count=round_count_before_mapping + 1,
        concept_block_length=concept_block_length,
        stream_seed=stream_seed,
    )
    server_round_oracle = build_server_round_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=class_count,
        **condition_overrides,
    )
    legacy_server = server_round_oracle["legacy_server"]
    legacy_clients = server_round_oracle["legacy_clients"]
    run_clients = server_round_oracle["run_clients"]
    stored_evaluation_sample_limit = legacy_clients[0].stored_data_limit
    observed_paths = set()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        for round_index in range(round_count_before_mapping):
            run_synchronization_round_in_both(
                server_round_oracle=server_round_oracle,
                client_streams=client_streams,
                round_index=round_index,
                distribute_in_both=distribute_to_each_client_in_both,
            )
        global_model_ids = sorted(legacy_server.global_models)
        assert len(global_model_ids) >= 2, global_model_ids
        if mapping_kind == "largest_into_smallest":
            receiving_model_id = global_model_ids[0]
            merged_model_ids = (
                global_model_ids[-2:] if len(global_model_ids) >= 3 else global_model_ids[-1:]
            )
        else:
            receiving_model_id = global_model_ids[-1]
            merged_model_ids = global_model_ids[:1]
            observed_paths.add("shared_feature_source_model_removed")
        model_id_mapping = {
            merged_model_id: receiving_model_id for merged_model_id in merged_model_ids
        }
        if len(merged_model_ids) >= 2:
            observed_paths.add("multiple_models_merged_into_one")
        current_model_ids_before = [
            legacy_client.current_model_id for legacy_client in legacy_clients
        ]
        processed_sample_count = round_count_before_mapping * ROUND_SAMPLE_COUNT
        for legacy_client in legacy_clients:
            merged_evaluation_sample_count = sum(
                len(legacy_client.stored_data.get(model_id, ()))
                for model_id in (receiving_model_id, *merged_model_ids)
            )
            if merged_evaluation_sample_count > stored_evaluation_sample_limit:
                observed_paths.add("evaluation_samples_resampled")
            if any(model_id < 0 for model_id in legacy_client.models):
                observed_paths.add("temporary_model_kept")
            if legacy_client.current_model_id < 0:
                observed_paths.add("current_training_model_is_temporary")
        distribution_applications = apply_distribution_to_each_client_in_both(
            server_round_oracle=server_round_oracle,
            model_id_mapping=model_id_mapping,
            legacy_global_models=exclude_remapped_models(
                legacy_server.global_models, model_id_mapping
            ),
            legacy_global_statistics=exclude_remapped_models(
                legacy_server.global_stats, model_id_mapping
            ),
        )
        assert_all_states_match_legacy_after_distribution(server_round_oracle)
        for run_client, legacy_client, distribution_application, current_model_id_before in zip(
            run_clients,
            legacy_clients,
            distribution_applications,
            current_model_ids_before,
            strict=True,
        ):
            assert distribution_application.held_model_ids == tuple(legacy_client.models)
            assert not set(distribution_application.held_model_ids) & set(model_id_mapping)
            adaptation_record_snapshot = (
                run_client.owners.adaptation_record_store.get_state_snapshot()
            )
            training_assignment_change = distribution_application.training_assignment_change
            if current_model_id_before in model_id_mapping:
                observed_paths.add("current_training_model_remapped")
                assert training_assignment_change is not None
                assert training_assignment_change.previous_model_id == current_model_id_before
                assert training_assignment_change.current_model_id == receiving_model_id
                # 付け替えの記録は、処理した標本数を位置にする（実旧のイベントとの照合は、上の全状態の照合）。
                assert adaptation_record_snapshot.server_remapped_sample_indices[-1] == (
                    processed_sample_count
                )
                assert legacy_client.adaptation_events[-1].action == "server_merge"
                assert legacy_client.adaptation_events[-1].detector == "server"
                assert adaptation_record_snapshot.adaptation_records[-1].detector_name == "server"
            else:
                assert training_assignment_change is None
                assert (
                    run_client.owners.current_training_model_assignment.current_training_model_id
                    == current_model_id_before
                )
        # 受取りの後も、両実装で1ラウンドを進められる（サーバは、集めたモデルを除いて配る）。
        run_synchronization_round_in_both(
            server_round_oracle=server_round_oracle,
            client_streams=client_streams,
            round_index=round_count_before_mapping,
            distribute_in_both=lambda server_round_oracle: (
                apply_distribution_to_each_client_in_both(
                    server_round_oracle=server_round_oracle,
                    model_id_mapping={},
                    legacy_global_models=exclude_remapped_models(
                        legacy_server.global_models, model_id_mapping
                    ),
                    legacy_global_statistics=exclude_remapped_models(
                        legacy_server.global_stats, model_id_mapping
                    ),
                )
            ),
        )
    MAPPING_OBSERVED_COVERAGE_BY_CONDITION[
        (class_count, concept_block_length, stream_seed, round_count_before_mapping)
    ] = observed_paths


def test_model_id_mapping_conditions_cover_required_paths():
    """上の対照が、ID対応の受取りの経路をすべて通っていること（全条件を実行したときだけ確かめる）。"""
    if len(MAPPING_OBSERVED_COVERAGE_BY_CONDITION) < len(MODEL_ID_MAPPING_CONDITIONS):
        pytest.skip("対照の全条件を実行したときだけ確かめる")
    observed_paths = set().union(*MAPPING_OBSERVED_COVERAGE_BY_CONDITION.values())
    assert observed_paths >= {
        "current_training_model_remapped",
        "multiple_models_merged_into_one",
        "shared_feature_source_model_removed",
        "evaluation_samples_resampled",
        "temporary_model_kept",
        "current_training_model_is_temporary",
    }, observed_paths


def build_synchronized_server_round_oracle(*, monkeypatch, valid_run_settings_mapping):
    """10ラウンド同期した状態。正式IDのモデルは0と2。client 1は、一時IDのモデルを保有し、それが現在の学習帰属。"""
    client_streams = make_client_streams(round_count=10, concept_block_length=10, stream_seed=50)
    server_round_oracle = build_server_round_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=2,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        for round_index in range(10):
            run_synchronization_round_in_both(
                server_round_oracle=server_round_oracle,
                client_streams=client_streams,
                round_index=round_index,
                distribute_in_both=distribute_to_each_client_in_both,
            )
    assert sorted(server_round_oracle["legacy_server"].global_models) == [0, 2]
    assert [
        legacy_client.current_model_id for legacy_client in server_round_oracle["legacy_clients"]
    ] == [2, -101, 2]
    return server_round_oracle


def get_distribution_application_arguments(*, server_round_oracle, client_index):
    """受取りの関数の、正常な引数（指定したclientのownerと、サーバが持つグローバルモデルと統計）。"""
    owners = server_round_oracle["run_clients"][client_index].owners
    legacy_server = server_round_oracle["legacy_server"]
    optimizer_settings = AdamParameterOptimizerSettings(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, adam_variant="amsgrad"
    )
    return dict(
        model_id_mapping={},
        distributed_parameter_snapshots=convert_legacy_global_models(legacy_server.global_models),
        distributed_loss_statistics=convert_legacy_global_statistics(legacy_server.global_stats),
        held_model_training_state_registry=owners.held_model_training_state_registry,
        shared_parameter_optimizer_state_holder=owners.shared_parameter_optimizer_state_holder,
        loss_statistics_store=owners.loss_statistics_store,
        training_sample_store=owners.training_sample_store,
        model_evaluation_sample_store=owners.model_evaluation_sample_store,
        model_training_and_assignment_counts_store=owners.model_training_and_assignment_counts_store,
        current_training_model_assignment=owners.current_training_model_assignment,
        adaptation_record_store=owners.adaptation_record_store,
        pending_training_assignment_buffer=owners.pending_training_assignment_buffer,
        rebuilt_model_parameter_optimizer_settings=optimizer_settings,
        reconnected_model_parameter_optimizer_settings=optimizer_settings,
        python_random_generator=server_round_oracle["python_random_generator"],
    )


def replace_first_distributed_parameters(arguments, change_parameter_snapshot):
    """最初に配布されるモデルのパラメータだけを、与えた操作で作り替えた列。"""
    (first_model_id, first_parameter_snapshot), *other_snapshots = arguments[
        "distributed_parameter_snapshots"
    ]
    return dict(
        distributed_parameter_snapshots=(
            (first_model_id, change_parameter_snapshot(dict(first_parameter_snapshot))),
            *other_snapshots,
        )
    )


def make_optimizer_settings_mutated_around_frozen(optimizer_settings):
    mutated_settings = replace(optimizer_settings)
    object.__setattr__(mutated_settings, "learning_rate", -1.0)
    return mutated_settings


DISTRIBUTION_OWNER_ARGUMENT_NAMES = (
    "held_model_training_state_registry",
    "shared_parameter_optimizer_state_holder",
    "loss_statistics_store",
    "training_sample_store",
    "model_evaluation_sample_store",
    "model_training_and_assignment_counts_store",
    "current_training_model_assignment",
    "adaptation_record_store",
    "pending_training_assignment_buffer",
)
DISTRIBUTION_OPTIMIZER_SETTINGS_ARGUMENT_NAMES = (
    "rebuilt_model_parameter_optimizer_settings",
    "reconnected_model_parameter_optimizer_settings",
)
FIRST_PARAMETER_NAME = "feature_extractor.hidden_layers.0.weight"

# 条件名 -> 正常な引数から、差し替える引数を作る操作。
INVALID_DISTRIBUTION_APPLICATION_ARGUMENT_CASES = (
    {
        "mapping_none": lambda arguments: dict(model_id_mapping=None),
        "mapping_list": lambda arguments: dict(model_id_mapping=[(2, 0)]),
        "mapping_subclass": lambda arguments: dict(
            model_id_mapping=type("DictSubclass", (dict,), {})({2: 0})
        ),
        "mapping_bool_key": lambda arguments: dict(model_id_mapping={True: 0}),
        "mapping_float_value": lambda arguments: dict(model_id_mapping={2: 0.0}),
        "mapping_text_key": lambda arguments: dict(model_id_mapping={"2": 0}),
        "parameters_list": lambda arguments: dict(
            distributed_parameter_snapshots=list(arguments["distributed_parameter_snapshots"])
        ),
        "parameters_empty": lambda arguments: dict(distributed_parameter_snapshots=()),
        "parameters_element_list": lambda arguments: dict(
            distributed_parameter_snapshots=(
                list(arguments["distributed_parameter_snapshots"][0]),
                *arguments["distributed_parameter_snapshots"][1:],
            )
        ),
        "parameters_element_triple": lambda arguments: dict(
            distributed_parameter_snapshots=(
                (*arguments["distributed_parameter_snapshots"][0], None),
                *arguments["distributed_parameter_snapshots"][1:],
            )
        ),
        "parameters_bool_id": lambda arguments: dict(
            distributed_parameter_snapshots=(
                (False, arguments["distributed_parameter_snapshots"][0][1]),
                *arguments["distributed_parameter_snapshots"][1:],
            )
        ),
        "parameters_negative_id": lambda arguments: dict(
            distributed_parameter_snapshots=(
                *arguments["distributed_parameter_snapshots"],
                (-101, arguments["distributed_parameter_snapshots"][0][1]),
            )
        ),
        "parameters_duplicate_id": lambda arguments: dict(
            distributed_parameter_snapshots=(
                *arguments["distributed_parameter_snapshots"],
                arguments["distributed_parameter_snapshots"][0],
            )
        ),
        "parameters_not_dict": lambda arguments: replace_first_distributed_parameters(
            arguments, lambda parameter_snapshot: tuple(parameter_snapshot.items())
        ),
        "parameters_missing_name": lambda arguments: replace_first_distributed_parameters(
            arguments,
            lambda parameter_snapshot: {
                parameter_name: parameter_values
                for parameter_name, parameter_values in parameter_snapshot.items()
                if parameter_name != FIRST_PARAMETER_NAME
            },
        ),
        "parameters_extra_name": lambda arguments: replace_first_distributed_parameters(
            arguments,
            lambda parameter_snapshot: parameter_snapshot | {"other": torch.zeros(1)},
        ),
        "parameters_non_text_name": lambda arguments: replace_first_distributed_parameters(
            arguments, lambda parameter_snapshot: parameter_snapshot | {0: torch.zeros(1)}
        ),
        "parameters_other_shape": lambda arguments: replace_first_distributed_parameters(
            arguments,
            lambda parameter_snapshot: (
                parameter_snapshot
                | {FIRST_PARAMETER_NAME: parameter_snapshot[FIRST_PARAMETER_NAME][:1]}
            ),
        ),
        "parameters_float64": lambda arguments: replace_first_distributed_parameters(
            arguments,
            lambda parameter_snapshot: (
                parameter_snapshot
                | {FIRST_PARAMETER_NAME: parameter_snapshot[FIRST_PARAMETER_NAME].double()}
            ),
        ),
        "parameters_not_tensor": lambda arguments: replace_first_distributed_parameters(
            arguments,
            lambda parameter_snapshot: (
                parameter_snapshot
                | {FIRST_PARAMETER_NAME: parameter_snapshot[FIRST_PARAMETER_NAME].tolist()}
            ),
        ),
        "parameters_parameter_type": lambda arguments: replace_first_distributed_parameters(
            arguments,
            lambda parameter_snapshot: (
                parameter_snapshot
                | {
                    FIRST_PARAMETER_NAME: torch.nn.Parameter(
                        parameter_snapshot[FIRST_PARAMETER_NAME]
                    )
                }
            ),
        ),
        "statistics_list": lambda arguments: dict(
            distributed_loss_statistics=list(arguments["distributed_loss_statistics"])
        ),
        "statistics_element_single": lambda arguments: dict(
            distributed_loss_statistics=((0,), *arguments["distributed_loss_statistics"][1:])
        ),
        "statistics_negative_id": lambda arguments: dict(
            distributed_loss_statistics=(
                *arguments["distributed_loss_statistics"],
                (-101, arguments["distributed_loss_statistics"][0][1]),
            )
        ),
        "statistics_duplicate_id": lambda arguments: dict(
            distributed_loss_statistics=(
                *arguments["distributed_loss_statistics"],
                arguments["distributed_loss_statistics"][0],
            )
        ),
        "statistics_legacy_dict_value": lambda arguments: dict(
            distributed_loss_statistics=((0, {"n": 3, "mean": 0.5, "M2": 0.0}),)
        ),
        # 受取りの後に、現在の学習帰属のモデルを保有しなくなる入力。
        "current_model_mapped_to_undistributed_id": lambda arguments: dict(
            model_id_mapping={
                arguments["current_training_model_assignment"].current_training_model_id: 99
            }
        ),
        "current_model_not_distributed": lambda arguments: dict(
            distributed_parameter_snapshots=tuple(
                distributed_snapshot
                for distributed_snapshot in arguments["distributed_parameter_snapshots"]
                if distributed_snapshot[0] != 2
            )
        ),
        "random_generator_module": lambda arguments: dict(python_random_generator=random),
        "random_generator_subclass": lambda arguments: dict(
            python_random_generator=type("RandomSubclass", (random.Random,), {})(3)
        ),
    }
    | {
        f"{owner_argument_name}_none": lambda arguments, owner_argument_name=owner_argument_name: {
            owner_argument_name: None
        }
        for owner_argument_name in DISTRIBUTION_OWNER_ARGUMENT_NAMES
    }
    | {
        f"{owner_argument_name}_subclass": lambda arguments, owner_argument_name=owner_argument_name: {
            owner_argument_name: make_subclass_copy(arguments[owner_argument_name])
        }
        for owner_argument_name in DISTRIBUTION_OWNER_ARGUMENT_NAMES
    }
    | {
        f"{settings_argument_name}_{invalid_kind}": lambda arguments, settings_argument_name=settings_argument_name, make_invalid_settings=make_invalid_settings: {
            settings_argument_name: make_invalid_settings(arguments[settings_argument_name])
        }
        for settings_argument_name in DISTRIBUTION_OPTIMIZER_SETTINGS_ARGUMENT_NAMES
        for invalid_kind, make_invalid_settings in (
            ("none", lambda optimizer_settings: None),
            ("subclass", make_subclass_copy),
            ("mutated", make_optimizer_settings_mutated_around_frozen),
        )
    }
)


def test_distribution_application_rejects_invalid_arguments_before_any_update(
    monkeypatch, valid_run_settings_mapping
):
    """不正な入力は、どのownerも乱数も変える前に拒否する（1つの状態で、全条件を順に確かめる）。"""
    server_round_oracle = build_synchronized_server_round_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    python_random_generator = server_round_oracle["python_random_generator"]
    # client 0は現在の学習帰属が正式ID 2、client 1は一時ID。
    for client_index in (0, 1):
        run_client = server_round_oracle["run_clients"][client_index]
        arguments = get_distribution_application_arguments(
            server_round_oracle=server_round_oracle, client_index=client_index
        )
        state_snapshot = snapshot_run_client_state(
            run_client=run_client, python_random_generator=python_random_generator
        )
        global_python_random_state = random.getstate()
        for (
            invalid_case_name,
            make_invalid_arguments,
        ) in INVALID_DISTRIBUTION_APPLICATION_ARGUMENT_CASES.items():
            if client_index == 1 and invalid_case_name == "current_model_not_distributed":
                # 一時IDのモデルは配布に含まれないので、この入力は不正にならない。
                continue
            with pytest.raises((TypeError, ValueError)):
                apply_global_model_distribution(**arguments | make_invalid_arguments(arguments))
            # 引数に不正な乱数生成器を渡した条件でも、clientの乱数生成器と状態は変わらない。
            assert_run_client_state_unchanged(
                state_snapshot=state_snapshot,
                run_client=run_client,
                python_random_generator=python_random_generator,
            )
            assert random.getstate() == global_python_random_state, invalid_case_name
    # 全条件の後も、実旧と同じ状態のまま。
    assert_all_states_match_legacy_after_distribution(server_round_oracle)


def test_client_rejects_invalid_distribution_through_its_operation(
    monkeypatch, valid_run_settings_mapping
):
    """clientの操作も、不正な入力を、何も変えずに拒否する。"""
    server_round_oracle = build_synchronized_server_round_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    python_random_generator = server_round_oracle["python_random_generator"]
    run_client = server_round_oracle["run_clients"][0]
    arguments = get_distribution_application_arguments(
        server_round_oracle=server_round_oracle, client_index=0
    )
    operation_arguments = {
        argument_name: arguments[argument_name]
        for argument_name in (
            "model_id_mapping",
            "distributed_parameter_snapshots",
            "distributed_loss_statistics",
        )
    }
    state_snapshot = snapshot_run_client_state(
        run_client=run_client, python_random_generator=python_random_generator
    )
    for invalid_case_name in (
        "mapping_bool_key",
        "parameters_empty",
        "parameters_other_shape",
        "statistics_duplicate_id",
        "current_model_mapped_to_undistributed_id",
    ):
        with pytest.raises((TypeError, ValueError)):
            run_client.apply_global_model_distribution(
                **operation_arguments
                | INVALID_DISTRIBUTION_APPLICATION_ARGUMENT_CASES[invalid_case_name](arguments)
            )
        assert_run_client_state_unchanged(
            state_snapshot=state_snapshot,
            run_client=run_client,
            python_random_generator=python_random_generator,
        )


# 受取りが変えない状態（読取りの項目名）。
STATES_KEPT_BY_DISTRIBUTION_APPLICATION = (
    "pending_assignment",
    "pending_sample_observations",
    "validation_assignment_sample_concept_ids",
    "monitoring",
    "alarm_records",
    "held_validation_session",
    "pending_training_request_count",
    "diagnostics",
    "fixed_share_prediction_weights",
    "sample_prediction_records",
    "next_temporary_model_id",
    "pending_model_upload",
)


@pytest.mark.parametrize("model_id_mapping", [{}, {2: 0}])
def test_distribution_application_keeps_states_outside_its_responsibility(
    model_id_mapping, monkeypatch, valid_run_settings_mapping
):
    """受取りは、保留・監視・候補検証・送信保留・学習要求・一時IDの採番・予測の重みと記録・診断証拠を変えない。"""
    server_round_oracle = build_synchronized_server_round_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    python_random_generator = server_round_oracle["python_random_generator"]
    legacy_server = server_round_oracle["legacy_server"]
    state_snapshots = [
        snapshot_run_client_state(
            run_client=run_client, python_random_generator=python_random_generator
        )
        for run_client in server_round_oracle["run_clients"]
    ]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        distribution_applications = apply_distribution_to_each_client_in_both(
            server_round_oracle=server_round_oracle,
            model_id_mapping=model_id_mapping,
            legacy_global_models=exclude_remapped_models(
                legacy_server.global_models, model_id_mapping
            ),
            legacy_global_statistics=exclude_remapped_models(
                legacy_server.global_stats, model_id_mapping
            ),
        )
    for run_client, state_snapshot, distribution_application in zip(
        server_round_oracle["run_clients"], state_snapshots, distribution_applications, strict=True
    ):
        current_snapshot = snapshot_run_client_state(
            run_client=run_client, python_random_generator=python_random_generator
        )
        for state_name in STATES_KEPT_BY_DISTRIBUTION_APPLICATION:
            if state_name in ("held_validation_session", "pending_model_upload"):
                assert current_snapshot[state_name] is state_snapshot[state_name], state_name
            else:
                assert current_snapshot[state_name] == state_snapshot[state_name], state_name
        # 保有モデルは、配布されたモデル（配布の順）と、残した一時IDのモデル。全部が1つの共有部につながる。
        held_model_training_states = run_client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        assert distribution_application.held_model_ids == tuple(
            held_state.model_id for held_state in held_model_training_states
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
        # 共有部のoptimizerの状態は、つなぎ先の共有部のパラメータを指す。
        shared_parameter_optimizer = run_client.owners.shared_parameter_optimizer_state_holder.held_shared_parameter_optimizer_state.parameter_optimizer
        assert all(
            optimizer_parameter is shared_parameter
            for optimizer_parameter, shared_parameter in zip(
                shared_parameter_optimizer.param_groups[0]["params"],
                held_model_training_states[0].classifier.feature_extractor.parameters(),
                strict=True,
            )
        )
