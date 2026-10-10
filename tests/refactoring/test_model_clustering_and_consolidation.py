"""クラスタリングと統合、サーバの1ラウンドの同期を、実旧のサーバのrun_round（クラスタリングあり）と照合する。"""

import math
import random
from dataclasses import replace

import pytest
import torch
from test_fedsda_run_client import (
    CROSS_EVALUATION_CLIENT_LIMIT,
    MAXIMUM_TOLERATED_MEAN_LOSS_INCREASE,
    run_in_both,
)
from test_global_model_distribution import (
    DIFFERENT_LEARNING_RATES,
    assert_all_states_match_legacy_after_distribution,
    make_client_streams,
)
from test_held_candidate_validation_progress import make_subclass_copy
from test_model_cross_evaluation import (
    MORE_EVALUATION_SAMPLES,
    derive_legacy_cross_evaluation_diagnostics,
)
from test_post_aggregation_prediction_recalibration import FIFO_REPLAY_RECALIBRATION
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping
from test_server_model_registration_and_aggregation import (
    assert_server_and_client_states_unchanged,
    build_server_round_oracle,
    process_round_samples_in_both,
    snapshot_server_and_client_states,
)

import federated_learning_experiments.runtime.server_round_synchronization as synchronization_module
from federated_learning_experiments.evaluation.cross_evaluation_record_store import (
    CrossEvaluationRecordStore,
)
from federated_learning_experiments.evaluation.model_clustering_record_store import (
    ModelClusteringRecordStore,
)
from federated_learning_experiments.methods.fedsda.consolidation.model_clustering_calculations import (
    ModelClusteringCriteria,
)
from federated_learning_experiments.runtime.model_clustering_and_consolidation import (
    ModelConsolidation,
    cluster_and_consolidate_global_models,
)
from federated_learning_experiments.runtime.model_cross_evaluation import (
    cross_evaluate_global_models,
)
from federated_learning_experiments.runtime.server_model_registration_and_aggregation import (
    aggregate_client_models_into_global_models,
)
from federated_learning_experiments.runtime.server_round_synchronization import (
    ServerRoundSynchronization,
    synchronize_models_in_server_round,
)

# 実旧のサーバと同じ基準: 閾値はサーバのdistance_threshold、評価の件数の下限は旧の既定（CLUSTER_MIN_EVAL_N）、信頼水準0.95。
MODEL_CLUSTERING_CRITERIA = ModelClusteringCriteria(
    maximum_same_cluster_decision_score=MAXIMUM_TOLERATED_MEAN_LOSS_INCREASE,
    minimum_pair_evaluation_sample_count=5,
    clustering_confidence_level=0.95,
)


def convert_missing_number(legacy_number):
    """旧のNaN（値なし）を、新のNoneへ対応させる。"""
    return None if math.isnan(legacy_number) else legacy_number


def assert_clustering_records_match_legacy(model_clustering_record_store, legacy_server):
    """クラスタリングの診断の記録を、実旧の来歴の、対の観測とモデルの観測の列と照合する。"""
    pair_observations = model_clustering_record_store.snapshot_pair_clustering_observations()
    legacy_pair_observations = legacy_server.model_lineage.clustering_pair_observations
    assert len(pair_observations) == len(legacy_pair_observations)
    for pair_observation, legacy_pair_observation in zip(
        pair_observations, legacy_pair_observations, strict=True
    ):
        assert (
            pair_observation.round_index,
            pair_observation.lower_model_id,
            pair_observation.higher_model_id,
            pair_observation.loss_increase_distance,
            pair_observation.decision_score,
            pair_observation.assigned_to_same_cluster,
            pair_observation.true_concepts_match,
            pair_observation.concept_specific_parameter_distance,
        ) == (
            legacy_pair_observation.round_index,
            legacy_pair_observation.left_model_id,
            legacy_pair_observation.right_model_id,
            legacy_pair_observation.distance,
            legacy_pair_observation.decision_score,
            legacy_pair_observation.same_cluster,
            legacy_pair_observation.oracle_same_concept,
            legacy_pair_observation.personalized_parameter_distance,
        )
    model_observations = model_clustering_record_store.snapshot_model_clustering_observations()
    legacy_model_observations = legacy_server.model_lineage.clustering_observations
    assert len(model_observations) == len(legacy_model_observations)
    for model_observation, legacy_model_observation in zip(
        model_observations, legacy_model_observations, strict=True
    ):
        assert (
            model_observation.round_index,
            model_observation.model_id,
            model_observation.nearest_model_id,
            model_observation.nearest_loss_increase_distance,
            model_observation.representative_model_id,
            model_observation.cluster_model_count,
            model_observation.maximum_within_cluster_distance,
            model_observation.evaluated_within_cluster_pair_count,
            model_observation.possible_within_cluster_pair_count,
            model_observation.merged_with_other_models,
            model_observation.absorbed_into_representative,
        ) == (
            legacy_model_observation.round_index,
            legacy_model_observation.model_id,
            # 旧は、最も近いモデルがないとき−1、距離がないときNaN。
            None
            if legacy_model_observation.nearest_model_id == -1
            else legacy_model_observation.nearest_model_id,
            convert_missing_number(legacy_model_observation.nearest_distance),
            legacy_model_observation.representative_model_id,
            legacy_model_observation.cluster_size,
            convert_missing_number(legacy_model_observation.cluster_max_distance),
            legacy_model_observation.cluster_evaluated_pairs,
            legacy_model_observation.cluster_possible_pairs,
            legacy_model_observation.participated_in_merge,
            legacy_model_observation.absorbed,
        )


def make_synchronization_owners(server_round_oracle):
    """同期の関数が使う、サーバ側のownerと基準（oracleが持たないものは、新しく作る）。"""
    return dict(
        run_clients=server_round_oracle["run_clients"],
        global_model_repository=server_round_oracle["global_model_repository"],
        communication_volume_record_store=server_round_oracle["communication_volume_record_store"],
        cross_evaluation_record_store=CrossEvaluationRecordStore(),
        model_clustering_record_store=ModelClusteringRecordStore(),
        model_clustering_criteria=MODEL_CLUSTERING_CRITERIA,
        maximum_evaluating_client_count_per_model=CROSS_EVALUATION_CLIENT_LIMIT,
        python_random_generator=server_round_oracle["python_random_generator"],
    )


def assert_server_side_records_match_legacy(synchronization_owners, legacy_server):
    assert_clustering_records_match_legacy(
        synchronization_owners["model_clustering_record_store"], legacy_server
    )
    all_evaluations, compared_evaluations, class_evaluations = (
        derive_legacy_cross_evaluation_diagnostics(
            synchronization_owners[
                "cross_evaluation_record_store"
            ].snapshot_client_cross_evaluation_records()
        )
    )
    assert all_evaluations == legacy_server.cross_evaluation_diagnostics
    assert compared_evaluations == legacy_server.pair_prediction_diagnostics
    assert class_evaluations == legacy_server.cross_evaluation_class_diagnostics


def run_round_with_clustering_in_both(
    *, server_round_oracle, synchronization_owners, client_streams, round_index
):
    """1ラウンド: 標本処理→保留中の学習→状態の報告→（旧はrun_round、新は同期の関数）→送信待ちの進行。

    クラスタリングは、旧の実験の枠と同じく、登録の前に、送信できるモデルを持つclientがいるラウンドだけ有効にする。
    戻り値: 新の同期の結果と、クラスタリングが有効だったか。
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
    model_clustering_enabled = any(
        legacy_client.has_pending_model() for legacy_client in legacy_clients
    )
    assert model_clustering_enabled == any(
        run_client.has_model_ready_for_server_registration() for run_client in run_clients
    )
    server_round_synchronization, _ = run_in_both(
        run_client=None,
        legacy_client=None,
        python_random_generator=server_round_oracle["python_random_generator"],
        legacy_operation=lambda: legacy_server.run_round(
            round_index, clustering_enabled=model_clustering_enabled
        ),
        operation=lambda: synchronize_models_in_server_round(
            **synchronization_owners,
            round_index=round_index,
            model_clustering_enabled=model_clustering_enabled,
        ),
    )
    assert type(server_round_synchronization) is ServerRoundSynchronization
    assert_all_states_match_legacy_after_distribution(server_round_oracle)
    assert_server_side_records_match_legacy(synchronization_owners, legacy_server)
    for run_client, legacy_client in zip(run_clients, legacy_clients, strict=True):
        legacy_client.promote_pending_to_ready()
        run_client.advance_new_model_upload_wait_after_synchronization(round_index=round_index)
    assert_all_states_match_legacy_after_distribution(server_round_oracle)
    return server_round_synchronization, model_clustering_enabled


CLUSTERING_ROUND_COUNT = 16
OBSERVED_COVERAGE_BY_CONDITION = {}
# (クラス数, 概念の区間長, 標本列のseed, 条件の上書き)。実旧だけで進めて、クラスタリングの起き方を調べて選んだ。
CLUSTERING_ROUND_CONDITIONS = [
    (2, 22, 11, {}),
    (2, 30, 3, {}),
    (2, 16, 23, {}),
    (2, 16, 23, MORE_EVALUATION_SAMPLES),
    (2, 13, 5, {}),
    (2, 9, 7, {}),
    (2, 10, 50, {}),
    (2, 25, 31, DIFFERENT_LEARNING_RATES),
    (4, 22, 11, {}),
    (4, 30, 3, {}),
    (4, 25, 31, MORE_EVALUATION_SAMPLES),
    (4, 13, 5, {}),
]


@pytest.mark.parametrize(
    "class_count,concept_block_length,stream_seed,condition_overrides",
    CLUSTERING_ROUND_CONDITIONS,
)
def test_server_round_with_clustering_matches_real_legacy_server_run_round(
    class_count,
    concept_block_length,
    stream_seed,
    condition_overrides,
    monkeypatch,
    valid_run_settings_mapping,
):
    client_streams = make_client_streams(
        round_count=CLUSTERING_ROUND_COUNT,
        concept_block_length=concept_block_length,
        stream_seed=stream_seed,
    )
    server_round_oracle = build_server_round_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=class_count,
        **FIFO_REPLAY_RECALIBRATION,
        **condition_overrides,
    )
    legacy_server = server_round_oracle["legacy_server"]
    synchronization_owners = make_synchronization_owners(server_round_oracle)
    model_clustering_record_store = synchronization_owners["model_clustering_record_store"]
    observed_paths = set()
    clustering_count = 0
    consolidated = False
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        for round_index in range(CLUSTERING_ROUND_COUNT):
            pair_observation_count = len(
                model_clustering_record_store.snapshot_pair_clustering_observations()
            )
            server_round_synchronization, model_clustering_enabled = (
                run_round_with_clustering_in_both(
                    server_round_oracle=server_round_oracle,
                    synchronization_owners=synchronization_owners,
                    client_streams=client_streams,
                    round_index=round_index,
                )
            )
            model_consolidation = server_round_synchronization.model_consolidation
            aggregated_model_ids = (
                server_round_synchronization.client_model_aggregation.aggregated_global_model_ids
            )
            if consolidated:
                observed_paths.add("round_after_consolidation")
            if model_consolidation is None:
                assert server_round_synchronization.model_cross_evaluation is None
                # クラスタリングを行わないのは、無効のラウンドか、集約の対象のモデルが1つ以下のとき。
                assert not model_clustering_enabled or len(aggregated_model_ids) <= 1
                observed_paths.add(
                    "clustering_enabled_with_single_model"
                    if model_clustering_enabled
                    else "clustering_disabled"
                )
                continue
            assert type(model_consolidation) is ModelConsolidation
            assert model_clustering_enabled and len(aggregated_model_ids) >= 2
            assert (
                server_round_synchronization.model_cross_evaluation.cross_evaluated_model_ids
                == aggregated_model_ids
            )
            clustering_count += 1
            if clustering_count >= 2:
                observed_paths.add("clustered_again_in_same_run")
            # クラスタと、ID対応は、実旧のこのラウンドの、モデルの観測（代表）と同じ。
            legacy_round_observations = [
                legacy_observation
                for legacy_observation in legacy_server.model_lineage.clustering_observations
                if legacy_observation.round_index == round_index
            ]
            assert sorted(
                model_id
                for model_cluster in model_consolidation.model_clusters
                for model_id in model_cluster
            ) == sorted(aggregated_model_ids)
            assert {
                model_id: min(model_cluster)
                for model_cluster in model_consolidation.model_clusters
                for model_id in model_cluster
            } == {
                legacy_observation.model_id: legacy_observation.representative_model_id
                for legacy_observation in legacy_round_observations
            }
            added_pair_observations = (
                model_clustering_record_store.snapshot_pair_clustering_observations()[
                    pair_observation_count:
                ]
            )
            possible_pair_count = len(aggregated_model_ids) * (len(aggregated_model_ids) - 1) // 2
            if len(added_pair_observations) < possible_pair_count:
                observed_paths.add("pair_without_loss_increase_distance")
            if any(
                not pair_observation.assigned_to_same_cluster
                for pair_observation in added_pair_observations
            ):
                observed_paths.add("evaluated_pair_kept_apart")
            if any(
                pair_observation.true_concepts_match is not None
                for pair_observation in added_pair_observations
            ):
                observed_paths.add("true_concept_match_recorded")
            maximum_cluster_model_count = max(
                len(model_cluster) for model_cluster in model_consolidation.model_clusters
            )
            if maximum_cluster_model_count == 1:
                observed_paths.add("clustered_without_consolidation")
                assert model_consolidation.model_id_mapping == {}
                assert model_consolidation.absorbed_model_ids == ()
                assert tuple(legacy_server.global_models) == aggregated_model_ids
                continue
            consolidated = True
            observed_paths.add(
                "two_models_consolidated"
                if maximum_cluster_model_count == 2
                else "three_or_more_models_consolidated"
            )
            # ID対応は、クラスタリングした全モデルを持つ（1モデルのクラスタは、自分自身への対応）。
            assert sorted(model_consolidation.model_id_mapping) == sorted(aggregated_model_ids)
            assert any(
                model_id == representative_model_id
                for model_id, representative_model_id in model_consolidation.model_id_mapping.items()
            )
            assert model_consolidation.absorbed_model_ids == tuple(
                sorted(
                    model_id
                    for model_id, representative_model_id in model_consolidation.model_id_mapping.items()
                    if model_id != representative_model_id
                )
            )
            assert not set(model_consolidation.absorbed_model_ids) & set(
                legacy_server.global_models
            )
            # 吸収されたモデルが現在の学習帰属だったclientは、代表へ付け替わっている。
            for distribution_application in server_round_synchronization.distribution_applications:
                assert not set(distribution_application.held_model_ids) & set(
                    model_consolidation.absorbed_model_ids
                )
                if distribution_application.training_assignment_change is not None:
                    observed_paths.add("current_training_model_remapped_by_consolidation")
    OBSERVED_COVERAGE_BY_CONDITION[
        (class_count, concept_block_length, stream_seed, tuple(condition_overrides))
    ] = observed_paths


def test_clustering_round_conditions_cover_required_paths():
    """上の対照が、要求の経路をすべて通っていること（全条件を実行したときだけ確かめる）。"""
    if len(OBSERVED_COVERAGE_BY_CONDITION) < len(CLUSTERING_ROUND_CONDITIONS):
        pytest.skip("対照の全条件を実行したときだけ確かめる")
    for required_path in (
        "clustering_disabled",
        "two_models_consolidated",
        "clustered_without_consolidation",
        "pair_without_loss_increase_distance",
        "evaluated_pair_kept_apart",
        "true_concept_match_recorded",
        "round_after_consolidation",
        "current_training_model_remapped_by_consolidation",
    ):
        observed_class_counts = {
            condition[0]
            for condition, observed_paths in OBSERVED_COVERAGE_BY_CONDITION.items()
            if required_path in observed_paths
        }
        assert observed_class_counts == {2, 4}, (required_path, observed_class_counts)
    all_observed_paths = set().union(*OBSERVED_COVERAGE_BY_CONDITION.values())
    # 3モデル以上が1つへ集まる統合と、同じrunでの2回めのクラスタリングは、どちらかのクラス数で通っていればよい。
    assert {
        "three_or_more_models_consolidated",
        "clustered_again_in_same_run",
    } <= all_observed_paths, all_observed_paths


def build_cross_evaluated_server_round_oracle(*, monkeypatch, valid_run_settings_mapping):
    """クラスタリングの直前の状態: 8ラウンド同期し、9ラウンドめの登録・集約・クロス評価まで済ませた状態。

    この標本列（2値、区間長16、seed 23）では、9ラウンドめに、新規モデルが3つ登録され、クラスタリングで、
    モデル1・2・3が1つへ集まる（下のtestが確かめる）。
    戻り値: (oracle, 同期のowner, クロス評価の結果, 集約の結果, ラウンド)。
    """
    client_streams = make_client_streams(round_count=9, concept_block_length=16, stream_seed=23)
    server_round_oracle = build_server_round_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=2,
        **FIFO_REPLAY_RECALIBRATION,
    )
    synchronization_owners = make_synchronization_owners(server_round_oracle)
    run_clients = server_round_oracle["run_clients"]
    legacy_server = server_round_oracle["legacy_server"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        for round_index in range(8):
            run_round_with_clustering_in_both(
                server_round_oracle=server_round_oracle,
                synchronization_owners=synchronization_owners,
                client_streams=client_streams,
                round_index=round_index,
            )
        round_index = 8
        process_round_samples_in_both(
            server_round_oracle=server_round_oracle,
            client_streams=client_streams,
            round_index=round_index,
        )
        # 旧は、ラウンドの部品を、ラウンドと同じ順に直接呼ぶ。
        legacy_server._register_new_models(round_index)
        legacy_active_model_ids = sorted(
            {
                model_id
                for legacy_client in server_round_oracle["legacy_clients"]
                for model_id in legacy_client.models
                if model_id >= 0
            }
        )
        legacy_aggregation_weights = legacy_server.update_global_models(legacy_active_model_ids)
        synchronization_module.register_ready_client_models(
            run_clients=run_clients,
            global_model_repository=server_round_oracle["global_model_repository"],
            round_index=round_index,
        )
        client_model_aggregation = aggregate_client_models_into_global_models(
            run_clients=run_clients,
            global_model_repository=server_round_oracle["global_model_repository"],
            communication_volume_record_store=server_round_oracle[
                "communication_volume_record_store"
            ],
        )
        model_cross_evaluation, legacy_statistics_matrix = run_in_both(
            run_client=None,
            legacy_client=None,
            python_random_generator=server_round_oracle["python_random_generator"],
            legacy_operation=lambda: legacy_server._cross_evaluate(
                legacy_active_model_ids, round_index=round_index
            ),
            operation=lambda: cross_evaluate_global_models(
                run_clients=run_clients,
                global_model_repository=server_round_oracle["global_model_repository"],
                communication_volume_record_store=server_round_oracle[
                    "communication_volume_record_store"
                ],
                cross_evaluation_record_store=synchronization_owners[
                    "cross_evaluation_record_store"
                ],
                cross_evaluated_model_ids=client_model_aggregation.aggregated_global_model_ids,
                round_index=round_index,
                maximum_evaluating_client_count_per_model=CROSS_EVALUATION_CLIENT_LIMIT,
                python_random_generator=server_round_oracle["python_random_generator"],
            ),
        )
    assert client_model_aggregation.aggregated_global_model_ids == (0, 1, 2, 3)
    return dict(
        server_round_oracle=server_round_oracle,
        synchronization_owners=synchronization_owners,
        model_cross_evaluation=model_cross_evaluation,
        client_model_aggregation=client_model_aggregation,
        legacy_statistics_matrix=legacy_statistics_matrix,
        legacy_active_model_ids=legacy_active_model_ids,
        legacy_aggregation_weights=legacy_aggregation_weights,
        round_index=round_index,
    )


def get_clustering_and_consolidation_arguments(cross_evaluated_state):
    synchronization_owners = cross_evaluated_state["synchronization_owners"]
    return dict(
        run_clients=synchronization_owners["run_clients"],
        model_cross_evaluation=cross_evaluated_state["model_cross_evaluation"],
        aggregated_training_sample_counts_by_model_id=cross_evaluated_state[
            "client_model_aggregation"
        ].aggregated_training_sample_counts_by_model_id,
        global_model_repository=synchronization_owners["global_model_repository"],
        model_clustering_record_store=synchronization_owners["model_clustering_record_store"],
        model_clustering_criteria=MODEL_CLUSTERING_CRITERIA,
        round_index=cross_evaluated_state["round_index"],
    )


def test_clustering_and_consolidation_match_real_legacy_merge_step(
    monkeypatch, valid_run_settings_mapping
):
    """クラスタリングと統合だけを、実旧の、クラスタリング→診断の記録→統合と照合する（配布の前）。"""
    cross_evaluated_state = build_cross_evaluated_server_round_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    server_round_oracle = cross_evaluated_state["server_round_oracle"]
    legacy_server = server_round_oracle["legacy_server"]
    global_model_repository = server_round_oracle["global_model_repository"]
    legacy_active_model_ids = cross_evaluated_state["legacy_active_model_ids"]
    round_index = cross_evaluated_state["round_index"]
    random_states = (random.getstate(), torch.get_rng_state().clone())
    next_global_model_id = global_model_repository.next_global_model_id
    registration_records = global_model_repository.snapshot_model_registration_records()
    client_state_snapshots = snapshot_server_and_client_states(server_round_oracle)["clients"]
    # 旧: クラスタリング→診断の記録→統合（_cluster_and_consolidateの、クロス評価より後の部分）。
    legacy_clusters = legacy_server.perform_hierarchical_clustering(
        legacy_active_model_ids, cross_evaluated_state["legacy_statistics_matrix"]
    )
    legacy_server.record_clustering_diagnostics(
        round_index, legacy_active_model_ids, legacy_clusters
    )
    assert len(legacy_clusters) < len(legacy_active_model_ids)
    legacy_model_id_mapping = legacy_server._merge_clusters(
        legacy_active_model_ids,
        legacy_clusters,
        cross_evaluated_state["legacy_aggregation_weights"],
        round_index,
    )
    model_consolidation = cluster_and_consolidate_global_models(
        **get_clustering_and_consolidation_arguments(cross_evaluated_state)
    )
    assert [
        list(model_cluster) for model_cluster in model_consolidation.model_clusters
    ] == legacy_clusters
    assert model_consolidation.model_clusters == ((0,), (1, 2, 3))
    assert model_consolidation.model_id_mapping == legacy_model_id_mapping
    assert list(model_consolidation.model_id_mapping) == list(legacy_model_id_mapping)
    assert model_consolidation.absorbed_model_ids == (2, 3)
    # グローバルモデル（統合した値、順）、統計、通信量、診断の記録、clientの状態は、実旧と同じ。
    assert_all_states_match_legacy_after_distribution(server_round_oracle)
    assert_server_side_records_match_legacy(
        cross_evaluated_state["synchronization_owners"], legacy_server
    )
    # 統合は、次の正式IDと来歴、clientの状態、乱数を変えない。
    assert global_model_repository.next_global_model_id == next_global_model_id
    assert global_model_repository.snapshot_model_registration_records() == registration_records
    assert_server_and_client_states_unchanged(
        state_snapshot=snapshot_server_and_client_states(server_round_oracle)
        | dict(clients=client_state_snapshots),
        server_round_oracle=server_round_oracle,
    )
    assert random.getstate() == random_states[0]
    assert torch.equal(torch.get_rng_state(), random_states[1])


def replace_cross_evaluation(arguments, **replaced_fields):
    return dict(
        model_cross_evaluation=replace(arguments["model_cross_evaluation"], **replaced_fields)
    )


def make_criteria_mutated_around_frozen(model_clustering_criteria):
    mutated_criteria = replace(model_clustering_criteria)
    object.__setattr__(mutated_criteria, "clustering_confidence_level", 1.5)
    return mutated_criteria


# 条件名 -> 正常な引数から、差し替える引数を作る操作。
INVALID_CLUSTERING_ARGUMENT_CASES = (
    {
        "clients_list": lambda arguments: dict(run_clients=list(arguments["run_clients"])),
        "clients_hold_none": lambda arguments: dict(run_clients=(*arguments["run_clients"], None)),
        "round_index_bool": lambda arguments: dict(round_index=True),
        "round_index_negative": lambda arguments: dict(round_index=-1),
        "criteria_mutated": lambda arguments: dict(
            model_clustering_criteria=make_criteria_mutated_around_frozen(
                arguments["model_clustering_criteria"]
            )
        ),
        "cross_evaluation_without_model_ids": lambda arguments: replace_cross_evaluation(
            arguments, cross_evaluated_model_ids=()
        ),
        # モデルが1つ（クラスタリングの対象がない。旧も、対象が1つ以下なら何もしない）。
        "cross_evaluation_single_model_id": lambda arguments: replace_cross_evaluation(
            arguments, cross_evaluated_model_ids=(0,)
        ),
        "cross_evaluation_model_ids_list": lambda arguments: replace_cross_evaluation(
            arguments, cross_evaluated_model_ids=[0, 1, 2, 3]
        ),
        "cross_evaluation_unknown_model_id": lambda arguments: replace_cross_evaluation(
            arguments, cross_evaluated_model_ids=(0, 1, 2, 99)
        ),
        "cross_evaluation_duplicate_model_id": lambda arguments: replace_cross_evaluation(
            arguments, cross_evaluated_model_ids=(0, 1, 1)
        ),
        "cross_evaluation_missing_loss_sums": lambda arguments: replace_cross_evaluation(
            arguments,
            loss_sums_by_candidate_and_target_model_id={
                model_pair: loss_sums
                for model_pair, loss_sums in arguments[
                    "model_cross_evaluation"
                ].loss_sums_by_candidate_and_target_model_id.items()
                if model_pair != (3, 1)
            },
        ),
        "cross_evaluation_loss_sums_tuple": lambda arguments: replace_cross_evaluation(
            arguments,
            loss_sums_by_candidate_and_target_model_id=arguments[
                "model_cross_evaluation"
            ].loss_sums_by_candidate_and_target_model_id
            | {(3, 1): (5, 1.0, 0.5)},
        ),
        "cross_evaluation_correctness_counts_list": lambda arguments: replace_cross_evaluation(
            arguments, unique_correctness_counts_by_model_pair=[]
        ),
        "sample_counts_list": lambda arguments: dict(
            aggregated_training_sample_counts_by_model_id=list(
                arguments["aggregated_training_sample_counts_by_model_id"].items()
            )
        ),
        "sample_counts_missing_model": lambda arguments: dict(
            aggregated_training_sample_counts_by_model_id={
                model_id: sample_count
                for model_id, sample_count in arguments[
                    "aggregated_training_sample_counts_by_model_id"
                ].items()
                if model_id != 2
            }
        ),
        "sample_counts_float": lambda arguments: dict(
            aggregated_training_sample_counts_by_model_id=arguments[
                "aggregated_training_sample_counts_by_model_id"
            ]
            | {2: 10.0}
        ),
    }
    | {
        f"{owner_argument_name}_none": lambda arguments, owner_argument_name=owner_argument_name: {
            owner_argument_name: None
        }
        for owner_argument_name in (
            "model_cross_evaluation",
            "global_model_repository",
            "model_clustering_record_store",
            "model_clustering_criteria",
        )
    }
    | {
        f"{owner_argument_name}_subclass": lambda arguments, owner_argument_name=owner_argument_name: {
            owner_argument_name: make_subclass_copy(arguments[owner_argument_name])
        }
        for owner_argument_name in (
            "model_cross_evaluation",
            "global_model_repository",
            "model_clustering_record_store",
            "model_clustering_criteria",
        )
    }
)


def test_clustering_and_consolidation_reject_invalid_arguments_before_any_update(
    monkeypatch, valid_run_settings_mapping
):
    """不正な入力は、診断の記録もグローバルモデルも変える前に拒否する（1つの状態で、全条件を順に確かめる）。"""
    cross_evaluated_state = build_cross_evaluated_server_round_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    server_round_oracle = cross_evaluated_state["server_round_oracle"]
    arguments = get_clustering_and_consolidation_arguments(cross_evaluated_state)
    state_snapshot = snapshot_server_and_client_states(server_round_oracle)
    model_clustering_record_store = arguments["model_clustering_record_store"]
    for invalid_case_name, make_invalid_arguments in INVALID_CLUSTERING_ARGUMENT_CASES.items():
        with pytest.raises((TypeError, ValueError)):
            cluster_and_consolidate_global_models(**arguments | make_invalid_arguments(arguments))
        assert_server_and_client_states_unchanged(
            state_snapshot=state_snapshot, server_round_oracle=server_round_oracle
        )
        assert model_clustering_record_store.snapshot_pair_clustering_observations() == (), (
            invalid_case_name
        )
        assert model_clustering_record_store.snapshot_model_clustering_observations() == ()


def test_clustering_without_consolidation_keeps_global_models(
    monkeypatch, valid_run_settings_mapping
):
    """閾値を、どの対も同じクラスタにならない値にすると、観測だけを足して、グローバルモデルを変えない。"""
    cross_evaluated_state = build_cross_evaluated_server_round_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    server_round_oracle = cross_evaluated_state["server_round_oracle"]
    arguments = get_clustering_and_consolidation_arguments(cross_evaluated_state)
    state_snapshot = snapshot_server_and_client_states(server_round_oracle)
    model_consolidation = cluster_and_consolidate_global_models(
        **arguments
        | dict(
            model_clustering_criteria=replace(
                MODEL_CLUSTERING_CRITERIA, maximum_same_cluster_decision_score=-1.0
            )
        )
    )
    assert model_consolidation.model_clusters == ((0,), (1,), (2,), (3,))
    assert model_consolidation.model_id_mapping == {}
    assert model_consolidation.absorbed_model_ids == ()
    assert_server_and_client_states_unchanged(
        state_snapshot=state_snapshot, server_round_oracle=server_round_oracle
    )
    model_observations = arguments[
        "model_clustering_record_store"
    ].snapshot_model_clustering_observations()
    assert [model_observation.model_id for model_observation in model_observations] == [0, 1, 2, 3]
    assert not any(
        model_observation.merged_with_other_models for model_observation in model_observations
    )
    assert arguments["model_clustering_record_store"].snapshot_pair_clustering_observations()


def test_server_round_synchronization_stops_at_failed_stage_without_rollback(
    monkeypatch, valid_run_settings_mapping
):
    """クラスタリングと統合の段が失敗したら、配布と再較正へ進まない。登録・集約・クロス評価の分は残る。"""
    client_streams = make_client_streams(round_count=9, concept_block_length=16, stream_seed=23)
    server_round_oracle = build_server_round_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=2,
        **FIFO_REPLAY_RECALIBRATION,
    )
    synchronization_owners = make_synchronization_owners(server_round_oracle)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        for round_index in range(8):
            run_round_with_clustering_in_both(
                server_round_oracle=server_round_oracle,
                synchronization_owners=synchronization_owners,
                client_streams=client_streams,
                round_index=round_index,
            )
        process_round_samples_in_both(
            server_round_oracle=server_round_oracle, client_streams=client_streams, round_index=8
        )
        global_model_repository = server_round_oracle["global_model_repository"]
        assert global_model_repository.global_model_ids == (0,)
        volume_before = server_round_oracle[
            "communication_volume_record_store"
        ].get_state_snapshot()
        called_stage_names = []
        for stage_name in (
            "distribute_global_models_to_clients",
            "cluster_and_consolidate_global_models",
        ):
            real_stage = getattr(synchronization_module, stage_name)

            def record_stage_call(*, stage_name=stage_name, real_stage=real_stage, **arguments):
                called_stage_names.append(stage_name)
                if stage_name == "cluster_and_consolidate_global_models":
                    raise RuntimeError("injected consolidation failure")
                return real_stage(**arguments)

            monkeypatch.setattr(synchronization_module, stage_name, record_stage_call)
        with pytest.raises(RuntimeError, match="injected consolidation failure"):
            synchronize_models_in_server_round(
                **synchronization_owners, round_index=8, model_clustering_enabled=True
            )
    assert called_stage_names == ["cluster_and_consolidate_global_models"]
    # 登録（3つ）と集約とクロス評価は済んでいる。統合と配布は行われていない。
    assert global_model_repository.global_model_ids == (0, 1, 2, 3)
    assert global_model_repository.next_global_model_id == 4
    assert synchronization_owners[
        "cross_evaluation_record_store"
    ].snapshot_client_cross_evaluation_records()
    assert (
        synchronization_owners[
            "model_clustering_record_store"
        ].snapshot_model_clustering_observations()
        == ()
    )
    volume_after = server_round_oracle["communication_volume_record_store"].get_state_snapshot()
    assert volume_after.uploaded_model_count > volume_before.uploaded_model_count
    assert volume_after.downloaded_message_count > volume_before.downloaded_message_count
    for run_client in server_round_oracle["run_clients"]:
        # 配布を受け取っていないので、clientは、ほかのclientが登録したモデルを保有していない。
        assert (
            len(
                run_client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
            )
            == 2
        )


def test_server_round_synchronization_rejects_invalid_clustering_flag(
    monkeypatch, valid_run_settings_mapping
):
    server_round_oracle = build_server_round_oracle(
        monkeypatch=monkeypatch,
        valid_run_settings_mapping=valid_run_settings_mapping,
        class_count=2,
    )
    synchronization_owners = make_synchronization_owners(server_round_oracle)
    state_snapshot = snapshot_server_and_client_states(server_round_oracle)
    for invalid_flag in (None, 1, "true"):
        with pytest.raises(TypeError, match="model_clustering_enabled"):
            synchronize_models_in_server_round(
                **synchronization_owners, round_index=0, model_clustering_enabled=invalid_flag
            )
        assert_server_and_client_states_unchanged(
            state_snapshot=state_snapshot, server_round_oracle=server_round_oracle
        )
    # 判定の基準が不正なら、登録・集約・クロス評価のどれより前に拒否する（クラスタリングが無効のラウンドでも）。
    for invalid_criteria in (
        None,
        make_subclass_copy(MODEL_CLUSTERING_CRITERIA),
        make_criteria_mutated_around_frozen(MODEL_CLUSTERING_CRITERIA),
    ):
        for model_clustering_enabled in (False, True):
            with pytest.raises((TypeError, ValueError)):
                synchronize_models_in_server_round(
                    **synchronization_owners | dict(model_clustering_criteria=invalid_criteria),
                    round_index=0,
                    model_clustering_enabled=model_clustering_enabled,
                )
            assert_server_and_client_states_unchanged(
                state_snapshot=state_snapshot, server_round_oracle=server_round_oracle
            )


@pytest.mark.parametrize("evaluated_sample_count", [4, 5])
def test_pair_is_evaluated_only_with_minimum_sample_count_in_all_four_loss_sums(
    evaluated_sample_count, monkeypatch, valid_run_settings_mapping
):
    """対は、4つの損失の統計の件数が、すべて下限（5件）以上のときだけ、距離を持つ（下限ちょうどは持つ）。

    クロス評価の結果の、1つの組の件数だけを、下限の前後へ書き換えて、実旧のクラスタリングと照合する。
    """
    cross_evaluated_state = build_cross_evaluated_server_round_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    legacy_server = cross_evaluated_state["server_round_oracle"]["legacy_server"]
    arguments = get_clustering_and_consolidation_arguments(cross_evaluated_state)
    model_cross_evaluation = arguments["model_cross_evaluation"]
    # モデル2を、モデル1の標本で評価した組の件数を書き換える（和は、そのままにする）。
    replaced_loss_sums = replace(
        model_cross_evaluation.loss_sums_by_candidate_and_target_model_id[(2, 1)],
        evaluated_sample_count=evaluated_sample_count,
    )
    legacy_statistics_matrix = {
        candidate_model_id: dict(target_statistics)
        for candidate_model_id, target_statistics in cross_evaluated_state[
            "legacy_statistics_matrix"
        ].items()
    }
    legacy_statistics_matrix[2][1] = (
        evaluated_sample_count,
        *legacy_statistics_matrix[2][1][1:],
    )
    legacy_server.perform_hierarchical_clustering(
        cross_evaluated_state["legacy_active_model_ids"], legacy_statistics_matrix
    )
    # 新は、閾値を、どの対も同じクラスタにならない値にして、観測だけを読む。
    cluster_and_consolidate_global_models(
        **arguments
        | dict(
            model_cross_evaluation=replace(
                model_cross_evaluation,
                loss_sums_by_candidate_and_target_model_id=(
                    model_cross_evaluation.loss_sums_by_candidate_and_target_model_id
                    | {(2, 1): replaced_loss_sums}
                ),
            ),
            model_clustering_criteria=replace(
                MODEL_CLUSTERING_CRITERIA, maximum_same_cluster_decision_score=-1.0
            ),
        )
    )
    pair_observations = arguments[
        "model_clustering_record_store"
    ].snapshot_pair_clustering_observations()
    assert {
        (pair_observation.lower_model_id, pair_observation.higher_model_id): (
            pair_observation.loss_increase_distance
        )
        for pair_observation in pair_observations
    } == legacy_server._last_pair_distances
    assert ((1, 2) in legacy_server._last_pair_distances) == (evaluated_sample_count >= 5)
    # ほかの対（1と3、2と3）は、どちらの件数でも、距離を持つ。
    assert {(1, 3), (2, 3)} <= set(legacy_server._last_pair_distances)
