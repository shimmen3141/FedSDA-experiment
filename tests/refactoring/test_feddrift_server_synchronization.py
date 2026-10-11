"""FedDrift同期を実旧のclient/serverの連続操作へ照合する。"""

import random
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from federated_drift_experiment import config
from federated_drift_experiment.clients.feddrift import FedDriftClient
from federated_drift_experiment.servers.feddrift import FedDriftServer
from federated_learning_experiments.data.observed_streams import ObservedSample
from federated_learning_experiments.evaluation.model_evaluation_sample_records import (
    ObservedEvaluationSample,
)
from federated_learning_experiments.learning.loss_statistics.bounded_loss_moments import (
    BoundedLossMoments,
)
from federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics import (
    ModelAndClassLossStatistics,
)
from federated_learning_experiments.learning.models.classifier_parameter_snapshot import (
    snapshot_classifier_parameters,
)
from federated_learning_experiments.learning.training.model_training_sample_records import (
    ObservedTrainingSample,
)
from federated_learning_experiments.methods.feddrift.loss_distance_model_clustering import (
    cluster_feddrift_model_ids,
)
from federated_learning_experiments.runtime.feddrift_run_client import assemble_feddrift_run_client
from federated_learning_experiments.runtime.feddrift_run_server import FeddriftRunServer
from federated_learning_experiments.runtime.feddrift_server_settings import FeddriftServerSettings
from tests.refactoring.test_feddrift_client_round_training import assert_values_equal
from tests.refactoring.test_feddrift_run_client import (
    assert_processing_matches_legacy,
    assert_rng_equal,
    build_assembly_inputs,
    processing_owners,
    rng_snapshot,
)


def run_pair(old_operation, new_operation, clients):
    """3RNGを同じ開始点に戻し、旧global Pythonと新run Randomを対応付ける。"""
    before = rng_snapshot()
    for client in clients:
        client._random.setstate(before[0])
    old_result = old_operation()
    expected_rng = rng_snapshot()
    random.setstate(before[0])
    np.random.set_state(before[1])
    torch.set_rng_state(before[2])
    new_result = new_operation()
    assert_rng_equal((before[0], expected_rng[1], expected_rng[2]))
    for client in clients:
        assert client._random.getstate() == expected_rng[0]
    # 次の対照の開始点を旧の終了点とする。
    random.setstate(expected_rng[0])
    return old_result, new_result


def assert_client_equal(old, new):
    assert_processing_matches_legacy(
        legacy=old, processor=new.detection_batch_processor, owners=processing_owners(new)
    )
    for model_id, legacy in old.models.items():
        state = new.owners.held_model_training_state_registry.get_held_model_training_state(
            model_id=model_id
        )
        for old_parameter, new_parameter in zip(
            legacy.parameters(), state.classifier.parameters(), strict=True
        ):
            assert torch.equal(old_parameter, new_parameter)
            if old_parameter.grad is None:
                assert new_parameter.grad is None
            else:
                assert torch.equal(old_parameter.grad, new_parameter.grad)
        assert_values_equal(
            legacy.optimizer.state_dict(),
            state.concept_specific_parameter_optimizer_state.parameter_optimizer.state_dict(),
        )
    snapshot = new.owners.model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts()
    assert {k: v for k, v in old.model_training_examples.items() if v} == {
        k: v for k, v in snapshot.trained_sample_counts_by_model_id.items() if v
    }
    assert {k: v for k, v in old.model_optimizer_steps.items() if v} == {
        k: v for k, v in snapshot.parameter_update_step_counts_by_model_id.items() if v
    }
    assert dict(old.model_concept_counts) == snapshot.assigned_sample_counts_by_model_and_concept_id
    assert (
        tuple(old.mapping_change_positions)
        == new.owners.adaptation_record_store.get_state_snapshot().server_remapped_sample_indices
    )


def distribution_pair(monkeypatch, class_count, optimizer_name):
    old, inputs = build_assembly_inputs(
        monkeypatch=monkeypatch, class_count=class_count, optimizer_name=optimizer_name
    )
    new = assemble_feddrift_run_client(**inputs)
    # 実旧の既定再構築lrはdatasetの0.01。旧model_cls()も新入力と同じ寸法にする。
    model_class = type(old.models[0])
    old.model_cls = lambda: model_class(dataset="sine2")
    for model_id in (2, -106):
        legacy = deepcopy(old.models[0])
        state = new.owners.held_model_training_state_registry.get_held_model_training_state(
            model_id=0
        )
        classifier, optimizer_state = deepcopy(
            (state.classifier, state.concept_specific_parameter_optimizer_state)
        )
        with torch.no_grad():
            for p in legacy.parameters():
                p.add_(model_id / 100)
            for p in classifier.parameters():
                p.add_(model_id / 100)
        old.models[model_id] = legacy
        new.owners.held_model_training_state_registry.register_held_model_training_state(
            model_id=model_id,
            classifier=classifier,
            concept_specific_parameter_optimizer_state=optimizer_state,
        )
    old.current_model_id = 2
    new.owners.current_training_model_assignment.assign_model_for_training(model_id=2)
    return old, new, inputs["run_client_settings"].new_model_optimizer_settings


def populate_distribution_stores(old, new, statistics_kind):
    for mid in (0, 2):
        samples = tuple(
            (torch.tensor([[i / 10, 0.25, -0.5]]), torch.tensor([[float(i % 2)]]))
            for i in range(12)
        )
        old.train_data_store[mid].extend(samples)
        new.owners.training_sample_store.append_model_training_samples(
            model_id=mid,
            training_samples=tuple(
                ObservedTrainingSample(input_features=x, observed_class_labels=y)
                for x, y in samples
            ),
        )
        for _ in range(2):
            run_pair(
                lambda mid=mid, samples=samples: old._store_evaluation_data(mid, samples),
                lambda mid=mid, samples=samples: (
                    new.owners.evaluation_sample_store.sample_and_append_model_evaluation_samples(
                        model_id=mid,
                        evaluation_samples=tuple(
                            ObservedEvaluationSample(input_features=x, observed_class_labels=y)
                            for x, y in samples
                        ),
                        python_random_generator=new._random,
                    )
                ),
                (new,),
            )
        old.model_training_examples[mid] = 8 + mid
        old.model_optimizer_steps[mid] = 2 + mid
        new.owners.model_training_and_assignment_counts_store.record_completed_model_training(
            model_id=mid,
            trained_sample_count=8 + mid,
            parameter_update_step_count=2 + mid,
        )
        old.model_concept_counts[mid][mid] += 1
        new.owners.model_training_and_assignment_counts_store.record_assigned_sample_concept(
            model_id=mid, observed_concept_id=mid
        )
    count = 3 if statistics_kind == "tie" else 9
    old.model_stats[2] = {"n": count, "mean": 0.5, "M2": 0.125}
    new.owners.loss_statistics_store.set_model_loss_statistics(
        model_id=2,
        loss_statistics=ModelAndClassLossStatistics(
            overall_loss_moments=BoundedLossMoments(
                observed_loss_count=count, mean_loss=0.5, sum_squared_loss_deviations=0.125
            )
        ),
    )
    if statistics_kind == "fallback":
        for mid in (0, 2):
            old.model_stats[mid] = {"n": 0, "mean": 0.0, "M2": 0.0}
            new.owners.loss_statistics_store.set_model_loss_statistics(
                model_id=mid,
                loss_statistics=ModelAndClassLossStatistics(
                    overall_loss_moments=BoundedLossMoments(
                        observed_loss_count=0, mean_loss=0.0, sum_squared_loss_deviations=0.0
                    )
                ),
            )


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("optimizer_name", ("sgd", "adam"))
@pytest.mark.parametrize("merge", (False, True))
@pytest.mark.parametrize("statistics_kind", ("tie", "larger", "fallback"))
def test_distribution_matches_real_apply_server_mapping(
    monkeypatch, class_count, optimizer_name, merge, statistics_kind
):
    old, new, settings = distribution_pair(monkeypatch, class_count, optimizer_name)
    populate_distribution_stores(old, new, statistics_kind)
    mapping = {0: 0, 2: 0} if merge else {}
    kept_ids = (0,) if merge else (2, 0)
    legacy_parameters = {mid: old.models[mid].get_params() for mid in kept_ids}
    new_parameters = {
        mid: snapshot_classifier_parameters(
            classifier=new.owners.held_model_training_state_registry.get_held_model_training_state(
                model_id=mid
            ).classifier
        )
        for mid in kept_ids
    }
    temporary_before = new.owners.held_model_training_state_registry.get_held_model_training_state(
        model_id=-106
    )
    server_statistics = ModelAndClassLossStatistics(
        overall_loss_moments=BoundedLossMoments(
            observed_loss_count=7, mean_loss=0.375, sum_squared_loss_deviations=0.0
        )
    )
    run_pair(
        lambda: old.apply_server_mapping(
            mapping, legacy_parameters, {0: {"n": 7, "mean": 0.375, "M2": 0.0}}
        ),
        lambda: new.apply_server_model_distribution(
            model_id_mapping=mapping,
            model_parameters_by_model_id=new_parameters,
            server_loss_statistics=((0, server_statistics),),
            rebuilt_model_optimizer_settings=settings,
        ),
        (new,),
    )
    assert_client_equal(old, new)
    assert (
        new.owners.held_model_training_state_registry.get_held_model_training_state(model_id=-106)
        is temporary_before
    )
    assert tuple(old.models) == tuple(
        s.model_id
        for s in new.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
    )


@pytest.mark.parametrize(
    "case",
    (
        "bool_id",
        "negative_id",
        "nan",
        "shape",
        "keys",
        "missing_current",
        "bad_mapping",
        "bad_settings",
        "bad_stats",
    ),
)
def test_distribution_rejects_before_owners_or_rng(case, monkeypatch):
    old, new, settings = distribution_pair(monkeypatch, 2, "adam")
    parameters = {
        2: snapshot_classifier_parameters(
            classifier=new.owners.held_model_training_state_registry.get_held_model_training_state(
                model_id=2
            ).classifier
        )
    }
    mapping = {}
    statistics = ()
    if case == "bool_id":
        parameters = {True: next(iter(parameters.values()))}
    elif case == "negative_id":
        parameters = {-1: next(iter(parameters.values()))}
    elif case == "nan":
        next(iter(parameters[2].values())).view(-1)[0] = float("nan")
    elif case == "shape":
        parameters[2][next(iter(parameters[2]))] = torch.ones(1)
    elif case == "keys":
        parameters[2].pop(next(iter(parameters[2])))
    elif case == "missing_current":
        parameters = {0: parameters[2]}
    elif case == "bad_mapping":
        mapping = {2: True}
    elif case == "bad_settings":
        settings = None
    elif case == "bad_stats":
        statistics = ((0, None),)
    before_rng = rng_snapshot()
    random_before = new._random.getstate()
    with pytest.raises((TypeError, ValueError)):
        new.apply_server_model_distribution(
            model_id_mapping=mapping,
            model_parameters_by_model_id=parameters,
            server_loss_statistics=statistics,
            rebuilt_model_optimizer_settings=settings,
        )
    assert_client_equal(old, new)
    assert_rng_equal(before_rng)
    assert new._random.getstate() == random_before


def settings_inputs(optimizer_settings, **overrides):
    result = dict(
        maximum_same_cluster_loss_increase=1.0,
        cluster_linkage="complete",
        minimum_cross_evaluation_sample_count=5,
        maximum_cross_evaluation_client_count=1,
        maximum_client_evaluation_sample_count=7,
        new_model_isolation_timestep_count=2,
        local_training_round_count=2,
        initial_model_aggregation_weight=3,
        rebuilt_model_optimizer_settings=optimizer_settings,
    )
    result.update(overrides)
    return result


def server_pair(
    monkeypatch,
    class_count=2,
    optimizer_name="sgd",
    *,
    evaluation_capacity=5,
    evaluation_add=2,
    **settings_overrides,
):
    old_clients, new_clients = [], []
    shared_random = random.Random()
    for client_id in (0, 1):
        old_reference, inputs = build_assembly_inputs(
            monkeypatch=monkeypatch, class_count=class_count, optimizer_name=optimizer_name
        )
        monkeypatch.setattr(config, "STORED_DATA_LIMIT", evaluation_capacity)
        inputs["run_client_settings"] = replace(
            inputs["run_client_settings"],
            maximum_stored_evaluation_sample_count_per_model=evaluation_capacity,
            added_evaluation_batch_sample_count=evaluation_add,
        )
        old = FedDriftClient(
            client_id,
            old_reference.models,
            initial_stats=old_reference.model_stats,
            distance_threshold=1.0,
            verbose=False,
        )
        model_class = type(old.models[0])
        old.model_cls = lambda model_class=model_class: model_class(dataset="sine2")
        inputs["client_id"] = client_id
        inputs["python_random_generator"] = shared_random
        new = assemble_feddrift_run_client(**inputs)
        old_clients.append(old)
        new_clients.append(new)
    monkeypatch.setattr(config, "EVAL_STORE_SAMPLE_SIZE", evaluation_add)
    options = settings_inputs(
        inputs["run_client_settings"].new_model_optimizer_settings, **settings_overrides
    )
    monkeypatch.setattr(
        config, "CROSS_EVAL_MAX_CLIENTS", options["maximum_cross_evaluation_client_count"]
    )
    monkeypatch.setattr(
        config, "EVAL_MAX_SAMPLES", options["maximum_client_evaluation_sample_count"]
    )
    monkeypatch.setattr(
        config, "CLUSTER_MIN_EVAL_N", options["minimum_cross_evaluation_sample_count"]
    )
    old_server = FedDriftServer(
        distance_threshold=options["maximum_same_cluster_loss_increase"],
        linkage=options["cluster_linkage"],
        isolation_timesteps=options["new_model_isolation_timestep_count"],
        verbose=False,
    )
    old_server.clients = old_clients
    old_server.register_model_params(0, old_clients[0].models[0].get_params())
    old_server.model_weights[0] = options["initial_model_aggregation_weight"]
    old_server.global_stats[0] = deepcopy(old_clients[0].model_stats[0])
    initial_state = new_clients[
        0
    ].owners.held_model_training_state_registry.get_held_model_training_state(model_id=0)
    new_server = FeddriftRunServer(
        run_clients=tuple(new_clients),
        server_settings=FeddriftServerSettings(**options),
        initial_parameter_snapshot=snapshot_classifier_parameters(
            classifier=initial_state.classifier
        ),
        initial_loss_statistics=new_clients[
            0
        ].owners.loss_statistics_store.get_model_loss_statistics(model_id=0),
        python_random_generator=shared_random,
    )
    return old_server, new_server, tuple(new_clients)


def assert_server_equal(old, new):
    repository = new.global_model_repository
    assert tuple(old.global_models) == repository.global_model_ids
    for mid, old_parameters in old.global_models.items():
        parameters = repository.get_global_model_parameters(model_id=mid)
        for old_value, new_value in zip(old_parameters.values(), parameters.values(), strict=True):
            assert torch.equal(old_value, new_value)
    assert old.model_weights == new.snapshot_model_weights()
    assert old.isolated_until == new.snapshot_isolation_until()
    statistics = dict(repository.snapshot_global_model_loss_statistics())
    assert tuple(old.global_stats) == tuple(statistics)
    for mid, value in old.global_stats.items():
        moments = statistics[mid].overall_loss_moments
        assert (value["n"], value["mean"], value["M2"]) == (
            moments.observed_loss_count,
            moments.mean_loss,
            moments.sum_squared_loss_deviations,
        )
    comm = new.communication_volume_record_store.get_state_snapshot()
    for old_name, new_name in (
        ("comm_models_up", "uploaded_model_count"),
        ("comm_models_down", "downloaded_model_count"),
        ("comm_messages_up", "uploaded_message_count"),
        ("comm_messages_down", "downloaded_message_count"),
        ("comm_parameter_values_up", "uploaded_parameter_value_count"),
        ("comm_parameter_values_down", "downloaded_parameter_value_count"),
        ("comm_bytes_up", "uploaded_byte_count"),
        ("comm_bytes_down", "downloaded_byte_count"),
    ):
        assert getattr(old, old_name) == getattr(comm, new_name)
    for old_client, new_client in zip(old.clients, new.run_clients, strict=True):
        assert_client_equal(old_client, new_client)
    pair_fields = {
        "round_index": "round_index",
        "left_model_id": "lower_model_id",
        "right_model_id": "higher_model_id",
        "distance": "loss_increase_distance",
        "decision_score": "decision_score",
        "same_cluster": "assigned_to_same_cluster",
        "oracle_same_concept": "true_concepts_match",
        "personalized_parameter_distance": "concept_specific_parameter_distance",
    }
    model_fields = {
        "round_index": "round_index",
        "model_id": "model_id",
        "nearest_model_id": "nearest_model_id",
        "nearest_distance": "nearest_loss_increase_distance",
        "representative_model_id": "representative_model_id",
        "cluster_size": "cluster_model_count",
        "cluster_max_distance": "maximum_within_cluster_distance",
        "cluster_evaluated_pairs": "evaluated_within_cluster_pair_count",
        "cluster_possible_pairs": "possible_within_cluster_pair_count",
        "participated_in_merge": "merged_with_other_models",
        "absorbed": "absorbed_into_representative",
    }
    for old_records, new_records, fields in (
        (
            old.model_lineage.clustering_pair_observations,
            new.model_clustering_record_store.snapshot_pair_clustering_observations(),
            pair_fields,
        ),
        (
            old.model_lineage.clustering_observations,
            new.model_clustering_record_store.snapshot_model_clustering_observations(),
            model_fields,
        ),
    ):
        for old_record, new_record in zip(old_records, new_records, strict=True):
            for old_name, new_name in fields.items():
                value = getattr(old_record, old_name)
                if old_name == "nearest_model_id" and value == -1:
                    value = None
                if isinstance(value, float) and np.isnan(value):
                    value = None
                assert value == getattr(new_record, new_name)
    for registration, record in zip(
        old.model_lineage.registrations,
        repository.snapshot_model_registration_records(),
        strict=True,
    ):
        assert registration.model_id == record.model_id
        assert (
            None if registration.round_index == -1 else registration.round_index
        ) == record.registered_round_index
        assert (
            None if registration.client_id == -1 else registration.client_id
        ) == record.registering_client_id


@pytest.mark.parametrize("linkage", ("connected", "average", "complete"))
@pytest.mark.parametrize("threshold", (-0.1, 0.0, 0.2, 1.0))
@pytest.mark.parametrize("missing", (False, True))
def test_loss_distance_clustering_matches_real_legacy(linkage, threshold, missing):
    ids = (7, 0, 3)
    distances = {(0, 3): 0.1, (0, 7): 0.3, (3, 7): 0.15}
    if missing:
        distances.pop((0, 7))
    statistics = {(mid, mid): (5, 1.0, 0.25) for mid in ids}
    for (left, right), distance in distances.items():
        statistics[(left, right)] = statistics[(right, left)] = (5, (0.2 + distance) * 5, 1.0)
    old_server = FedDriftServer(distance_threshold=threshold, linkage=linkage, verbose=False)
    matrix = {
        mid: {target: statistics.get((mid, target), (0, 0.0, 0.0)) for target in ids} for mid in ids
    }
    # 旧の実距離計算も呼び、単にcluster_modelsの代役へしない。
    old_server._personalized_parameter_distance = lambda left, right: 0.0
    expected = old_server.perform_hierarchical_clustering(sorted(ids), matrix)
    result = cluster_feddrift_model_ids(
        model_ids=ids,
        loss_sums_by_model_pair=statistics,
        minimum_evaluated_sample_count=5,
        maximum_same_cluster_loss_increase=threshold,
        cluster_linkage=linkage,
    )
    assert result == tuple(tuple(c) for c in expected)


@pytest.mark.parametrize(
    "case",
    (
        "bool_rounds",
        "zero_rounds",
        "nan_distance",
        "wrong_linkage",
        "bool_client_limit",
        "zero_isolation",
        "bad_optimizer",
    ),
)
def test_settings_reject_invalid_values(case, monkeypatch):
    _, inputs = build_assembly_inputs(monkeypatch=monkeypatch)
    kwargs = settings_inputs(inputs["run_client_settings"].new_model_optimizer_settings)
    names_values = {
        "bool_rounds": ("local_training_round_count", True),
        "zero_rounds": ("local_training_round_count", 0),
        "nan_distance": ("maximum_same_cluster_loss_increase", float("nan")),
        "wrong_linkage": ("cluster_linkage", "ward"),
        "bool_client_limit": ("maximum_cross_evaluation_client_count", True),
        "zero_isolation": ("new_model_isolation_timestep_count", 0),
        "bad_optimizer": ("rebuilt_model_optimizer_settings", None),
    }
    name, value = names_values[case]
    kwargs[name] = value
    with pytest.raises((TypeError, ValueError)):
        FeddriftServerSettings(**kwargs)


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("optimizer_name", ("sgd", "adam"))
def test_prepare_empty_and_ready_registration_matches_legacy(
    monkeypatch, class_count, optimizer_name
):
    old, new, clients = server_pair(monkeypatch, class_count, optimizer_name)
    run_pair(
        lambda: old.prepare_timestep(0), lambda: new.prepare_timestep(timestep_index=0), clients
    )
    assert_server_equal(old, new)
    # 実旧の通常観測から即時readyを作る。前回最小値0なら必ずドリフト。
    features = torch.linspace(-1, 1, 9).reshape(3, 3)
    labels = torch.tensor([[0.0], [1.0], [0.0]])
    for old_client, client in zip(old.clients, clients, strict=True):
        old_client.last_min_loss = 0.0
        old_client.distance_threshold = 0.0
        client.detection_batch_processor._previous_loss = 0.0
        client.detection_batch_processor._threshold = 0.0
        observed = tuple(
            ObservedSample(feature_values=tuple(float(v) for v in x), class_label=int(y.item()))
            for x, y in zip(features, labels, strict=True)
        )
        run_pair(
            lambda old_client=old_client: old_client.process_batch(
                tuple(
                    (x.reshape(1, -1), y.reshape(1, 1))
                    for x, y in zip(features, labels, strict=True)
                ),
                (0, 0, 0),
            ),
            lambda client=client, observed=observed: client.process_observed_block(
                observed_samples=observed, evaluation_concept_ids=(0, 0, 0)
            ),
            (client,),
        )
    run_pair(
        lambda: old.prepare_timestep(1), lambda: new.prepare_timestep(timestep_index=1), clients
    )
    assert_server_equal(old, new)
    assert old.next_model_id == new.global_model_repository.next_global_model_id == 3
    assert new.snapshot_isolation_until() == {1: 3, 2: 3}
    # 登録IDのparameterは最初の集約で上がり、全clientへ渡る。隔離解除まで続ける。
    for timestep in (1, 3):
        for _ in range(2):
            run_pair(
                lambda timestep=timestep: (
                    [c.local_train(k_steps=3) for c in old.clients],
                    old.run_training_round(timestep),
                ),
                lambda timestep=timestep: (
                    [c.train_local_round() for c in clients],
                    new.aggregate_and_distribute_training_round(timestep_index=timestep),
                ),
                clients,
            )
            assert_server_equal(old, new)
        run_pair(
            lambda timestep=timestep: old.prepare_timestep(timestep),
            lambda timestep=timestep: new.prepare_timestep(timestep_index=timestep),
            clients,
        )
        assert_server_equal(old, new)


def seed_multiple_cached_models(old, new, clients):
    """本番のprepare入口と同じ全client保有状態を、対応する配布操作で用意する。"""
    for mid in (2, 3):
        old.global_models[mid] = {
            name: value + mid / 100 for name, value in old.global_models[0].items()
        }
        reference = new.global_model_repository.get_global_model_parameters(model_id=0)
        new.global_model_repository.set_global_model_parameters(
            model_id=mid,
            parameter_snapshot={name: value + mid / 100 for name, value in reference.items()},
        )
        old.global_stats[mid] = deepcopy(old.global_stats[0])
        new.global_model_repository.set_global_model_loss_statistics(
            model_id=mid,
            loss_statistics=new.global_model_repository.get_global_model_loss_statistics(
                model_id=0
            ),
        )
        old.model_weights[mid] = new._model_weights[mid] = mid + 5
    old.next_model_id = new.global_model_repository._next_global_model_id = 4
    old.isolated_until[3] = new._isolation_until[3] = 10
    for old_client, client in zip(old.clients, clients, strict=True):
        # serverの保存値とは別のclient自身の値で統合されることを確かめる。
        old_parameters = {
            mid: {name: value + client.client_id / 10 for name, value in params.items()}
            for mid, params in old.global_models.items()
        }
        new_parameters = {
            mid: {
                name: value + client.client_id / 10
                for name, value in new.global_model_repository.get_global_model_parameters(
                    model_id=mid
                ).items()
            }
            for mid in (0, 2, 3)
        }
        run_pair(
            lambda old_client=old_client, old_parameters=old_parameters: (
                old_client.apply_server_mapping({}, old_parameters, old.global_stats)
            ),
            lambda client=client, new_parameters=new_parameters: (
                client.apply_server_model_distribution(
                    model_id_mapping={},
                    model_parameters_by_model_id=new_parameters,
                    server_loss_statistics=new.global_model_repository.snapshot_global_model_loss_statistics(),
                    rebuilt_model_optimizer_settings=new._settings.rebuilt_model_optimizer_settings,
                )
            ),
            (client,),
        )
        old_client.current_model_id = 2
        client.owners.current_training_model_assignment.assign_model_for_training(model_id=2)
        populate_distribution_stores(old_client, client, "larger")


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("optimizer_name", ("sgd", "adam"))
@pytest.mark.parametrize("linkage", ("connected", "average", "complete"))
@pytest.mark.parametrize("threshold", (-1.0, 1.0))
def test_prepare_mature_cached_models_and_isolation_matches_legacy(
    monkeypatch, class_count, optimizer_name, linkage, threshold
):
    old, new, clients = server_pair(
        monkeypatch,
        class_count,
        optimizer_name,
        evaluation_capacity=12,
        evaluation_add=6,
        cluster_linkage=linkage,
        maximum_same_cluster_loss_increase=threshold,
    )
    seed_multiple_cached_models(old, new, clients)
    original_cross_evaluate = old._cross_evaluate
    matrices = []

    def capture_cross_evaluation(*args, **kwargs):
        matrix = original_cross_evaluate(*args, **kwargs)
        matrices.append(matrix)
        return matrix

    old._cross_evaluate = capture_cross_evaluation
    for timestep in (5, 10):
        run_pair(
            lambda timestep=timestep: old.prepare_timestep(timestep),
            lambda timestep=timestep: new.prepare_timestep(timestep_index=timestep),
            clients,
        )
        assert_server_equal(old, new)
        observed = {}
        for record in new.cross_evaluation_record_store.snapshot_client_cross_evaluation_records():
            if record.round_index != timestep:
                continue
            key = (record.candidate_model_id, record.target_model_id)
            count, loss_sum, squares = observed.get(key, (0, 0.0, 0.0))
            observed[key] = (
                count + record.evaluated_sample_count,
                loss_sum + record.bounded_loss_sum,
                squares + record.squared_bounded_loss_sum,
            )
        assert observed == {
            (candidate, target): value
            for candidate, targets in matrices[-1].items()
            for target, value in targets.items()
        }
    assert new.cross_evaluation_record_store.snapshot_client_cross_evaluation_records()
    if threshold > 0:
        assert 2 not in new.global_model_repository.global_model_ids
    else:
        assert 2 in new.global_model_repository.global_model_ids


@pytest.mark.parametrize(
    "case",
    (
        "bool_time",
        "negative_time",
        "bad_settings",
        "duplicate_clients",
        "missing_model",
        "nan_client",
        "bad_weight",
        "bad_expiry",
        "bad_next_id",
        "colliding_next_id",
        "bad_pending",
        "nan_evaluation",
    ),
)
def test_prepare_rejects_bad_operation_before_records_and_rng(monkeypatch, case):
    old, new, clients = server_pair(monkeypatch)
    timestep = 0
    if case == "bool_time":
        timestep = True
    elif case == "negative_time":
        timestep = -1
    elif case == "bad_settings":
        object.__setattr__(new._settings, "local_training_round_count", False)
    elif case == "duplicate_clients":
        new.run_clients = (clients[0], clients[0])
    elif case == "missing_model":
        new.global_model_repository.set_global_model_parameters(
            model_id=7,
            parameter_snapshot=new.global_model_repository.get_global_model_parameters(model_id=0),
        )
    elif case == "bad_weight":
        new._model_weights[0] = True
    elif case == "bad_expiry":
        new._isolation_until[0] = -1
    elif case == "bad_next_id":
        new.global_model_repository._next_global_model_id = True
    elif case == "colliding_next_id":
        new.global_model_repository._next_global_model_id = 0
    elif case == "bad_pending":
        clients[0].owners.pending_model_upload_state.queue_ready_model_upload(
            model_id=0,
            parameter_snapshot=new.global_model_repository.get_global_model_parameters(model_id=0),
        )
    elif case == "nan_evaluation":
        clients[0].owners.evaluation_sample_store._evaluation_samples_by_model_id[0] = [
            ObservedEvaluationSample(
                input_features=torch.full((1, 3), float("nan")),
                observed_class_labels=torch.zeros((1, 1)),
            )
        ]
    else:
        with torch.no_grad():
            next(
                iter(
                    clients[0]
                    .owners.held_model_training_state_registry.get_held_model_training_state(
                        model_id=0
                    )
                    .classifier.parameters()
                )
            ).fill_(float("nan"))
    before = rng_snapshot()
    borrowed_before = clients[0]._random.getstate()
    comm = new.communication_volume_record_store.get_state_snapshot()
    registrations = new.global_model_repository.snapshot_model_registration_records()
    with pytest.raises((TypeError, ValueError)):
        new.prepare_timestep(timestep_index=timestep)
    assert_rng_equal(before)
    assert clients[0]._random.getstate() == borrowed_before
    assert comm == new.communication_volume_record_store.get_state_snapshot()
    assert registrations == new.global_model_repository.snapshot_model_registration_records()


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("optimizer_name", ("sgd", "adam"))
@pytest.mark.parametrize("round_count", (1, 3))
@pytest.mark.parametrize("contribution_pattern", ("zero", "both", "single", "unequal"))
def test_fedavg_rounds_match_real_legacy(
    monkeypatch, class_count, optimizer_name, round_count, contribution_pattern
):
    old, new, clients = server_pair(
        monkeypatch, class_count, optimizer_name, evaluation_capacity=12, evaluation_add=6
    )
    seed_multiple_cached_models(old, new, clients)
    if contribution_pattern == "zero":
        for old_client, client in zip(old.clients, clients, strict=True):
            old_client.train_data_store.clear()
            client.owners.training_sample_store._training_samples_by_model_id.clear()
    elif contribution_pattern in ("single", "unequal"):
        kept_count = 0 if contribution_pattern == "single" else 7
        for mid in (0, 2):
            old.clients[1].train_data_store[mid] = old.clients[1].train_data_store[mid][:kept_count]
            clients[1].owners.training_sample_store._training_samples_by_model_id[mid] = clients[
                1
            ].owners.training_sample_store._training_samples_by_model_id[mid][:kept_count]
    for _ in range(round_count):

        def old_round():
            for client in old.clients:
                client.local_train(k_steps=3)
            old.run_training_round(5)

        def new_round():
            for client in clients:
                client.train_local_round()
            new.aggregate_and_distribute_training_round(timestep_index=5)

        run_pair(old_round, new_round, clients)
        assert_server_equal(old, new)


@pytest.mark.parametrize("class_count", (2, 4))
@pytest.mark.parametrize("optimizer_name", ("sgd", "adam"))
@pytest.mark.parametrize("round_count", (1, 3))
def test_synchronize_matches_prepare_and_exact_r_rounds(
    monkeypatch, class_count, optimizer_name, round_count
):
    old, new, clients = server_pair(
        monkeypatch,
        class_count,
        optimizer_name,
        evaluation_capacity=12,
        evaluation_add=6,
        local_training_round_count=round_count,
    )
    seed_multiple_cached_models(old, new, clients)
    processed_before = tuple(client._stream_processor.processed_sample_count for client in clients)

    def old_timestep():
        old.prepare_timestep(5)
        for _ in range(round_count):
            for client in old.clients:
                client.local_train(k_steps=3)
            old.run_training_round(5)

    run_pair(old_timestep, lambda: new.synchronize_timestep(timestep_index=5), clients)
    assert_server_equal(old, new)
    assert (
        tuple(client._stream_processor.processed_sample_count for client in clients)
        == processed_before
    )


@pytest.mark.parametrize(
    "operation", ("aggregate_and_distribute_training_round", "synchronize_timestep")
)
def test_round_and_synchronize_validate_before_learning_or_rng(monkeypatch, operation):
    old, new, clients = server_pair(monkeypatch)
    before = rng_snapshot()
    comm = new.communication_volume_record_store.get_state_snapshot()
    with pytest.raises((TypeError, ValueError)):
        getattr(new, operation)(timestep_index=True)
    assert_rng_equal(before)
    assert new.communication_volume_record_store.get_state_snapshot() == comm
    assert_server_equal(old, new)


@pytest.mark.parametrize("broken_model_id", (-106, 0, 2))
def test_distribution_rejects_corrupt_held_optimizer_before_any_owner_or_rng(
    monkeypatch, broken_model_id
):
    old, client, settings = distribution_pair(monkeypatch, 2, "sgd")
    populate_distribution_stores(old, client, "larger")
    owners = client.owners
    registry = owners.held_model_training_state_registry
    parameters = {
        0: snapshot_classifier_parameters(
            classifier=registry.get_held_model_training_state(model_id=0).classifier
        )
    }
    broken = registry.get_held_model_training_state(model_id=broken_model_id)
    optimizer = broken.concept_specific_parameter_optimizer_state.parameter_optimizer
    optimizer.param_groups[0]["params"][0] = torch.nn.Parameter(
        optimizer.param_groups[0]["params"][0].detach().clone()
    )
    rng_before = rng_snapshot()
    borrowed_before = client.python_random_generator.getstate()
    statistics_before = owners.loss_statistics_store.get_state_snapshot()
    evaluation_before = owners.evaluation_sample_store.snapshot_ordered_model_evaluation_samples()
    training_before = owners.training_sample_store.snapshot_ordered_model_training_samples()
    counts_before = owners.model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts()
    adaptation_before = owners.adaptation_record_store.get_state_snapshot()
    held_before = registry.snapshot_ordered_held_model_training_states()
    with pytest.raises(ValueError):
        client.apply_server_model_distribution(
            model_id_mapping={0: 0, 2: 0},
            model_parameters_by_model_id=parameters,
            server_loss_statistics=(),
            rebuilt_model_optimizer_settings=settings,
        )
    assert_rng_equal(rng_before)
    assert client.python_random_generator.getstate() == borrowed_before
    assert owners.loss_statistics_store.get_state_snapshot() == statistics_before
    assert (
        owners.evaluation_sample_store.snapshot_ordered_model_evaluation_samples()
        == evaluation_before
    )
    assert owners.training_sample_store.snapshot_ordered_model_training_samples() == training_before
    assert (
        owners.model_training_and_assignment_counts_store.snapshot_model_training_and_assignment_counts()
        == counts_before
    )
    assert owners.adaptation_record_store.get_state_snapshot() == adaptation_before
    assert registry.snapshot_ordered_held_model_training_states() == held_before
    assert owners.current_training_model_assignment.current_training_model_id == 2
