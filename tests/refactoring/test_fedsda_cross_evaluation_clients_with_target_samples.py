"""対象の標本を返せるclientの選択と、評価に使う標本の規則を守る。"""

import random
from dataclasses import replace

import pytest
import torch
from behavior_golden_cases import build_case_settings
from test_model_cross_evaluation import (
    build_cross_evaluation_server_round_oracle,
    get_cross_evaluation_arguments,
)
from test_model_cross_evaluation import (
    valid_run_settings_mapping as valid_run_settings_mapping,
)

from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.runtime.client_model_cross_evaluation import (
    target_model_cross_evaluation_samples_are_available,
)
from federated_learning_experiments.runtime.model_cross_evaluation import (
    cross_evaluate_global_models,
)
from federated_learning_experiments.runtime.server_round_synchronization import (
    synchronize_models_in_server_round,
)
from federated_learning_experiments.sweep.experiment_configurations import (
    EXPERIMENT_CONFIGURATION_NAMES,
    build_experiment_run_settings,
)

OLD = "clients_holding_target_model"
NEW = "clients_with_target_model_samples"


@pytest.mark.parametrize(
    "configuration_name",
    [name for name in EXPERIMENT_CONFIGURATION_NAMES if name.startswith("fedsda")],
)
def test_all_fedsda_configurations_choose_expected_clients(configuration_name):
    conditions = build_case_settings("fedsda", 11).execution_settings.experiment_run_conditions
    settings = build_experiment_run_settings(
        configuration_name=configuration_name, experiment_run_conditions=conditions
    )
    assert settings.run_participant_settings.cross_evaluation_client_selection_policy == (NEW)


@pytest.mark.parametrize("invalid", [None, True, "samples"])
def test_invalid_policy_is_rejected_before_synchronization(invalid):
    settings = build_case_settings("fedsda", 11).run_participant_settings
    with pytest.raises(RunSettingsValidationError) as caught:
        replace(settings, cross_evaluation_client_selection_policy=invalid)
    assert caught.value.configuration_parameter_name == "cross_evaluation_client_selection_policy"
    # 他の引数を読む前に拒否するので、ownerを用意しなくても検査の順が分かる。
    import inspect

    arguments = {
        name: None for name in inspect.signature(synchronize_models_in_server_round).parameters
    }
    arguments.update(
        model_clustering_enabled=True, cross_evaluation_client_selection_policy=invalid
    )
    with pytest.raises((TypeError, ValueError), match="cross_evaluation_client_selection_policy"):
        synchronize_models_in_server_round(**arguments)


def test_selection_preserves_order_sampling_and_excludes_missing_models(
    monkeypatch, valid_run_settings_mapping
):
    oracle = build_cross_evaluation_server_round_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    arguments = get_cross_evaluation_arguments(oracle)
    arguments["run_clients"] = tuple(reversed(arguments["run_clients"]))
    clients = arguments["run_clients"]
    for client in clients:
        client._run_client_settings = replace(
            client._run_client_settings,
            scalar_settings=replace(
                client._run_client_settings.scalar_settings,
                maximum_cross_evaluation_sample_count=100000,
            ),
        )
    # 同じclientの実体を使い、順序と上限だけを変えてサーバの選択を確かめる。
    for limit in (1, len(clients)):
        state = arguments["python_random_generator"].getstate()
        expected_random = random.Random()
        expected_random.setstate(state)
        expected = []
        model_ids = arguments["cross_evaluated_model_ids"]
        for candidate in model_ids:
            for target in model_ids:
                eligible = [
                    c for c in clients if target in c.get_model_ids_with_cross_evaluation_samples()
                ]
                if len(eligible) > limit:
                    eligible = expected_random.sample(eligible, limit)
                expected.extend((candidate, target, c.client_id) for c in eligible)
        result_arguments = dict(arguments, maximum_evaluating_client_count_per_model=limit)
        cross_evaluate_global_models(
            **result_arguments, cross_evaluation_client_selection_policy=NEW
        )
        records = result_arguments[
            "cross_evaluation_record_store"
        ].snapshot_client_cross_evaluation_records()
        assert [(r.candidate_model_id, r.target_model_id, r.client_id) for r in records] == expected
        assert records and all(r.evaluated_sample_count > 0 for r in records)
        assert arguments["python_random_generator"].getstate() == expected_random.getstate()
        arguments = dict(get_cross_evaluation_arguments(oracle), run_clients=clients)
    old_arguments = get_cross_evaluation_arguments(oracle)
    cross_evaluate_global_models(**old_arguments, cross_evaluation_client_selection_policy=OLD)
    assert any(
        r.evaluated_sample_count == 0
        for r in old_arguments[
            "cross_evaluation_record_store"
        ].snapshot_client_cross_evaluation_records()
    )

    client = next(c for c in clients if c.get_model_ids_with_cross_evaluation_samples())
    sample_holding_target = next(iter(client.get_model_ids_with_cross_evaluation_samples()))
    registry = client.owners.held_model_training_state_registry
    registry.replace_held_model_training_states(held_model_training_states=())
    assert target_model_cross_evaluation_samples_are_available(
        target_model_id=sample_holding_target,
        model_evaluation_sample_store=client.owners.model_evaluation_sample_store,
        training_sample_store=client.owners.training_sample_store,
        current_training_model_assignment=client.owners.current_training_model_assignment,
    )
    assert client.get_model_ids_with_cross_evaluation_samples() == frozenset()
    empty_arguments = get_cross_evaluation_arguments(oracle)
    empty_arguments.update(run_clients=(client,))
    state = empty_arguments["python_random_generator"].getstate()
    volume_before = empty_arguments["communication_volume_record_store"].get_state_snapshot()
    cross_evaluate_global_models(**empty_arguments, cross_evaluation_client_selection_policy=NEW)
    assert (
        empty_arguments["cross_evaluation_record_store"].snapshot_client_cross_evaluation_records()
        == ()
    )
    assert empty_arguments["python_random_generator"].getstate() == state
    assert (
        empty_arguments["communication_volume_record_store"].get_state_snapshot() == volume_before
    )


def test_availability_matches_real_evaluation_at_sample_boundaries(
    monkeypatch, valid_run_settings_mapping
):
    oracle = build_cross_evaluation_server_round_oracle(
        monkeypatch=monkeypatch, valid_run_settings_mapping=valid_run_settings_mapping
    )
    client = oracle["run_clients"][0]
    owners = client.owners
    target = owners.current_training_model_assignment.current_training_model_id
    parameters = oracle["global_model_repository"].get_global_model_parameters(model_id=target)
    evaluation_store = owners.model_evaluation_sample_store
    training_store = owners.training_sample_store
    # ownerの実体に標本のsnapshotだけを差し替え、評価標本優先・帰属条件・境界を実評価と比べる。
    from types import SimpleNamespace

    sample = next(
        s
        for collection in training_store.snapshot_ordered_model_training_samples()
        for s in collection.training_samples
    )
    for evaluation_count, training_count, is_current in (
        (5, 10, True),
        (6, 0, False),
        (5, 11, True),
        (5, 11, False),
        (6, 11, True),
    ):
        owners.current_training_model_assignment.assign_model_for_training(
            model_id=target
            if is_current
            else next(
                s.model_id
                for s in owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
                if s.model_id != target
            )
        )
        monkeypatch.setattr(
            type(evaluation_store),
            "snapshot_ordered_model_evaluation_samples",
            lambda self, count=evaluation_count: (
                SimpleNamespace(model_id=target, evaluation_samples=(sample,) * count),
            ),
        )
        monkeypatch.setattr(
            type(training_store),
            "snapshot_ordered_model_training_samples",
            lambda self, count=training_count: (
                SimpleNamespace(model_id=target, training_samples=(sample,) * count),
            ),
        )
        state = oracle["python_random_generator"].getstate()
        torch_state = torch.get_rng_state().clone()
        available = target_model_cross_evaluation_samples_are_available(
            target_model_id=target,
            model_evaluation_sample_store=evaluation_store,
            training_sample_store=training_store,
            current_training_model_assignment=owners.current_training_model_assignment,
        )
        assert oracle["python_random_generator"].getstate() == state
        assert torch.equal(torch.get_rng_state(), torch_state)
        result = client.evaluate_candidate_model_on_target_model_samples(
            candidate_parameter_snapshot=parameters,
            target_model_id=target,
            compare_correctness_with_held_target_model=True,
        )
        assert available == (result.evaluated_sample_count > 0)
        assert available == (evaluation_count >= 6 or (is_current and training_count >= 11))
        if available:
            assert result.evaluated_sample_count == (
                evaluation_count if evaluation_count >= 6 else training_count
            )
