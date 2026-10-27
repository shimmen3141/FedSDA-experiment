"""全参加者の準備を実旧setupに照合し、観測・同期へ接続する。"""

import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from federated_drift_experiment import config, experiment
from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.data.observed_sample_generation import (
    create_observed_sample_generator,
)
from federated_learning_experiments.execution.run_random_sources import create_run_random_sources
from federated_learning_experiments.learning.training.initial_model_pretraining_settings import (
    InitialModelPretrainingSettings,
)
from federated_learning_experiments.learning.training.parameter_optimizer_settings import (
    AdamParameterOptimizerSettings,
    SgdParameterOptimizerSettings,
)
from federated_learning_experiments.runtime.feddrift_run_client_settings import (
    FeddriftRunClientSettings,
)
from federated_learning_experiments.runtime.feddrift_run_participant_factory import (
    FeddriftRunParticipantFactory,
    FeddriftRunParticipants,
    FeddriftRunParticipantSettings,
)
from federated_learning_experiments.runtime.feddrift_server_settings import FeddriftServerSettings
from tests.refactoring.test_feddrift_server_synchronization import assert_server_equal, run_pair
from tests.refactoring.test_initial_model_pretraining import assert_numpy_random_states_equal


def make_inputs(monkeypatch, dataset="sine2", optimizer="sgd", epochs=1, clients=2):
    """モデル寸法だけ縮小し、実旧の生成/事前学習/setupをそのまま使う。"""
    original_spec = config.dataset_spec
    monkeypatch.setattr(
        config,
        "dataset_spec",
        lambda dataset=None: replace(original_spec(dataset), hidden_dims=(7, 5)),
    )
    for key, value in dict(
        DATASET=dataset,
        PRETRAIN_SAMPLES=7,
        PRETRAIN_EPOCHS=epochs,
        PRETRAIN_BATCH_SIZE=3,
        N_CLIENTS=clients,
        BASE_LR=0.01,
        OPTIMIZER=optimizer,
        WEIGHT_DECAY=0.0,
        AMSGRAD=False,
        NEW_MODEL_LR=0.02,
        NEW_MODEL_TRAINING="fixed",
        CLIENT_BATCH_SIZE=3,
        STORED_DATA_LIMIT=5,
        EVAL_STORE_SAMPLE_SIZE=2,
        FEDDRIFT_DETECTION_BATCH_SIZE=3,
        UPDATES_PER_SAMPLE=2,
        FEDDRIFT_CLUSTER_LINKAGE="complete",
        FEDDRIFT_ISOLATION_TIMESTEPS=2,
        FEDDRIFT_ROUNDS=2,
        CLUSTER_MIN_EVAL_N=5,
        CROSS_EVAL_MAX_CLIENTS=1,
        EVAL_MAX_SAMPLES=7,
    ).items():
        monkeypatch.setattr(config, key, value)
    initial_lr = config.dataset_spec().learning_rate or config.BASE_LR

    def optimizer_settings(lr):
        return (
            SgdParameterOptimizerSettings(learning_rate=lr)
            if optimizer == "sgd"
            else AdamParameterOptimizerSettings(
                learning_rate=lr, weight_decay=0.0, adam_variant="standard"
            )
        )

    settings = FeddriftRunParticipantSettings(
        run_client_settings=FeddriftRunClientSettings(
            model_comparison_criterion="bounded_loss",
            new_model_parameter_source="current_training_model",
            detection_reference_after_drift="new_model_batch_value",
            loss_increase_threshold=1.0,
            detection_batch_sample_count=3,
            local_training_batch_sample_count=3,
            updates_per_sample=2,
            maximum_stored_evaluation_sample_count_per_model=5,
            added_evaluation_batch_sample_count=2,
            new_model_optimizer_settings=optimizer_settings(0.02),
        ),
        server_settings=FeddriftServerSettings(
            model_comparison_criterion="bounded_loss",
            training_model_eligibility="held_models_with_training_samples",
            consolidation_weight_timing="weights_before_current_timestep",
            maximum_same_cluster_loss_increase=1.0,
            cluster_linkage="complete",
            minimum_cross_evaluation_sample_count=5,
            maximum_cross_evaluation_client_count=1,
            maximum_client_evaluation_sample_count=7,
            new_model_isolation_timestep_count=2,
            local_training_round_count=2,
            initial_model_aggregation_weight=7,
            rebuilt_model_optimizer_settings=optimizer_settings(initial_lr),
        ),
        initial_model_pretraining_settings=InitialModelPretrainingSettings(
            pretraining_sample_count=7,
            pretraining_epoch_count=epochs,
            pretraining_batch_sample_count=3,
        ),
        hidden_layer_widths=(7, 5),
    )
    sources = create_run_random_sources(random_seed=37)
    return settings, dict(
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name=dataset,
            random_seed=37,
            client_count=clients,
            per_client_sample_count=12,
            server_aggregation_interval_per_client_samples=3,
        ),
        run_random_sources=sources,
        sample_generator=create_observed_sample_generator(
            dataset_name=dataset,
            numpy_random_generator=sources.numpy_random_generator,
        ),
    )


def prepare_pair(monkeypatch, dataset="sine2", optimizer="sgd", epochs=1, clients=2):
    settings, inputs = make_inputs(monkeypatch, dataset, optimizer, epochs, clients)
    random.seed(37)
    np.random.seed(37)
    torch.manual_seed(37)
    old, old_clients = experiment._setup_server_and_clients(
        experiment.MODE_SPECS["FedDrift"], 1.0, False
    )
    assert old.clients == old_clients
    expected = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    torch.manual_seed(37)
    factory = FeddriftRunParticipantFactory(run_participant_settings=settings)
    assert factory.prepared_run_participants is None
    result = factory.prepare_run(**inputs)
    assert type(result) is FeddriftRunParticipants
    assert factory.prepared_run_participants is result
    assert random.getstate() == expected[0]
    assert_numpy_random_states_equal(np.random.get_state(), expected[1])
    assert inputs["run_random_sources"].python_random_generator.getstate() == expected[0]
    assert_numpy_random_states_equal(
        inputs["run_random_sources"].numpy_random_generator.get_state(), expected[1]
    )
    assert torch.equal(torch.get_rng_state(), expected[2])
    assert_server_equal(old, result.run_server)
    return old, result, factory, inputs


@pytest.mark.parametrize("dataset", ("sine2", "sea4", "mnist2"))
@pytest.mark.parametrize("optimizer", ("sgd", "adam"))
@pytest.mark.parametrize("epochs", (0, 2))
@pytest.mark.parametrize("clients", (1, 3))
def test_prepare_matches_real_setup_and_isolates_state(
    monkeypatch, dataset, optimizer, epochs, clients
):
    _, result, _, inputs = prepare_pair(monkeypatch, dataset, optimizer, epochs, clients)
    assert result.run_server.run_clients is result.run_clients
    owner_ids = set()
    parameter_ids = set()
    optimizer_ids = set()
    for client in result.run_clients:
        assert (
            client.python_random_generator is inputs["run_random_sources"].python_random_generator
        )
        for owner in vars(client.owners).values():
            assert id(owner) not in owner_ids
            owner_ids.add(id(owner))
        state = client.owners.held_model_training_state_registry.get_held_model_training_state(
            model_id=0
        )
        for parameter in state.classifier.parameters():
            assert id(parameter) not in parameter_ids
            parameter_ids.add(id(parameter))
        optimizer_id = id(state.concept_specific_parameter_optimizer_state.parameter_optimizer)
        assert optimizer_id not in optimizer_ids
        optimizer_ids.add(optimizer_id)
    client = result.run_clients[0]
    classifier = client.owners.held_model_training_state_registry.get_held_model_training_state(
        model_id=0
    ).classifier
    before = result.run_server.global_model_repository.get_global_model_parameters(model_id=0)
    with torch.no_grad():
        next(classifier.parameters()).add_(10)
    after = result.run_server.global_model_repository.get_global_model_parameters(model_id=0)
    assert all(torch.equal(before[k], after[k]) for k in before)


@pytest.mark.parametrize(
    "invalid",
    (
        "settings_type",
        "width_bool",
        "width_empty",
        "threshold",
        "weight",
        "condition_bool",
        "interval",
        "sources_type",
        "python_source",
        "numpy_source",
        "generator_type",
        "generator_dataset",
        "generator_source",
        "disabled_grad",
    ),
)
def test_prepare_rejects_before_rng_and_preserves_previous(monkeypatch, invalid):
    _, result, factory, inputs = prepare_pair(monkeypatch)
    settings = factory._settings
    if invalid == "settings_type":
        factory._settings = None
    elif invalid == "width_bool":
        object.__setattr__(settings, "hidden_layer_widths", (True,))
    elif invalid == "width_empty":
        object.__setattr__(settings, "hidden_layer_widths", ())
    elif invalid == "threshold":
        object.__setattr__(settings.server_settings, "maximum_same_cluster_loss_increase", 0.0)
    elif invalid == "weight":
        object.__setattr__(settings.server_settings, "initial_model_aggregation_weight", 0)
    elif invalid == "condition_bool":
        object.__setattr__(inputs["experiment_run_conditions"], "client_count", True)
    elif invalid == "interval":
        inputs["experiment_run_conditions"] = replace(
            inputs["experiment_run_conditions"], server_aggregation_interval_per_client_samples=2
        )
    elif invalid == "sources_type":
        inputs["run_random_sources"] = None
    elif invalid == "python_source":
        inputs["run_random_sources"].python_random_generator = None
    elif invalid == "numpy_source":
        inputs["run_random_sources"].numpy_random_generator = None
    elif invalid == "generator_type":
        inputs["sample_generator"] = None
    elif invalid == "generator_dataset":
        inputs["sample_generator"] = create_observed_sample_generator(
            dataset_name="sea2",
            numpy_random_generator=inputs["run_random_sources"].numpy_random_generator,
        )
    elif invalid == "generator_source":
        inputs["sample_generator"].numpy_random_generator = np.random.RandomState(37)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    borrowed = result.run_clients[0].python_random_generator
    borrowed_before = borrowed.getstate()
    generator = inputs["sample_generator"]
    generator_rng = getattr(generator, "numpy_random_generator", None)
    generator_before = generator_rng.get_state() if generator_rng is not None else None
    with torch.set_grad_enabled(invalid != "disabled_grad"):
        with pytest.raises((TypeError, ValueError)):
            factory.prepare_run(**inputs)
    assert factory.prepared_run_participants is result
    assert random.getstate() == before[0]
    assert_numpy_random_states_equal(np.random.get_state(), before[1])
    assert torch.equal(torch.get_rng_state(), before[2])
    assert borrowed.getstate() == borrowed_before
    if generator_before is not None:
        assert_numpy_random_states_equal(generator_rng.get_state(), generator_before)


def test_repeated_preparation_has_new_owners_and_validation_is_read_only(monkeypatch):
    _, first, factory, inputs = prepare_pair(monkeypatch)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    factory.validate_configuration()
    assert factory.prepared_run_participants is first
    assert random.getstate() == before[0]
    assert_numpy_random_states_equal(np.random.get_state(), before[1])
    assert torch.equal(torch.get_rng_state(), before[2])
    second = factory.prepare_run(**inputs)
    assert second is factory.prepared_run_participants and second is not first
    assert second.run_server is not first.run_server
    assert all(
        a.owners is not b.owners for a, b in zip(first.run_clients, second.run_clients, strict=True)
    )


@pytest.mark.parametrize("dataset", ("sine2", "mnist2"))
@pytest.mark.parametrize("optimizer", ("sgd", "adam"))
def test_factory_output_connects_to_observation_and_exact_r_sync(monkeypatch, dataset, optimizer):
    old, new, _, inputs = prepare_pair(monkeypatch, dataset, optimizer)
    generator = inputs["sample_generator"]
    for t in range(3):
        blocks = [
            tuple(generator.generate_sample(concept_id=t % 2) for _ in range(3))
            for _ in new.run_clients
        ]

        def old_step(t=t, blocks=blocks):
            data = [
                [
                    (torch.tensor(s.feature_values), torch.tensor([float(s.class_label)]))
                    for s in block
                ]
                for block in blocks
            ]
            experiment._run_feddrift_timestep(
                old.clients, old, data, [[t % 2] * 3 for _ in blocks], t, True, False
            )

        def new_step(t=t, blocks=blocks):
            for client, block in zip(new.run_clients, blocks, strict=True):
                client.process_observed_block(
                    observed_samples=block, evaluation_concept_ids=(t % 2,) * 3
                )
            new.run_server.synchronize_timestep(timestep_index=t)

        run_pair(old_step, new_step, new.run_clients)
        assert_server_equal(old, new.run_server)
