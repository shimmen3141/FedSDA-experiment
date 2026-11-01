"""WindowとAdaptive-FedAvgの公式参照・設定・実体の窓と学習率を守る。"""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.methods.feddrift.adaptive_fedavg_learning_rate import (
    AdaptiveFedavgLearningRateSettings,
    AdaptiveFedavgLearningRateState,
)
from federated_learning_experiments.runtime.feddrift_experiment_run import (
    execute_feddrift_experiment_run,
)
from federated_learning_experiments.runtime.feddrift_run_client import FeddriftRunClient
from federated_learning_experiments.runtime.feddrift_run_record_tables import (
    build_feddrift_run_record_tables,
)
from federated_learning_experiments.runtime.feddrift_run_server import FeddriftRunServer
from federated_learning_experiments.sweep import experiment_configurations as configurations
from federated_learning_experiments.sweep.experiment_sweep_plan import (
    build_planned_experiment_runs,
    make_run_identifier,
)
from tests.refactoring import behavior_golden_cases as golden
from tests.refactoring.test_feddrift_experiment_sweep import plan
from tests.refactoring.test_feddrift_oracle_concept_assignment_configuration import settings

CONFIGURATIONS = ("feddrift-window", "feddrift-adaptive-fedavg-window")


# 公式AdaStateをそのまま複写した参照（FedAvgEnsDataLoader.py）。
class AdaState:
    def __init__(self, init_lr=1e-2, beta1=0.5, beta2=0.5, beta3=0.5):
        self.init_lr = init_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3

        self.eta = init_lr
        self.mu = None
        self.s = 0
        self.gam = 0

    def update(self, theta, t):
        # count starting at 1 instead of 0
        t = t + 1

        # copy estimates from t-1
        prev_mu = self.mu
        if prev_mu is None:
            prev_mu = np.zeros(theta.shape)
        prev_s = self.s
        prev_gam = self.gam
        if t != 1:
            prev_muh = prev_mu / (1 - self.beta1 ** (t - 1))
            prev_sh = prev_s / (1 - self.beta2 ** (t - 1))
        else:
            prev_muh = 0
            prev_sh = 0

        # compute new estimates for t
        new_mu = self.beta1 * prev_mu + (1 - self.beta1) * theta
        new_s = self.beta2 * prev_s + (1 - self.beta2) * np.mean(
            (theta - prev_muh) * (theta - prev_muh)
        )
        new_sh = new_s / (1 - self.beta2**t)

        if prev_sh != 0:
            ratio = new_sh / prev_sh
        else:
            ratio = 1

        new_gam = self.beta3 * prev_gam + (1 - self.beta3) * ratio
        new_gamh = new_gam / (1 - self.beta3**t)

        # store estimates
        self.eta = min(self.init_lr, (self.init_lr * new_gamh) / t)

        self.mu = copy.deepcopy(new_mu)
        self.s = new_s
        self.gam = new_gam

    def current_lr(self):
        return self.eta


@pytest.mark.parametrize("initial_rate", (0.1, 0.001))
@pytest.mark.parametrize("sequence", ("random", "jump", "constant", "zero"))
def test_learning_rate_matches_official_reference(initial_rate, sequence):
    values = np.random.RandomState(51).normal(size=(30, 7))
    if sequence == "jump":
        values[15:] += 100
    if sequence == "constant":
        values[:] = 3
    if sequence == "zero":
        values[:] = 0
    reference = AdaState(init_lr=initial_rate)
    state = AdaptiveFedavgLearningRateState(
        settings=AdaptiveFedavgLearningRateSettings(initial_learning_rate=initial_rate)
    )
    for t, theta in enumerate(values):
        reference.update(theta, t)
        record = state.update(aggregated_parameters=theta)
        assert record.aggregation_count == t + 1
        assert record.learning_rate_for_next_timestep == pytest.approx(
            reference.current_lr(), rel=1e-12, abs=0
        )


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_configuration_registration_defaults_and_identifier(configuration):
    actual = settings(configuration=configuration)
    default = settings(configuration="feddrift")
    client = actual.run_participant_settings.run_client_settings
    assert client.model_assignment_policy == "single_global_model_assignment"
    assert client.training_sample_window == "current_timestep"
    assert client.cluster_model_initialization == "all_pretrained"
    assert actual == replace(
        default,
        run_participant_settings=replace(
            default.run_participant_settings,
            run_client_settings=replace(
                default.run_participant_settings.run_client_settings,
                model_assignment_policy="single_global_model_assignment",
                training_sample_window="current_timestep",
                adaptive_learning_rate_settings=AdaptiveFedavgLearningRateSettings()
                if configuration == CONFIGURATIONS[1]
                else None,
            ),
        ),
    )
    assert configurations.EXPERIMENT_CONFIGURATION_NAMES[-2:] == CONFIGURATIONS
    assert (
        golden.CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN[configuration]
        == "tests/refactoring/test_feddrift_window_and_adaptive_fedavg_configuration.py"
    )
    saved = convert_settings_to_plain_mapping(default)["run_participant_settings"][
        "run_client_settings"
    ]
    assert "adaptive_learning_rate_settings" not in saved
    assert (
        make_run_identifier(configuration_name="feddrift", run_settings=default)
        == "sine2_interval50_seed0_1fbf192c89f9"
    )


@pytest.mark.parametrize("rate", (0.0, -0.1, float("nan"), float("inf")))
def test_invalid_rates(rate):
    with pytest.raises(ValueError):
        AdaptiveFedavgLearningRateSettings(initial_learning_rate=rate)
    with pytest.raises(ValueError):
        replace(
            plan(),
            configuration_names=CONFIGURATIONS,
            adaptive_fedavg_initial_learning_rates=(rate,),
        )


def test_sweep_rejects_inactive_and_changes_only_adaptive_identifiers():
    with pytest.raises(ValueError):
        plan(adaptive_fedavg_initial_learning_rates=(0.1,))
    runs = build_planned_experiment_runs(
        sweep_plan=replace(
            plan(),
            configuration_names=CONFIGURATIONS,
            random_seeds=(0,),
            server_aggregation_intervals_per_client_samples=(50,),
            adaptive_fedavg_initial_learning_rates=(0.1, 0.001),
        )
    )
    assert len(runs) == 3
    assert len({r.run_identifier for r in runs[1:]}) == 2
    assert [
        r.run_settings.run_participant_settings.run_client_settings.adaptive_learning_rate_settings.initial_learning_rate
        for r in runs[1:]
    ] == [0.1, 0.001]


@pytest.mark.slow
@pytest.mark.parametrize("configuration", CONFIGURATIONS)
@pytest.mark.parametrize("dataset", ("sine2", "mnist4"))
def test_full_run_window_optimizer_learning_rates_and_single_model(
    monkeypatch, configuration, dataset
):
    adaptive = configuration == CONFIGURATIONS[1]
    reference = AdaState()
    rates = []
    windows = []
    train = FeddriftRunClient.train_local_round
    aggregate = FeddriftRunServer.aggregate_and_distribute_training_round

    def inspect_training(client, *, eligible_model_ids=None):
        end = client.stream_processor.processed_sample_count
        assert client.get_held_model_ids() == {0}
        assert eligible_model_ids == frozenset({0})
        states = client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()
        optimizer = states[0].concept_specific_parameter_optimizer_state.parameter_optimizer
        assert type(optimizer) is (torch.optim.SGD if adaptive else torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(
            reference.current_lr() if adaptive else (0.01 if dataset == "sine2" else 0.001),
            rel=1e-12,
        )
        if adaptive:
            assert optimizer.param_groups[0]["momentum"] == 0
            assert optimizer.param_groups[0]["weight_decay"] == 0
        collections = client.owners.training_sample_store.snapshot_ordered_model_training_samples()
        assert len(collections) == 1 and collections[0].model_id == 0
        assert len(collections[0].training_samples) == 50
        for offset, sample in enumerate(collections[0].training_samples):
            windows.append(
                (
                    client.client_id,
                    end - 50 + offset,
                    sample.input_features.clone(),
                    sample.observed_class_labels.item(),
                )
            )
        return train(client, eligible_model_ids=eligible_model_ids)

    def inspect_aggregation(server, *, timestep_index):
        aggregate(server, timestep_index=timestep_index)
        theta = np.concatenate(
            [
                p.detach().cpu().numpy().astype(np.float64).flatten()
                for p in server.global_model_repository.get_global_model_parameters(
                    model_id=0
                ).values()
            ]
        )
        if adaptive:
            reference.update(theta, timestep_index)
            rates.append(reference.current_lr())

    monkeypatch.setattr(FeddriftRunClient, "train_local_round", inspect_training)
    monkeypatch.setattr(
        FeddriftRunServer, "aggregate_and_distribute_training_round", inspect_aggregation
    )
    run = execute_feddrift_experiment_run(run_settings=settings(dataset, configuration))
    tables = build_feddrift_run_record_tables(experiment_run=run)
    concepts = tables["sample_prediction.observed_concept_id"]
    changes = int(np.count_nonzero(concepts[:, 1:] != concepts[:, :-1]))
    assert changes > 0
    streams = run.measured_run.run_result.observed_client_streams
    assert len(windows) == 25000
    for client_id, index, features, label in windows:
        sample = streams[client_id].observed_samples[index]
        torch.testing.assert_close(
            features, torch.tensor((sample.feature_values,), dtype=torch.float32), rtol=0, atol=0
        )
        assert label == sample.class_label
    assert tables["model_registration.model_id"].tolist() == [0]
    assert tables["model_registration.registered_server_timestep_index"].tolist() == [-1]
    assert len(tables["model_clustering.model_id"]) == 0
    assert len(tables["detection_batch.client_id"]) == 0
    for column in ("sample_prediction.prediction_model_id",):
        assert np.all(tables[column] == 0)
    for column in (
        "sample_prediction.held_model_count_at_prediction",
        "server_timestep.global_model_count_after_synchronization",
        "server_timestep.client_held_model_count_after_synchronization",
    ):
        assert np.all(tables[column] == 1)
    extra = ""
    if adaptive:
        np.testing.assert_allclose(
            tables["adaptive_learning_rate.learning_rate_for_next_timestep"],
            rates,
            rtol=1e-12,
            atol=0,
        )
        np.testing.assert_array_equal(
            tables["adaptive_learning_rate.aggregation_count"], np.arange(1, 51)
        )
        changed_timesteps = (
            np.flatnonzero(np.any(concepts[:, 1:] != concepts[:, :-1], axis=0)) // 50
        )
        after_first_change = np.arange(1, 50) > changed_timesteps.min()
        increases = int(np.count_nonzero((np.diff(rates) > 0) & after_first_change))
        extra = f", eta_min={min(rates)}, eta_max={max(rates)}, increases_after_change={increases}"
    print(
        f"{configuration}/{dataset}: changes={changes}, windows={len(windows)}, optimizer_and_rate_checks=500, models=1, registrations=1, merges=0{extra}"
    )


def test_nonfinite_estimates_do_not_advance_state():
    state = AdaptiveFedavgLearningRateState(settings=AdaptiveFedavgLearningRateSettings())
    state.update(aggregated_parameters=np.ones(7))
    before = state.snapshot_records()
    for theta in (np.full(7, np.inf), np.full(7, 1e308), np.ones(8)):
        with pytest.raises(ValueError):
            state.update(aggregated_parameters=theta)
        assert state.snapshot_records() == before


def test_saved_initial_rate_omission_and_explicit_default_identity():
    default = settings(configuration=CONFIGURATIONS[1])
    plain = convert_settings_to_plain_mapping(default)
    assert plain["run_participant_settings"]["run_client_settings"][
        "adaptive_learning_rate_settings"
    ] == {"settings_type": "AdaptiveFedavgLearningRateSettings"}
    runs = build_planned_experiment_runs(
        sweep_plan=replace(
            plan(),
            configuration_names=(CONFIGURATIONS[1],),
            random_seeds=(0,),
            client_count=10,
            per_client_sample_count=2500,
            server_aggregation_intervals_per_client_samples=(50,),
            adaptive_fedavg_initial_learning_rates=(0.01,),
        )
    )
    assert runs[0].run_settings == default
    assert runs[0].run_identifier == make_run_identifier(
        configuration_name=CONFIGURATIONS[1], run_settings=default
    )


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
def test_short_run_and_corrupt_learning_rate_record(configuration):
    from federated_learning_experiments.configuration.experiment_run_conditions import (
        ExperimentRunConditions,
    )

    run_settings = configurations.build_experiment_run_settings(
        configuration_name=configuration,
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name="sine2",
            random_seed=0,
            client_count=1,
            per_client_sample_count=6,
            server_aggregation_interval_per_client_samples=3,
        ),
    )
    run = execute_feddrift_experiment_run(run_settings=run_settings)
    tables = build_feddrift_run_record_tables(experiment_run=run)
    assert tables["model_registration.model_id"].tolist() == [0]
    state = run.measured_run.participants.run_server._adaptive_learning_rate_state
    if state is None:
        assert not any(name.startswith("adaptive_learning_rate.") for name in tables)
    else:
        assert tables["adaptive_learning_rate.timestep_index"].tolist() == [0, 1]
        assert tables["adaptive_learning_rate.learning_rate_for_next_timestep"][0] == 0.01
        state._records.pop()
        with pytest.raises(ValueError, match="同期件数"):
            build_feddrift_run_record_tables(experiment_run=run)


def test_cli_initial_learning_rates_parse_and_build_plan():
    from federated_learning_experiments.sweep import experiment_sweep_command as command

    args = command._build_argument_parser().parse_args(
        [
            "--configurations",
            CONFIGURATIONS[1],
            "--datasets",
            "sine2",
            "--seeds",
            "0",
            "--aggregation-intervals",
            "50",
            "--adaptive-fedavg-initial-learning-rates",
            "0.1",
            "0.001",
        ]
    )
    actual = command._build_sweep_plan(args)
    assert actual.adaptive_fedavg_initial_learning_rates == (0.1, 0.001)


@pytest.mark.parametrize(
    "forbidden",
    (
        "torch",
        "federated_drift_experiment",
        "federated_learning_experiments.runtime.feddrift_run_server",
        "federated_learning_experiments.sweep.experiment_sweep_plan",
    ),
)
def test_learning_rate_component_rejects_runtime_and_legacy_dependencies(forbidden):
    from tests.refactoring.test_single_run_dependency_boundaries import (
        collect_dependency_boundary_violations,
    )

    assert collect_dependency_boundary_violations(
        source_module_path="methods/feddrift/adaptive_fedavg_learning_rate.py",
        source_text=f"import {forbidden}",
    )


def test_initial_single_model_copies_pretraining_without_advancing_rng(monkeypatch):
    from federated_learning_experiments.runtime.feddrift_run_client import (
        assemble_feddrift_run_client,
    )
    from tests.refactoring.test_feddrift_run_client import (
        assert_rng_equal,
        build_assembly_inputs,
        rng_snapshot,
    )

    _, inputs = build_assembly_inputs(monkeypatch=monkeypatch)
    before = rng_snapshot()
    python_before = inputs["python_random_generator"].getstate()
    for adaptive in (None, AdaptiveFedavgLearningRateSettings()):
        client = assemble_feddrift_run_client(
            **{
                **inputs,
                "run_client_settings": replace(
                    inputs["run_client_settings"],
                    model_assignment_policy="single_global_model_assignment",
                    training_sample_window="current_timestep",
                    adaptive_learning_rate_settings=adaptive,
                ),
            }
        )
        assert client.get_held_model_ids() == {0}
        state = client.owners.held_model_training_state_registry.snapshot_ordered_held_model_training_states()[
            0
        ]
        for name, value in inputs["initial_classifier"].state_dict().items():
            torch.testing.assert_close(state.classifier.state_dict()[name], value, rtol=0, atol=0)
        assert state.classifier is not inputs["initial_classifier"]
    assert_rng_equal(before)
    assert inputs["python_random_generator"].getstate() == python_before


def test_mixed_axes_preserve_existing_feddrift_oracle_and_ifca_runs():
    existing = (
        "feddrift",
        "feddrift-oracle-concept-assignment",
        "feddrift-ifca-window",
        "feddrift-ifca-window-pretrained-cluster0",
    )
    old_plan = replace(
        plan(),
        configuration_names=existing,
        random_seeds=(0,),
        server_aggregation_intervals_per_client_samples=(50,),
        feddrift_loss_increase_thresholds=(0.1, 0.2),
    )
    old_runs = build_planned_experiment_runs(sweep_plan=old_plan)
    mixed = build_planned_experiment_runs(
        sweep_plan=replace(
            old_plan,
            configuration_names=existing + CONFIGURATIONS,
            adaptive_fedavg_initial_learning_rates=(0.1, 0.001),
        )
    )
    assert mixed[: len(old_runs)] == old_runs
    assert len(old_runs) == 8
    assert [r.configuration_name for r in mixed[len(old_runs) :]] == [
        CONFIGURATIONS[0],
        CONFIGURATIONS[1],
        CONFIGURATIONS[1],
    ]
    assert all(
        r.run_settings.run_participant_settings.run_client_settings.loss_increase_threshold == 0.2
        for r in mixed[len(old_runs) :]
    )
