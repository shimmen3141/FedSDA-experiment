"""再来の合成3データ（stagger3・agrawal4・randomrbf4）の4手法の既定、全体run、掃引の計画を確認する。"""

from dataclasses import replace

import pytest

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.runtime.dynamic_weighted_majority_experiment_run import (
    execute_dynamic_weighted_majority_experiment_run,
)
from federated_learning_experiments.runtime.fedavg_experiment_run import (
    execute_fedavg_experiment_run,
)
from federated_learning_experiments.runtime.feddrift_experiment_run import (
    execute_feddrift_experiment_run,
)
from federated_learning_experiments.runtime.feddrift_run_settings import FeddriftRunSettings
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.sweep import (
    experiment_sweep_command,
    experiment_sweep_execution,
)
from federated_learning_experiments.sweep.experiment_configurations import (
    build_experiment_run_settings,
)
from federated_learning_experiments.sweep.experiment_sweep_plan import (
    ExperimentSweepPlan,
    build_planned_experiment_runs,
)

NEW_DATASET_NAMES = ("stagger3", "agrawal4", "randomrbf4")
CONFIGURATION_NAMES = ("fedsda", "feddrift", "fedavg", "dwm-neural-network")


def build_run_settings(configuration_name, dataset_name):
    return build_experiment_run_settings(
        configuration_name=configuration_name,
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name=dataset_name,
            random_seed=0,
            client_count=10,
            per_client_sample_count=100,
            server_aggregation_interval_per_client_samples=50,
        ),
    )


def _with_conditions_of(settings, other):
    """dataset名だけが違う実行条件を、比べる相手の条件へ置き換える。"""
    return replace(
        settings,
        execution_settings=replace(
            settings.execution_settings,
            experiment_run_conditions=other.execution_settings.experiment_run_conditions,
        ),
    )


@pytest.mark.parametrize("dataset_name", NEW_DATASET_NAMES)
@pytest.mark.parametrize("configuration_name", CONFIGURATION_NAMES)
def test_new_dataset_defaults_equal_the_synthetic_sea4_defaults(configuration_name, dataset_name):
    """FedDriftのδ以外の完全設定（ネットワーク・学習率を含む）が、合成データのsea4と一致する。"""
    actual = build_run_settings(configuration_name, dataset_name)
    sea4 = build_run_settings(configuration_name, "sea4")
    if configuration_name == "feddrift":
        assert isinstance(actual, FeddriftRunSettings)
        client_settings = actual.run_participant_settings.run_client_settings
        server_settings = actual.run_participant_settings.server_settings
        assert client_settings.loss_increase_threshold == 0.10
        assert server_settings.maximum_same_cluster_loss_increase == 0.10
        # sea4のδ（0.04）だけを、新しいデータの既定（0.10）へそろえて比べる。
        sea4 = replace(
            sea4,
            run_participant_settings=replace(
                sea4.run_participant_settings,
                run_client_settings=replace(
                    sea4.run_participant_settings.run_client_settings,
                    loss_increase_threshold=0.10,
                ),
                server_settings=replace(
                    sea4.run_participant_settings.server_settings,
                    maximum_same_cluster_loss_increase=0.10,
                ),
            ),
        )
    assert actual == _with_conditions_of(sea4, actual)


@pytest.mark.parametrize("dataset_name", NEW_DATASET_NAMES)
@pytest.mark.parametrize("configuration_name", CONFIGURATION_NAMES)
@pytest.mark.slow
def test_whole_new_dataset_run_finishes(configuration_name, dataset_name):
    """要求7.2(iv)の規模で、事前学習から指標の導出まで実行する。"""
    execute = {
        "fedsda": execute_fedsda_experiment_run,
        "feddrift": execute_feddrift_experiment_run,
        "fedavg": execute_fedavg_experiment_run,
        "dwm-neural-network": execute_dynamic_weighted_majority_experiment_run,
    }[configuration_name]
    result = execute(run_settings=build_run_settings(configuration_name, dataset_name))
    run_result = result.measured_run.run_result
    assert run_result.generated_sample_count_per_client == 100
    assert run_result.processed_sample_count_per_client == 100
    assert run_result.synchronization_interval_count == 2
    assert len(run_result.observed_client_streams) == 10
    assert all(len(stream.observed_samples) == 100 for stream in run_result.observed_client_streams)


def test_command_accepts_new_datasets_without_creating_output(tmp_path, capsys):
    output = tmp_path / "not-created"
    assert (
        experiment_sweep_command.main(
            [
                "--datasets",
                *NEW_DATASET_NAMES,
                "--configurations",
                *CONFIGURATION_NAMES,
                "--seeds",
                "0",
                "--aggregation-intervals",
                "50",
                "--output-directory",
                str(output),
                "--print-plan",
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert all(name in printed for name in NEW_DATASET_NAMES)
    assert not output.exists()


def test_new_dataset_cost_matches_the_synthetic_datasets():
    runs = build_planned_experiment_runs(
        sweep_plan=ExperimentSweepPlan(
            configuration_names=("fedsda",),
            dataset_names=("sea4", *NEW_DATASET_NAMES),
            random_seeds=(0,),
            client_count=10,
            per_client_sample_count=100,
            server_aggregation_intervals_per_client_samples=(50,),
        )
    )
    costs = {
        run.run_settings.execution_settings.experiment_run_conditions.dataset_name: experiment_sweep_execution._estimate_run_cost(
            run
        )
        for run in runs
    }
    for dataset_name in NEW_DATASET_NAMES:
        assert costs[dataset_name] == costs["sea4"]
