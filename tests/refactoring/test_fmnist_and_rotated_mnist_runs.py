"""追加画像データの3手法の既定、全体run、掃引の計画を確認する。"""

from dataclasses import replace

import pytest

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.data.mnist.mnist_training_data import (
    resolve_fashion_mnist_data_directory,
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

NEW_DATASET_NAMES = ("fmnist2", "fmnist4", "rotmnist2", "rotmnist4")


def build_image_run_settings(configuration_name, dataset_name):
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


@pytest.mark.parametrize("dataset_name", NEW_DATASET_NAMES)
@pytest.mark.parametrize("configuration_name", ["fedsda", "feddrift", "fedavg"])
def test_image_defaults_equal_mnist(configuration_name, dataset_name):
    """ネットワーク・学習率以外も含めて、dataset名を除いた完全設定がMNISTと一致する。"""
    actual = build_image_run_settings(configuration_name, dataset_name)
    expected = build_image_run_settings(configuration_name, f"mnist{dataset_name[-1]}")
    assert actual == replace(
        expected,
        execution_settings=replace(
            expected.execution_settings,
            experiment_run_conditions=actual.execution_settings.experiment_run_conditions,
        ),
    )
    if configuration_name == "feddrift":
        assert isinstance(actual, FeddriftRunSettings)
        assert actual.run_participant_settings.run_client_settings.loss_increase_threshold == 0.10
        assert (
            actual.run_participant_settings.server_settings.maximum_same_cluster_loss_increase
            == 0.10
        )


@pytest.mark.parametrize("dataset_name", NEW_DATASET_NAMES)
@pytest.mark.parametrize("configuration_name", ["fedsda", "feddrift", "fedavg"])
@pytest.mark.slow
def test_whole_image_run_finishes(configuration_name, dataset_name):
    """要求4.2(vi)の規模で、事前学習から指標の導出まで実行する。"""
    if dataset_name in ("fmnist2", "fmnist4"):
        directory = resolve_fashion_mnist_data_directory()
        if not all(
            (directory / name).is_file()
            for name in ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
        ):
            pytest.skip(f"Fashion-MNISTの学習用IDXファイルがありません: {directory}")
    execute = {
        "fedsda": execute_fedsda_experiment_run,
        "feddrift": execute_feddrift_experiment_run,
        "fedavg": execute_fedavg_experiment_run,
    }[configuration_name]
    result = execute(run_settings=build_image_run_settings(configuration_name, dataset_name))
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
                "fedsda",
                "feddrift",
                "fedavg",
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


def test_new_dataset_cost_matches_mnist():
    runs = build_planned_experiment_runs(
        sweep_plan=ExperimentSweepPlan(
            configuration_names=("fedsda",),
            dataset_names=("mnist2", "mnist4", *NEW_DATASET_NAMES),
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
    assert costs["mnist2"] != costs["mnist4"]
    for dataset_name in NEW_DATASET_NAMES:
        assert costs[dataset_name] == costs[f"mnist{dataset_name[-1]}"]


@pytest.mark.parametrize("dataset_name", ["fmnist2", "rotmnist2"])
def test_command_rejects_missing_image_files_before_output(
    tmp_path, monkeypatch, capsys, dataset_name
):
    monkeypatch.setenv("FDE_MNIST_DATA_DIR", str(tmp_path / "missing-mnist"))
    monkeypatch.setenv("FDE_FASHION_MNIST_DATA_DIR", str(tmp_path / "missing-fashion"))
    output = tmp_path / "not-created"
    assert (
        experiment_sweep_command.main(
            [
                "--datasets",
                dataset_name,
                "--seeds",
                "0",
                "--aggregation-intervals",
                "50",
                "--output-directory",
                str(output),
                "--purpose",
                "入力不足の拒否を確認する",
            ]
        )
        == 2
    )
    error_text = capsys.readouterr().err
    expected_variable = (
        "FDE_FASHION_MNIST_DATA_DIR" if dataset_name == "fmnist2" else "FDE_MNIST_DATA_DIR"
    )
    assert expected_variable in error_text
    assert "data is not available" in error_text
    assert not output.exists()
