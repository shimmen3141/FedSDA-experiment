"""局所更新を集約の間隔ごとにまとめるFedAvgの構成（`fedavg-local-updates-per-aggregation`）を確認する。

この構成はgoldenの事例にしない（`behavior_golden_cases.CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`）。
設定が`fedavg`と局所更新の間隔だけ違うこと、roundの中の予測が全部、そのroundの始めに配布されたモデルで
行われること、更新の回数、掃引のコマンドでの保存を、ここで確かめる。
"""

import csv
from copy import deepcopy

import pytest
import torch

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.runtime.fedavg_experiment_run import (
    execute_fedavg_experiment_run,
)
from federated_learning_experiments.runtime.fedavg_run_server import FedavgRunServer
from federated_learning_experiments.sweep.experiment_configurations import (
    FEDAVG_LOCAL_UPDATES_PER_AGGREGATION_CONFIGURATION_NAME,
    build_experiment_run_settings,
)
from federated_learning_experiments.sweep.experiment_sweep_command import main

_BATCH_SAMPLE_COUNT = 32


def _conditions(*, interval, samples, seed=0, client_count=2):
    return ExperimentRunConditions(
        dataset_name="sine2",
        random_seed=seed,
        client_count=client_count,
        per_client_sample_count=samples,
        server_aggregation_interval_per_client_samples=interval,
    )


def _differing_paths(left, right, path=()):
    if isinstance(left, dict) and isinstance(right, dict):
        differing = []
        for key in sorted(set(left) | set(right)):
            differing += _differing_paths(left.get(key), right.get(key), (*path, key))
        return differing
    return [] if left == right else [path]


@pytest.mark.parametrize("interval", [3, 50])
def test_run_settings_differ_from_fedavg_only_in_local_update_interval(interval):
    conditions = _conditions(interval=interval, samples=interval * 2)
    fedavg_mapping = convert_settings_to_plain_mapping(
        build_experiment_run_settings(
            configuration_name="fedavg", experiment_run_conditions=conditions
        )
    )
    run_settings = build_experiment_run_settings(
        configuration_name=FEDAVG_LOCAL_UPDATES_PER_AGGREGATION_CONFIGURATION_NAME,
        experiment_run_conditions=conditions,
    )
    mapping = convert_settings_to_plain_mapping(run_settings)
    assert _differing_paths(fedavg_mapping, mapping) == [
        ("run_participant_settings", "local_update_interval_per_client_samples")
    ]
    assert (
        run_settings.run_participant_settings.local_update_interval_per_client_samples == interval
    )
    assert (
        fedavg_mapping["run_participant_settings"]["local_update_interval_per_client_samples"] == 1
    )


def _run_and_record_round_start_parameters(monkeypatch, *, configuration_name, interval, samples):
    """runを実行し、各roundの始めにserverが持つ（配布済みの）parameterを、roundの順に返す。"""
    round_start_parameters = []
    original_synchronize = FedavgRunServer.synchronize_models

    def recording_synchronize(self, **arguments):
        # 同期の前のserverのparameterは、このroundの始めに配布したもの（round 0は事前学習のモデル）。
        round_start_parameters.append(deepcopy(self.global_parameters))
        return original_synchronize(self, **arguments)

    monkeypatch.setattr(FedavgRunServer, "synchronize_models", recording_synchronize)
    run_settings = build_experiment_run_settings(
        configuration_name=configuration_name,
        experiment_run_conditions=_conditions(interval=interval, samples=samples),
    )
    experiment_run = execute_fedavg_experiment_run(run_settings=run_settings)
    return experiment_run, round_start_parameters


def _predict_with_parameters(classifier, parameters, observed_samples):
    model = deepcopy(classifier)
    model.load_state_dict(parameters, strict=True)
    with torch.no_grad():
        scores = model(
            torch.tensor([s.feature_values for s in observed_samples], dtype=torch.float32)
        )
    if model.class_count == 2:
        return [int(score > 0.5) for score in scores[:, 0].tolist()]
    return scores.argmax(dim=1).tolist()


def _count_round_prediction_mismatches(experiment_run, round_start_parameters, interval):
    participants = experiment_run.measured_run.participants
    streams = experiment_run.measured_run.run_result.observed_client_streams
    mismatch_count = 0
    for client, stream in zip(participants.client_operations, streams, strict=True):
        for round_index, parameters in enumerate(round_start_parameters):
            start = round_index * interval
            expected = _predict_with_parameters(
                client.classifier, parameters, stream.observed_samples[start : start + interval]
            )
            recorded = [
                p.predicted_class_id for p in client.sample_predictions[start : start + interval]
            ]
            mismatch_count += sum(e != r for e, r in zip(expected, recorded, strict=True))
    return mismatch_count


@pytest.mark.parametrize(
    ("interval", "updated_round_count"),
    [
        # 3 roundとも、roundの最後の標本の時点で、累積の保持標本がbatchの件数以上。
        (40, 3),
        # 最初のroundは累積の保持標本（20件）がbatchの件数（32）に足りず、その更新は学習されずに捨てられる。
        (20, 2),
    ],
)
def test_predictions_in_each_round_use_the_model_distributed_at_round_start(
    monkeypatch, interval, updated_round_count
):
    experiment_run, round_start_parameters = _run_and_record_round_start_parameters(
        monkeypatch,
        configuration_name=FEDAVG_LOCAL_UPDATES_PER_AGGREGATION_CONFIGURATION_NAME,
        interval=interval,
        samples=interval * 3,
    )
    assert len(round_start_parameters) == 3
    assert _count_round_prediction_mismatches(experiment_run, round_start_parameters, interval) == 0
    run_settings = experiment_run.run_settings.run_participant_settings
    assert run_settings.local_training_batch_sample_count == _BATCH_SAMPLE_COUNT
    for client in experiment_run.measured_run.participants.client_operations:
        assert client.parameter_update_step_count == (
            updated_round_count * interval * run_settings.updates_per_sample
        )


def test_round_start_model_check_detects_per_sample_local_updates(monkeypatch):
    """検出力: 1標本ごとに局所更新する`fedavg`では、round内の予測が配布モデルの予測と食い違う。"""
    interval = 40
    experiment_run, round_start_parameters = _run_and_record_round_start_parameters(
        monkeypatch, configuration_name="fedavg", interval=interval, samples=interval * 3
    )
    assert _count_round_prediction_mismatches(experiment_run, round_start_parameters, interval) > 0


def test_sweep_command_saves_runs_under_the_configuration_directory(tmp_path):
    output = tmp_path / "out"
    arguments = [
        "--configurations",
        FEDAVG_LOCAL_UPDATES_PER_AGGREGATION_CONFIGURATION_NAME,
        "--datasets",
        "sine2",
        "--seeds",
        "0",
        "--aggregation-intervals",
        "3",
        "--client-count",
        "2",
        "--per-client-sample-count",
        "7",
        "--workers",
        "1",
        "--purpose",
        "局所更新を集約ごとにまとめるFedAvgのtest",
        "--output-directory",
        str(output),
    ]
    assert main(arguments) == 0
    configuration_directory = output / FEDAVG_LOCAL_UPDATES_PER_AGGREGATION_CONFIGURATION_NAME
    assert len(list((configuration_directory / "runs").glob("*.json"))) == 1
    with (configuration_directory / "run_summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["configuration_name"] for row in rows] == [
        FEDAVG_LOCAL_UPDATES_PER_AGGREGATION_CONFIGURATION_NAME
    ]
    assert not (output / "fedavg").exists()
