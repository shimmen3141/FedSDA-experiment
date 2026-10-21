"""FedDriftの既定条件・専用軸・既存sweepへの統合を確かめる。"""

import csv
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from run_provenance_examples import TEST_RUN_PROVENANCE

from federated_drift_experiment import config
from federated_drift_experiment.data.specs import get_dataset_spec
from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.runtime.feddrift_default_run_settings import (
    build_default_feddrift_run_settings,
)
from federated_learning_experiments.runtime.feddrift_experiment_run import (
    execute_feddrift_experiment_run,
)
from federated_learning_experiments.runtime.feddrift_run_artifact_files import (
    load_feddrift_run_artifacts,
)
from federated_learning_experiments.runtime.feddrift_run_record_tables import (
    build_feddrift_run_record_tables,
)
from federated_learning_experiments.runtime.feddrift_run_settings import FeddriftRunSettings
from federated_learning_experiments.sweep import experiment_sweep_command
from federated_learning_experiments.sweep.experiment_configurations import (
    build_experiment_run_settings,
)
from federated_learning_experiments.sweep.experiment_sweep_execution import (
    execute_experiment_sweep,
    find_existing_run_state,
)
from federated_learning_experiments.sweep.experiment_sweep_plan import (
    ExperimentSweepPlan,
    build_planned_experiment_runs,
    make_run_identifier,
)
from tests.refactoring.test_feddrift_run_client import assert_rng_equal, rng_snapshot


def plan(**changes):
    return ExperimentSweepPlan(
        **dict(
            configuration_names=("fedsda", "feddrift"),
            dataset_names=("sine2",),
            random_seeds=(0, 1),
            client_count=2,
            per_client_sample_count=5,
            server_aggregation_intervals_per_client_samples=(3, 4),
            **changes,
        )
    )


def test_mixed_plan_expands_only_applicable_axis_in_order_and_preserves_fedsda_hash():
    runs = build_planned_experiment_runs(
        sweep_plan=plan(feddrift_loss_increase_thresholds=(0.0, 0.2))
    )
    assert len(runs) == 12
    keys = [
        (
            r.configuration_name,
            r.run_settings.execution_settings.experiment_run_conditions.server_aggregation_interval_per_client_samples,
            r.run_settings.execution_settings.experiment_run_conditions.random_seed,
            r.run_settings.run_participant_settings.run_client_settings.loss_increase_threshold
            if type(r.run_settings) is FeddriftRunSettings
            else None,
        )
        for r in runs
    ]
    assert keys == [("fedsda", interval, seed, None) for interval in (3, 4) for seed in (0, 1)] + [
        ("feddrift", interval, seed, threshold)
        for interval in (3, 4)
        for seed in (0, 1)
        for threshold in (0.0, 0.2)
    ]
    assert runs[0].run_identifier == "sine2_interval3_seed0_73df845459ba"
    assert len({(r.configuration_name, r.run_identifier) for r in runs}) == 12
    for r in runs[4:]:
        participant = r.run_settings.run_participant_settings
        assert (
            participant.server_settings.maximum_same_cluster_loss_increase
            == participant.run_client_settings.loss_increase_threshold
        )
        assert (
            participant.run_client_settings.detection_batch_sample_count
            == r.run_settings.execution_settings.experiment_run_conditions.server_aggregation_interval_per_client_samples
        )
        assert (
            make_run_identifier(
                configuration_name=r.configuration_name, run_settings=r.run_settings
            )
            == r.run_identifier
        )


def test_default_threshold_matches_explicit_default_axis():
    assert build_planned_experiment_runs(sweep_plan=plan()) == build_planned_experiment_runs(
        sweep_plan=plan(feddrift_loss_increase_thresholds=(0.1,))
    )


@pytest.mark.parametrize(
    "axis", ((), (0.1, 0.1), (-0.1,), (float("nan"),), (float("inf"),), (True,), (1,), [0.1], "0.1")
)
def test_plan_rejects_invalid_threshold_axis(axis):
    with pytest.raises((TypeError, ValueError)):
        plan(feddrift_loss_increase_thresholds=axis)


def test_plan_rejects_explicit_axis_without_feddrift_and_revalidates():
    with pytest.raises(ValueError):
        replace(plan(), configuration_names=("fedsda",), feddrift_loss_increase_thresholds=(0.1,))
    corrupt = plan()
    object.__setattr__(corrupt, "feddrift_loss_increase_thresholds", (float("nan"),))
    with pytest.raises(ValueError):
        build_planned_experiment_runs(sweep_plan=corrupt)


def small_feddrift_runs():
    runs = build_planned_experiment_runs(
        sweep_plan=replace(
            plan(),
            configuration_names=("feddrift",),
            server_aggregation_intervals_per_client_samples=(3,),
            feddrift_loss_increase_thresholds=(0.0,),
        )
    )
    result = []
    for run in runs:
        participant = run.run_settings.run_participant_settings
        settings = replace(
            run.run_settings,
            run_participant_settings=replace(
                participant,
                hidden_layer_widths=(3,),
                initial_model_pretraining_settings=replace(
                    participant.initial_model_pretraining_settings,
                    pretraining_sample_count=8,
                    pretraining_epoch_count=1,
                ),
                server_settings=replace(
                    participant.server_settings, initial_model_aggregation_weight=8
                ),
            ),
            execution_settings=replace(
                run.run_settings.execution_settings,
                concept_schedule_settings=replace(
                    run.run_settings.execution_settings.concept_schedule_settings,
                    minimum_sample_index_gap_before_change_trial=0,
                    per_eligible_sample_concept_change_probability=1.0,
                ),
            ),
        )
        result.append(
            replace(
                run,
                run_settings=settings,
                run_identifier=make_run_identifier(
                    configuration_name="feddrift", run_settings=settings
                ),
            )
        )
    return tuple(result)


@pytest.mark.parametrize("workers", (1, 2))
@pytest.mark.slow
def test_actual_sweep_matches_single_run_and_resumes_without_execution(
    tmp_path, monkeypatch, workers
):
    from federated_learning_experiments.sweep import experiment_sweep_execution as execution

    runs = small_feddrift_runs()
    before = rng_snapshot()
    outcomes = execute_experiment_sweep(
        planned_runs=runs,
        output_directory=tmp_path,
        worker_count=workers,
        run_provenance=TEST_RUN_PROVENANCE,
    )
    assert_rng_equal(before)
    assert [o.run_status for o in outcomes] == ["completed", "completed"]
    assert [o.run_identifier for o in outcomes] == [r.run_identifier for r in runs]
    for run in runs:
        single = execute_feddrift_experiment_run(run_settings=run.run_settings)
        summary, tables = load_feddrift_run_artifacts(
            summary_file_path=tmp_path / "feddrift" / "runs" / (run.run_identifier + ".json")
        )
        for name, value in (
            ("run_settings", run.run_settings),
            ("run_metrics", single.run_metrics),
        ):
            assert summary[name] == json.loads(
                json.dumps(convert_settings_to_plain_mapping(value, non_finite_float_as_none=True))
            )
        expected = build_feddrift_run_record_tables(experiment_run=single)
        assert tables.keys() == expected.keys()
        for key in tables:
            np.testing.assert_array_equal(tables[key], expected[key])
            assert tables[key].dtype == expected[key].dtype
        assert (
            find_existing_run_state(planned_run=run, output_directory=tmp_path).run_state
            == "completed"
        )

    def reject(**kwargs):
        raise AssertionError("completed run must not execute")

    monkeypatch.setattr(execution, "execute_feddrift_experiment_run", reject)
    snapshot = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert [
        o.run_status
        for o in execute_experiment_sweep(
            planned_runs=runs,
            output_directory=tmp_path,
            worker_count=workers,
            run_provenance=TEST_RUN_PROVENANCE,
        )
    ] == ["skipped_existing", "skipped_existing"]
    assert snapshot == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "corruption", ("wrong_method", "wrong_settings", "partial", "corrupt_json", "corrupt_npz")
)
def test_conflicting_feddrift_artifacts_are_not_overwritten(tmp_path, corruption):
    run = small_feddrift_runs()[0]
    assert (
        execute_experiment_sweep(
            planned_runs=(run,),
            output_directory=tmp_path,
            worker_count=1,
            run_provenance=TEST_RUN_PROVENANCE,
        )[0].run_status
        == "completed"
    )
    directory = tmp_path / "feddrift" / "runs"
    summary_file = directory / (run.run_identifier + ".json")
    table_file = directory / (run.run_identifier + ".npz")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if corruption == "wrong_method":
        summary["method_name"] = "fedsda"
    elif corruption == "wrong_settings":
        summary["run_settings"]["run_participant_settings"]["run_client_settings"][
            "loss_increase_threshold"
        ] = 99.0
    elif corruption == "partial":
        table_file.unlink()
    elif corruption == "corrupt_json":
        summary_file.write_text("not json", encoding="utf-8")
    else:
        table_file.write_bytes(b"not npz")
    if corruption.startswith("wrong"):
        summary_file.write_text(json.dumps(summary), encoding="utf-8")
    before = {p: p.read_bytes() for p in directory.iterdir()}
    assert (
        find_existing_run_state(planned_run=run, output_directory=tmp_path).run_state
        == "conflicting"
    )
    assert (
        execute_experiment_sweep(
            planned_runs=(run,),
            output_directory=tmp_path,
            worker_count=1,
            run_provenance=TEST_RUN_PROVENANCE,
        )[0].run_status
        == "failed"
    )
    assert before == {p: p.read_bytes() for p in directory.iterdir()}


def conditions(dataset="sine2"):
    return ExperimentRunConditions(
        dataset_name=dataset,
        random_seed=0,
        client_count=2,
        per_client_sample_count=5,
        server_aggregation_interval_per_client_samples=3,
    )


@pytest.mark.parametrize("dataset", ("sine2", "sea2", "sea4", "circle2", "mnist2", "mnist4"))
def test_default_settings_match_actual_legacy_values_without_rng_changes(dataset):
    before = rng_snapshot()
    run = build_default_feddrift_run_settings(experiment_run_conditions=conditions(dataset))
    assert_rng_equal(before)
    assert type(run) is FeddriftRunSettings
    assert (
        build_experiment_run_settings(
            configuration_name="feddrift", experiment_run_conditions=conditions(dataset)
        )
        == run
    )
    participant = run.run_participant_settings
    client, server = participant.run_client_settings, participant.server_settings
    assert participant.hidden_layer_widths == get_dataset_spec(dataset).hidden_dims
    for name, legacy in (
        ("local_training_batch_sample_count", "CLIENT_BATCH_SIZE"),
        ("updates_per_sample", "UPDATES_PER_SAMPLE"),
        ("maximum_stored_evaluation_sample_count_per_model", "STORED_DATA_LIMIT"),
        ("added_evaluation_batch_sample_count", "EVAL_STORE_SAMPLE_SIZE"),
        ("loss_increase_threshold", "FEDDRIFT_DISTANCE_THRESHOLD"),
    ):
        assert getattr(client, name) == getattr(config, legacy)
    assert client.detection_batch_sample_count == 3
    for name, legacy in (
        ("maximum_same_cluster_loss_increase", "FEDDRIFT_DISTANCE_THRESHOLD"),
        ("cluster_linkage", "FEDDRIFT_CLUSTER_LINKAGE"),
        ("minimum_cross_evaluation_sample_count", "CLUSTER_MIN_EVAL_N"),
        ("maximum_cross_evaluation_client_count", "CROSS_EVAL_MAX_CLIENTS"),
        ("maximum_client_evaluation_sample_count", "EVAL_MAX_SAMPLES"),
        ("new_model_isolation_timestep_count", "FEDDRIFT_ISOLATION_TIMESTEPS"),
        ("local_training_round_count", "FEDDRIFT_ROUNDS"),
        ("initial_model_aggregation_weight", "PRETRAIN_SAMPLES"),
    ):
        assert getattr(server, name) == getattr(config, legacy)
    pretrain = participant.initial_model_pretraining_settings
    assert (
        pretrain.pretraining_sample_count,
        pretrain.pretraining_epoch_count,
        pretrain.pretraining_batch_sample_count,
    ) == (config.PRETRAIN_SAMPLES, config.PRETRAIN_EPOCHS, config.PRETRAIN_BATCH_SIZE)
    optimizer = client.new_model_optimizer_settings
    assert optimizer == server.rebuilt_model_optimizer_settings
    assert config.OPTIMIZER == "adam" and config.AMSGRAD
    assert optimizer.adam_variant == "amsgrad"
    assert optimizer.weight_decay == config.WEIGHT_DECAY
    assert optimizer.learning_rate == (
        get_dataset_spec(dataset).learning_rate or config.NEW_MODEL_LR
    )
    schedule = run.execution_settings.concept_schedule_settings
    assert schedule.minimum_sample_index_gap_before_change_trial == config.MIN_STABLE_PERIOD
    assert schedule.per_eligible_sample_concept_change_probability == config.DRIFT_PROB
    assert run.run_metric_settings.maximum_detection_delay_sample_count == config.DELAY_TOLERANCE
    assert run.run_metric_settings.post_change_recovery_window_sample_count == config.STABLE_WINDOW


@pytest.mark.parametrize("threshold", (0.0, 0.03, 1.0))
def test_threshold_applies_to_both_owners(threshold):
    run = build_default_feddrift_run_settings(
        experiment_run_conditions=conditions(), loss_increase_threshold=threshold
    )
    assert run.run_participant_settings.run_client_settings.loss_increase_threshold == threshold
    assert (
        run.run_participant_settings.server_settings.maximum_same_cluster_loss_increase == threshold
    )


@pytest.mark.parametrize("invalid", (None, True, 1, -1.0, float("nan"), float("inf")))
def test_default_settings_reject_invalid_threshold_without_rng_changes(invalid):
    before = rng_snapshot()
    with pytest.raises((TypeError, ValueError)):
        build_default_feddrift_run_settings(
            experiment_run_conditions=conditions(), loss_increase_threshold=invalid
        )
    assert_rng_equal(before)


@pytest.mark.parametrize("invalid", (None, "corrupted", "seed_overflow"))
def test_default_settings_revalidate_conditions(invalid):
    value = conditions()
    if invalid == "corrupted":
        object.__setattr__(value, "client_count", 0)
    elif invalid == "seed_overflow":
        value = replace(value, random_seed=2**32)
    else:
        value = invalid
    before = rng_snapshot()
    with pytest.raises((TypeError, ValueError)):
        build_default_feddrift_run_settings(experiment_run_conditions=value)
    assert_rng_equal(before)


def test_default_settings_do_not_read_mutable_legacy_config(monkeypatch):
    expected = build_default_feddrift_run_settings(experiment_run_conditions=conditions())
    monkeypatch.setattr(config, "FEDDRIFT_DISTANCE_THRESHOLD", 99.0)
    monkeypatch.setattr(config, "FEDDRIFT_ROUNDS", 9)
    monkeypatch.setattr(config, "BASE_LR", 0.99)
    assert build_default_feddrift_run_settings(experiment_run_conditions=conditions()) == expected


def command_arguments(output):
    return [
        "--configurations",
        "feddrift",
        "--datasets",
        "sine2",
        "--seeds",
        "0",
        "--client-count",
        "2",
        "--per-client-sample-count",
        "5",
        "--aggregation-intervals",
        "3",
        "--feddrift-loss-increase-thresholds",
        "0.0",
        "0.2",
        "--workers",
        "1",
        "--purpose",
        "feddrift test",
        "--output-directory",
        str(output),
    ]


def test_cli_writes_threshold_columns_records_and_resumes(tmp_path, capsys):
    output = tmp_path / "output"
    args = command_arguments(output)
    assert experiment_sweep_command.main(args + ["--print-plan"]) == 0
    assert "planned runs: 2" in capsys.readouterr().out
    assert not output.exists()
    assert experiment_sweep_command.main(args) == 0
    assert "completed: 2" in capsys.readouterr().out
    with (output / "feddrift" / "run_summary.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {float(row["loss_increase_threshold"]) for row in rows} == {0.0, 0.2}
    assert {int(row["detection_batch_sample_count"]) for row in rows} == {3}
    for row in rows:
        summary = json.loads(
            (output / "feddrift" / "runs" / (row["run_identifier"] + ".json")).read_text(
                encoding="utf-8"
            )
        )
        assert float(row["prediction_accuracy"]) == summary["run_metrics"]["prediction_accuracy"]
        assert (
            float(row["loss_increase_threshold"])
            == summary["run_settings"]["run_participant_settings"]["run_client_settings"][
                "loss_increase_threshold"
            ]
        )
    record = json.loads(next(output.glob("sweep_record_*.json")).read_text(encoding="utf-8"))
    assert record["sweep_plan"]["feddrift_loss_increase_thresholds"] == [0.0, 0.2]
    assert record["run_status_counts"] == dict(completed=2, skipped_existing=0, failed=0)
    assert experiment_sweep_command.main(args) == 0
    assert "skipped (existing): 2" in capsys.readouterr().out


@pytest.mark.parametrize("values", ([], ["-1"], ["nan"], ["inf"], ["0.1", "0.1"]))
def test_cli_rejects_invalid_axis_before_outputs(tmp_path, values):
    args = command_arguments(tmp_path / "output")
    start = args.index("--feddrift-loss-increase-thresholds")
    args[start + 1 : start + 3] = values
    assert experiment_sweep_command.main(args) == 2
    assert list(tmp_path.iterdir()) == []


def test_cli_rejects_fedsda_only_explicit_axis_and_handles_missing_mnist(tmp_path, monkeypatch):
    args = command_arguments(tmp_path / "output")
    args[1] = "fedsda"
    assert experiment_sweep_command.main(args) == 2
    args[1] = "feddrift"
    args[3] = "mnist2"
    monkeypatch.setenv("FDE_MNIST_DATA_DIR", str(tmp_path / "missing-mnist"))
    assert experiment_sweep_command.main(args + ["--print-plan"]) == 0
    assert experiment_sweep_command.main(args) == 2
    assert list(tmp_path.iterdir()) == []


def test_feddrift_cli_reports_execution_failure(tmp_path, monkeypatch):
    from federated_learning_experiments.sweep import experiment_sweep_execution

    def fail(**kwargs):
        raise RuntimeError("simulated FedDrift failure")

    monkeypatch.setattr(experiment_sweep_execution, "execute_feddrift_experiment_run", fail)
    output = tmp_path / "output"
    assert experiment_sweep_command.main(command_arguments(output)) == 1
    record = json.loads(next(output.glob("sweep_record_*.json")).read_text(encoding="utf-8"))
    assert record["run_status_counts"]["failed"] == 2
    assert all(
        "simulated FedDrift failure" in o["failure_description"] for o in record["run_outcomes"]
    )


@pytest.mark.slow
def test_real_cli_help_and_spawn_execution(tmp_path):
    root = Path(__file__).resolve().parents[2]
    help_result = subprocess.run(
        [sys.executable, str(root / "run_experiment_sweep.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        # 日本語のhelpを、端末の文字コードに依らず、UTF-8で受け取る。
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert help_result.returncode == 0, help_result.stderr
    assert (
        "--feddrift-loss-increase-thresholds" in help_result.stdout
        and "feddrift" in help_result.stdout
    )
    output = tmp_path / "output"
    args = command_arguments(output)
    args[args.index("--workers") + 1] = "2"
    result = subprocess.run(
        [sys.executable, str(root / "run_experiment_sweep.py"), *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "completed: 2" in result.stdout
    for summary_file in (output / "feddrift" / "runs").glob("*.json"):
        summary, tables = load_feddrift_run_artifacts(summary_file_path=summary_file)
        assert summary["method_name"] == "feddrift" and tables[
            "sample_prediction.sample_index"
        ].shape == (2, 3)
