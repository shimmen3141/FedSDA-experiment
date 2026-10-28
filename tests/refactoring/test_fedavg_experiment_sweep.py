"""FedAvgの構成登録、混合掃引、実行・再開・一覧を確認する。"""

import csv
import json
from dataclasses import replace

import pytest
from run_provenance_examples import TEST_RUN_PROVENANCE

from federated_learning_experiments.runtime.fedavg_run_settings import FedavgRunSettings
from federated_learning_experiments.sweep.experiment_sweep_command import main
from federated_learning_experiments.sweep.experiment_sweep_execution import (
    execute_experiment_sweep,
    find_existing_run_state,
)
from federated_learning_experiments.sweep.experiment_sweep_plan import (
    ExperimentSweepPlan,
    PlannedExperimentRun,
    build_planned_experiment_runs,
    make_run_identifier,
)
from federated_learning_experiments.sweep.experiment_sweep_records import write_run_summary_table
from tests.refactoring.test_fedavg_experiment_run import small_run_settings


def test_mixed_plan_uses_feddrift_axis_only_and_preserves_existing_identity():
    runs = build_planned_experiment_runs(
        sweep_plan=ExperimentSweepPlan(
            configuration_names=("fedsda", "feddrift", "fedavg"),
            dataset_names=("sine2",),
            random_seeds=(0, 1),
            client_count=2,
            per_client_sample_count=5,
            server_aggregation_intervals_per_client_samples=(3,),
            feddrift_loss_increase_thresholds=(0.0, 0.2),
        )
    )
    assert [r.configuration_name for r in runs] == ["fedsda"] * 2 + ["feddrift"] * 4 + [
        "fedavg"
    ] * 2
    # FedSDAの識別子は、候補検証の設定の項目を消したので変わった（基準版15のspec。前はfc28fa415515）。
    assert runs[0].run_identifier == "sine2_interval3_seed0_f9bff53a0cdc"
    assert all(type(r.run_settings) is FedavgRunSettings for r in runs[-2:])
    assert len({r.run_identifier for r in runs}) == 8


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.slow
def test_real_sweep_saves_resumes_and_refuses_conflict(tmp_path, workers):
    settings = small_run_settings(samples=5, interval=3)
    second = replace(
        settings,
        execution_settings=replace(
            settings.execution_settings,
            experiment_run_conditions=replace(
                settings.execution_settings.experiment_run_conditions, random_seed=38
            ),
        ),
    )
    runs = tuple(
        PlannedExperimentRun(
            configuration_name="fedavg",
            run_identifier=make_run_identifier(configuration_name="fedavg", run_settings=s),
            run_settings=s,
        )
        for s in (settings, second)
    )
    first = execute_experiment_sweep(
        planned_runs=runs,
        output_directory=tmp_path,
        worker_count=workers,
        run_provenance=TEST_RUN_PROVENANCE,
    )
    assert [o.run_status for o in first] == ["completed", "completed"], first
    repeated = execute_experiment_sweep(
        planned_runs=runs,
        output_directory=tmp_path,
        worker_count=workers,
        run_provenance=TEST_RUN_PROVENANCE,
    )
    assert [o.run_status for o in repeated] == ["skipped_existing"] * 2
    csv_path = write_run_summary_table(
        configuration_directory=tmp_path / "fedavg", configuration_name="fedavg"
    ).table_file_path
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and {r["configuration_name"] for r in rows} == {"fedavg"}
    assert all(r["final_global_model_count"] == "1" for r in rows)
    path = tmp_path / "fedavg" / "runs" / (runs[0].run_identifier + ".json")
    summary = json.loads(path.read_text("utf-8"))
    summary["method_name"] = "feddrift"
    path.write_text(json.dumps(summary), encoding="utf-8")
    assert (
        find_existing_run_state(planned_run=runs[0], output_directory=tmp_path).run_state
        == "conflicting"
    )
    assert (
        execute_experiment_sweep(
            planned_runs=runs,
            output_directory=tmp_path,
            worker_count=workers,
            run_provenance=TEST_RUN_PROVENANCE,
        )[0].run_status
        == "failed"
    )
    assert json.loads(path.read_text("utf-8"))["method_name"] == "feddrift"


def test_print_plan_is_read_only_and_old_name_is_rejected(tmp_path, capsys):
    output = tmp_path / "absent"
    arguments = [
        "--configurations",
        "fedavg",
        "--datasets",
        "sine2",
        "--seeds",
        "0",
        "--aggregation-intervals",
        "3",
        "--client-count",
        "2",
        "--per-client-sample-count",
        "5",
        "--workers",
        "1",
        "--purpose",
        "fedavg test",
        "--output-directory",
        str(output),
        "--print-plan",
    ]
    assert main(arguments) == 0
    assert "fedavg" in capsys.readouterr().out
    assert not output.exists()
    arguments[1] = "Oblivious"
    assert main(arguments) == 2
    assert not output.exists()
