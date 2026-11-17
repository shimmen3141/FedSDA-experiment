"""構成`dwm-neural-network`の掃引への接続（登録、計画、実行・保存・一覧・再開・競合）を確かめる。"""

import csv
import json

import pytest
from run_provenance_examples import TEST_RUN_PROVENANCE

from federated_learning_experiments.runtime.dynamic_weighted_majority_run_settings import (
    DynamicWeightedMajorityRunSettings,
)
from federated_learning_experiments.sweep import experiment_configurations as configurations
from federated_learning_experiments.sweep.experiment_sweep_command import main
from federated_learning_experiments.sweep.experiment_sweep_execution import (
    execute_experiment_sweep,
    find_existing_run_state,
)
from federated_learning_experiments.sweep.experiment_sweep_plan import (
    ExperimentSweepPlan,
    build_planned_experiment_runs,
)
from federated_learning_experiments.sweep.experiment_sweep_records import write_run_summary_table
from tests.refactoring import behavior_golden_cases as golden

NAME = "dwm-neural-network"


def test_configuration_is_registered_last_and_protected_by_its_own_test():
    assert configurations.DYNAMIC_WEIGHTED_MAJORITY_NEURAL_NETWORK_CONFIGURATION_NAME == NAME
    assert configurations.EXPERIMENT_CONFIGURATION_NAMES[-1] == NAME
    assert configurations.EXPERIMENT_CONFIGURATION_NAMES[-2] == (
        "fedsda-soft-training-single-model-prediction"
    )
    assert (
        golden.CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN[NAME]
        == "tests/refactoring/test_dynamic_weighted_majority_experiment_run.py"
    )


def test_plan_does_not_multiply_by_method_specific_axes_and_keeps_existing_identity():
    plan = dict(
        dataset_names=("sine2",),
        random_seeds=(0, 1),
        client_count=1,
        per_client_sample_count=6,
        server_aggregation_intervals_per_client_samples=(3,),
    )
    with_dwm = build_planned_experiment_runs(
        sweep_plan=ExperimentSweepPlan(
            configuration_names=("fedsda", "feddrift", "fedavg", NAME),
            feddrift_loss_increase_thresholds=(0.0, 0.2),
            **plan,
        )
    )
    without = build_planned_experiment_runs(
        sweep_plan=ExperimentSweepPlan(
            configuration_names=("fedsda", "feddrift", "fedavg"),
            feddrift_loss_increase_thresholds=(0.0, 0.2),
            **plan,
        )
    )
    dwm_runs = [r for r in with_dwm if r.configuration_name == NAME]
    assert len(dwm_runs) == 2
    assert all(type(r.run_settings) is DynamicWeightedMajorityRunSettings for r in dwm_runs)
    assert [r.run_identifier for r in with_dwm if r.configuration_name != NAME] == [
        r.run_identifier for r in without
    ]
    assert len({r.run_identifier for r in with_dwm}) == len(with_dwm)


@pytest.mark.slow
def test_real_sweep_saves_lists_resumes_and_refuses_conflict(tmp_path):
    runs = build_planned_experiment_runs(
        sweep_plan=ExperimentSweepPlan(
            configuration_names=(NAME,),
            dataset_names=("sine2",),
            random_seeds=(0, 1),
            client_count=1,
            per_client_sample_count=60,
            server_aggregation_intervals_per_client_samples=(20,),
        )
    )
    first = execute_experiment_sweep(
        planned_runs=runs,
        output_directory=tmp_path,
        worker_count=1,
        run_provenance=TEST_RUN_PROVENANCE,
    )
    assert [o.run_status for o in first] == ["completed", "completed"], first
    repeated = execute_experiment_sweep(
        planned_runs=runs,
        output_directory=tmp_path,
        worker_count=1,
        run_provenance=TEST_RUN_PROVENANCE,
    )
    assert [o.run_status for o in repeated] == ["skipped_existing"] * 2
    csv_path = write_run_summary_table(
        configuration_directory=tmp_path / NAME, configuration_name=NAME
    ).table_file_path
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and {r["configuration_name"] for r in rows} == {NAME}
    assert all(0.0 <= float(r["prediction_accuracy"]) <= 1.0 for r in rows)
    assert all(int(r["final_global_model_count"]) >= 1 for r in rows)
    # 片方だけのファイル（要約の`method_name`が違う）は競合として扱い、上書きしない。
    path = tmp_path / NAME / "runs" / (runs[0].run_identifier + ".json")
    summary = json.loads(path.read_text("utf-8"))
    summary["method_name"] = "fedavg"
    path.write_text(json.dumps(summary), encoding="utf-8")
    assert (
        find_existing_run_state(planned_run=runs[0], output_directory=tmp_path).run_state
        == "conflicting"
    )
    assert (
        execute_experiment_sweep(
            planned_runs=runs,
            output_directory=tmp_path,
            worker_count=1,
            run_provenance=TEST_RUN_PROVENANCE,
        )[0].run_status
        == "failed"
    )
    assert json.loads(path.read_text("utf-8"))["method_name"] == "fedavg"
    (tmp_path / NAME / "runs" / (runs[1].run_identifier + ".npz")).unlink()
    assert (
        find_existing_run_state(planned_run=runs[1], output_directory=tmp_path).run_state
        == "conflicting"
    )


def test_print_plan_accepts_the_configuration_and_is_read_only(tmp_path, capsys):
    output = tmp_path / "absent"
    arguments = [
        "--configurations",
        NAME,
        "--datasets",
        "sine2",
        "--seeds",
        "0",
        "--aggregation-intervals",
        "3",
        "--client-count",
        "1",
        "--per-client-sample-count",
        "6",
        "--workers",
        "1",
        "--purpose",
        "dwm test",
        "--output-directory",
        str(output),
        "--print-plan",
    ]
    assert main(arguments) == 0
    assert NAME in capsys.readouterr().out
    assert not output.exists()
