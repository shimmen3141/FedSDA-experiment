"""実験の掃引（計画、逐次・並列の実行、再開、一覧の表、実行の記録、コマンド）を確かめる。"""

import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_final_configuration_run_settings import (
    build_final_configuration_fedsda_run_settings,
)
from federated_learning_experiments.runtime.fedsda_run_artifact_files import (
    load_fedsda_run_artifacts,
)
from federated_learning_experiments.runtime.fedsda_run_record_tables import (
    build_fedsda_run_record_tables,
)
from federated_learning_experiments.runtime.fedsda_run_settings import FedsdaRunSettings
from federated_learning_experiments.sweep import (
    experiment_sweep_command,
    experiment_sweep_execution,
)
from federated_learning_experiments.sweep.experiment_configurations import (
    EXPERIMENT_CONFIGURATION_NAMES,
    PROPOSED_FEDSDA_CONFIGURATION_NAME,
    build_experiment_run_settings,
)
from federated_learning_experiments.sweep.experiment_sweep_execution import (
    ExistingRunState,
    ExperimentRunOutcome,
    execute_experiment_sweep,
    find_existing_run_state,
)
from federated_learning_experiments.sweep.experiment_sweep_plan import (
    ExperimentSweepPlan,
    PlannedExperimentRun,
    build_planned_experiment_runs,
    make_run_identifier,
)
from federated_learning_experiments.sweep.experiment_sweep_records import (
    RunSummaryTableResult,
    describe_code_revision,
    write_run_summary_table,
    write_sweep_record,
)

REPOSITORY_DIRECTORY = Path(__file__).resolve().parents[2]
# 小さい規模の計画（1 runが数秒）。
SMALL_SWEEP_PLAN = ExperimentSweepPlan(
    configuration_names=(PROPOSED_FEDSDA_CONFIGURATION_NAME,),
    dataset_names=("circle2",),
    random_seeds=(0, 1),
    client_count=2,
    per_client_sample_count=65,
    server_aggregation_intervals_per_client_samples=(10,),
)
SMALL_COMMAND_ARGUMENTS = [
    "--datasets",
    "circle2",
    "--seeds",
    "0",
    "--client-count",
    "2",
    "--per-client-sample-count",
    "65",
    "--aggregation-intervals",
    "10",
]


# ---- 構成の名前 ----


def test_proposed_configuration_builds_final_configuration_run_settings():
    """提案構成の名前は、最終構成の既定値の、完全なrun設定を作る。"""
    assert PROPOSED_FEDSDA_CONFIGURATION_NAME == "fedsda"
    assert EXPERIMENT_CONFIGURATION_NAMES == ("fedsda",)
    conditions = ExperimentRunConditions(
        dataset_name="sine2",
        random_seed=3,
        client_count=4,
        per_client_sample_count=200,
        server_aggregation_interval_per_client_samples=50,
    )
    run_settings = build_experiment_run_settings(
        configuration_name="fedsda", experiment_run_conditions=conditions
    )
    assert type(run_settings) is FedsdaRunSettings
    assert run_settings == build_final_configuration_fedsda_run_settings(
        experiment_run_conditions=conditions
    )
    with pytest.raises(ValueError, match="unknown experiment configuration"):
        build_experiment_run_settings(
            configuration_name="fedsda-unknown", experiment_run_conditions=conditions
        )
    with pytest.raises(TypeError):
        build_experiment_run_settings(configuration_name=1, experiment_run_conditions=conditions)
    with pytest.raises(TypeError):
        build_experiment_run_settings(configuration_name="fedsda", experiment_run_conditions=None)


# ---- 計画 ----


def test_planned_runs_cover_every_combination_in_fixed_order():
    """計画は、全部の組合せを、構成・dataset・集約間隔・seedの順で返す。"""
    sweep_plan = ExperimentSweepPlan(
        configuration_names=("fedsda",),
        dataset_names=("sine2", "sea4"),
        random_seeds=(1, 0),
        client_count=3,
        per_client_sample_count=120,
        server_aggregation_intervals_per_client_samples=(50, 20),
    )
    planned_runs = build_planned_experiment_runs(sweep_plan=sweep_plan)
    assert type(planned_runs) is tuple
    assert all(type(planned_run) is PlannedExperimentRun for planned_run in planned_runs)
    conditions = [
        planned_run.run_settings.execution_settings.experiment_run_conditions
        for planned_run in planned_runs
    ]
    assert [
        (
            condition.dataset_name,
            condition.server_aggregation_interval_per_client_samples,
            condition.random_seed,
        )
        for condition in conditions
    ] == [
        (dataset_name, aggregation_interval, random_seed)
        for dataset_name in ("sine2", "sea4")
        for aggregation_interval in (50, 20)
        for random_seed in (1, 0)
    ]
    assert {condition.client_count for condition in conditions} == {3}
    assert {condition.per_client_sample_count for condition in conditions} == {120}
    assert {planned_run.configuration_name for planned_run in planned_runs} == {"fedsda"}
    for planned_run in planned_runs:
        assert planned_run.run_settings == build_experiment_run_settings(
            configuration_name="fedsda",
            experiment_run_conditions=(
                planned_run.run_settings.execution_settings.experiment_run_conditions
            ),
        )
        assert planned_run.run_identifier == make_run_identifier(
            configuration_name="fedsda", run_settings=planned_run.run_settings
        )
    with pytest.raises(FrozenInstanceError):
        planned_runs[0].run_identifier = "other"  # type: ignore[misc]


def test_run_identifiers_are_readable_stable_and_unique():
    """識別子は、dataset・集約間隔・seedと、12桁のhashを持ち、同じ条件なら同じ、条件が違えば違う。"""
    planned_runs = build_planned_experiment_runs(
        sweep_plan=replace(
            SMALL_SWEEP_PLAN,
            dataset_names=("circle2", "sine2"),
            server_aggregation_intervals_per_client_samples=(10, 20),
        )
    )
    run_identifiers = [planned_run.run_identifier for planned_run in planned_runs]
    assert len(set(run_identifiers)) == len(run_identifiers) == 8
    assert re.fullmatch(r"circle2_interval10_seed0_[0-9a-f]{12}", run_identifiers[0])
    assert run_identifiers == [
        planned_run.run_identifier
        for planned_run in build_planned_experiment_runs(
            sweep_plan=replace(
                SMALL_SWEEP_PLAN,
                dataset_names=("circle2", "sine2"),
                server_aggregation_intervals_per_client_samples=(10, 20),
            )
        )
    ]
    # 読める部分が同じでも、規模が違えば、hashが違う。
    other_scale_identifier = build_planned_experiment_runs(
        sweep_plan=replace(SMALL_SWEEP_PLAN, per_client_sample_count=66)
    )[0].run_identifier
    assert (
        other_scale_identifier
        != build_planned_experiment_runs(sweep_plan=SMALL_SWEEP_PLAN)[0].run_identifier
    )
    assert other_scale_identifier.startswith("circle2_interval10_seed0_")
    # 構成の名前も、hashに入る。
    run_settings = planned_runs[0].run_settings
    assert make_run_identifier(
        configuration_name="fedsda", run_settings=run_settings
    ) != make_run_identifier(configuration_name="fedsda-other", run_settings=run_settings)
    with pytest.raises(TypeError):
        make_run_identifier(configuration_name="fedsda", run_settings=None)
    with pytest.raises(ValueError):
        make_run_identifier(configuration_name="", run_settings=run_settings)


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "error_type"),
    [
        ("configuration_names", (), ValueError),
        ("configuration_names", ["fedsda"], TypeError),
        ("configuration_names", ("fedsda", "fedsda"), ValueError),
        ("configuration_names", ("fedsda-unknown",), ValueError),
        ("configuration_names", (1,), TypeError),
        ("dataset_names", (), ValueError),
        ("dataset_names", ("circle2", "circle2"), ValueError),
        ("dataset_names", (2,), TypeError),
        ("random_seeds", (), ValueError),
        ("random_seeds", (0, 0), ValueError),
        ("random_seeds", (True,), TypeError),
        ("random_seeds", [0], TypeError),
        ("client_count", 2.0, TypeError),
        ("client_count", True, TypeError),
        ("per_client_sample_count", "65", TypeError),
        ("server_aggregation_intervals_per_client_samples", (), ValueError),
        ("server_aggregation_intervals_per_client_samples", (10, 10), ValueError),
        ("server_aggregation_intervals_per_client_samples", (10.0,), TypeError),
    ],
)
def test_sweep_plan_rejects_invalid_values(field_name, invalid_value, error_type):
    with pytest.raises(error_type):
        replace(SMALL_SWEEP_PLAN, **{field_name: invalid_value})


@pytest.mark.parametrize(
    "changed_fields",
    [
        dict(dataset_names=("blobs",)),
        dict(random_seeds=(-1,)),
        dict(client_count=0),
        dict(per_client_sample_count=0),
        dict(server_aggregation_intervals_per_client_samples=(0,)),
    ],
)
def test_planned_runs_reject_invalid_conditions_before_running(changed_fields):
    """知らないdatasetや、範囲の外の値は、計画を作るときに、拒否する。"""
    with pytest.raises(ValueError):
        build_planned_experiment_runs(sweep_plan=replace(SMALL_SWEEP_PLAN, **changed_fields))
    with pytest.raises(TypeError):
        build_planned_experiment_runs(sweep_plan=None)


# ---- 実行 ----


def read_run_files(output_directory, planned_run):
    return load_fedsda_run_artifacts(
        summary_file_path=(
            output_directory
            / planned_run.configuration_name
            / "runs"
            / f"{planned_run.run_identifier}.json"
        )
    )


def assert_record_tables_equal(record_tables, other_record_tables):
    assert set(record_tables) == set(other_record_tables)
    for table_column_name, column in record_tables.items():
        other_column = other_record_tables[table_column_name]
        assert column.dtype == other_column.dtype, table_column_name
        assert column.shape == other_column.shape, table_column_name
        assert np.array_equal(column, other_column, equal_nan=column.dtype.kind == "f"), (
            table_column_name
        )


@pytest.fixture(scope="module")
def sequential_sweep(tmp_path_factory):
    """小さい計画を、逐次で、1回実行する。"""
    output_directory = tmp_path_factory.mktemp("sequential-sweep")
    planned_runs = build_planned_experiment_runs(sweep_plan=SMALL_SWEEP_PLAN)
    progress_messages = []
    thread_count_before = torch.get_num_threads()
    outcomes = execute_experiment_sweep(
        planned_runs=planned_runs,
        output_directory=output_directory,
        worker_count=1,
        report_progress=progress_messages.append,
    )
    # 逐次の実行は、呼出し側のprocessの、計算のthreadの数を、元へ戻す。
    assert torch.get_num_threads() == thread_count_before
    return output_directory, planned_runs, outcomes, progress_messages


def test_sequential_sweep_saves_the_same_content_as_direct_run(sequential_sweep):
    """逐次の実行が保存した内容は、同じ設定で、1 runを直接実行して保存する内容と、同じ。"""
    output_directory, planned_runs, outcomes, progress_messages = sequential_sweep
    assert outcomes == tuple(
        ExperimentRunOutcome(
            configuration_name="fedsda",
            run_identifier=planned_run.run_identifier,
            run_status="completed",
            failure_description=None,
        )
        for planned_run in planned_runs
    )
    assert sorted(path.name for path in (output_directory / "fedsda" / "runs").iterdir()) == sorted(
        f"{planned_run.run_identifier}{suffix}"
        for planned_run in planned_runs
        for suffix in (".json", ".npz")
    )
    assert len(progress_messages) == len(planned_runs)
    assert all("completed" in message for message in progress_messages)
    for planned_run in planned_runs:
        summary, record_tables = read_run_files(output_directory, planned_run)
        experiment_run = execute_fedsda_experiment_run(run_settings=planned_run.run_settings)
        assert summary["run_settings"] == json.loads(
            json.dumps(convert_settings_to_plain_mapping(planned_run.run_settings))
        )
        assert summary["run_metrics"] == json.loads(
            json.dumps(
                convert_settings_to_plain_mapping(
                    experiment_run.run_metrics, non_finite_float_as_none=True
                )
            )
        )
        assert_record_tables_equal(
            record_tables, build_fedsda_run_record_tables(experiment_run=experiment_run)
        )
    # 2つのrun（seedが違う）は、違う結果になっている。
    assert (
        read_run_files(output_directory, planned_runs[0])[0]["run_metrics"]
        != read_run_files(output_directory, planned_runs[1])[0]["run_metrics"]
    )


def test_completed_runs_are_skipped_and_left_unchanged(sequential_sweep, monkeypatch):
    """済んだrunは、実行せずに飛ばし、ファイルを変えない。"""
    output_directory, planned_runs, _, _ = sequential_sweep
    runs_directory = output_directory / "fedsda" / "runs"
    file_contents_before = {path.name: path.read_bytes() for path in runs_directory.iterdir()}
    for planned_run in planned_runs:
        assert find_existing_run_state(
            planned_run=planned_run, output_directory=output_directory
        ) == ExistingRunState(run_state="completed", description=None)

    def reject_execution(**_):
        raise AssertionError("a completed run must not be executed again")

    monkeypatch.setattr(
        experiment_sweep_execution, "execute_fedsda_experiment_run", reject_execution
    )
    progress_messages = []
    outcomes = execute_experiment_sweep(
        planned_runs=planned_runs,
        output_directory=output_directory,
        worker_count=1,
        report_progress=progress_messages.append,
    )
    assert [outcome.run_status for outcome in outcomes] == ["skipped_existing"] * 2
    assert all(outcome.failure_description is None for outcome in outcomes)
    assert all("skipped_existing" in message for message in progress_messages)
    assert {
        path.name: path.read_bytes() for path in runs_directory.iterdir()
    } == file_contents_before


def copy_sweep_output(source_directory, destination_directory):
    for source_path in source_directory.rglob("*"):
        destination_path = destination_directory / source_path.relative_to(source_directory)
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
        else:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.write_bytes(source_path.read_bytes())


@pytest.mark.parametrize(
    "damage", ["different_run_settings", "missing_record_table", "missing_summary", "invalid_json"]
)
def test_conflicting_existing_files_are_reported_and_not_overwritten(
    sequential_sweep, tmp_path, monkeypatch, damage
):
    """同じ識別子のファイルが、片方しかない、または、計画と違うとき、上書きせず、失敗として報告する。"""
    source_directory, planned_runs, _, _ = sequential_sweep
    copy_sweep_output(source_directory, tmp_path)
    damaged_run, intact_run = planned_runs
    runs_directory = tmp_path / "fedsda" / "runs"
    summary_file_path = runs_directory / f"{damaged_run.run_identifier}.json"
    record_table_file_path = runs_directory / f"{damaged_run.run_identifier}.npz"
    if damage == "different_run_settings":
        summary = json.loads(summary_file_path.read_text(encoding="utf-8"))
        summary["run_settings"]["run_metric_settings"]["maximum_detection_delay_sample_count"] += 1
        summary_file_path.write_text(json.dumps(summary), encoding="utf-8")
    elif damage == "missing_record_table":
        record_table_file_path.unlink()
    elif damage == "missing_summary":
        summary_file_path.unlink()
    else:
        summary_file_path.write_text("{not json", encoding="utf-8")
    file_contents_before = {path.name: path.read_bytes() for path in runs_directory.iterdir()}
    existing_run_state = find_existing_run_state(planned_run=damaged_run, output_directory=tmp_path)
    assert existing_run_state.run_state == "conflicting"
    assert existing_run_state.description

    def reject_execution(**_):
        raise AssertionError("no run must be executed here")

    monkeypatch.setattr(
        experiment_sweep_execution, "execute_fedsda_experiment_run", reject_execution
    )
    outcomes = execute_experiment_sweep(
        planned_runs=planned_runs, output_directory=tmp_path, worker_count=1
    )
    assert [outcome.run_status for outcome in outcomes] == ["failed", "skipped_existing"]
    assert outcomes[0].run_identifier == damaged_run.run_identifier
    assert outcomes[0].failure_description == existing_run_state.description
    assert outcomes[1].run_identifier == intact_run.run_identifier
    assert {
        path.name: path.read_bytes() for path in runs_directory.iterdir()
    } == file_contents_before


def test_failed_run_is_reported_and_other_runs_continue(tmp_path, monkeypatch):
    """あるrunが例外で失敗しても、ほかのrunを続け、失敗したrunと理由を報告する。"""
    planned_runs = build_planned_experiment_runs(sweep_plan=SMALL_SWEEP_PLAN)
    real_execution = experiment_sweep_execution.execute_fedsda_experiment_run

    def fail_first_seed(*, run_settings):
        if run_settings.execution_settings.experiment_run_conditions.random_seed == 0:
            raise RuntimeError("simulated run failure")
        return real_execution(run_settings=run_settings)

    monkeypatch.setattr(
        experiment_sweep_execution, "execute_fedsda_experiment_run", fail_first_seed
    )
    progress_messages = []
    outcomes = execute_experiment_sweep(
        planned_runs=planned_runs,
        output_directory=tmp_path,
        worker_count=1,
        report_progress=progress_messages.append,
    )
    assert [outcome.run_status for outcome in outcomes] == ["failed", "completed"]
    assert "RuntimeError: simulated run failure" in outcomes[0].failure_description
    assert outcomes[1].failure_description is None
    assert sorted(path.name for path in (tmp_path / "fedsda" / "runs").iterdir()) == [
        f"{planned_runs[1].run_identifier}.json",
        f"{planned_runs[1].run_identifier}.npz",
    ]
    assert "failed" in progress_messages[0] and "simulated run failure" in progress_messages[0]


def test_parallel_sweep_saves_the_same_content_as_sequential_sweep(
    sequential_sweep, tmp_path, monkeypatch
):
    """別々のprocessでの並列の実行が保存した内容は、逐次の実行と、同じ。

    並列の実行は、workerのprocessを作る前に、数値計算のライブラリのthreadの数の環境変数を、未設定なら1にする
    （workerのprocessは、その環境を受け継ぐ）。設定されている値は、変えない。
    """
    sequential_directory, planned_runs, _, _ = sequential_sweep
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")
    progress_messages = []
    outcomes = execute_experiment_sweep(
        planned_runs=planned_runs,
        output_directory=tmp_path,
        worker_count=2,
        report_progress=progress_messages.append,
    )
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "3"
    # 結果は、計画の順で返る。
    assert [(outcome.run_identifier, outcome.run_status) for outcome in outcomes] == [
        (planned_run.run_identifier, "completed") for planned_run in planned_runs
    ]
    assert len(progress_messages) == len(planned_runs)
    for planned_run in planned_runs:
        summary, record_tables = read_run_files(tmp_path, planned_run)
        sequential_summary, sequential_record_tables = read_run_files(
            sequential_directory, planned_run
        )
        assert summary == sequential_summary
        assert_record_tables_equal(record_tables, sequential_record_tables)


def test_parallel_sweep_cancels_waiting_runs_when_interrupted(tmp_path):
    """並列の実行の途中で、例外（進み具合の報告の失敗、中断）が出たら、workerへ渡す前の、待ちのrunを取り消す。"""
    planned_runs = build_planned_experiment_runs(
        sweep_plan=replace(SMALL_SWEEP_PLAN, random_seeds=tuple(range(8)))
    )

    def interrupt_on_first_outcome(progress_message):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute_experiment_sweep(
            planned_runs=planned_runs,
            output_directory=tmp_path,
            worker_count=2,
            report_progress=interrupt_on_first_outcome,
        )
    # 戻った時点で、workerのprocessは終わっている（この後、ファイルは増えない）。
    saved_run_count = len(list((tmp_path / "fedsda" / "runs").glob("*.json")))
    assert 1 <= saved_run_count < len(planned_runs)
    # 済んだrunは、次の実行で飛ばされ、残りだけが実行される。
    outcomes = execute_experiment_sweep(
        planned_runs=planned_runs, output_directory=tmp_path, worker_count=2
    )
    assert [outcome.run_status for outcome in outcomes].count("skipped_existing") == saved_run_count
    assert [outcome.run_status for outcome in outcomes].count("completed") == (
        len(planned_runs) - saved_run_count
    )


def test_sweep_execution_rejects_invalid_arguments(tmp_path):
    planned_runs = build_planned_experiment_runs(sweep_plan=SMALL_SWEEP_PLAN)
    valid_arguments = dict(planned_runs=planned_runs, output_directory=tmp_path, worker_count=1)
    for changed_arguments, error_type in (
        (dict(planned_runs=list(planned_runs)), TypeError),
        (dict(planned_runs=(planned_runs[0], None)), TypeError),
        (dict(planned_runs=()), ValueError),
        (dict(planned_runs=(planned_runs[0], planned_runs[0])), ValueError),
        (dict(output_directory=str(tmp_path)), TypeError),
        (dict(worker_count=0), ValueError),
        (dict(worker_count=True), TypeError),
        (dict(worker_count=1.0), TypeError),
        (dict(report_progress="print"), TypeError),
    ):
        with pytest.raises(error_type):
            execute_experiment_sweep(**{**valid_arguments, **changed_arguments})
    # 拒否した呼出しは、何も作らない。
    assert list(tmp_path.iterdir()) == []


# ---- 一覧の表と、実行の記録 ----


def test_run_summary_table_holds_one_row_per_saved_run(sequential_sweep, tmp_path):
    """一覧の表は、`runs/`の全部の要約から、1 runを1行として作り直される。"""
    source_directory, planned_runs, _, _ = sequential_sweep
    copy_sweep_output(source_directory, tmp_path)
    configuration_directory = tmp_path / "fedsda"
    table_result = write_run_summary_table(
        configuration_directory=configuration_directory, configuration_name="fedsda"
    )
    assert type(table_result) is RunSummaryTableResult
    assert table_result.unreadable_run_summary_descriptions == ()
    table_file_path = table_result.table_file_path
    assert table_file_path == configuration_directory / "run_summary.csv"
    with table_file_path.open(encoding="utf-8", newline="") as table_file:
        rows = list(csv.DictReader(table_file))
    assert [row["run_identifier"] for row in rows] == [
        planned_run.run_identifier for planned_run in planned_runs
    ]
    column_names = list(rows[0])
    assert column_names[:7] == [
        "run_identifier",
        "configuration_name",
        "dataset_name",
        "random_seed",
        "client_count",
        "per_client_sample_count",
        "server_aggregation_interval_per_client_samples",
    ]
    assert "prediction_accuracy" in column_names
    assert "communication_volume.uploaded_model_count" in column_names
    assert (
        "prediction_diagnostic_metrics.prediction_correct_counts.predicted_sample_count"
        in column_names
    )
    assert not [column_name for column_name in column_names if "settings_type" in column_name]
    for row, planned_run in zip(rows, planned_runs, strict=True):
        summary, _ = read_run_files(tmp_path, planned_run)
        assert row["configuration_name"] == "fedsda"
        assert row["dataset_name"] == "circle2"
        assert row["random_seed"] == str(
            planned_run.run_settings.execution_settings.experiment_run_conditions.random_seed
        )
        assert row["client_count"] == "2" and row["per_client_sample_count"] == "65"
        assert row["server_aggregation_interval_per_client_samples"] == "10"
        assert float(row["prediction_accuracy"]) == summary["run_metrics"]["prediction_accuracy"]
        assert (
            int(row["communication_volume.uploaded_model_count"])
            == (summary["run_metrics"]["communication_volume"]["uploaded_model_count"])
        )
    # 値のない指標（null）は、空の欄。
    null_metric_paths = [
        "model_diagnostic_metrics.clustering_true_concept_agreement_metrics"
        ".loss_increase_distance_same_concept_auc"
    ]
    for null_metric_path in null_metric_paths:
        assert {row[null_metric_path] for row in rows} == {""}
    # 要約を1つ除いて作り直すと、行が減る（表は、いまある要約だけから作る）。
    (configuration_directory / "runs" / f"{planned_runs[0].run_identifier}.json").unlink()
    write_run_summary_table(
        configuration_directory=configuration_directory, configuration_name="fedsda"
    )
    with table_file_path.open(encoding="utf-8", newline="") as table_file:
        assert [row["run_identifier"] for row in csv.DictReader(table_file)] == [
            planned_runs[1].run_identifier
        ]
    assert sorted(path.name for path in configuration_directory.iterdir()) == [
        "run_summary.csv",
        "runs",
    ]
    # 読めない要約（JSONでない、要るキーがない）は、表から外して、理由を返す。ほかの行は、作る。
    (configuration_directory / "runs" / "broken.json").write_text("{not json", encoding="utf-8")
    (configuration_directory / "runs" / "incomplete.json").write_text(
        json.dumps({"run_settings": {}}), encoding="utf-8"
    )
    # 実行条件の値の型が違う要約（行の並べ替えを壊す）。
    mistyped_summary = json.loads(
        (configuration_directory / "runs" / f"{planned_runs[1].run_identifier}.json").read_text(
            encoding="utf-8"
        )
    )
    mistyped_summary["run_settings"]["execution_settings"]["experiment_run_conditions"][
        "random_seed"
    ] = None
    (configuration_directory / "runs" / "mistyped.json").write_text(
        json.dumps(mistyped_summary), encoding="utf-8"
    )
    broken_table_result = write_run_summary_table(
        configuration_directory=configuration_directory, configuration_name="fedsda"
    )
    assert len(broken_table_result.unreadable_run_summary_descriptions) == 3
    assert "broken.json" in broken_table_result.unreadable_run_summary_descriptions[0]
    assert "incomplete.json" in broken_table_result.unreadable_run_summary_descriptions[1]
    assert "mistyped.json" in broken_table_result.unreadable_run_summary_descriptions[2]
    with table_file_path.open(encoding="utf-8", newline="") as table_file:
        assert [row["run_identifier"] for row in csv.DictReader(table_file)] == [
            planned_runs[1].run_identifier
        ]
    with pytest.raises(FileNotFoundError):
        write_run_summary_table(
            configuration_directory=tmp_path / "missing", configuration_name="fedsda"
        )
    with pytest.raises(TypeError):
        write_run_summary_table(configuration_directory=str(tmp_path), configuration_name="fedsda")


def test_code_revision_describes_commit_or_reason(tmp_path):
    """コードの版は、gitのcommitと、未コミットの変更の有無。取れなければ、理由。"""
    code_revision = describe_code_revision(repository_directory=REPOSITORY_DIRECTORY)
    assert set(code_revision) == {"commit", "has_uncommitted_changes", "unavailable_reason"}
    if code_revision["unavailable_reason"] is None:
        assert re.fullmatch(r"[0-9a-f]{40}", code_revision["commit"])
        assert type(code_revision["has_uncommitted_changes"]) is bool
    else:
        # gitが、この作業ツリーを読めない環境（WSLから見た、Windowsのworktree）。
        assert code_revision["commit"] is None
        assert code_revision["has_uncommitted_changes"] is None
    # Windowsの基準環境では、取れる。
    assert code_revision["unavailable_reason"] is None or sys.platform != "win32"
    # gitのリポジトリでない場所。
    outside_revision = describe_code_revision(repository_directory=tmp_path)
    assert outside_revision["commit"] is None
    assert outside_revision["has_uncommitted_changes"] is None
    assert outside_revision["unavailable_reason"]
    with pytest.raises(TypeError):
        describe_code_revision(repository_directory=str(tmp_path))


def test_sweep_record_holds_plan_command_revision_and_outcomes(tmp_path):
    outcomes = (
        ExperimentRunOutcome(
            configuration_name="fedsda",
            run_identifier="circle2_interval10_seed0_000000000000",
            run_status="completed",
            failure_description=None,
        ),
        ExperimentRunOutcome(
            configuration_name="fedsda",
            run_identifier="circle2_interval10_seed1_111111111111",
            run_status="failed",
            failure_description="RuntimeError: simulated",
        ),
    )
    arguments = dict(
        output_directory=tmp_path,
        started_at="2026-10-11T12:00:00+09:00",
        finished_at="2026-10-11T12:00:09+09:00",
        command_arguments=("--datasets", "circle2"),
        sweep_plan=SMALL_SWEEP_PLAN,
        worker_count=2,
        outcomes=outcomes,
        code_revision=dict(commit="a" * 40, has_uncommitted_changes=False, unavailable_reason=None),
        unreadable_run_summary_descriptions=("fedsda/runs/broken.json: JSONDecodeError",),
    )
    record_file_path = write_sweep_record(**arguments)
    assert record_file_path == tmp_path / "sweep_record_20261011_120000.json"
    record = json.loads(record_file_path.read_text(encoding="utf-8"))
    assert list(record) == [
        "record_format_version",
        "started_at",
        "finished_at",
        "command_arguments",
        "sweep_plan",
        "worker_count",
        "code_revision",
        "environment",
        "run_status_counts",
        "run_outcomes",
        "unreadable_run_summary_descriptions",
    ]
    assert record["record_format_version"] == 1
    assert record["command_arguments"] == ["--datasets", "circle2"]
    assert record["sweep_plan"] == dict(
        configuration_names=["fedsda"],
        dataset_names=["circle2"],
        random_seeds=[0, 1],
        client_count=2,
        per_client_sample_count=65,
        server_aggregation_intervals_per_client_samples=[10],
    )
    assert record["worker_count"] == 2
    assert record["code_revision"]["commit"] == "a" * 40
    assert set(record["environment"]) == {
        "system",
        "machine",
        "python",
        "numpy",
        "torch",
        "torch_thread_count",
        "thread_count_environment_variables",
    }
    assert record["environment"]["torch_thread_count"] == torch.get_num_threads()
    assert set(record["environment"]["thread_count_environment_variables"]) == {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    }
    assert record["unreadable_run_summary_descriptions"] == [
        "fedsda/runs/broken.json: JSONDecodeError"
    ]
    assert record["run_status_counts"] == dict(completed=1, skipped_existing=0, failed=1)
    assert record["run_outcomes"] == [
        dict(
            configuration_name=outcome.configuration_name,
            run_identifier=outcome.run_identifier,
            run_status=outcome.run_status,
            failure_description=outcome.failure_description,
        )
        for outcome in outcomes
    ]
    # 同じ開始時刻の記録は、上書きせず、番号をつけた別の名前にする。
    record_text_before = record_file_path.read_text(encoding="utf-8")
    assert write_sweep_record(**arguments) == tmp_path / "sweep_record_20261011_120000_2.json"
    assert write_sweep_record(**arguments) == tmp_path / "sweep_record_20261011_120000_3.json"
    assert record_file_path.read_text(encoding="utf-8") == record_text_before
    with pytest.raises(ValueError):
        write_sweep_record(**{**arguments, "started_at": "yesterday"})
    with pytest.raises(TypeError):
        write_sweep_record(**{**arguments, "outcomes": list(outcomes)})


# ---- コマンド ----


def test_print_plan_lists_runs_and_creates_nothing(tmp_path, capsys):
    """`--print-plan`は、計画のrunと、済んだrunを表示し、何も実行せず、何も作らない。"""
    output_directory = tmp_path / "not-created"
    exit_code = experiment_sweep_command.main(
        [*SMALL_COMMAND_ARGUMENTS, "--output-directory", str(output_directory), "--print-plan"]
    )
    assert exit_code == 0
    assert not output_directory.exists()
    printed_text = capsys.readouterr().out
    planned_run = build_planned_experiment_runs(
        sweep_plan=replace(SMALL_SWEEP_PLAN, random_seeds=(0,))
    )[0]
    assert planned_run.run_identifier in printed_text
    assert "missing" in printed_text
    assert str(output_directory) in printed_text


def test_command_runs_sweep_and_writes_table_and_record(tmp_path, capsys):
    """コマンドは、runを実行し、一覧の表と、実行の記録を書き、終了コード0を返す。2回目は、済んだrunを飛ばす。"""
    output_directory = tmp_path / "command-output"
    command_arguments = [*SMALL_COMMAND_ARGUMENTS, "--output-directory", str(output_directory)]
    assert experiment_sweep_command.main(command_arguments) == 0
    planned_run = build_planned_experiment_runs(
        sweep_plan=replace(SMALL_SWEEP_PLAN, random_seeds=(0,))
    )[0]
    summary, _ = read_run_files(output_directory, planned_run)
    assert summary["run_settings"]["execution_settings"]["experiment_run_conditions"] == dict(
        settings_type="ExperimentRunConditions",
        dataset_name="circle2",
        random_seed=0,
        client_count=2,
        per_client_sample_count=65,
        server_aggregation_interval_per_client_samples=10,
    )
    with (output_directory / "fedsda" / "run_summary.csv").open(
        encoding="utf-8", newline=""
    ) as table_file:
        assert [row["run_identifier"] for row in csv.DictReader(table_file)] == [
            planned_run.run_identifier
        ]
    record_file_paths = sorted(output_directory.glob("sweep_record_*.json"))
    assert len(record_file_paths) == 1
    record = json.loads(record_file_paths[0].read_text(encoding="utf-8"))
    assert record["command_arguments"] == command_arguments
    assert record["run_status_counts"] == dict(completed=1, skipped_existing=0, failed=0)
    assert record["code_revision"]["commit"] or record["code_revision"]["unavailable_reason"]
    assert "completed" in capsys.readouterr().out
    # `--print-plan`は、済んだrunを、済んだものとして表示する。
    assert experiment_sweep_command.main([*command_arguments, "--print-plan"]) == 0
    assert "completed" in capsys.readouterr().out
    assert len(sorted(output_directory.glob("sweep_record_*.json"))) == 1


def test_command_writes_record_even_when_a_run_summary_is_unreadable(tmp_path, capsys):
    """出力先に、読めない要約があっても、ほかのrunの表と、実行の記録を書き、失敗として報告する。"""
    output_directory = tmp_path / "broken-output"
    run_directory = output_directory / "fedsda" / "runs"
    run_directory.mkdir(parents=True)
    (run_directory / "unrelated_broken_run.json").write_text("{not json", encoding="utf-8")
    command_arguments = [*SMALL_COMMAND_ARGUMENTS, "--output-directory", str(output_directory)]
    assert experiment_sweep_command.main(command_arguments) == 1
    record = json.loads(
        next(output_directory.glob("sweep_record_*.json")).read_text(encoding="utf-8")
    )
    assert record["run_status_counts"] == dict(completed=1, skipped_existing=0, failed=0)
    assert len(record["unreadable_run_summary_descriptions"]) == 1
    assert "unrelated_broken_run.json" in record["unreadable_run_summary_descriptions"][0]
    with (output_directory / "fedsda" / "run_summary.csv").open(
        encoding="utf-8", newline=""
    ) as table_file:
        assert len(list(csv.DictReader(table_file))) == 1
    assert "unrelated_broken_run.json" in capsys.readouterr().out


def test_command_returns_nonzero_when_a_run_fails(tmp_path, monkeypatch, capsys):
    def fail_execution(*, run_settings):
        raise RuntimeError("simulated run failure")

    monkeypatch.setattr(experiment_sweep_execution, "execute_fedsda_experiment_run", fail_execution)
    output_directory = tmp_path / "failed-output"
    assert (
        experiment_sweep_command.main(
            [*SMALL_COMMAND_ARGUMENTS, "--output-directory", str(output_directory)]
        )
        == 1
    )
    record = json.loads(
        next(output_directory.glob("sweep_record_*.json")).read_text(encoding="utf-8")
    )
    assert record["run_status_counts"] == dict(completed=0, skipped_existing=0, failed=1)
    assert "simulated run failure" in record["run_outcomes"][0]["failure_description"]
    assert "failed" in capsys.readouterr().out


def test_command_stops_before_running_when_mnist_files_are_missing(tmp_path, monkeypatch, capsys):
    """MNISTのdatasetを計画に含み、ファイルが置かれていないとき、実行の前に、置き場所を示して止まる。"""
    empty_data_directory = tmp_path / "empty-mnist"
    empty_data_directory.mkdir()
    monkeypatch.setenv("FDE_MNIST_DATA_DIR", str(empty_data_directory))
    output_directory = tmp_path / "mnist-output"
    exit_code = experiment_sweep_command.main(
        [
            "--datasets",
            "mnist2",
            "--seeds",
            "0",
            "--client-count",
            "2",
            "--per-client-sample-count",
            "65",
            "--aggregation-intervals",
            "10",
            "--output-directory",
            str(output_directory),
        ]
    )
    assert exit_code == 2
    assert not output_directory.exists()
    error_text = capsys.readouterr().err
    assert str(empty_data_directory.resolve()) in error_text
    assert "FDE_MNIST_DATA_DIR" in error_text


@pytest.mark.parametrize(
    "invalid_arguments",
    [
        ["--datasets", "blobs", "--seeds", "0", "--aggregation-intervals", "10"],
        ["--datasets", "circle2", "--seeds", "0", "0", "--aggregation-intervals", "10"],
        ["--datasets", "circle2", "--seeds", "0", "--aggregation-intervals", "0"],
        [
            "--configurations",
            "fedsda-unknown",
            "--datasets",
            "circle2",
            "--seeds",
            "0",
            "--aggregation-intervals",
            "10",
        ],
        [
            "--datasets",
            "circle2",
            "--seeds",
            "0",
            "--aggregation-intervals",
            "10",
            "--workers",
            "0",
        ],
    ],
)
def test_command_rejects_invalid_plans_before_running(tmp_path, capsys, invalid_arguments):
    output_directory = tmp_path / "rejected-output"
    assert (
        experiment_sweep_command.main(
            [*invalid_arguments, "--output-directory", str(output_directory)]
        )
        == 2
    )
    assert not output_directory.exists()
    assert capsys.readouterr().err


def test_repository_script_runs_without_installation_or_environment_variables(tmp_path):
    """リポジトリ直下のscriptは、packageのinstallや、読込みの経路の設定なしに、実行できる。"""
    output_directory = tmp_path / "script-output"
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in ("PYTHONPATH", "FDE_MNIST_DATA_DIR")
    }
    completed_process = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_DIRECTORY / "run_experiment_sweep.py"),
            *SMALL_COMMAND_ARGUMENTS,
            "--output-directory",
            str(output_directory),
            "--print-plan",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        env=environment,
        timeout=300,
        check=False,
    )
    assert completed_process.returncode == 0, completed_process.stderr
    assert "circle2_interval10_seed0_" in completed_process.stdout
    assert not output_directory.exists()


def test_repository_script_runs_parallel_sweep_in_separate_processes(tmp_path, sequential_sweep):
    """直下のscriptでの並列の実行（workerのprocessが、scriptを読み直す経路）が、逐次の実行と、同じ内容を保存する。

    scriptは、数値計算のライブラリのthreadの数の環境変数を、未設定なら1にする。
    """
    sequential_directory, planned_runs, _, _ = sequential_sweep
    output_directory = tmp_path / "script-parallel-output"
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    completed_process = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_DIRECTORY / "run_experiment_sweep.py"),
            "--datasets",
            "circle2",
            "--seeds",
            "0",
            "1",
            "--client-count",
            "2",
            "--per-client-sample-count",
            "65",
            "--aggregation-intervals",
            "10",
            "--workers",
            "2",
            "--output-directory",
            str(output_directory),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        env=environment,
        timeout=600,
        check=False,
    )
    assert completed_process.returncode == 0, completed_process.stdout + completed_process.stderr
    for planned_run in planned_runs:
        summary, record_tables = read_run_files(output_directory, planned_run)
        sequential_summary, sequential_record_tables = read_run_files(
            sequential_directory, planned_run
        )
        assert summary == sequential_summary
        assert_record_tables_equal(record_tables, sequential_record_tables)
    record = json.loads(
        next(output_directory.glob("sweep_record_*.json")).read_text(encoding="utf-8")
    )
    assert record["environment"]["thread_count_environment_variables"] == dict(
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    assert record["worker_count"] == 2
