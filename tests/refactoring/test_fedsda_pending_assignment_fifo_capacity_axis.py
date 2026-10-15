"""掃引の軸`--fedsda-pending-assignment-fifo-capacities`（FedSDAの帰属の保留の容量）を確認する。

軸の値で、帰属の保留の容量とFixed-Shareの時間尺度の両方を替える（今の設定の検査が、同じ値を求める）。
"""

import csv
import json

import pytest

from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.sweep import experiment_sweep_command
from federated_learning_experiments.sweep.experiment_sweep_plan import (
    ExperimentSweepPlan,
    build_planned_experiment_runs,
)
from federated_learning_experiments.sweep.experiment_sweep_records import write_run_summary_table


def small_plan(**changes):
    return ExperimentSweepPlan(
        **(
            dict(
                configuration_names=("fedsda",),
                dataset_names=("sine2",),
                random_seeds=(0,),
                client_count=2,
                per_client_sample_count=50,
                server_aggregation_intervals_per_client_samples=(10,),
            )
            | changes
        )
    )


def capacity_and_time_scale(planned_run):
    client = planned_run.run_settings.run_participant_settings.run_client_settings
    return (
        client.training_data_assignment_settings.pending_assignment_buffer_capacity_samples,
        client.prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples,
    )


def without_capacity(planned_run):
    mapping = convert_settings_to_plain_mapping(planned_run.run_settings)
    client = mapping["run_participant_settings"]["run_client_settings"]
    del client["training_data_assignment_settings"]["pending_assignment_buffer_capacity_samples"]
    del client["prediction_combination_settings"][
        "fixed_share_weight_redistribution_time_scale_samples"
    ]
    return mapping


def test_capacity_axis_makes_one_fedsda_variant_per_capacity():
    planned_runs = build_planned_experiment_runs(
        sweep_plan=small_plan(
            configuration_names=("fedsda", "fedsda-independent-models", "feddrift", "fedavg"),
            fedsda_pending_assignment_fifo_capacities=(30, 60),
        )
    )
    by_configuration = {}
    for planned_run in planned_runs:
        by_configuration.setdefault(planned_run.configuration_name, []).append(planned_run)
    assert {name: len(runs) for name, runs in by_configuration.items()} == {
        "fedsda": 2,
        "fedsda-independent-models": 2,
        "feddrift": 1,
        "fedavg": 1,
    }
    for name in ("fedsda", "fedsda-independent-models"):
        runs = by_configuration[name]
        assert [capacity_and_time_scale(run) for run in runs] == [(30, 30), (60, 60)]
        # 容量と時間尺度のほかの設定は、変わらない。
        assert without_capacity(runs[0]) == without_capacity(runs[1])
    assert len({run.run_identifier for run in planned_runs}) == len(planned_runs)


def test_default_capacity_keeps_the_plan_without_the_axis():
    assert build_planned_experiment_runs(sweep_plan=small_plan()) == build_planned_experiment_runs(
        sweep_plan=small_plan(fedsda_pending_assignment_fifo_capacities=(30,))
    )


def test_two_fedsda_axes_make_every_combination_in_order():
    planned_runs = build_planned_experiment_runs(
        sweep_plan=small_plan(
            fedsda_training_sample_selections=("uniform", "decay:0.997"),
            fedsda_pending_assignment_fifo_capacities=(30, 60),
        )
    )
    combinations = [
        (
            run.run_settings.run_participant_settings.run_client_settings.local_training_settings.training_sample_selection_policy,
            capacity_and_time_scale(run)[0],
        )
        for run in planned_runs
    ]
    assert combinations == [
        ("uniform_over_all_stored_samples", 30),
        ("uniform_over_all_stored_samples", 60),
        ("recency_decayed_stored_samples", 30),
        ("recency_decayed_stored_samples", 60),
    ]


@pytest.mark.parametrize(
    "capacities,error_type",
    [
        ((), ValueError),
        ([60], TypeError),
        ((60, 60), ValueError),
        ((0,), ValueError),
        ((-5,), ValueError),
        ((True,), TypeError),
    ],
)
def test_capacity_axis_rejects_invalid_values(capacities, error_type):
    with pytest.raises(error_type):
        small_plan(fedsda_pending_assignment_fifo_capacities=capacities)


def test_capacity_axis_requires_fedsda_and_values_accepted_by_run_settings():
    with pytest.raises(ValueError, match="fedsda_pending_assignment_fifo_capacities"):
        build_planned_experiment_runs(
            sweep_plan=small_plan(
                configuration_names=("feddrift", "fedavg"),
                fedsda_pending_assignment_fifo_capacities=(60,),
            )
        )
    # 容量1は、Fixed-Shareの時間尺度の下限（2）で拒否される。
    with pytest.raises(RunSettingsValidationError):
        build_planned_experiment_runs(
            sweep_plan=small_plan(fedsda_pending_assignment_fifo_capacities=(1,))
        )
    # 警報を待つ扱い（既定）では、容量が変化区間の最小件数（5）より小さいと拒否される。
    with pytest.raises(RunSettingsValidationError):
        build_planned_experiment_runs(
            sweep_plan=small_plan(fedsda_pending_assignment_fifo_capacities=(4,))
        )
    # 警報を即時にする構成では、受理される。
    (planned_run,) = build_planned_experiment_runs(
        sweep_plan=small_plan(
            configuration_names=("fedsda-classwise-adwin-detection",),
            fedsda_pending_assignment_fifo_capacities=(4,),
        )
    )
    assert capacity_and_time_scale(planned_run) == (4, 4)


def base_arguments(output_directory):
    return [
        "--datasets",
        "sine2",
        "--seeds",
        "0",
        "--client-count",
        "2",
        "--per-client-sample-count",
        "30",
        "--aggregation-intervals",
        "10",
        "--purpose",
        "test",
        "--output-directory",
        str(output_directory),
    ]


def test_command_rejects_invalid_capacities_before_running(tmp_path, capsys):
    output_directory = tmp_path / "output"
    for capacities in (["4"], ["1"], ["60", "60"]):
        assert (
            experiment_sweep_command.main(
                [
                    *base_arguments(output_directory),
                    "--fedsda-pending-assignment-fifo-capacities",
                    *capacities,
                ]
            )
            == 2
        )
        assert "invalid sweep plan" in capsys.readouterr().err
    # 整数でない値は、引数の解析で拒否される（終了コード2）。
    assert (
        experiment_sweep_command.main(
            [
                *base_arguments(output_directory),
                "--fedsda-pending-assignment-fifo-capacities",
                "6.5",
            ]
        )
        == 2
    )
    assert (
        experiment_sweep_command.main(
            [
                *base_arguments(output_directory),
                "--configurations",
                "feddrift",
                "--fedsda-pending-assignment-fifo-capacities",
                "60",
            ]
        )
        == 2
    )
    assert "fedsda_pending_assignment_fifo_capacities" in capsys.readouterr().err
    assert not output_directory.exists()


def test_command_saves_runs_with_capacity_column_and_record_format_version(tmp_path):
    output_directory = tmp_path / "output"
    assert (
        experiment_sweep_command.main(
            [
                *base_arguments(output_directory),
                "--fedsda-pending-assignment-fifo-capacities",
                "30",
                "60",
            ]
        )
        == 0
    )
    with (output_directory / "fedsda" / "run_summary.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert sorted(row["pending_assignment_fifo_capacity"] for row in rows) == ["30", "60"]
    (record_path,) = output_directory.glob("sweep_record_*.json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["record_format_version"] == 5
    assert record["sweep_plan"]["fedsda_pending_assignment_fifo_capacities"] == [30, 60]
    # 要約の形式の版1と同じ形（来歴を持たない。run設定は今と同じ形）の要約から作った表でも、列が埋まる。
    # 実物の版1の要約は作っていない（版1でもこの項目が在ることは、gitの履歴で確かめた。spec のtasks.md）。
    for summary_path in (output_directory / "fedsda" / "runs").glob("*.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        del summary["run_provenance"]
        summary["format_version"] = 1
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    table_result = write_run_summary_table(
        configuration_directory=output_directory / "fedsda", configuration_name="fedsda"
    )
    assert table_result.unreadable_run_summary_descriptions == ()
    with table_result.table_file_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert sorted(row["pending_assignment_fifo_capacity"] for row in rows) == ["30", "60"]
