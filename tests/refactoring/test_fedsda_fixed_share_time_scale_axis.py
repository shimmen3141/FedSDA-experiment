"""Fixed-Shareの時間尺度を、帰属の保留の容量から分けて指定する（IMPROVE-015）ことと、掃引の軸
`--fedsda-fixed-share-time-scales`を確認する。"""

import csv
import json
from dataclasses import replace

import pytest
from test_run_settings_validation import valid_run_settings_mapping as valid_run_settings_mapping

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.configuration.run_settings import (
    ValidatedExperimentRunSettingsSubset,
)
from federated_learning_experiments.configuration.run_settings_validation import (
    validate_experiment_run_settings,
)
from federated_learning_experiments.core.configuration_errors import RunSettingsValidationError
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.sweep import experiment_sweep_command
from federated_learning_experiments.sweep.experiment_configurations import (
    SYNCHRONIZED_LOCAL_MIXTURE_PREDICTION_CONFIGURATION_NAME,
    build_experiment_run_settings,
)
from federated_learning_experiments.sweep.experiment_sweep_plan import (
    ExperimentSweepPlan,
    build_planned_experiment_runs,
)
from federated_learning_experiments.sweep.experiment_sweep_records import write_run_summary_table

_CONDITIONS = ExperimentRunConditions(
    dataset_name="sine2",
    random_seed=0,
    client_count=2,
    per_client_sample_count=50,
    server_aggregation_interval_per_client_samples=10,
)


def replace_client(run_settings, **client_changes):
    participant = run_settings.run_participant_settings
    return replace(
        run_settings,
        run_participant_settings=replace(
            participant,
            run_client_settings=replace(participant.run_client_settings, **client_changes),
        ),
    )


def test_fedsda_run_settings_accept_different_capacity_and_time_scale():
    run_settings = build_experiment_run_settings(
        configuration_name="fedsda", experiment_run_conditions=_CONDITIONS
    )
    client = run_settings.run_participant_settings.run_client_settings
    only_time_scale = replace_client(
        run_settings,
        prediction_combination_settings=replace(
            client.prediction_combination_settings,
            fixed_share_weight_redistribution_time_scale_samples=60,
        ),
    )
    only_capacity = replace_client(
        run_settings,
        training_data_assignment_settings=replace(
            client.training_data_assignment_settings,
            pending_assignment_buffer_capacity_samples=60,
        ),
    )
    for changed, expected in ((only_time_scale, (30, 60)), (only_capacity, (60, 120))):
        changed_client = changed.run_participant_settings.run_client_settings
        assert (
            changed_client.training_data_assignment_settings.pending_assignment_buffer_capacity_samples,
            changed_client.prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples,
        ) == expected
    # 警報を即時にする構成では、容量1も、T（既定の120）と違う値のまま受理される。
    immediate = build_experiment_run_settings(
        configuration_name="fedsda-classwise-adwin-detection", experiment_run_conditions=_CONDITIONS
    )
    immediate_client = immediate.run_participant_settings.run_client_settings
    capacity_one = replace_client(
        immediate,
        training_data_assignment_settings=replace(
            immediate_client.training_data_assignment_settings,
            pending_assignment_buffer_capacity_samples=1,
        ),
    )
    assert (
        capacity_one.run_participant_settings.run_client_settings.prediction_combination_settings.fixed_share_weight_redistribution_time_scale_samples
        == 120
    )
    # Tの下限（2）は、今どおり拒否される。
    with pytest.raises(RunSettingsValidationError):
        replace(
            client.prediction_combination_settings,
            fixed_share_weight_redistribution_time_scale_samples=1,
        )


def test_overall_validation_accepts_different_capacity_and_time_scale(valid_run_settings_mapping):
    unvalidated_run_settings = valid_run_settings_mapping.copy()
    unvalidated_run_settings["prediction_combination_settings"] = replace(
        unvalidated_run_settings["prediction_combination_settings"],
        fixed_share_weight_redistribution_time_scale_samples=60,
    )
    assert validate_experiment_run_settings(unvalidated_run_settings) is None
    ValidatedExperimentRunSettingsSubset(**unvalidated_run_settings)


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


def without_time_scale(planned_run):
    mapping = convert_settings_to_plain_mapping(planned_run.run_settings)
    client = mapping["run_participant_settings"]["run_client_settings"]
    del client["prediction_combination_settings"][
        "fixed_share_weight_redistribution_time_scale_samples"
    ]
    return mapping


def test_time_scale_axis_makes_one_fedsda_variant_per_time_scale():
    planned_runs = build_planned_experiment_runs(
        sweep_plan=small_plan(
            configuration_names=("fedsda", "fedsda-independent-models", "feddrift", "fedavg"),
            fedsda_fixed_share_time_scales=(30, 60),
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
        # Tだけが替わり、容量は構成の値（30）のまま。
        assert [capacity_and_time_scale(run) for run in runs] == [(30, 30), (30, 60)]
        assert without_time_scale(runs[0]) == without_time_scale(runs[1])
    assert len({run.run_identifier for run in planned_runs}) == len(planned_runs)


def test_capacity_axis_keeps_the_time_scale():
    planned_runs = build_planned_experiment_runs(
        sweep_plan=small_plan(fedsda_pending_assignment_fifo_capacities=(30, 60))
    )
    assert [capacity_and_time_scale(run) for run in planned_runs] == [(30, 120), (60, 120)]


def test_capacity_and_time_scale_axes_make_every_pair():
    planned_runs = build_planned_experiment_runs(
        sweep_plan=small_plan(
            fedsda_pending_assignment_fifo_capacities=(30, 60),
            fedsda_fixed_share_time_scales=(30, 60),
        )
    )
    # 容量の順の中で、Tの順。
    assert [capacity_and_time_scale(run) for run in planned_runs] == [
        (30, 30),
        (30, 60),
        (60, 30),
        (60, 60),
    ]
    assert len({run.run_identifier for run in planned_runs}) == 4


def test_default_time_scale_keeps_the_plan_without_the_axis():
    assert build_planned_experiment_runs(sweep_plan=small_plan()) == build_planned_experiment_runs(
        sweep_plan=small_plan(fedsda_fixed_share_time_scales=(120,))
    )


@pytest.mark.parametrize(
    "time_scales,error_type",
    [
        ((), ValueError),
        ([60], TypeError),
        ((60, 60), ValueError),
        ((0,), ValueError),
        ((-5,), ValueError),
        ((True,), TypeError),
    ],
)
def test_time_scale_axis_rejects_invalid_values(time_scales, error_type):
    with pytest.raises(error_type):
        small_plan(fedsda_fixed_share_time_scales=time_scales)


def test_time_scale_axis_requires_fedsda_and_values_accepted_by_run_settings():
    with pytest.raises(ValueError, match="fedsda_fixed_share_time_scales"):
        build_planned_experiment_runs(
            sweep_plan=small_plan(
                configuration_names=("feddrift", "fedavg"),
                fedsda_fixed_share_time_scales=(60,),
            )
        )
    with pytest.raises(ValueError) as raised_error:
        build_planned_experiment_runs(
            sweep_plan=small_plan(
                configuration_names=("feddrift",),
                fedsda_pending_assignment_fifo_capacities=(60,),
                fedsda_fixed_share_time_scales=(60,),
            )
        )
    assert "fedsda_pending_assignment_fifo_capacities" in str(raised_error.value)
    assert "fedsda_fixed_share_time_scales" in str(raised_error.value)
    # T＝1は、Tの下限（2）で拒否される。
    with pytest.raises(RunSettingsValidationError):
        build_planned_experiment_runs(sweep_plan=small_plan(fedsda_fixed_share_time_scales=(1,)))


def test_mixture_share_rate_follows_the_time_scale_axis():
    (planned_run,) = build_planned_experiment_runs(
        sweep_plan=small_plan(
            configuration_names=(SYNCHRONIZED_LOCAL_MIXTURE_PREDICTION_CONFIGURATION_NAME,),
            fedsda_fixed_share_time_scales=(60,),
        )
    )
    run = execute_fedsda_experiment_run(run_settings=planned_run.run_settings)
    # 混ぜる重みの分け合いの割合は、clientの組立てで1/Tになる（内部の値を直接読む）。
    assert {
        client.synchronized_and_local_prediction_mixture._share_rate
        for client in run.measured_run.participants.client_operations
    } == {1.0 / 60}


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


def test_command_rejects_invalid_time_scales_before_running(tmp_path, capsys):
    output_directory = tmp_path / "output"
    for time_scales in (["1"], ["60", "60"], ["0"]):
        assert (
            experiment_sweep_command.main(
                [
                    *base_arguments(output_directory),
                    "--fedsda-fixed-share-time-scales",
                    *time_scales,
                ]
            )
            == 2
        )
        assert "invalid sweep plan" in capsys.readouterr().err
    # 整数でない値は、引数の解析で拒否される（終了コード2）。
    assert (
        experiment_sweep_command.main(
            [*base_arguments(output_directory), "--fedsda-fixed-share-time-scales", "6.5"]
        )
        == 2
    )
    assert (
        experiment_sweep_command.main(
            [
                *base_arguments(output_directory),
                "--configurations",
                "feddrift",
                "--fedsda-fixed-share-time-scales",
                "60",
            ]
        )
        == 2
    )
    assert "fedsda_fixed_share_time_scales" in capsys.readouterr().err
    assert not output_directory.exists()


def test_command_saves_runs_with_time_scale_column_and_record_format_version(tmp_path):
    output_directory = tmp_path / "output"
    assert (
        experiment_sweep_command.main(
            [
                *base_arguments(output_directory),
                "--fedsda-pending-assignment-fifo-capacities",
                "30",
                "60",
                "--fedsda-fixed-share-time-scales",
                "60",
            ]
        )
        == 0
    )
    with (output_directory / "fedsda" / "run_summary.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert sorted(
        (row["pending_assignment_fifo_capacity"], row["fixed_share_time_scale"]) for row in rows
    ) == [("30", "60"), ("60", "60")]
    (record_path,) = output_directory.glob("sweep_record_*.json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["record_format_version"] == 7
    assert record["sweep_plan"]["fedsda_fixed_share_time_scales"] == [60]
    assert record["sweep_plan"]["fedsda_pending_assignment_fifo_capacities"] == [30, 60]
    # 時間尺度の項目を持たない要約でも、行を外さず、列を空にする。
    for summary_path in (output_directory / "fedsda" / "runs").glob("*.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        del summary["run_settings"]["run_participant_settings"]["run_client_settings"][
            "prediction_combination_settings"
        ]
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    table_result = write_run_summary_table(
        configuration_directory=output_directory / "fedsda", configuration_name="fedsda"
    )
    assert table_result.unreadable_run_summary_descriptions == ()
    with table_result.table_file_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["fixed_share_time_scale"] for row in rows] == ["", ""]
