"""修正の有無の比較script（直下の`run_behavior_fix_comparison.py`と`analyze_behavior_fix_comparison.py`）。

登録の水準に、現行モデルとして待った後に登録する扱い（`waited-as-current`）を足し、集計を、データと集約間隔ごとの
絶対値（平均±標準偏差）と対の差（平均±標準誤差）の形にした。種類の違うデータを越えて平均した行は出さない。
"""

import contextlib
import csv
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from federated_learning_experiments.methods.fedsda.model_registration.pending_model_registration_settings import (
    KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME,
    REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME,
    REGISTER_PENDING_MODEL_AFTER_WAITING_AS_CURRENT_MODEL_POLICY_NAME,
)

REPOSITORY_DIRECTORY = Path(__file__).resolve().parents[2]


def load_script_module(file_name):
    specification = importlib.util.spec_from_file_location(
        file_name.removesuffix(".py"), REPOSITORY_DIRECTORY / file_name
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


run_script = load_script_module("run_behavior_fix_comparison.py")
analyze_script = load_script_module("analyze_behavior_fix_comparison.py")


def test_registration_levels_name_every_policy_and_keep_the_default_plan():
    assert run_script.REGISTRATION_POLICY_BY_LEVEL == {
        "legacy": KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME,
        "fixed": REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME,
        "waited-as-current": REGISTER_PENDING_MODEL_AFTER_WAITING_AS_CURRENT_MODEL_POLICY_NAME,
    }
    assert run_script.REGISTRATION_LEVELS == analyze_script.REGISTRATION_LEVELS
    assert run_script.SHORT_ALARM_LEVELS == analyze_script.SHORT_ALARM_LEVELS == ("legacy", "fixed")
    # 登録の水準を指定しないときは、2026-10-13の480 runと同じ計画。
    assert run_script.DEFAULT_REGISTRATION_LEVELS == ("legacy", "fixed")


def test_short_alarm_axis_rejects_a_registration_level(tmp_path):
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        run_script.main(
            [
                "--datasets",
                "sine2",
                "--seeds",
                "0",
                "--aggregation-intervals",
                "50",
                "--short-alarm-levels",
                "waited-as-current",
                "--output-directory",
                str(tmp_path / "unused"),
                "--print-plan",
            ]
        )
    assert not (tmp_path / "unused").exists()


def test_planned_runs_use_each_registration_policy_and_keep_existing_combination_names():
    planned_runs = run_script.build_planned_runs(
        SimpleNamespace(
            registration_levels=["legacy", "fixed", "waited-as-current"],
            short_alarm_levels=["fixed"],
            alarm_wait_levels=["off"],
            datasets=["sine2"],
            aggregation_intervals=[50],
            seeds=[0],
            client_count=2,
            per_client_sample_count=60,
        )
    )
    assert [planned_run.configuration_name for planned_run in planned_runs] == [
        "registration-legacy__short-alarm-fixed",
        "registration-fixed__short-alarm-fixed",
        "registration-waited-as-current__short-alarm-fixed",
    ]
    assert [
        planned_run.run_settings.run_participant_settings.run_client_settings.pending_model_registration_settings.pending_model_registration_policy
        for planned_run in planned_runs
    ] == [
        KEEP_LATEST_PENDING_MODEL_AND_REASSIGN_CURRENT_MODEL_ID_POLICY_NAME,
        REGISTER_EACH_PENDING_MODEL_BY_ITS_OWN_ID_POLICY_NAME,
        REGISTER_PENDING_MODEL_AFTER_WAITING_AS_CURRENT_MODEL_POLICY_NAME,
    ]


def write_run_summary(output_directory, combination_name, rows):
    combination_directory = output_directory / combination_name
    combination_directory.mkdir(parents=True)
    columns = [
        "run_identifier",
        "dataset_name",
        "server_aggregation_interval_per_client_samples",
        "random_seed",
        "prediction_accuracy",
    ]
    with (combination_directory / "run_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as table_file:
        writer = csv.DictWriter(table_file, fieldnames=columns)
        writer.writeheader()
        for dataset_name, seed, accuracy in rows:
            writer.writerow(
                dict(
                    run_identifier=f"{dataset_name}_{seed}",
                    dataset_name=dataset_name,
                    server_aggregation_interval_per_client_samples=50,
                    random_seed=seed,
                    prediction_accuracy=accuracy,
                )
            )


def test_analysis_reports_absolute_values_and_differences_per_dataset_only(tmp_path):
    write_run_summary(
        tmp_path,
        "registration-fixed__short-alarm-fixed",
        [("sine2", 0, 0.90), ("sine2", 1, 0.92), ("sea4", 0, 0.80)],
    )
    write_run_summary(
        tmp_path,
        "registration-waited-as-current__short-alarm-fixed",
        [("sine2", 0, 0.91), ("sine2", 1, 0.97), ("sea4", 0, 0.79)],
    )
    standard_output = io.StringIO()
    with contextlib.redirect_stdout(standard_output):
        assert analyze_script.main([str(tmp_path)]) == 0
    report = standard_output.getvalue()
    # 絶対値: 平均±標準偏差（n−1）。runが1つなら、標準偏差は「—」。
    assert "| sine2 | 50 | fixed | fixed | off | 2 | 91.00±1.41 |" in report
    assert "| sine2 | 50 | waited-as-current | fixed | off | 2 | 94.00±4.24 |" in report
    assert "| sea4 | 50 | fixed | fixed | off | 1 | 80.00±— |" in report
    # 対の差（fixed→waited-as-current）: sine2は+1と+5（平均+3、標準誤差2。`*`は付かない）、sea4は−1（runが1つ）。
    assert (
        "### 登録の扱いの効果（fixed→waited-as-current。短い警報はfixed、警報を待つ扱いはoff）"
        in report
    )
    assert "| sine2 | 50 | +3.00±2.00（↑2/↓0/n=2） |" in report
    assert "| sea4 | 50 | -1.00（↑0/↓1/n=1） |" in report
    # 種類の違うデータを越えて平均した行は出さない。
    assert "全部" not in report
    # 結果の無い組合せの比較は、出さない。
    assert "legacy→fixed" not in report
