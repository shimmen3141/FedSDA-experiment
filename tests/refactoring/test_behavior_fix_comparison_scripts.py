"""修正の有無の比較script（直下の`run_behavior_fix_comparison.py`と`analyze_behavior_fix_comparison.py`）。

登録の水準に、現行モデルとして待った後に登録する扱い（`waited-as-current`）を足し、集計を、データと集約間隔ごとの
絶対値（平均±標準偏差）と対の差（平均±標準誤差）の形にした。種類の違うデータを越えて平均した行は出さない。
"""

import contextlib
import csv
import importlib.util
import io
import json
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


ANALYSIS_FIXTURE_COMBINATION_NAMES = (
    "registration-fixed__short-alarm-fixed",
    "registration-waited-as-current__short-alarm-fixed",
    "registration-waited-as-current__short-alarm-fixed__alarm-wait-on",
)
# 表の数値を変えないことを確かめる期待値（`## 1.`から末尾）。5ca896fの`analyze_behavior_fix_comparison.py`で、
# `write_analysis_fixture`の入力から`--chains`ありで出した標準出力から作った（手で書かない。入力を変えたら作り直す）。
ANALYSIS_EXPECTED_TABLES_PATH = Path(__file__).with_name("behavior_fix_analysis_expected_tables.md")


def write_analysis_fixture(output_directory):
    """分析の表の全部（絶対値、対の差、連鎖）に行が出る、決まった入力を作る。"""
    import numpy as np

    metric_columns = list(analyze_script.METRICS)
    for combination_position, combination_name in enumerate(ANALYSIS_FIXTURE_COMBINATION_NAMES):
        runs_directory = output_directory / combination_name / "runs"
        runs_directory.mkdir(parents=True)
        rows = []
        for dataset_position, dataset_name in enumerate(("sea4", "sine2")):
            for interval in (50, 200):
                for seed in range(3):
                    run_identifier = f"{dataset_name}_interval{interval}_seed{seed}"
                    base = 1 + combination_position * 0.37 + dataset_position * 1.9 + seed * 0.23
                    row = dict(
                        run_identifier=run_identifier,
                        dataset_name=dataset_name,
                        server_aggregation_interval_per_client_samples=interval,
                        random_seed=seed,
                        code_commit="a" * 40,
                        behavior_baseline_version=5,
                    )
                    for metric_position, column in enumerate(metric_columns):
                        row[column] = round(
                            base * (metric_position + 1) * (interval / 50) ** 0.5, 6
                        )
                    rows.append(row)
                    # 短い警報の連鎖: clientごとに、短い警報が続く並びと、ほかの警報を挟む。
                    outcomes = []
                    client_ids = []
                    for client_id in range(2):
                        chain_length = 1 + (seed + client_id + combination_position) % 6
                        pattern = (
                            ["alarm_change_interval_too_short"] * chain_length
                            + ["alarm_reuse_held_model", "no_alarm_adaptation"]
                            + ["alarm_change_interval_too_short"] * (dataset_position + 1)
                        )
                        outcomes.extend(pattern)
                        client_ids.extend([client_id] * len(pattern))
                    np.savez(
                        runs_directory / f"{run_identifier}.npz",
                        **{
                            "adaptation.client_id": np.array(client_ids),
                            "adaptation.adaptation_sample_index": np.arange(len(outcomes)),
                            "adaptation.adaptation_outcome": np.array(outcomes),
                        },
                    )
        with (output_directory / combination_name / "run_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as table_file:
            writer = csv.DictWriter(table_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


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
        assert analyze_script.main([str(tmp_path), "--stdout"]) == 0
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


def test_comparison_stops_with_a_warning_when_existing_runs_have_another_baseline(
    tmp_path, monkeypatch, capsys
):
    """比較scriptも、基準版の違う済んだrunがあれば、警告を出して2を返し、比較の記録を書かない。"""
    from federated_learning_experiments.sweep import experiment_sweep_execution

    mismatch_description = "existing run summary was saved with behavior baseline version 4"

    def stop_sweep(**_):
        raise experiment_sweep_execution.ExistingRunBehaviorBaselineMismatchError(
            (mismatch_description,)
        )

    monkeypatch.setattr(
        experiment_sweep_execution,
        "collect_behavior_baseline_mismatch_descriptions",
        lambda **_: (mismatch_description,),
    )
    monkeypatch.setattr(experiment_sweep_execution, "execute_experiment_sweep", stop_sweep)
    output_directory = tmp_path / "comparison"
    command_arguments = [
        "--datasets",
        "sine2",
        "--seeds",
        "0",
        "--aggregation-intervals",
        "50",
        "--purpose",
        "baseline stop",
        "--output-directory",
        str(output_directory),
    ]
    assert run_script.main(command_arguments) == 2
    assert f"warning: {mismatch_description}" in capsys.readouterr().err
    assert not output_directory.exists()
    # `--print-plan`は、状態の理由も表示する。

    def report_mismatch(**_):
        return experiment_sweep_execution.ExistingRunState(
            run_state="different_behavior_baseline", description=mismatch_description
        )

    monkeypatch.setattr(experiment_sweep_execution, "find_existing_run_state", report_mismatch)
    assert run_script.main([*command_arguments, "--print-plan"]) == 0
    assert f"different_behavior_baseline — {mismatch_description}" in capsys.readouterr().out


def test_comparison_record_holds_code_revision_and_behavior_baseline(tmp_path, monkeypatch):
    """比較の記録は、コードの版の辞書と、基準版を持つ（要約へ渡したのと同じ来歴）。"""
    from federated_learning_experiments.core.run_provenance import BEHAVIOR_BASELINE_VERSION
    from federated_learning_experiments.sweep import experiment_sweep_execution

    passed_run_provenances = []

    def record_run_provenance(*, planned_runs, run_provenance, **_):
        passed_run_provenances.append(run_provenance)
        return ()

    monkeypatch.setattr(
        experiment_sweep_execution, "execute_experiment_sweep", record_run_provenance
    )
    output_directory = tmp_path / "comparison"
    assert (
        run_script.main(
            [
                "--datasets",
                "sine2",
                "--seeds",
                "0",
                "--aggregation-intervals",
                "50",
                "--purpose",
                "record test",
                "--output-directory",
                str(output_directory),
            ]
        )
        == 0
    )
    (run_provenance,) = passed_run_provenances
    record = json.loads(next(output_directory.glob("comparison_record_*.json")).read_text("utf-8"))
    assert record["behavior_baseline_version"] == BEHAVIOR_BASELINE_VERSION
    assert record["code_revision"] == dict(
        commit=run_provenance.code_commit,
        has_uncommitted_changes=run_provenance.has_uncommitted_changes,
        unavailable_reason=run_provenance.code_revision_unavailable_reason,
    )


def test_analysis_tables_are_unchanged_from_the_previous_script(tmp_path):
    """出どころを足した後も、表（`## 1.`から末尾）は、変更前のscriptの出力と文字単位で同じ。"""
    write_analysis_fixture(tmp_path)
    standard_output = io.StringIO()
    with contextlib.redirect_stdout(standard_output):
        assert analyze_script.main(["--stdout", str(tmp_path), "--chains"]) == 0
    report = standard_output.getvalue()
    expected_tables = ANALYSIS_EXPECTED_TABLES_PATH.read_text(encoding="utf-8")
    assert "| sine2 | 200 |" in expected_tables.split("## 3.")[1]
    assert report[report.index("## 1.") :] == expected_tables
    assert not list(tmp_path.glob("analysis_*.md"))


def test_analysis_writes_a_file_with_provenance_in_the_input_directory(tmp_path, monkeypatch):
    """既定では、入力のディレクトリの`analysis_<作成日時>.md`へ書き、同じ名前があれば番号を付ける。"""
    write_analysis_fixture(tmp_path)

    class FixedDatetime(analyze_script.datetime):
        @classmethod
        def now(cls, tz=None):
            return analyze_script.datetime(2026, 10, 14, 12, 0, 0, tzinfo=tz)

    monkeypatch.setattr(analyze_script, "datetime", FixedDatetime)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        assert analyze_script.main([str(tmp_path)]) == 0
        assert analyze_script.main([str(tmp_path)]) == 0
    first = tmp_path / "analysis_20261014_120000.md"
    second = tmp_path / "analysis_20261014_120000_2.md"
    assert printed.getvalue().splitlines() == [f"analysis: {first}", f"analysis: {second}"]
    report = first.read_text(encoding="utf-8")
    standard_output = io.StringIO()
    with contextlib.redirect_stdout(standard_output):
        assert analyze_script.main([str(tmp_path), "--stdout"]) == 0
    # 中身は、`--stdout`と同じ（引数の行だけが違う）。
    assert [line for line in report.splitlines() if "分析のコマンドの引数" not in line] == [
        line
        for line in standard_output.getvalue().splitlines()
        if "分析のコマンドの引数" not in line
    ]
    lines = report.splitlines()
    assert lines[0] == "# 挙動の修正（LEGACY-016・LEGACY-002）の、適用の有無の比較"
    assert lines[2] == "## 出どころ"
    assert "警告" not in report
    for expected_line in (
        "- 作成の日時: 2026-10-14T12:00:00",
        f"- 分析のコマンドの引数: `{tmp_path}`",
        "- 分析のコードの版: commit ",
        f"- 入力のディレクトリ: `{tmp_path}`",
        "| registration-fixed__short-alarm-fixed | 12 | aaaaaaaaaaaa | 5 |",
        "| registration-waited-as-current__short-alarm-fixed__alarm-wait-on | 12 | aaaaaaaaaaaa | 5 |",
    ):
        assert any(line.startswith(expected_line) for line in lines), expected_line


def test_analysis_warns_right_after_the_title_when_baseline_versions_are_mixed(tmp_path):
    """基準版が揃っていない（記録なしを含む）とき、見出しの行の直後に警告を書き、集計は続ける。"""
    write_analysis_fixture(tmp_path)
    write_run_summary(
        tmp_path,
        "registration-legacy__short-alarm-fixed",
        [("sine2", 0, 0.90)],
    )
    standard_output = io.StringIO()
    with contextlib.redirect_stdout(standard_output):
        assert analyze_script.main([str(tmp_path), "--stdout"]) == 0
    lines = standard_output.getvalue().splitlines()
    assert (
        lines[2]
        == "> 警告: 入力のrunの基準版が揃っていない（5、記録なし）。版の違う結果を比べている可能性がある。"
    )
    assert lines[4] == "## 出どころ"
    assert "| registration-legacy__short-alarm-fixed | 1 | 記録なし | 記録なし |" in lines
    assert any(line.startswith("## 1.") for line in lines)


def test_analysis_without_input_tables_returns_one_and_writes_nothing(tmp_path, capsys):
    assert analyze_script.main([str(tmp_path)]) == 1
    assert "no run_summary.csv" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_comparison_requires_purpose_and_places_results_under_its_study(
    tmp_path, monkeypatch, capsys
):
    """比較scriptは、目的が要り、出力先を省略すると、results/behavior-fix-comparison/<開始日時>_<ラベル>に置く。"""
    from federated_learning_experiments.sweep import experiment_sweep_execution

    base_arguments = ["--datasets", "sine2", "--seeds", "0", "--aggregation-intervals", "50"]
    monkeypatch.setattr(run_script, "REPOSITORY_DIRECTORY", tmp_path)
    with pytest.raises(SystemExit) as raised:
        run_script.main(base_arguments)
    assert raised.value.code == 2
    assert "--purpose is required to run" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run_script.main([*base_arguments, "--purpose", "p", "--label", "a/b"])
    assert "--label must be a plain directory name" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setattr(experiment_sweep_execution, "execute_experiment_sweep", lambda **_: ())
    assert run_script.main([*base_arguments, "--purpose", "study placement", "--label", "try"]) == 0
    (output_directory,) = (tmp_path / "results" / "behavior-fix-comparison").iterdir()
    assert output_directory.name.endswith("_try")
    readme_text = (output_directory / "README.md").read_text(encoding="utf-8")
    assert "- 目的: study placement" in readme_text
    assert "- 研究の問い: behavior-fix-comparison" in readme_text
    record = json.loads(next(output_directory.glob("comparison_record_*.json")).read_text("utf-8"))
    assert record["purpose"] == "study placement"
    assert record["study_name"] == "behavior-fix-comparison"


def test_comparison_record_is_not_overwritten_within_the_same_second(tmp_path, monkeypatch):
    from federated_learning_experiments.sweep import experiment_sweep_execution

    class FixedDatetime(run_script.datetime):
        @classmethod
        def now(cls, tz=None):
            return run_script.datetime(2026, 10, 14, 12, 0, 0, tzinfo=tz)

    monkeypatch.setattr(run_script, "datetime", FixedDatetime)
    monkeypatch.setattr(experiment_sweep_execution, "execute_experiment_sweep", lambda **_: ())
    arguments = [
        "--datasets",
        "sine2",
        "--seeds",
        "0",
        "--aggregation-intervals",
        "50",
        "--output-directory",
        str(tmp_path),
    ]
    assert run_script.main([*arguments, "--purpose", "first"]) == 0
    assert run_script.main([*arguments, "--purpose", "second"]) == 0
    assert sorted(path.name for path in tmp_path.glob("comparison_record_*.json")) == [
        "comparison_record_20261014_120000.json",
        "comparison_record_20261014_120000_2.json",
    ]
    assert (
        json.loads((tmp_path / "comparison_record_20261014_120000.json").read_text("utf-8"))[
            "purpose"
        ]
        == "first"
    )
