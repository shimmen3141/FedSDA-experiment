"""同じ条件の対だけを比較し、欠測と来歴差を隠さないCLIの契約。"""

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.sweep.experiment_configurations import (
    build_experiment_run_settings,
)

_ROOT = Path(__file__).resolve().parents[2]
_CONFIGURATION = "fedsda-without-candidate-validation"


@pytest.fixture
def analyzer():
    specification = importlib.util.spec_from_file_location(
        "_component_ablation_test_module", _ROOT / "analyze_fedsda_component_ablation.py"
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _artifact(root, configuration, seed, accuracy, *, name=None):
    directory = root / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    conditions = ExperimentRunConditions(
        dataset_name="circle2",
        random_seed=seed,
        client_count=2,
        per_client_sample_count=10,
        server_aggregation_interval_per_client_samples=5,
    )
    stem = name or f"seed{seed}"
    payload = {
        "format_version": 2,
        "run_settings": convert_settings_to_plain_mapping(
            build_experiment_run_settings(
                configuration_name=configuration, experiment_run_conditions=conditions
            )
        ),
        "run_metrics": {
            "prediction_accuracy": accuracy,
            "stable_period_prediction_accuracy": None,
            "processed_sample_count": 20,
            "communication_volume": {"uploaded_byte_count": 100, "downloaded_byte_count": 200},
        },
        "environment": dict(system="Linux", machine="x86_64", python="3.10", numpy="2", torch="2"),
        "run_provenance": {
            "code_commit": "a" * 40,
            "has_uncommitted_changes": False,
            "code_revision_unavailable_reason": None,
            "behavior_baseline_version": 6,
        },
        "record_table_file_name": f"{stem}.npz",
    }
    path = directory / f"{stem}.json"
    _write(path, payload)
    # JSON集計で配列を読まないことも保証する。NPZ内部の検証は契約外。
    path.with_suffix(".npz").write_bytes(b"not loaded by summary analysis")
    return path


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _arguments(root, seeds=(0, 1)):
    (root / "ablation").mkdir(exist_ok=True)
    return [
        "--baseline-directory",
        str(root / "baseline"),
        "--ablation-directory",
        str(root / "ablation"),
        "--configurations",
        _CONFIGURATION,
        "--datasets",
        "circle2",
        "--seeds",
        *map(str, seeds),
        "--aggregation-intervals",
        "5",
        "--client-count",
        "2",
        "--per-client-sample-count",
        "10",
    ]


def _populate(root):
    for seed, baseline, variant in [(0, 0.8, 0.82), (1, 0.7, 0.68)]:
        _artifact(root / "baseline", "fedsda", seed, baseline)
        _artifact(root / "ablation" / _CONFIGURATION, _CONFIGURATION, seed, variant)


def _outputs(root):
    directory = sorted((root / "ablation").glob("analysis_*"))[-1]
    with (directory / "paired.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return (directory / "report.md").read_text(encoding="utf-8"), rows


def test_complete_grid_has_hand_calculated_paired_statistics(analyzer, tmp_path):
    _populate(tmp_path)
    assert analyzer.main(_arguments(tmp_path)) == 0
    report, rows = _outputs(tmp_path)
    accuracy_rows = [row for row in rows if row["metric"] == "prediction_accuracy"]
    assert [float(row["delta"]) for row in accuracy_rows] == pytest.approx([2, -2])
    cells = next(line for line in report.splitlines() if line.startswith("| prediction_accuracy |"))
    # 基準平均75、基準SD sqrt(50)、変更平均75、変更SD sqrt(98)、差平均0、SE2。
    numbers = [float(cell.strip()) for cell in cells.split("|")[4:12]]
    assert numbers == pytest.approx([75, 50**0.5, 75, 98**0.5, 0, 2, -2, 2], abs=1e-4)
    assert "1/0/1" in cells
    assert "自動採否なし" in report
    communication = [
        row
        for row in rows
        if row["metric"] == "communication_volume.uploaded_byte_count/processed_sample_count"
    ]
    assert all(float(row["baseline"]) == 5 for row in communication)


def test_missing_run_does_not_bias_paired_mean(analyzer, tmp_path):
    _populate(tmp_path)
    (tmp_path / "ablation" / _CONFIGURATION / "runs" / "seed1.json").unlink()
    assert analyzer.main(_arguments(tmp_path)) == 1
    report, rows = _outputs(tmp_path)
    assert "不足" in report and "seed=1" in report
    assert {row["seed"] for row in rows} == {"0"}
    cells = next(line for line in report.splitlines() if line.startswith("| prediction_accuracy |"))
    assert "未定義" in cells and "1/2" in cells


@pytest.mark.parametrize(
    "change,reason",
    [
        ("environment", "環境不一致"),
        ("dirty", "来歴"),
        ("commit", "来歴差"),
        ("missing_npz", "NPZ"),
        ("unsafe_npz", "NPZ"),
        ("wrong_settings", "設定不一致"),
        ("incomplete", "処理標本数"),
        ("invalid_json", "JSON"),
    ],
)
def test_incomparable_input_is_excluded(analyzer, tmp_path, change, reason):
    _populate(tmp_path)
    path = tmp_path / "ablation" / _CONFIGURATION / "runs" / "seed1.json"
    payload = json.loads(path.read_text())
    if change == "environment":
        payload["environment"]["torch"] = "different"
    elif change == "dirty":
        payload["run_provenance"]["has_uncommitted_changes"] = True
    elif change == "commit":
        payload["run_provenance"]["code_commit"] = "b" * 40
    elif change == "missing_npz":
        path.with_suffix(".npz").unlink()
    elif change == "unsafe_npz":
        payload["record_table_file_name"] = "../outside.npz"
    elif change == "wrong_settings":
        payload["run_settings"]["execution_settings"]["experiment_run_conditions"][
            "client_count"
        ] = True
    elif change == "incomplete":
        payload["run_metrics"]["processed_sample_count"] = 19
    if change == "invalid_json":
        path.write_text("{broken", encoding="utf-8")
    else:
        _write(path, payload)
    assert analyzer.main(_arguments(tmp_path)) == 1
    report, rows = _outputs(tmp_path)
    assert reason in report
    assert {row["seed"] for row in rows} == {"0"}


def test_provenance_difference_requires_acknowledgement_and_is_reported(analyzer, tmp_path):
    _populate(tmp_path)
    path = tmp_path / "baseline" / "runs" / "seed1.json"
    payload = json.loads(path.read_text())
    payload["run_provenance"].update(code_commit="b" * 40, behavior_baseline_version=5)
    _write(path, payload)
    assert analyzer.main(_arguments(tmp_path) + ["--allow-provenance-difference"]) == 0
    report, _ = _outputs(tmp_path)
    assert "b" * 40 in report and "来歴差を許可" in report
    assert "同等性を保証しない" in report


def test_duplicate_baseline_invalidates_key_but_other_time_scale_is_just_excluded(
    analyzer, tmp_path
):
    _populate(tmp_path)
    path = _artifact(tmp_path / "baseline", "fedsda", 1, 0.5, name="other-time-scale")
    payload = json.loads(path.read_text())
    # 別設定は同条件の重複とはしない。任意の設定差を用い、正しい基準は残す。
    payload["run_settings"]["unknown_option"] = 240
    _write(path, payload)
    assert analyzer.main(_arguments(tmp_path)) == 0
    _artifact(tmp_path / "baseline", "fedsda", 1, 0.1, name="duplicate")
    assert analyzer.main(_arguments(tmp_path)) == 1
    report, rows = _outputs(tmp_path)
    assert "重複" in report
    assert {row["seed"] for row in rows} == {"0"}


@pytest.mark.parametrize("value", [None, True, "0.9", float("nan"), float("inf")])
def test_undefined_metric_is_never_zero_filled(analyzer, tmp_path, value):
    _populate(tmp_path)
    path = tmp_path / "ablation" / _CONFIGURATION / "runs" / "seed1.json"
    payload = json.loads(path.read_text())
    payload["run_metrics"]["prediction_accuracy"] = value
    _write(path, payload)
    assert analyzer.main(_arguments(tmp_path)) == 0
    report, rows = _outputs(tmp_path)
    accuracy_rows = [row for row in rows if row["metric"] == "prediction_accuracy"]
    assert len(accuracy_rows) == 1
    assert "指標欠測" in report and "seed=1" in report


def test_stdout_read_only_and_unique_file_output(analyzer, tmp_path, capsys):
    _populate(tmp_path)
    arguments = _arguments(tmp_path)
    before = sorted(tmp_path.rglob("*"))
    assert analyzer.main(arguments + ["--stdout"]) == 0
    assert "全体精度" in capsys.readouterr().out
    assert sorted(tmp_path.rglob("*")) == before
    assert analyzer.main(arguments) == 0
    first = {p: p.read_bytes() for p in (tmp_path / "ablation").glob("analysis_*/*")}
    assert analyzer.main(arguments) == 0
    assert len(list((tmp_path / "ablation").glob("analysis_*"))) == 2
    assert all(p.read_bytes() == content for p, content in first.items())


def test_invalid_input_and_duplicate_grid_arguments_create_no_outputs(analyzer, tmp_path):
    arguments = _arguments(tmp_path)
    with pytest.raises(SystemExit) as error:
        analyzer.main(arguments)
    assert error.value.code == 2
    _populate(tmp_path)
    with pytest.raises(SystemExit) as error:
        analyzer.main(_arguments(tmp_path, seeds=(0, 0)))
    assert error.value.code == 2
    assert not list((tmp_path / "ablation").glob("analysis_*"))


def test_cli_complete_grid_in_fresh_process(tmp_path):
    _populate(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "analyze_fedsda_component_ablation.py"),
            *_arguments(tmp_path),
            "--stdout",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0, result.stderr
    assert "自動採否なし" in result.stdout


@pytest.mark.parametrize("value", [0, 0.125, 0.999, 1])
def test_equal_metrics_have_zero_difference_independent_of_file_order(analyzer, tmp_path, value):
    # 契約の性質: 同じ値なら差0。ファイル名とseedの順序は対付けを変えない。
    for seed in (1, 0):
        _artifact(tmp_path / "baseline", "fedsda", seed, value, name=f"base{1 - seed}")
        _artifact(
            tmp_path / "ablation" / _CONFIGURATION,
            _CONFIGURATION,
            seed,
            value,
            name=f"variant{seed}",
        )
    assert analyzer.main(_arguments(tmp_path)) == 0
    _, rows = _outputs(tmp_path)
    assert all(float(row["delta"]) == 0 for row in rows)


def test_non_finite_environment_extension_is_rejected_without_crash(analyzer, tmp_path):
    _populate(tmp_path)
    path = tmp_path / "ablation" / _CONFIGURATION / "runs" / "seed1.json"
    payload = json.loads(path.read_text())
    payload["environment"]["unexpected"] = float("nan")
    _write(path, payload)
    assert analyzer.main(_arguments(tmp_path)) == 1


def test_all_declared_configuration_defaults_accept_current_canonical_settings(analyzer, tmp_path):
    # 構成名をディレクトリ名へ対応付け、誤った一部だけを処理しない。
    for configuration in (
        "fedsda-without-candidate-validation",
        "fedsda-without-prediction-weight-recalibration",
        "fedsda-loss-distance-clustering",
        "fedsda-overall-loss-esr-detection",
        "fedsda-without-concept-specific-adapter",
        "fedsda-independent-models",
    ):
        _artifact(tmp_path / "ablation" / configuration, configuration, 0, 0.8)
    _artifact(tmp_path / "baseline", "fedsda", 0, 0.8)
    arguments = _arguments(tmp_path, seeds=(0,))
    start = arguments.index("--configurations")
    del arguments[start : start + 2]
    assert analyzer.main(arguments) == 0
    report, rows = _outputs(tmp_path)
    assert "有効対 6/6" in report
    assert len({row["configuration"] for row in rows}) == 6
