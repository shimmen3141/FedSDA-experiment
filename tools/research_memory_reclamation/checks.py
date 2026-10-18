"""研究用ランナーの明示実行専用checks。通常pytestへは登録しない。"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def runner():
    path = Path(__file__).with_name("run.py")
    spec = importlib.util.spec_from_file_location("research_memory_reclamation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reclaim_follows_successful_sync_without_changing_result(runner, monkeypatch):
    calls = []
    state = {"samples": object(), "model": object()}
    before = dict(state)
    sentinel = object()

    def sync(owner, **kwargs):
        calls.append(("sync", owner, kwargs))
        return sentinel

    def trim():
        assert state == before
        calls.append(("trim",))
        return 0

    readings = iter(
        [{"rss_bytes": 900, "high_water_bytes": 950}, {"rss_bytes": 400, "high_water_bytes": 950}]
    )
    monkeypatch.setattr(runner, "_read_memory", lambda: next(readings))
    events = []
    wrapped = runner._make_sync_wrapper(sync, trim, events.append)
    owner = object()
    assert wrapped(owner, round_index=3, new_model_registration_available=True) is sentinel
    assert [v[0] for v in calls] == ["sync", "trim"]
    assert events[0]["before_rss_bytes"] == 900
    assert events[0]["after_rss_bytes"] == 400
    assert events[0]["trim_result"] == 0
    assert state == before


def test_sync_failure_does_not_reclaim_or_record_success(runner):
    events = []

    def fail(owner, **kwargs):
        raise LookupError("original")

    def trim():
        pytest.fail("失敗した同期の後で回収してはいけない")

    with pytest.raises(LookupError, match="original"):
        runner._make_sync_wrapper(fail, trim, events.append)(object(), round_index=0)
    assert not events


def test_control_does_not_reclaim(runner, monkeypatch):
    monkeypatch.setattr(runner, "_read_memory", lambda: {"rss_bytes": 100, "high_water_bytes": 100})
    events = []
    assert (
        runner._make_sync_wrapper(lambda owner, **kw: 17, None, events.append)(
            object(), round_index=0
        )
        == 17
    )
    assert events[0]["trim_result"] is None


@pytest.mark.parametrize("count,purpose", [(0, "a"), (51, "a"), (50, ""), (50, "a\nb")])
def test_invalid_arguments_are_rejected(runner, tmp_path, count, purpose):
    args = SimpleNamespace(sample_count=count, purpose=purpose, output_directory=tmp_path / "new")
    with pytest.raises(ValueError):
        runner._validate_arguments(args)
    assert not args.output_directory.exists()


def test_plan_creates_no_directory(runner, tmp_path, capsys):
    target = tmp_path / "not-created"
    assert (
        runner.main(
            [
                "--sample-count",
                "50",
                "--purpose",
                "smoke",
                "--output-directory",
                str(target),
                "--print-plan",
            ]
        )
        == 0
    )
    assert not target.exists()
    assert json.loads(capsys.readouterr().out)["arms"] == ["control", "reclaim"]


@pytest.fixture
def pair(tmp_path):
    for mode in ("control", "reclaim"):
        arm = tmp_path / mode
        runs = arm / "cli/fedsda/runs"
        runs.mkdir(parents=True)
        values = {"value": np.array([0.0, -0.0, np.nan], dtype=np.float64)}
        values.update({f"column_{i}": np.array([i], dtype=np.int64) for i in range(132)})
        np.savez_compressed(runs / "run.npz", **values)
        summary = dict(
            run_settings=dict(
                execution_settings=dict(experiment_run_conditions=dict(per_client_sample_count=50))
            ),
            run_metrics=dict(accuracy=0.9),
            record_table_file_name="run.npz",
        )
        (runs / "run.json").write_text(json.dumps(summary))
        for name in ("environment.json", "environment-after.json"):
            (arm / name).write_text(json.dumps({"identity": "same"}))
        (arm / "parameters.json").write_text(json.dumps({"0": {"weight": {"sha256": "same"}}}))
        rows = [
            dict(
                stage="synchronization_end",
                round_index=0,
                trim_result=None if mode == "control" else 0,
                before_rss_bytes=900,
                after_rss_bytes=400,
                trim_seconds=0.1,
            ),
            dict(stage="run_return", high_water_bytes=1000),
            dict(stage="finished"),
        ]
        (arm / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (arm / "gnu-time.txt").write_text(
            "Maximum resident set size (kbytes): 1024\n"
            "Elapsed (wall clock) time (h:mm:ss or m:ss): 1:02.50\nExit status: 0\n"
            "User time (seconds): 60\nSystem time (seconds): 1.5\n"
        )
    return tmp_path


def test_identical_artifacts_pass_with_trim_returning_zero(runner, pair):
    report = runner._compare(pair)
    assert report["status"] == "passed"
    assert report["exact_array_count"] == 133
    assert report["arms"]["control"]["elapsed_seconds"] == 62.5


@pytest.mark.parametrize(
    "difference",
    [
        "metrics",
        "settings",
        "value",
        "dtype",
        "shape",
        "parameters",
        "environment",
        "incomplete",
        "missing_round",
        "control_trim",
    ],
)
def test_mismatches_are_not_reported_as_success(runner, pair, difference):
    arm = pair / "reclaim"
    if difference in ("metrics", "settings"):
        path = arm / "cli/fedsda/runs/run.json"
        obj = json.loads(path.read_text())
        obj["run_metrics" if difference == "metrics" else "run_settings"]["extra"] = 1
        path.write_text(json.dumps(obj))
    elif difference in ("value", "dtype", "shape"):
        value = np.array([0.0, -0.0, np.nan], dtype=np.float64)
        if difference == "value":
            value[1] = 0.0  # 符号付き0も検出する。
        if difference == "dtype":
            value = value.astype(np.float32)
        if difference == "shape":
            value = value.reshape(1, 3)
        path = arm / "cli/fedsda/runs/run.npz"
        with np.load(path) as original:
            values = {key: original[key] for key in original.files}
        values["value"] = value
        np.savez_compressed(path, **values)
    elif difference in ("parameters", "environment"):
        path = arm / ("parameters.json" if difference == "parameters" else "environment-after.json")
        path.write_text(json.dumps({"different": True}))
    else:
        arm = pair / "control" if difference == "control_trim" else arm
        path = arm / "events.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        if difference == "incomplete":
            rows.pop()
        elif difference == "missing_round":
            rows = rows[1:]
        else:
            rows[0]["trim_result"] = 1
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError):
        runner._compare(pair)


def test_same_missing_array_in_both_arms_is_rejected(runner, pair):
    for mode in ("control", "reclaim"):
        path = pair / mode / "cli/fedsda/runs/run.npz"
        with np.load(path) as original:
            values = {key: original[key] for key in original.files if key != "column_0"}
        np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="133"):
        runner._compare(pair)


def test_existing_output_is_not_modified(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_build_reclaimer", lambda: lambda: 0)
    sentinel = tmp_path / "keep"
    sentinel.write_bytes(b"original")
    args = SimpleNamespace(output_directory=tmp_path, sample_count=50, purpose="test")
    with pytest.raises(FileExistsError):
        runner._pair(args, runner._source_identity())
    assert sentinel.read_bytes() == b"original"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep"]


def test_unsupported_platform_rejected_before_output(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner.sys, "platform", "unsupported")
    output = tmp_path / "new"
    assert (
        runner.main(
            ["--sample-count", "50", "--purpose", "test", "--output-directory", str(output)]
        )
        == 1
    )
    assert not output.exists()


def test_context_restores_real_integration_on_exception(runner, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(runner._ROOT / "src"))
    from federated_learning_experiments.runtime.fedsda_measured_run_execution import (
        _RoundMeasuringServerOperations,
    )
    from federated_learning_experiments.sweep import (
        experiment_sweep_execution as execution,
    )

    original_sync = _RoundMeasuringServerOperations.synchronize_models
    original_run = execution.execute_fedsda_experiment_run
    original_argv = runner.sys.argv
    monkeypatch.setattr(runner, "_build_reclaimer", lambda: lambda: 0)
    monkeypatch.setattr(runner, "_read_memory", lambda: dict(rss_bytes=100, high_water_bytes=100))
    monkeypatch.setattr(
        runner,
        "_environment",
        lambda: dict(identity=dict(source_sha256=runner._EXPECTED_SOURCE), threads=1),
    )
    for name in runner._THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")

    def fail(*args, **kwargs):
        assert _RoundMeasuringServerOperations.synchronize_models is not original_sync
        assert execution.execute_fedsda_experiment_run is not original_run
        raise LookupError("inside original CLI")

    monkeypatch.setattr(runner.runpy, "run_path", fail)
    args = SimpleNamespace(
        output_directory=tmp_path, sample_count=50, purpose="test", _arm="control"
    )
    with pytest.raises(LookupError, match="inside original CLI"):
        runner._arm(args)
    assert _RoundMeasuringServerOperations.synchronize_models is original_sync
    assert execution.execute_fedsda_experiment_run is original_run
    assert runner.sys.argv is original_argv
    assert not (tmp_path / "environment-after.json").exists()


def test_first_child_failure_stops_pair_and_preserves_failure(runner, tmp_path, monkeypatch):
    import subprocess

    identity = runner._source_identity()
    monkeypatch.setattr(runner, "_build_reclaimer", lambda: lambda: 0)
    monkeypatch.setattr(runner, "_source_identity", lambda: identity)
    commands = []

    def fail(command, **kwargs):
        commands.append(command)
        raise subprocess.CalledProcessError(4, command)

    monkeypatch.setattr(runner.subprocess, "run", fail)
    output = tmp_path / "new"
    args = SimpleNamespace(output_directory=output, sample_count=50, purpose="test")
    with pytest.raises(subprocess.CalledProcessError):
        runner._pair(args, identity)
    assert len(commands) == 1
    assert (output / "control").is_dir()
    assert not (output / "reclaim").exists()
    assert json.loads((output / "failure.json").read_text())["status"] == "failed"
