"""本規模研究CLIの再開・進行表示・既定再利用の拒否条件を検査する。"""

import hashlib
import json
import uuid
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.research_s3_fullscale import analyze, baseline, run


def new_output():
    return Path("results/_test_tmp") / str(uuid.uuid4())


def plan():
    return dict(datasets=["sea2"], intervals=[100], seeds=[0, 1, 2, 3], sample_count=5000)


def fake_row(j):
    signature = hashlib.sha256(str((j["dataset"], j["interval"], j["seed"])).encode()).hexdigest()
    accuracy = 0.8 + (0.01 if j["arm"] == "full-score-head-train" else 0)
    return dict(
        job=j,
        input_signature=signature,
        metrics=dict(
            **{k: v for k, v in j.items() if k != "arm"},
            research_variant=j["arm"],
            variant="product-default",
            check=False,
            accuracy=accuracy,
            record_accuracy=accuracy,
            uploaded_bytes=100,
            downloaded_bytes=200,
            model_computation_counts=dict(
                forward_multiply_accumulate_count=30, backward_multiply_accumulate_count=60
            ),
            preparation_model_computation_counts=dict(forward_multiply_accumulate_count=10),
            seconds=2,
            concept_accuracies={"0": accuracy},
            prediction_period_accuracies={},
            research_events=[],
        ),
    )


class InlinePool:
    def __init__(self, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, fn, job):
        future = Future()
        try:
            future.set_result(fn(job))
        except Exception as error:
            future.set_exception(error)
        return future


@pytest.fixture
def execution(monkeypatch):
    monkeypatch.setattr(run, "source_hashes", lambda: {"source": "hash"})
    monkeypatch.setattr(analyze, "source_hashes", lambda: {"source": "hash"})
    monkeypatch.setattr(run, "environment", lambda: {"fake": "environment"})
    monkeypatch.setattr(baseline, "environment", lambda: {"fake": "environment"})
    monkeypatch.setattr(run, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(run, "execute", fake_row)


def test_full_plan_and_only_two_arms():
    jobs = run.make_jobs(list(run.DATASETS), [50, 100, 200, 500], list(range(10)), 5000)
    assert len(jobs) == 480
    assert {j["arm"] for j in jobs} == {"product-default", "full-score-head-train"}


@pytest.mark.parametrize(
    "change",
    [
        dict(sample_count=2500),
        dict(seeds=[0, 1, 2]),
        dict(seeds=[0, 1, 2, 2]),
        dict(intervals=[0]),
        dict(datasets=["unknown"]),
    ],
)
def test_bad_plans(change):
    p = plan() | change
    with pytest.raises(ValueError):
        run.make_jobs(**p)


def test_resume_failure_retries_only_missing_run(execution, monkeypatch, capsys):
    output = new_output()
    p = plan()
    jobs = run.make_jobs(**p)
    calls = []

    def fail_one(j):
        calls.append(run.job_id(j))
        if j == jobs[0]:
            raise RuntimeError("intentional failure")
        return fake_row(j)

    monkeypatch.setattr(run, "execute", fail_one)
    with pytest.raises(RuntimeError, match="失敗run"):
        run.execute_plan(output, jobs, p, 2, False)
    assert len(list((output / "runs").glob("*.json"))) == 7
    assert not (output / "complete.json").exists()

    def succeed(j):
        calls.append(run.job_id(j))
        return fake_row(j)

    monkeypatch.setattr(run, "execute", succeed)
    run.execute_plan(output, jobs, p, 1, True)
    assert len(calls) == 9 and calls[-1] == run.job_id(jobs[0])
    assert "NEW 8/8" in capsys.readouterr().out
    result = analyze.analyze(output)
    assert result["summaries"][1]["default_delta_pp"] == pytest.approx(1.0)
    assert result["direct_input_pair_groups_verified"] == 4
    assert (output / "standard_metrics_by_seed.csv").exists()
    run.execute_plan(output, jobs, p, 1, True)
    assert len(calls) == 9


@pytest.mark.parametrize("corruption", ["source", "row", "checkpoint", "failure", "plan", "input"])
def test_analysis_fails_closed(execution, corruption):
    output = new_output()
    p = plan()
    jobs = run.make_jobs(**p)
    run.execute_plan(output, jobs, p, 2, False)
    if corruption == "source":
        m = json.loads((output / "metadata.json").read_text())
        m["source_hashes"] = {}
        run.atomic_json(output / "metadata.json", m)
    elif corruption in ("row", "input"):
        rows = json.loads((output / "results.json").read_text())
        if corruption == "row":
            rows.pop()
        else:
            rows[0]["input_signature"] = "0" * 64
        run.atomic_json(output / "results.json", rows)
        run.atomic_json(
            output / "complete.json",
            dict(
                run_count=len(rows),
                source_hashes_verified=True,
                results_sha256=run.sha(output / "results.json"),
            ),
        )
    elif corruption == "checkpoint":
        run.atomic_json(output / "runs" / "unplanned.json", fake_row(jobs[0]))
    elif corruption == "failure":
        run.atomic_json(output / "failures.json", ["failed"])
    else:
        m = json.loads((output / "metadata.json").read_text())
        m["jobs"].pop()
        run.atomic_json(output / "metadata.json", m)
    with pytest.raises(ValueError):
        analyze.analyze(output)
    assert not (output / "summary.json").exists()


def settings_for(j):
    return dict(
        execution_settings=dict(
            experiment_run_conditions=dict(
                dataset_name=j["dataset"],
                random_seed=j["seed"],
                client_count=10,
                per_client_sample_count=j["sample_count"],
                server_aggregation_interval_per_client_samples=j["interval"],
            )
        )
    )


def native_record(j):
    return dict(
        run_settings=settings_for(j),
        environment={"fake": "environment"},
        run_provenance=dict(
            code_commit="a" * 40, has_uncommitted_changes=False, behavior_baseline_version=12
        ),
        run_metrics=dict(
            processed_sample_count=50000,
            prediction_accuracy=0.8,
            prediction_diagnostic_metrics=dict(
                prediction_correct_counts=dict(
                    predicted_sample_count=50000, combined_prediction_correct_count=40000
                ),
                combined_prediction_period_accuracies={},
            ),
            communication_volume=dict(uploaded_byte_count=100, downloaded_byte_count=200),
            model_computation_counts=dict(
                forward_multiply_accumulate_count=30, backward_multiply_accumulate_count=60
            ),
        ),
    )


@pytest.fixture
def native_validation(monkeypatch):
    monkeypatch.setattr(
        baseline, "native_tree_proof", lambda c: dict(commit=c, native_tree_verified=True)
    )
    monkeypatch.setattr(baseline, "expected_settings", settings_for)


def test_baseline_reuse_only_executes_B_and_keeps_unknowns(
    execution, native_validation, monkeypatch, capsys
):
    output = new_output()
    p = plan()
    jobs = run.make_jobs(**p)
    source = new_output()
    source.mkdir(parents=True)
    for j in jobs:
        if j["arm"] == "product-default":
            run.atomic_json(source / (run.job_id(j) + ".json"), native_record(j))
    calls = []

    def execute_B(j):
        calls.append(j)
        return fake_row(j)

    monkeypatch.setattr(run, "execute", execute_B)
    run.execute_plan(output, jobs, p, 2, False, source)
    assert len(calls) == 4 and all(j["arm"] == "full-score-head-train" for j in calls)
    assert "NEW 4/4" in capsys.readouterr().out
    result = analyze.analyze(output)
    assert result["reused_default_count"] == 4 and result["direct_input_pair_groups_verified"] == 0
    assert result["summaries"][0]["seconds"] is None
    assert result["summaries"][0]["preparation_mac"] is None
    assert "N/A" in (output / "analysis_standard_metrics.md").read_text(encoding="utf-8")
    run.execute_plan(output, jobs, p, 1, True, source)
    assert len(calls) == 4
    path = next((output / "baseline").glob("*.json"))
    path.write_text("{}")
    with pytest.raises(ValueError):
        analyze.analyze(output)


@pytest.mark.parametrize("corruption", ["dirty", "version", "settings", "environment", "count"])
def test_baseline_rejects_incompatible_records(native_validation, corruption):
    j = run.make_jobs(**plan())[0]
    r = native_record(j)
    if corruption == "dirty":
        r["run_provenance"]["has_uncommitted_changes"] = True
    elif corruption == "version":
        r["run_provenance"]["behavior_baseline_version"] = 11
    elif corruption == "settings":
        r["run_settings"] = {}
    elif corruption == "environment":
        r["environment"] = {}
    else:
        r["run_metrics"]["processed_sample_count"] = 1
    with pytest.raises(ValueError):
        baseline.validate_record(r, j, {}, dict(fake="environment"))


def test_invalid_commit_is_not_executed():
    with pytest.raises(ValueError):
        baseline.native_tree_proof("--not-a-commit")


def test_input_signatures_are_grouped_by_dataset_and_interval():
    jobs = run.make_jobs(["sea2", "sea4"], [100, 500], [0, 1, 2, 3], 5000)
    rows = [fake_row(j) for j in jobs]
    assert len(run.validate_rows(rows, jobs)) == 16


@pytest.mark.parametrize("corruption", [None, "blob", "files"])
def test_native_tree_proof_checks_contents_and_file_set(monkeypatch, corruption):
    root = new_output()
    path = root / "src/federated_learning_experiments/a.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"value = 1\r\n")
    data = b"value = 1\n"
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if corruption == "blob":
        blob = "0" * 40
    stdout = f"100644 blob {blob}\tsrc/federated_learning_experiments/a.py\n"
    if corruption == "files":
        stdout = ""
    monkeypatch.setattr(baseline, "ROOT", root)
    monkeypatch.setattr(baseline.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=stdout))
    if corruption:
        with pytest.raises(ValueError):
            baseline.native_tree_proof("a" * 40)
    else:
        assert baseline.native_tree_proof("a" * 40)["native_file_count"] == 1


def test_resume_rejects_source_and_environment_changes(execution, monkeypatch):
    output = new_output()
    p = plan()
    jobs = run.make_jobs(**p)
    run.execute_plan(output, jobs, p, 2, False)
    monkeypatch.setattr(run, "source_hashes", lambda: {})
    with pytest.raises(ValueError):
        run.execute_plan(output, jobs, p, 2, True)
    monkeypatch.setattr(run, "source_hashes", lambda: {"source": "hash"})
    monkeypatch.setattr(run, "environment", lambda: {})
    with pytest.raises(ValueError):
        run.execute_plan(output, jobs, p, 2, True)
