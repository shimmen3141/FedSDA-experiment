"""限定Wの参照対応・再開と、予測系列の入力/被覆/累積差を検査する。"""

import json
import uuid
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.research_s3_warm_control import analyze, cumulative, run, trace


def output_path():
    return Path("results/_test_tmp") / str(uuid.uuid4())


def predictions(n=6, reverse=False):
    return [
        dict(
            client_id=c,
            sample_index=s,
            label=0,
            concept=0,
            input_hash=str((c, s)),
            predicted_label=0 if (s < 3) != reverse else 1,
            correct=(s < 3) != reverse,
        )
        for c in range(10)
        for s in range(n)
    ]


def test_cumulative_uses_correct_prefix_and_short_final_window():
    b, c = predictions(), predictions(reverse=True)
    result = cumulative.compare(b, list(reversed(c)), 6, 4)
    assert result["blocks"][0]["window_delta_pp"] == -50
    assert result["blocks"][1]["window_delta_pp"] == 100
    assert result["blocks"][1]["window_observations"] == 20
    assert result["final_delta_pp"] == 0
    for p in b:
        p["probabilities"] = [0, 1] + [0] * 8 if p.pop("predicted_label") else [1] + [0] * 9
    assert cumulative.compare(b, c, 6, 3)["blocks"][0]["cumulative_delta_pp"] == -100


@pytest.mark.parametrize("bad", ["missing", "duplicate", "input", "label", "concept", "correct"])
def test_cumulative_rejects_invalid_or_unpaired_records(bad):
    b, c = predictions(), predictions(reverse=True)
    if bad == "missing":
        c.pop()
    elif bad == "duplicate":
        c[-1] = c[0]
    elif bad == "input":
        c[0]["input_hash"] = "different"
    elif bad == "label":
        c[0]["label"] = 10
    elif bad == "concept":
        c[0]["concept"] = 1
    else:
        c[0]["correct"] = True
    with pytest.raises(ValueError):
        cumulative.compare(b, c, 6, 4)


def test_job_plan_is_only_W_twenty_runs():
    jobs = run.jobs_for(list(range(10)), [100, 500])
    assert len(jobs) == 20 and {j["arm"] for j in jobs} == {"warm-matched"}
    assert {j["sample_count"] for j in jobs} == {5000}


@pytest.mark.parametrize(
    "seeds,intervals",
    [([0, 1, 2], [100]), ([0, 1, 2, 2], [100]), ([0, 1, 2, 3], [50]), ([0, 1, 2, 3], [100, 100])],
)
def test_bad_plans(seeds, intervals):
    with pytest.raises(ValueError):
        run.jobs_for(seeds, intervals)


def test_reference_resolution_requires_unique_completed_result():
    root = output_path()
    (root / "one").mkdir(parents=True)
    (root / "one/complete.json").write_text("{}")
    assert run.resolve_reference(root) == root / "one"
    (root / "two").mkdir()
    (root / "two/complete.json").write_text("{}")
    with pytest.raises(ValueError):
        run.resolve_reference(root)


def fake_row(j):
    accuracy = {"product-default": 0.8, "warm-matched": 0.81, "full-score-head-train": 0.82}[
        j["arm"]
    ]
    return dict(
        job=j,
        input_signature="0" * 64,
        metrics=dict(
            **{k: v for k, v in j.items() if k != "arm"},
            research_variant=j["arm"],
            variant="product-default",
            check=False,
            accuracy=accuracy,
            record_accuracy=accuracy,
            uploaded_bytes=100,
            downloaded_bytes=200,
            model_computation_counts=dict(forward_multiply_accumulate_count=30),
            preparation_model_computation_counts={},
            seconds=1,
            concept_accuracies={},
            prediction_period_accuracies={},
            prediction_record_digest="fake",
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

    def submit(self, fn, *args):
        f = Future()
        try:
            f.set_result(fn(*args))
        except Exception as e:
            f.set_exception(e)
        return f


@pytest.fixture
def ready(monkeypatch):
    monkeypatch.setattr(run.full, "source_hashes", lambda: {"code": "same"})
    monkeypatch.setattr(run, "source_hashes", lambda: {"code": "same", "warm": "same"})
    monkeypatch.setattr(analyze, "source_hashes", lambda: {"code": "same", "warm": "same"})
    monkeypatch.setattr(run.full, "environment", lambda: {"env": "same"})
    monkeypatch.setattr(run, "environment", lambda: {"env": "same"})
    monkeypatch.setattr(run.full, "execute", fake_row)
    monkeypatch.setattr(run.full, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(run, "ProcessPoolExecutor", InlinePool)
    ref = output_path()
    p = dict(datasets=["mnist4"], intervals=[100, 500], seeds=[0, 1, 2, 3], sample_count=5000)
    run.full.execute_plan(ref, run.full.make_jobs(**p), p, 2, False)

    def fake_trace_check(row, output, refs):
        expected = refs[run.full.job_key(row["job"])[:-1] + ("full-score-head-train",)][
            "input_signature"
        ]
        if row["input_signature"] != expected:
            raise ValueError("input mismatch")

    monkeypatch.setattr(run, "validate_trace", fake_trace_check)
    monkeypatch.setattr(analyze, "validate_trace", fake_trace_check)
    return ref


def test_resume_retries_only_failed_W_and_decomposes(ready, monkeypatch, capsys):
    output = output_path()
    jobs = run.jobs_for([0, 1, 2, 3], [100, 500])
    calls = []

    def fail(j, expected, out):
        calls.append(run.full.job_key(j))
        if j == jobs[0]:
            raise RuntimeError("intentional")
        return fake_row(j)

    monkeypatch.setattr(run, "execute", fail)
    with pytest.raises(RuntimeError):
        run.execute_plan(ready, output, jobs, 2, False)
    assert len(list((output / "runs").glob("*.json"))) == 7

    def good(j, expected, out):
        calls.append(run.full.job_key(j))
        return fake_row(j)

    monkeypatch.setattr(run, "execute", good)
    run.execute_plan(ready, output, jobs, 1, True)
    assert len(calls) == 9 and "W 8/8" in capsys.readouterr().out
    result = analyze.analyze(output, ready)
    assert result["new_W_runs"] == 8 and result["reused_D_B_runs"] == 16
    assert result["contrasts"][0]["initialization_pp"] == pytest.approx(1)
    assert result["contrasts"][0]["training_scope_pp"] == pytest.approx(1)
    assert result["contrasts"][0]["B_minus_default_pp"] == pytest.approx(2)
    run.execute_plan(ready, output, jobs, 1, True)
    assert len(calls) == 9


@pytest.mark.parametrize("bad", ["environment", "reference", "source", "missing"])
def test_reference_rejects_mismatch(ready, monkeypatch, bad):
    jobs = run.jobs_for([0, 1, 2, 3], [100, 500])
    if bad == "environment":
        monkeypatch.setattr(run, "environment", lambda: {})
    else:
        path = ready / ("metadata.json" if bad == "source" else "results.json")
        data = json.loads(path.read_text())
        if bad == "source":
            data["source_hashes"] = {}
        elif bad == "missing":
            data.pop()
        else:
            data[0]["metrics"]["accuracy"] = 0.1
        run.full.atomic_json(path, data)
    with pytest.raises(ValueError):
        run.load_reference(ready, jobs, True)


def test_actual_trace_validation_checks_artifact_and_input():
    tmp_path = output_path()
    ps = predictions(6)
    inputs, digest = trace.signature(ps, 6)
    j = dict(
        dataset="mnist4",
        fixed_concept=None,
        interval=500,
        seed=0,
        sample_count=6,
        arm="warm-matched",
    )
    row = fake_row(j)
    row["input_signature"] = inputs
    row["metrics"].update(accuracy=0.5, record_accuracy=0.5, prediction_record_digest=digest)
    relative = "predictions/" + run.full.job_id(j) + ".json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(predictions=ps)))
    row.update(trace_file=relative, trace_sha256=run.full.sha(path), extra_model_forward_count=0)
    refs = {run.full.job_key(j)[:-1] + ("full-score-head-train",): dict(input_signature=inputs)}
    run.validate_trace(row, tmp_path, refs)
    path.write_text("{}")
    with pytest.raises(ValueError):
        run.validate_trace(row, tmp_path, refs)


@pytest.mark.parametrize("raises", [False, True])
def test_recorder_preserves_result_and_restores_native_hooks(monkeypatch, raises):
    import torch

    import federated_learning_experiments.runtime.fedsda_run_client as client
    import federated_learning_experiments.runtime.observed_sample_processing as processing

    result = SimpleNamespace(
        combined_prediction_probabilities=torch.tensor([0.8, 0.1] + [0.0125] * 8),
        sample_prediction_record=SimpleNamespace(combined_prediction_is_correct=True),
    )
    observation = SimpleNamespace(
        sample_index=0,
        observed_concept_id=0,
        training_sample=SimpleNamespace(
            input_features=torch.tensor([[1.0]]), observed_class_labels=torch.tensor(0)
        ),
    )

    def predict(**kw):
        if raises:
            raise RuntimeError("intentional")
        return result

    def process(**kw):
        return processing.predict_observed_sample_and_update_prediction_weights(
            indexed_observation=observation
        )

    monkeypatch.setattr(client, "process_observed_sample", process)
    monkeypatch.setattr(
        processing, "predict_observed_sample_and_update_prediction_weights", predict
    )
    tracker = trace.PredictionTrace()
    if raises:
        with pytest.raises(RuntimeError), tracker.activated():
            client.process_observed_sample(client_id=3)
    else:
        with tracker.activated():
            assert client.process_observed_sample(client_id=3) is result
        assert tracker.predictions[0]["predicted_label"] == 0
    assert tracker.client_id is None
    assert client.process_observed_sample is process
    assert processing.predict_observed_sample_and_update_prediction_weights is predict


def test_digit_nine_is_valid_in_four_concept_mnist():
    rows = predictions()
    for p in rows:
        p.update(label=9, predicted_label=9, correct=True, probabilities=[0] * 9 + [1])
    legacy = [{k: v for k, v in p.items() if k != "predicted_label"} for p in rows]
    assert cumulative.compare(legacy, rows, 6, 4)["final_delta_pp"] == 0


@pytest.mark.parametrize(
    "probabilities", [[float("nan")] + [0] * 9, [0.2] * 10, [1.1] + [0] * 9, [1, 0, 0, 0]]
)
def test_legacy_probability_validation(probabilities):
    rows = predictions()
    rows[0]["probabilities"] = probabilities
    with pytest.raises(ValueError):
        cumulative.compare(rows, predictions(), 6, 3)


@pytest.mark.parametrize("bad", ["source", "rectangle", "binding", "failure", "input"])
def test_analysis_rejects_corrupt_W_artifacts(ready, monkeypatch, bad):
    output = output_path()
    jobs = run.jobs_for([0, 1, 2, 3], [100, 500])
    monkeypatch.setattr(run, "execute", lambda j, *args: fake_row(j))
    run.execute_plan(ready, output, jobs, 2, False)
    if bad in ("source", "rectangle", "binding"):
        m = json.loads((output / "metadata.json").read_text())
        if bad == "source":
            m["source_hashes"] = {}
        elif bad == "rectangle":
            m["jobs"].pop()
        else:
            m["reference_results_sha256"] = "bad"
        run.full.atomic_json(output / "metadata.json", m)
    elif bad == "failure":
        run.full.atomic_json(output / "failures.json", ["failed"])
    else:
        rows = json.loads((output / "results.json").read_text())
        rows[0]["input_signature"] = "1" * 64
        run.full.atomic_json(output / "results.json", rows)
        run.full.atomic_json(output / "runs" / (run.full.job_id(rows[0]["job"]) + ".json"), rows[0])
        run.full.atomic_json(
            output / "complete.json",
            dict(
                run_count=len(rows),
                source_hashes_verified=True,
                results_sha256=run.full.sha(output / "results.json"),
            ),
        )
    with pytest.raises(ValueError):
        analyze.analyze(output, ready)
    assert not (output / "summary.json").exists()


def test_resume_print_plan_is_readonly_and_counts_completed_W(ready, monkeypatch, capsys):
    output = output_path()
    jobs = run.jobs_for([0, 1, 2, 3], [100, 500])
    monkeypatch.setattr(run, "execute", lambda j, *args: fake_row(j))
    run.execute_plan(ready, output, jobs, 2, False)
    before = {p: p.stat().st_mtime_ns for p in output.rglob("*") if p.is_file()}
    capsys.readouterr()
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--reference",
            str(ready),
            "--output",
            str(output),
            "--workers",
            "2",
            "--seeds",
            "0",
            "1",
            "2",
            "3",
            "--resume",
            "--print-plan",
        ],
    )
    run.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["completed_W_runs"] == 8 and printed["pending_W_runs"] == 0
    assert before == {p: p.stat().st_mtime_ns for p in output.rglob("*") if p.is_file()}
