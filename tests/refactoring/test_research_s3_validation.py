"""未使用seedの計画・対応付け・失敗拒否と標準指標を検査する。"""

import hashlib
import json
import uuid
from pathlib import Path

import pytest

from tools.research_s3_validation import analyze, run
from tools.research_s3_validation.trace import validate_predictions


def rows_for(jobs):
    rows = []
    for j in jobs:
        accuracy = 0.8 + 0.001 * j["seed"] + 0.002 * run.ARMS.index(j["arm"])
        rows.append(
            dict(
                job=j,
                input_signature=str(j["seed"]),
                metrics=dict(
                    **{k: v for k, v in j.items() if k != "arm"},
                    research_variant=j["arm"],
                    variant="product-default",
                    check=False,
                    accuracy=accuracy,
                    uploaded_bytes=100,
                    downloaded_bytes=200,
                    model_computation_counts=dict(
                        forward_multiply_accumulate_count=40,
                        backward_multiply_accumulate_count=80,
                        other_count=999,
                    ),
                    seconds=2.5,
                    concept_accuracies={},
                    prediction_period_accuracies={},
                ),
            )
        )
    return rows


@pytest.mark.parametrize(
    "seeds,count",
    [([0, 1, 2, 3], 2500), ([4, 5, 6], 2500), ([4, 5, 6, 6], 2500), ([4, 5, 6, 7], 2499)],
)
def test_plan_rejects_explored_or_insufficient_conditions(seeds, count):
    with pytest.raises(ValueError):
        run.make_jobs(seeds, count)


def test_fixed_rectangle_and_signatures():
    jobs = run.make_jobs([4, 5, 6, 7], 2500)
    assert len(jobs) == 16
    assert run.validate_rows(rows_for(jobs), jobs) == {s: str(s) for s in range(4, 8)}


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "input", "condition"])
def test_invalid_rows_rejected(corruption):
    jobs = run.make_jobs([4, 5, 6, 7], 2500)
    rows = rows_for(jobs)
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows[-1] = rows[0]
    elif corruption == "input":
        rows[1]["input_signature"] = "other"
    else:
        rows[0]["metrics"]["seed"] = 99
    with pytest.raises(ValueError):
        run.validate_rows(rows, jobs)


@pytest.mark.parametrize("corruption", [None, "rectangle", "source", "incomplete", "failure"])
def test_analysis_standard_metrics_and_rejections(monkeypatch, corruption):
    output = Path("results/_test_tmp") / str(uuid.uuid4())
    output.mkdir(parents=True)
    jobs = run.make_jobs([4, 5, 6, 7], 2500)
    meta = dict(jobs=jobs, seeds=[4, 5, 6, 7], sample_count=2500, source_hashes={"a": "x"})
    rows = rows_for(jobs)
    done = dict(run_count=16, source_hashes_verified=True)
    if corruption == "rectangle":
        rows.pop()
        meta["jobs"] = jobs[:-1]
        done["run_count"] = 15
    elif corruption == "source":
        meta["source_hashes"] = {}
    elif corruption == "incomplete":
        done["source_hashes_verified"] = False
    elif corruption == "failure":
        (output / "failures.json").write_text("[]")
    for name, obj in (("metadata", meta), ("results", rows), ("complete", done)):
        (output / (name + ".json")).write_text(json.dumps(obj), encoding="utf-8")
    monkeypatch.setattr(analyze, "source_hashes", lambda: {"a": "x"})
    if corruption:
        with pytest.raises(ValueError):
            analyze.analyze(output)
        assert not (output / "summary.json").exists()
    else:
        result = analyze.analyze(output)
        b = result["summaries"][2]
        assert b["default_delta_pp"] == pytest.approx(0.4)
        assert b["matched_delta_pp"] == pytest.approx(0.2)
        assert b["s3_delta_pp"] == pytest.approx(-0.2)
        assert b["positive_seeds"] == 4
        assert b["mac"] == 120
        assert b["uploaded_bytes"] == 100
        assert b["downloaded_bytes"] == 200
        assert b["seconds"] == 2.5


@pytest.mark.parametrize("corruption", [None, "coverage", "probability", "digest"])
def test_trace_checks_exact_inputs_and_prediction_records(corruption):
    predictions = [
        dict(client_id=c, sample_index=0, label=0, correct=True, probabilities=[1, 0, 0, 0])
        for c in range(10)
    ]
    metrics = dict(
        accuracy=1.0,
        prediction_record_digest=hashlib.sha256(json.dumps([True] * 10).encode()).hexdigest(),
    )
    if corruption == "coverage":
        predictions[-1]["client_id"] = 0
    elif corruption == "probability":
        predictions[0]["probabilities"] = [0, 1, 0, 0]
    elif corruption == "digest":
        metrics["prediction_record_digest"] = "wrong"
    if corruption:
        with pytest.raises(ValueError):
            validate_predictions(dict(predictions=predictions), metrics, 1)
    else:
        validate_predictions(dict(predictions=predictions), metrics, 1)


@pytest.mark.parametrize("missing_side", ["base", "measured"])
def test_first_decision_includes_either_one_sided_event(missing_side):
    from tools.research_s3_contribution.analyze import first_decision_difference

    def record(sample):
        return dict(
            client_id=4,
            sample_index=sample,
            prediction_order=sample * 10 + 4,
            selection=dict(
                selected_hypothesis="held_model",
                change_interval_sample_count=30,
                destination_model_id=0,
                new_model_learner_name="",
            ),
            held_parameter_hashes={},
            observation_indices=[],
        )

    early, late = record(109), record(126)
    base, measured = ([late], [early, late]) if missing_side == "base" else ([early, late], [late])
    result = first_decision_difference(base, measured)
    assert result["sample_index"] == 109
    assert result["baseline" if missing_side == "base" else "measured"] is None
