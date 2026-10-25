"""研究専用の対応差・再現拒否・読み取り専用planの検査。通常CIへ追加しない。"""

import json
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from tools.research_s3_b_replay import analyze, run
from tools.research_s3_b_replay.analyze import paired_details, standard_cell
from tools.research_s3_warm_control.cumulative import compare
from tools.research_s3_warm_control.trace import signature


class ReplayChecks(unittest.TestCase):
    def test_jobs(self):
        jobs = run.jobs_for([0, 3, 7, 8], [100, 500])
        self.assertEqual(len(jobs), 8)
        self.assertEqual({j["arm"] for j in jobs}, {run.ARM})
        self.assertEqual({j["sample_count"] for j in jobs}, {5000})
        for seeds, intervals in (
            ([0, 3], [100, 500]),
            ([0, 3, 7, 7], [100]),
            ([0, 3, 7, 8], [200]),
        ):
            with self.assertRaises(ValueError):
                run.jobs_for(seeds, intervals)

    def test_only_seconds_may_differ(self):
        original = dict(
            job={"arm": run.ARM},
            input_signature="abc",
            metrics=dict(
                seconds=3,
                accuracy=0.9,
                model_computation_counts={"mac": 10},
                research_events=[{"event": "train", "count": 1}],
            ),
        )
        row = deepcopy(original)
        row["metrics"]["seconds"] = 9
        run.require_replay_match(row, original)
        for field, value in (
            ("accuracy", 0.91),
            ("model_computation_counts", {"mac": 11}),
            ("research_events", []),
            ("unknown_new_metric", 1),
        ):
            changed = deepcopy(row)
            changed["metrics"][field] = value
            with self.assertRaises(ValueError):
                run.require_replay_match(changed, original)
        row["input_signature"] = "different"
        with self.assertRaises(ValueError):
            run.require_replay_match(row, original)

    def test_trace_execute_requires_parity(self):
        original = dict(
            job={"arm": run.ARM}, input_signature="abc", metrics={"accuracy": 0.9}
        )
        with patch.object(run.warm, "execute", return_value=deepcopy(original)):
            self.assertTrue(
                run.execute(original["job"], original, Path("unused"))[
                    "original_B_metrics_verified"
                ]
            )
        altered = deepcopy(original)
        altered["metrics"]["accuracy"] = 0.8
        with patch.object(run.warm, "execute", return_value=altered):
            with self.assertRaises(ValueError):
                run.execute(original["job"], original, Path("unused"))

    def test_cumulative_and_concepts(self):
        base = [
            dict(
                client_id=c,
                sample_index=s,
                input_hash=f"{c}-{s}",
                label=0,
                concept=s % 2,
                predicted_label=0,
                correct=True,
            )
            for c in range(10)
            for s in range(3)
        ]
        candidate = deepcopy(base)
        candidate[0].update(predicted_label=1, correct=False)
        result = compare(base, candidate, 3, 2)
        self.assertAlmostEqual(result["final_delta_pp"], -100 / 30)
        self.assertAlmostEqual(result["blocks"][0]["window_delta_pp"], -5)
        concepts = paired_details(base, candidate)
        self.assertEqual(concepts[0]["harmed"], 1)
        self.assertEqual(concepts[0]["rescued"], 0)
        self.assertEqual(concepts[0]["delta_pp"], -5)
        candidate[1]["input_hash"] = "different"
        with self.assertRaises(ValueError):
            compare(base, candidate, 3, 2)
        with self.assertRaises(ValueError):
            compare(base[:-1], base[:-1], 3, 2)

    def test_standard_metrics_and_missing_time(self):
        metrics = dict(
            accuracy=0.92,
            uploaded_bytes=10,
            downloaded_bytes=20,
            model_computation_counts=dict(
                forward_multiply_accumulate_count=4,
                backward_multiply_accumulate_count=6,
                ignored=100,
            ),
            seconds=None,
        )
        cell = standard_cell(
            dict(interval=100, seed=0),
            "B",
            metrics,
            {"accuracy": 0.9},
            {"accuracy": 0.91},
        )
        self.assertAlmostEqual(cell["default_delta_pp"], 2)
        self.assertAlmostEqual(cell["warm_delta_pp"], 1)
        self.assertEqual(cell["mac"], 10)
        self.assertIsNone(cell["seconds"])

    def test_plan_does_not_create_output(self):
        output = full_output = run.full.ROOT / "results" / ("plan-check-" + uuid4().hex)
        context = (Path("ref"), {"execution_environment": {}}, {}, {}, {})
        with (
            patch.object(run, "load_context", return_value=context),
            patch.object(run, "source_hashes", return_value={}),
        ):
            run.run(
                Path("ref"),
                Path("warm"),
                output,
                run.jobs_for([0, 3, 7, 8], [100, 500]),
                64,
                False,
                True,
            )
        self.assertFalse(full_output.exists())

    def test_analysis_pipeline_with_controlled_traces(self):
        # NNを回さず、ファイルI/Oだけを置き換えて8対の検査・集計を通す。
        output, warm_output = Path("fixture-b"), Path("fixture-w")
        jobs = [dict(j, sample_count=3) for j in run.jobs_for([0, 3, 7, 8], [100, 500])]
        base = [
            dict(
                client_id=c,
                sample_index=s,
                input_hash=f"{c}-{s}",
                label=0,
                concept=s % 2,
                predicted_label=0,
                correct=True,
            )
            for c in range(10)
            for s in range(3)
        ]
        files, rows, refs, ws = {}, [], {}, {}
        digest = "a" * 64
        for job in jobs:
            candidate = deepcopy(base)
            harmed = [0, 3, 7, 8].index(job["seed"]) + 1
            for c in range(harmed):
                candidate[3 * c].update(predicted_label=1, correct=False)
            inputs, bd = signature(candidate, 3)
            _, wd = signature(base, 3)
            metrics = dict(
                job,
                variant="product-default",
                check=False,
                research_variant=run.ARM,
                accuracy=(30 - harmed) / 30,
                record_accuracy=(30 - harmed) / 30,
                uploaded_bytes=10,
                downloaded_bytes=20,
                model_computation_counts={"forward_multiply_accumulate_count": 4},
                seconds=2,
                prediction_record_digest=bd,
            )
            metrics.pop("arm")
            filename = "predictions/" + run.full.job_id(job) + ".json"
            row = dict(
                job=job,
                metrics=metrics,
                input_signature=inputs,
                trace_file=filename,
                trace_sha256=digest,
                extra_model_forward_count=0,
                original_B_metrics_verified=True,
            )
            rows.append(row)
            refs[run.full.job_key(job)] = row
            refs[run.full.job_key(job)[:-1] + ("product-default",)] = {
                "metrics": dict(metrics, accuracy=0.9, seconds=None)
            }
            ws[run.full.job_key(job)[:-1] + ("warm-matched",)] = dict(
                metrics=dict(metrics, accuracy=1, prediction_record_digest=wd),
                trace_file=filename,
                trace_sha256=digest,
            )
            files[output / filename] = dict(predictions=candidate)
            files[warm_output / filename] = dict(predictions=base)
        files.update(
            {
                output / "metadata.json": dict(
                    jobs=jobs,
                    source_hashes={},
                    execution_environment={},
                    reference="fixture-ref",
                    warm_output=str(warm_output),
                ),
                output / "results.json": rows,
                output / "failures.json": [],
                output / "complete.json": dict(
                    run_count=8, source_hashes_verified=True, results_sha256=digest
                ),
            }
        )
        with (
            patch.object(
                run,
                "load_context",
                return_value=(
                    Path("fixture-ref"),
                    {"execution_environment": {}},
                    refs,
                    ws,
                    {},
                ),
            ),
            patch.object(analyze, "load_context", run.load_context),
            patch.object(analyze, "jobs_for", return_value=jobs),
            patch.object(analyze, "source_hashes", return_value={}),
            patch.object(run.full, "sha", return_value=digest),
            patch.object(run.full, "load_checkpoints", return_value=rows),
            patch.object(Path, "read_text", lambda path, **kw: json.dumps(files[path])),
            patch.object(Path, "write_text"),
            patch.object(run.full, "atomic_json"),
        ):
            result = analyze.analyze(output, window_size=2)
        self.assertEqual(len(result["pairs"]), 8)
        self.assertEqual(len(result["standard_metrics_by_seed"]), 24)
        self.assertAlmostEqual(result["summaries"][2]["warm_delta_pp"], -100 * 2.5 / 30)
        self.assertAlmostEqual(
            result["cumulative_summaries"][1]["cumulative_delta_pp"], -100 * 2.5 / 30
        )
        self.assertEqual(result["concept_summaries"][0]["harmed"], 10)
        self.assertEqual(result["concept_summaries"][0]["delta_pp"], -12.5)


if __name__ == "__main__":
    unittest.main()
