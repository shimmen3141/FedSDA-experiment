"""②の採用実験と①のoptimizer対照を別出力・別processで実行する研究CLI。"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
from statistics import mean
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def execute_job(job):
    try:
        return _execute_job(job)
    except Exception:
        raise RuntimeError(traceback.format_exc()) from None


def _execute_job(job):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    from tools.research_provisional_candidate_expert.run import _execute_job as run
    from tools.research_shadow_learner_diagnostic.controls import (
        stationary_concept_policy,
    )
    from tools.research_shadow_learner_diagnostic.diagnostic import (
        shadow_learner_diagnostic_policy,
    )
    from tools.research_shadow_learner_diagnostic.adoption import adopt_shadow
    from tools.research_shadow_learner_diagnostic.optimizer_control import (
        optimizer_reset_control,
    )

    common = {k: job[k] for k in ("dataset", "interval", "seed", "sample_count")}
    common.update(
        variant="product-default", check=False, record_prediction_period_metrics=True
    )
    policy = dict(dataset=job["dataset"], concept_id=job["fixed_concept"])
    with stationary_concept_policy(**policy):
        baseline = run(common)
    diagnostics, trace = {}, []
    handler = (
        adopt_shadow
        if job["task"] == "adoption" and not job.get("disabled", False)
        else None
    )
    with (
        stationary_concept_policy(**policy),
        optimizer_reset_control() if job["task"] == "optimizer" else nullcontext(),
        shadow_learner_diagnostic_policy(
            diagnostics=diagnostics,
            trace=trace,
            horizon_samples=job["horizon_samples"],
            online_current_control=job["task"] == "optimizer",
            adoption_handler=handler,
        ),
    ):
        measured = run(common)
    if handler is None:
        excluded = {"seconds", "model_computation_counts"}
        assert all(
            baseline.get(k) == measured.get(k)
            for k in set(baseline) | set(measured)
            if k not in excluded
        )
        extra = diagnostics["extra_model_computation_counts"]
        assert {
            k: v - extra.get(k, 0)
            for k, v in measured["model_computation_counts"].items()
        } == baseline["model_computation_counts"]
    if "output" in job:
        name = (
            f"{job['dataset']}_{job['stream']}_i{job['interval']}_s{job['seed']}.jsonl"
        )
        (Path(job["output"]) / name).write_text(
            "".join(json.dumps(t) + "\n" for t in trace), encoding="utf-8"
        )
        diagnostics["trace_file"] = name
    diagnostics["trace_row_count"] = len(trace)
    return dict(
        job=job, baseline=baseline, measured=measured, shadow_diagnostic=diagnostics
    )


def analyze(output):
    from tools.research_shadow_learner_diagnostic.summarize_standard_metrics import (
        total_mac,
    )

    metadata = json.loads((output / "research-source.json").read_text(encoding="utf-8"))
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    if len(rows) != len(metadata["jobs"]) or (output / "failures.json").exists():
        raise ValueError("未完了/失敗のある実験です")

    def keys(j):
        return j["dataset"], j["stream"], j["interval"], j["seed"]

    assert sorted(keys(r["job"]) for r in rows) == sorted(
        keys(j) for j in metadata["jobs"]
    )
    per_run = []
    for r in rows:
        j, b, m, d = r["job"], r["baseline"], r["measured"], r["shadow_diagnostic"]
        trace = [
            json.loads(line)
            for line in (output / d["trace_file"])
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(trace) == d["trace_row_count"]
        if j["task"] == "optimizer":
            by = {}
            for t in trace:
                by.setdefault(t["episode_id"], []).append(t)
            for e in d["episodes"]:
                score = e["initial_score"]
                first = None
                for t in by.get(e["episode_id"], []):
                    score += t["reset_current_log_loss"] - t["shadow_log_loss"]
                    assert abs(score - t["reset_control_score"]) < 1e-10
                    if first is None and score > 0:
                        first = (t["age"], t["same_alarm_concept_so_far"])
                assert e["reset_control_first_crossing_age"] == (
                    first[0] if first else None
                )
                assert e["reset_control_first_crossing_same_alarm_concept"] == (
                    first[1] if first else None
                )
        for e in d["episodes"]:
            if "adopted_model_id" in e:
                assert e["last_score"] > 0 and e["end_reason"] == "shadow_adopted"
                assert (
                    e["adoption_sample_index"]
                    == e["alarm_sample_index"] + e["observed_sample_count"]
                )
        per_run.append(
            dict(
                **{k: j[k] for k in ("dataset", "stream", "interval", "seed")},
                baseline_accuracy=b["accuracy"],
                measured_accuracy=m["accuracy"],
                accuracy_delta_pp=100 * (m["accuracy"] - b["accuracy"]),
                baseline_up=b["uploaded_bytes"],
                measured_up=m["uploaded_bytes"],
                baseline_down=b["downloaded_bytes"],
                measured_down=m["downloaded_bytes"],
                baseline_mac=total_mac(b["model_computation_counts"]),
                measured_mac=total_mac(m["model_computation_counts"]),
                baseline_seconds=b["seconds"],
                measured_seconds=m["seconds"],
                episodes=len(d["episodes"]),
                adoptions=sum("adopted_model_id" in e for e in d["episodes"]),
                keep_crossings=sum(
                    e.get("control_first_crossing_same_alarm_concept") is True
                    for e in d["episodes"]
                ),
                reset_crossings=sum(
                    e.get("reset_control_first_crossing_same_alarm_concept") is True
                    for e in d["episodes"]
                )
                if j["task"] == "optimizer"
                else None,
            )
        )
    lines = [
        "# S1の追加実験（標準指標と既定との差）",
        "",
        "各行seed平均。MB=10^6byte、G MAC=10^9。経過秒は並列負荷と実行順序の影響を含む。",
        "",
        "| データ | 入力 | 間隔 | 精度% 既定→変種 | 差pp | 上りMB 既定→変種 | 下りMB 既定→変種 | G MAC 既定→変種 | 秒 既定→変種 | 採用数 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted({(r["dataset"], r["stream"], r["interval"]) for r in per_run}):
        rs = [r for r in per_run if (r["dataset"], r["stream"], r["interval"]) == key]

        def avg(k):
            return mean(r[k] for r in rs)

        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {100 * avg('baseline_accuracy'):.3f}→{100 * avg('measured_accuracy'):.3f} | {avg('accuracy_delta_pp'):+.3f} | {avg('baseline_up') / 1e6:.3f}→{avg('measured_up') / 1e6:.3f} | {avg('baseline_down') / 1e6:.3f}→{avg('measured_down') / 1e6:.3f} | {avg('baseline_mac') / 1e9:.3f}→{avg('measured_mac') / 1e9:.3f} | {avg('baseline_seconds'):.2f}→{avg('measured_seconds'):.2f} | {sum(r['adoptions'] for r in rs)} |"
        )
    (output / "analysis_followup.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "summary_followup.json").write_text(
        json.dumps(dict(runs=per_run), indent=2) + "\n", encoding="utf-8"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", choices=["adoption", "optimizer"], required=True)
    p.add_argument(
        "--datasets", nargs="+", choices=["sea4", "sine2"], default=["sea4", "sine2"]
    )
    p.add_argument(
        "--streams", nargs="+", choices=["drift", "fixed"], default=["drift", "fixed"]
    )
    p.add_argument("--intervals", type=int, nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--sample-count", type=int, default=5000)
    p.add_argument("--horizon-samples", type=int, default=240)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--disabled", action="store_true")
    p.add_argument("--print-plan", action="store_true")
    a = p.parse_args()
    if (
        min(a.intervals + [a.sample_count, a.horizon_samples, a.workers]) < 1
        or min(a.seeds) < 0
        or any(a.sample_count % i for i in a.intervals)
    ):
        p.error("正の値・間隔の倍数の標本数・非負seedが必要です")
    jobs = [
        dict(
            dataset=d,
            stream="drift" if c is None else f"fixed{c}",
            fixed_concept=c,
            interval=i,
            seed=s,
            sample_count=a.sample_count,
            horizon_samples=a.horizon_samples,
            task=a.task,
            disabled=a.disabled,
            output=str(a.output),
        )
        for d in a.datasets
        for c in (
            ([None] if "drift" in a.streams else [])
            + (([0, 2] if d == "sea4" else [0, 1]) if "fixed" in a.streams else [])
        )
        for i in a.intervals
        for s in a.seeds
    ]
    if len(
        {(j["dataset"], j["stream"], j["interval"], j["seed"]) for j in jobs}
    ) != len(jobs):
        p.error("条件が重複しています")
    if a.print_plan:
        print(
            json.dumps(
                dict(
                    jobs=jobs,
                    job_count=len(jobs),
                    run_count=2 * len(jobs),
                    workers=a.workers,
                    exists=a.output.exists(),
                ),
                indent=2,
            )
        )
        return
    a.output.mkdir(parents=True, exist_ok=False)
    sources = (
        list((ROOT / "src").rglob("*.py"))
        + list((ROOT / "tools/research_shadow_learner_diagnostic").glob("*.py"))
        + list((ROOT / "tools/research_provisional_candidate_expert").glob("*.py"))
    )
    (a.output / "research-source.json").write_text(
        json.dumps(
            dict(
                jobs=jobs,
                workers=a.workers,
                sha256={
                    f.relative_to(ROOT).as_posix(): hashlib.sha256(
                        f.read_bytes()
                    ).hexdigest()
                    for f in sources
                },
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    rows, errors = [], []
    with ProcessPoolExecutor(
        max_workers=a.workers, mp_context=get_context("spawn")
    ) as pool:
        pending = {pool.submit(execute_job, j): j for j in jobs}
        for f in as_completed(pending):
            try:
                r = f.result()
            except Exception:
                errors.append(dict(job=pending[f], traceback=traceback.format_exc()))
                (a.output / "failures.json").write_text(
                    json.dumps(errors, indent=2), encoding="utf-8"
                )
                continue
            rows.append(r)
            temp = a.output / "results.json.tmp"
            temp.write_text(json.dumps(rows, indent=1), encoding="utf-8")
            temp.replace(a.output / "results.json")
            print(
                r["job"]["dataset"],
                r["job"]["stream"],
                r["job"]["seed"],
                "adoptions",
                sum(
                    "adopted_model_id" in e for e in r["shadow_diagnostic"]["episodes"]
                ),
                flush=True,
            )
    if errors:
        raise SystemExit(1)
    analyze(a.output)


if __name__ == "__main__":
    main()
