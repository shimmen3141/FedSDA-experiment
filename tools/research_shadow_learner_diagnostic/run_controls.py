"""S1の無変化・オンライン継続対照。各条件で同環境の既定も実行して照合する。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def execute_control_job(job):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    from tools.research_provisional_candidate_expert.run import _execute_job
    from tools.research_shadow_learner_diagnostic.controls import (
        stationary_concept_policy,
    )
    from tools.research_shadow_learner_diagnostic.diagnostic import (
        shadow_learner_diagnostic_policy,
    )

    baseline_job = {k: job[k] for k in ("dataset", "interval", "seed", "sample_count")}
    baseline_job.update(
        variant="product-default", check=False, record_prediction_period_metrics=True
    )
    diagnostics, trace = {}, []
    policy = dict(dataset=job["dataset"], concept_id=job["fixed_concept"])
    with stationary_concept_policy(**policy):
        baseline = _execute_job(baseline_job)
    with (
        stationary_concept_policy(**policy),
        shadow_learner_diagnostic_policy(
            diagnostics=diagnostics,
            trace=trace,
            horizon_samples=job["horizon_samples"],
            online_current_control=True,
        ),
    ):
        measured = _execute_job(baseline_job)
    excluded = {"seconds", "model_computation_counts"}
    mismatches = [
        k
        for k in set(baseline) | set(measured)
        if k not in excluded and baseline.get(k) != measured.get(k)
    ]
    if mismatches:
        raise AssertionError(f"既定の保存値が変化: {mismatches}")
    if {
        k: v - diagnostics["extra_model_computation_counts"].get(k, 0)
        for k, v in measured["model_computation_counts"].items()
    } != baseline["model_computation_counts"]:
        raise AssertionError("追加計算を差し引いたモデル計算が既定と不一致")
    if job["fixed_concept"] is not None:
        if set(measured["concept_accuracies"]) != {str(job["fixed_concept"])}:
            raise AssertionError("無変化対照の評価データに別概念が混在")
    diagnostics["trace_row_count"] = len(trace)
    if "output" in job:
        name = (
            f"{job['dataset']}_{job['stream']}_i{job['interval']}_s{job['seed']}.jsonl"
        )
        (Path(job["output"]) / name).write_text(
            "".join(json.dumps(t) + "\n" for t in trace), encoding="utf-8"
        )
        diagnostics["trace_file"] = name
    return dict(
        job=job,
        baseline=baseline,
        measured=measured,
        shadow_diagnostic=diagnostics,
        baseline_parity=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intervals", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--horizon-samples", type=int, default=240)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()
    if (
        min(args.intervals + [args.sample_count, args.horizon_samples, args.workers])
        < 1
        or min(args.seeds) < 0
    ):
        parser.error("間隔・標本数・上限・workersは正、seedは非負が必要です")
    if args.sample_count % max(args.intervals) or any(
        args.sample_count % i for i in args.intervals
    ):
        parser.error(
            "未処理の末尾を避けるため、標本数は全ての集約間隔の倍数にしてください"
        )
    streams = [
        ("sea4", None),
        ("sea4", 0),
        ("sea4", 2),
        ("sine2", None),
        ("sine2", 0),
        ("sine2", 1),
    ]
    jobs = [
        dict(
            dataset=d,
            fixed_concept=c,
            stream="drift" if c is None else f"fixed{c}",
            interval=i,
            seed=s,
            sample_count=args.sample_count,
            horizon_samples=args.horizon_samples,
            output=str(args.output),
        )
        for d, c in streams
        for i in args.intervals
        for s in args.seeds
    ]
    keys = [(j["dataset"], j["stream"], j["interval"], j["seed"]) for j in jobs]
    if len(keys) != len(set(keys)):
        parser.error("条件が重複しています")
    plan = dict(
        output=str(args.output),
        exists=args.output.exists(),
        jobs=jobs,
        job_count=len(jobs),
        run_count=2 * len(jobs),
        workers=args.workers,
    )
    if args.print_plan:
        print(json.dumps(plan, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    sources = (
        sorted((ROOT / "src").rglob("*.py"))
        + sorted((ROOT / "tools/research_shadow_learner_diagnostic").glob("*.py"))
        + sorted((ROOT / "tools/research_provisional_candidate_expert").glob("*.py"))
    )
    hashes = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
    }
    (args.output / "research-source.json").write_text(
        json.dumps(
            dict(
                **plan,
                sha256=hashes,
                purpose="S1の無変化・オンライン継続対照。採用はしない。",
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    results, failures = [], []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn")
    ) as pool:
        pending = {pool.submit(execute_control_job, j): j for j in jobs}
        for future in as_completed(pending):
            job = pending[future]
            try:
                result = future.result()
            except Exception:
                failures.append(dict(job=job, traceback=traceback.format_exc()))
                (args.output / "failures.json").write_text(
                    json.dumps(failures, indent=2), encoding="utf-8"
                )
                print("FAILED", job, flush=True)
                continue
            results.append(result)
            temporary = args.output / "results.json.tmp"
            temporary.write_text(json.dumps(results, indent=1), encoding="utf-8")
            temporary.replace(args.output / "results.json")
            print(
                job["dataset"],
                job["stream"],
                job["interval"],
                job["seed"],
                len(result["shadow_diagnostic"]["episodes"]),
                flush=True,
            )
    print(f"completed {len(results)}, failed {len(failures)}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
