"""S1の診断用CLI。製品の既定runを包み、影の軌跡を別ファイルへ保存する。"""

import argparse
import hashlib
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def execute_diagnostic_job(job):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    from tools.research_provisional_candidate_expert.run import _execute_job
    from tools.research_shadow_learner_diagnostic.diagnostic import (
        shadow_learner_diagnostic_policy,
    )

    diagnostics, trace = {}, []
    baseline_job = {k: job[k] for k in ("dataset", "interval", "seed", "sample_count")}
    baseline_job.update(
        variant="product-default", check=False, record_prediction_period_metrics=True
    )
    try:
        with shadow_learner_diagnostic_policy(
            diagnostics=diagnostics,
            trace=trace,
            horizon_samples=job["horizon_samples"],
        ):
            result = _execute_job(baseline_job)
    except Exception:
        # 製品の例外のpickle復元に依存せず、元のtracebackを残して親へ通知する。
        error = traceback.format_exc()
        if "output" in job:
            path = (
                Path(job["output"])
                / f"{job['dataset']}_i{job['interval']}_s{job['seed']}.error.txt"
            )
            path.write_text(error, encoding="utf-8")
        raise RuntimeError(error) from None
    if "output" in job:
        path = (
            Path(job["output"])
            / f"{job['dataset']}_i{job['interval']}_s{job['seed']}.jsonl"
        )
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trace),
            encoding="utf-8",
        )
        diagnostics["trace_file"] = path.name
    diagnostics["trace_row_count"] = len(trace)
    result["shadow_diagnostic"] = diagnostics
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=["sea4", "sine2"], required=True
    )
    parser.add_argument("--intervals", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=1500)
    parser.add_argument("--horizon-samples", type=int, default=240)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()
    if (
        min(args.intervals + [args.sample_count, args.horizon_samples, args.workers])
        < 1
        or min(args.seeds) < 0
    ):
        parser.error("間隔・標本数・上限・workersは正、seedは非負にしてください")
    jobs = [
        dict(
            dataset=d,
            interval=i,
            seed=s,
            sample_count=args.sample_count,
            horizon_samples=args.horizon_samples,
            output=str(args.output),
        )
        for d in args.datasets
        for i in args.intervals
        for s in args.seeds
    ]
    if len({(j["dataset"], j["interval"], j["seed"]) for j in jobs}) != len(jobs):
        parser.error("条件の重複は受理しません")
    if args.print_plan:
        print(
            json.dumps(
                dict(
                    output=str(args.output),
                    exists=args.output.exists(),
                    workers=args.workers,
                    jobs=jobs,
                ),
                indent=1,
            )
        )
        return
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        Path(__file__).with_name("diagnostic.py"),
        ROOT / "tools/research_provisional_candidate_expert/run.py",
        ROOT
        / "src/federated_learning_experiments/learning/training/prequential_code_length.py",
    ]
    (args.output / "research-source.json").write_text(
        json.dumps(
            dict(
                base_commit="186e531b",
                behavior_baseline_version=12,
                workers=args.workers,
                purpose="影の符号長差が初期の不利を取り返せるか。採用・帰属は変更しない。",
                sha256={
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sources
                },
                jobs=jobs,
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
        pending = {pool.submit(execute_diagnostic_job, job): job for job in jobs}
        for future in as_completed(pending):
            try:
                result = future.result()
            except Exception as error:
                failures.append(dict(job=pending[future], error=str(error)))
                print("FAILED", pending[future], str(error), flush=True)
                (args.output / "failures.json").write_text(
                    json.dumps(failures, indent=1), encoding="utf-8"
                )
                continue
            results.append(result)
            temporary = args.output / "results.json.tmp"
            temporary.write_text(
                json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            temporary.replace(args.output / "results.json")
            print(
                result["dataset"],
                result["interval"],
                result["seed"],
                len(result["shadow_diagnostic"]["episodes"]),
                result["shadow_diagnostic"]["trace_row_count"],
                flush=True,
            )
    print(f"completed {len(results)}, failed {len(failures)}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
