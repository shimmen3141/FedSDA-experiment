"""参照の転送を要しない、固定4構成の未使用seed対照。"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import json
from multiprocessing import get_context
import os
from pathlib import Path

from tools.research_s3_factorial.run import ROOT, sha, source_hashes as factorial_hashes

ARMS = (
    "product-default",
    "warm-matched",
    "full-score-head-train",
    "output-layer-matched",
)


def source_hashes():
    hashes = factorial_hashes()
    names = ["tools/research_s3_contribution/diagnostic.py"]
    for package in (
        "research_adopted_candidate_removal",
        "research_current_mixture_benefit",
    ):
        names += [
            p.relative_to(ROOT).as_posix()
            for p in (ROOT / "tools" / package).glob("*.py")
        ]
    names += [
        "tools/research_s3_validation/" + p
        for p in ("__init__.py", "run.py", "trace.py")
    ]
    hashes.update({name: sha(ROOT / name) for name in names})
    return dict(sorted(hashes.items()))


def make_jobs(seeds, sample_count):
    if (
        len(seeds) < 4
        or len(set(seeds)) != len(seeds)
        or min(seeds) < 4
        or sample_count < 2500
    ):
        raise ValueError("探索済み0〜3を除く異なる4seed以上・2500標本以上が必要です")
    return [
        dict(
            dataset="mnist4",
            fixed_concept=None,
            interval=500,
            sample_count=sample_count,
            seed=s,
            arm=a,
        )
        for s in seeds
        for a in ARMS
    ]


def execute(job, trace_output=None):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    from tools.research_provisional_candidate_expert.run import _execute_job
    from tools.research_s3_factorial.policy import (
        capture_input_signature,
        factorial_policy,
        input_signature,
    )

    events, inputs = [], {}
    with ExitStack() as stack:
        tracker = None
        if trace_output is not None:
            from tools.research_s3_contribution.diagnostic import ContributionTracker

            # この名前は既存trackerのhead候補の表示だけに使われ、学習を変更しない。
            tracker = ContributionTracker("output-layer-matched")
            stack.enter_context(tracker.activated())
        stack.enter_context(capture_input_signature(inputs))
        if job["arm"] != "product-default":
            stack.enter_context(factorial_policy(arm=job["arm"], events=events))
        metrics = _execute_job(
            dict(
                job,
                variant="product-default",
                check=False,
                record_prediction_period_metrics=True,
            )
        )
    metrics.update(
        research_variant=job["arm"], fixed_concept=None, research_events=events
    )
    row = dict(
        job=job,
        metrics=metrics,
        input_signature=input_signature(inputs, job["sample_count"]),
    )
    if tracker is not None:
        data = dict(
            predictions=tracker.predictions,
            candidates=tracker.records,
            decisions=tracker.decisions,
            joint_removals=tracker.joint_removals,
        )
        trace_output.write_text(
            json.dumps(data, separators=(",", ":")), encoding="utf-8"
        )
        row.update(
            diagnostic_file=trace_output.name,
            diagnostic_sha256=sha(trace_output),
            extra_model_forward_count=0,
        )
    return row


def validate_rows(rows, jobs):
    def key(j):
        return tuple(
            j[k]
            for k in (
                "dataset",
                "fixed_concept",
                "interval",
                "sample_count",
                "seed",
                "arm",
            )
        )

    expected = [key(j) for j in jobs]
    actual = [key(r["job"]) for r in rows]
    if (
        len(set(actual)) != len(actual)
        or set(actual) != set(expected)
        or len(actual) != len(expected)
    ):
        raise ValueError("条件の欠落・重複があります")
    signatures = {}
    for r in rows:
        j, m = r["job"], r["metrics"]
        if (
            any(m[k] != j[k] for k in ("dataset", "interval", "sample_count", "seed"))
            or m["research_variant"] != j["arm"]
            or m["fixed_concept"] is not None
            or m["check"] is not False
            or m["variant"] != "product-default"
        ):
            raise ValueError("条件と指標が一致しません")
        signature = r["input_signature"]
        if j["seed"] in signatures and signatures[j["seed"]] != signature:
            raise ValueError("同じseedの構成間で入力列が異なります")
        signatures[j["seed"]] = signature
    return signatures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[4, 5, 6, 7])
    p.add_argument("--sample-count", type=int, default=2500)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--print-plan", action="store_true")
    a = p.parse_args()
    try:
        jobs = make_jobs(a.seeds, a.sample_count)
    except ValueError as error:
        p.error(str(error))
    if a.workers < 1 or a.output.exists():
        p.error("正のworkersと未使用の出力先が必要です")
    if a.print_plan:
        print(
            json.dumps(
                dict(run_count=len(jobs), workers=a.workers, jobs=jobs), indent=2
            )
        )
        return
    a.output.mkdir(parents=True)
    meta = dict(
        jobs=jobs,
        seeds=a.seeds,
        sample_count=a.sample_count,
        workers=a.workers,
        source_hashes=source_hashes(),
        protocol="fixed-B-v1",
        exploratory_seeds=[0, 1, 2, 3],
    )
    (a.output / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    rows, failures = [], []
    with ProcessPoolExecutor(
        max_workers=min(a.workers, len(jobs)), mp_context=get_context("spawn")
    ) as pool:
        futures = {pool.submit(execute, j): j for j in jobs}
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as error:
                failures.append(dict(job=futures[future], error=repr(error)))
                (a.output / "failures.json").write_text(
                    json.dumps(failures, indent=2), encoding="utf-8"
                )
                continue
            rows.append(row)
            tmp = a.output / "results.tmp"
            tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            tmp.replace(a.output / "results.json")
            print(
                f"{len(rows)}/{len(jobs)} seed{row['job']['seed']} {row['job']['arm']} acc={row['metrics']['accuracy']:.6f}",
                flush=True,
            )
    if failures:
        raise RuntimeError("失敗runがあります。failures.jsonを確認してください")
    validate_rows(rows, jobs)
    if meta["source_hashes"] != source_hashes():
        raise RuntimeError("実行中に数値ソースが変化しました")
    (a.output / "complete.json").write_text(
        json.dumps(dict(run_count=len(rows), source_hashes_verified=True)),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
