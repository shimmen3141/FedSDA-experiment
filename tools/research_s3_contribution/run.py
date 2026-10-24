"""S3寄与診断を、保存済みの同条件の実行と照合して行う。"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("product-default", "warm-matched", "output-layer-matched")
KEYS = (
    "dataset",
    "fixed_concept",
    "interval",
    "sample_count",
    "seed",
    "research_variant",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execution_sources():
    paths = list((ROOT / "src/federated_learning_experiments").rglob("*.py"))
    paths += list((ROOT / "tools").rglob("*.py"))
    # 解析は実験終了後の別処理。数値実行のfingerprintとは分ける。
    paths = [
        p for p in paths if p != ROOT / "tools/research_s3_contribution/analyze.py"
    ]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def reference_rows(directory):
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    hashes = {k.replace("\\", "/"): v for k, v in metadata["source_hashes"].items()}
    required = {
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "src/federated_learning_experiments").rglob("*.py")
    }
    required.update(
        "tools/research_output_layer_learner/" + p
        for p in ("__init__.py", "policy.py", "run.py")
    )
    required.update(
        (
            "tools/research_provisional_candidate_expert/run.py",
            "tools/research_shadow_learner_diagnostic/controls.py",
        )
    )
    if not required <= set(hashes) or any(
        digest(ROOT / p) != hashes[p] for p in required
    ):
        raise ValueError("参照実験と数値実行ソースが一致しません")
    complete = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    rows = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    if (
        complete.get("source_hashes_verified") is not True
        or complete["run_count"] != len(rows)
        or len(rows) != len(metadata["jobs"])
    ):
        raise ValueError("参照実験が未完了です")
    indexed = {tuple(r[k] for k in KEYS): r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != {
        tuple(j[k] for k in KEYS) for j in metadata["jobs"]
    }:
        raise ValueError("参照条件の欠落・重複があります")
    return indexed


def execute(job, reference, output):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    from tools.research_output_layer_learner.run import execute as run_s3
    from tools.research_s3_contribution.diagnostic import ContributionTracker

    started = time.perf_counter()
    tracker = ContributionTracker(job["research_variant"])
    with tracker.activated():
        metrics = run_s3(job)
    differences = [
        k for k, v in reference.items() if k != "seconds" and metrics[k] != v
    ]
    if differences:
        raise ValueError(f"計測なし参照と相違: {differences}")
    predictions = tracker.predictions
    ordered = sorted(predictions, key=lambda p: (p["client_id"], p["sample_index"]))
    if len(ordered) != 10 * job["sample_count"] or len(
        {(p["client_id"], p["sample_index"]) for p in ordered}
    ) != len(ordered):
        raise ValueError("予測記録に欠落・重複があります")
    trace_digest = hashlib.sha256(
        json.dumps([p["correct"] for p in ordered]).encode()
    ).hexdigest()
    if trace_digest != metrics["prediction_record_digest"]:
        raise ValueError("全標本traceと製品の正誤digestが一致しません")
    filename = f"s{job['seed']}_{job['research_variant']}.json"
    target = output / filename
    target.write_text(
        json.dumps(
            dict(
                predictions=predictions,
                candidates=tracker.records,
                decisions=tracker.decisions,
                joint_removals=tracker.joint_removals,
            ),
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return dict(
        job=job,
        metrics=metrics,
        reference_non_time_fields_verified=True,
        diagnostic_file=filename,
        diagnostic_sha256=digest(target),
        diagnostic_wall_seconds=time.perf_counter() - started,
        extra_model_forward_count=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--sample-count", type=int, default=2500)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()
    if (
        len(set(args.seeds)) != len(args.seeds)
        or len(args.seeds) < 4
        or min(args.seeds) < 0
        or args.sample_count < 2500
        or args.workers < 1
    ):
        parser.error("異なる4seed以上・2500標本以上・正のworkersが必要です")
    if args.output.exists():
        parser.error("保存先が存在します。保存済みの条件を確認してください")
    refs = reference_rows(args.reference)
    jobs = [
        dict(
            dataset="mnist4",
            fixed_concept=None,
            interval=500,
            sample_count=args.sample_count,
            seed=s,
            research_variant=v,
        )
        for s in args.seeds
        for v in VARIANTS
    ]
    if any(tuple(j[k] for k in KEYS) not in refs for j in jobs):
        parser.error("対応する計測なし参照がありません")
    if args.print_plan:
        print(
            json.dumps(
                dict(run_count=len(jobs), jobs=jobs, reference=str(args.reference)),
                indent=2,
            )
        )
        return
    args.output.mkdir(parents=True)
    hashes = execution_sources()
    metadata = dict(
        jobs=jobs,
        workers=args.workers,
        source_hashes=hashes,
        reference=str(args.reference),
        reference_results_sha256=digest(args.reference / "results.json"),
    )
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(execute, j, refs[tuple(j[k] for k in KEYS)], args.output)
            for j in jobs
        ]
        for future in as_completed(futures):
            r = future.result()
            rows.append(r)
            temporary = args.output / "results.tmp"
            temporary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            temporary.replace(args.output / "results.json")
            print(
                f"{len(rows)}/{len(jobs)} seed{r['job']['seed']} {r['job']['research_variant']} reference=exact",
                flush=True,
            )
    if (
        hashes != execution_sources()
        or digest(args.reference / "results.json")
        != metadata["reference_results_sha256"]
    ):
        raise RuntimeError("実行中にソースまたは参照が変化しました")
    (args.output / "complete.json").write_text(
        json.dumps(dict(run_count=len(rows), source_hashes_verified=True)),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
