"""既定とBだけを比べ、検証できた既定結果を再利用する本規模研究CLI。"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from multiprocessing import get_context
from pathlib import Path

from tools.research_s3_validation.run import (
    ROOT,
    execute,
    sha,
    source_hashes as validation_hashes,
)
from tools.research_s3_fullscale.baseline import (
    convert,
    environment,
    load_records,
    validate_record,
)

DATASETS = ("sea2", "sea4", "sine2", "circle2", "mnist2", "mnist4")
ARMS = ("product-default", "full-score-head-train")


def source_hashes():
    hashes = validation_hashes()
    for name in ("__init__.py", "run.py", "baseline.py"):
        path = "tools/research_s3_fullscale/" + name
        hashes[path] = sha(ROOT / path)
    return dict(sorted(hashes.items()))


def make_jobs(datasets, intervals, seeds, sample_count):
    if (
        not datasets
        or not set(datasets) <= set(DATASETS)
        or not intervals
        or min(intervals) < 1
        or len(seeds) < 4
        or min(seeds) < 0
        or sample_count < 5000
        or any(len(set(xs)) != len(xs) for xs in (datasets, intervals, seeds))
    ):
        raise ValueError(
            "対応データ・正の間隔・異なる4seed以上・5000標本以上が必要です"
        )
    return [
        dict(
            dataset=d,
            fixed_concept=None,
            interval=i,
            sample_count=sample_count,
            seed=s,
            arm=a,
        )
        for d in datasets
        for i in intervals
        for s in seeds
        for a in ARMS
    ]


def job_key(job):
    return tuple(
        job[k]
        for k in ("dataset", "fixed_concept", "interval", "sample_count", "seed", "arm")
    )


def job_id(job):
    import hashlib

    return hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:20]


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def validate_one(row, job):
    m = row["metrics"]
    signature = row["input_signature"]
    if (
        row["job"] != job
        or any(m[k] != job[k] for k in ("dataset", "interval", "seed", "sample_count"))
        or m["research_variant"] != job["arm"]
        or m["fixed_concept"] is not None
        or m["variant"] != "product-default"
        or m["check"] is not False
        or m["accuracy"] != m["record_accuracy"]
    ):
        raise ValueError("checkpointの条件・指標・入力署名が不正です")
    if row.get("reused_default"):
        if (
            job["arm"] != "product-default"
            or signature is not None
            or m["seconds"] is not None
        ):
            raise ValueError("再利用した既定の来歴が不正です")
    elif (
        not isinstance(signature, str)
        or len(signature) != 64
        or any(c not in "0123456789abcdef" for c in signature)
    ):
        raise ValueError("新規runの入力署名が不正です")


def validate_rows(rows, jobs):
    expected = {job_key(j): j for j in jobs}
    actual = [job_key(r["job"]) for r in rows]
    if (
        len(expected) != len(jobs)
        or len(set(actual)) != len(actual)
        or set(actual) != set(expected)
    ):
        raise ValueError("条件の欠落・重複があります")
    signatures = {}
    for row in rows:
        j = row["job"]
        validate_one(row, expected[job_key(j)])
        group = j["dataset"], j["interval"], j["sample_count"], j["seed"]
        if row["input_signature"] is None:
            continue
        if group in signatures and signatures[group] != row["input_signature"]:
            raise ValueError("同じ条件の構成間で入力列が異なります")
        signatures[group] = row["input_signature"]
    return signatures


def load_checkpoints(output, jobs):
    planned = {job_id(j): j for j in jobs}
    if len(planned) != len(jobs):
        raise ValueError("job IDが衝突しています")
    rows = []
    for path in sorted((output / "runs").glob("*.json")):
        if path.stem not in planned:
            raise ValueError("計画外のcheckpointがあります")
        row = json.loads(path.read_text(encoding="utf-8"))
        validate_one(row, planned[path.stem])
        rows.append(row)
    return rows


def validate_saved_baselines(output, jobs, meta):
    proofs = {}
    indexed = {job_id(j): j for j in jobs if j["arm"] == "product-default"}
    reused_ids = {
        job_id(r["job"])
        for r in load_checkpoints(output, jobs)
        if r.get("reused_default")
    }
    if reused_ids != set(meta["baseline_record_hashes"]):
        raise ValueError("再利用checkpointと来歴の集合が一致しません")
    for identifier, digest in meta["baseline_record_hashes"].items():
        path = output / "baseline" / (identifier + ".json")
        if identifier not in indexed or sha(path) != digest:
            raise ValueError("再利用した既定の原記録が変化しています")
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_record(
            record, indexed[identifier], proofs, meta["execution_environment"]
        )
        checkpoint = json.loads(
            (output / "runs" / (identifier + ".json")).read_text(encoding="utf-8")
        )
        if checkpoint != convert(record, indexed[identifier], digest):
            raise ValueError("既定のcheckpointと原記録が一致しません")
    if meta["baseline_source"] is not None and set(
        meta["baseline_record_hashes"]
    ) != set(indexed):
        raise ValueError("既定の再利用記録に欠落があります")


def prepare(output, jobs, plan, resume, baseline=None):
    hashes = source_hashes()
    if output.exists():
        if not resume:
            raise ValueError("既存出力です。再開する場合だけ--resumeを指定してください")
        meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        if (
            meta["jobs"] != jobs
            or meta["plan"] != plan
            or meta["source_hashes"] != hashes
            or meta["baseline_source"]
            != (str(baseline) if baseline is not None else None)
            or meta["execution_environment"] != environment()
        ):
            raise ValueError("再開時の条件または数値ソースが元と異なります")
        validate_saved_baselines(output, jobs, meta)
    else:
        if resume:
            raise ValueError("再開先が存在しません")
        reused, proof = (
            load_records(baseline, jobs, require_environment=True)
            if baseline is not None
            else ([], {})
        )
        output.mkdir(parents=True)
        (output / "runs").mkdir()
        (output / "baseline").mkdir()
        baseline_hashes = {}
        for job, record, _ in reused:
            identifier = job_id(job)
            path = output / "baseline" / (identifier + ".json")
            atomic_json(path, record)
            digest = sha(path)
            baseline_hashes[identifier] = digest
            atomic_json(
                output / "runs" / (identifier + ".json"), convert(record, job, digest)
            )
        meta = dict(
            jobs=jobs,
            plan=plan,
            source_hashes=hashes,
            protocol="fixed-B-fullscale-v1",
            execution_environment=environment(),
            baseline_source=str(baseline) if baseline is not None else None,
            baseline_record_hashes=baseline_hashes,
            baseline_proof=proof,
        )
        atomic_json(output / "metadata.json", meta)
    return meta, load_checkpoints(output, jobs)


def execute_plan(output, jobs, plan, workers, resume, baseline=None):
    meta, rows = prepare(output, jobs, plan, resume, baseline)
    completed = {job_key(r["job"]) for r in rows}
    pending = [j for j in jobs if job_key(j) not in completed]
    reused_defaults = sum(bool(r.get("reused_default")) for r in rows)
    new_total = len(jobs) - reused_defaults
    print(
        f"START new_runs={len(rows) - reused_defaults}/{new_total} reused_default={reused_defaults} remaining_new={len(pending)}",
        flush=True,
    )
    # 過去の失敗は履歴へ残し、今回の試行とは分ける。完了済みrunは再実行しない。
    with (output / "attempts.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                dict(
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    workers=workers,
                    reused=len(rows),
                    pending=len(pending),
                )
            )
            + "\n"
        )
    failures = []
    if pending:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(pending)), mp_context=get_context("spawn")
        ) as pool:
            futures = {pool.submit(execute, j): j for j in pending}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    row = future.result()
                    validate_one(row, job)
                except Exception as error:
                    failures.append(dict(job=job, error=repr(error)))
                    with (output / "attempts.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(failures[-1]) + "\n")
                    atomic_json(output / "failures.json", failures)
                    print(f"FAILED {job_id(job)} {error!r}", flush=True)
                    continue
                atomic_json(output / "runs" / (job_id(job) + ".json"), row)
                rows.append(row)
                print(
                    f"NEW {len(rows) - reused_defaults}/{new_total} (compared={len(rows)}/{len(jobs)} reused_default={reused_defaults}) {job['dataset']} i{job['interval']} s{job['seed']} {job['arm']} acc={row['metrics']['accuracy']:.6f}",
                    flush=True,
                )
    if meta["source_hashes"] != source_hashes():
        raise RuntimeError("実行中に数値ソースが変化しました。再開・解析を拒否します")
    atomic_json(output / "failures.json", failures)
    if failures:
        raise RuntimeError("失敗runがあります。同じ条件・ソースで--resumeしてください")
    validate_rows(rows, jobs)
    validate_saved_baselines(output, jobs, meta)
    ordered = sorted(rows, key=lambda r: job_key(r["job"]))
    atomic_json(output / "results.json", ordered)
    atomic_json(
        output / "complete.json",
        dict(
            run_count=len(rows),
            source_hashes_verified=True,
            results_sha256=sha(output / "results.json"),
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--intervals", type=int, nargs="+", default=[50, 100, 200, 500])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--sample-count", type=int, default=5000)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--baseline",
        type=Path,
        help="検証して再利用するネイティブ既定run JSONのディレクトリ",
    )
    p.add_argument("--print-plan", action="store_true")
    a = p.parse_args()
    try:
        jobs = make_jobs(a.datasets, a.intervals, a.seeds, a.sample_count)
        if a.workers < 1:
            raise ValueError("workersは正数が必要です")
        if a.output.exists() and not a.resume:
            raise ValueError("既存出力です。--resumeで明示してください")
        plan = dict(
            datasets=a.datasets,
            intervals=a.intervals,
            seeds=a.seeds,
            sample_count=a.sample_count,
        )
        if a.print_plan:
            reused = 0
            if a.resume:
                meta = json.loads(
                    (a.output / "metadata.json").read_text(encoding="utf-8")
                )
                if (
                    meta["jobs"] != jobs
                    or meta["plan"] != plan
                    or meta["source_hashes"] != source_hashes()
                ):
                    raise ValueError("再開条件・ソースが異なります")
                reused = len(load_checkpoints(a.output, jobs))
            elif a.baseline is not None:
                reused = len(
                    load_records(a.baseline, jobs, require_environment=False)[0]
                )
            print(
                json.dumps(
                    dict(
                        run_count=len(jobs),
                        reused=reused,
                        pending=len(jobs) - reused,
                        workers=a.workers,
                        baseline_environment_check="required_at_execution"
                        if a.baseline is not None
                        else None,
                        plan=plan,
                        arms=ARMS,
                    ),
                    indent=2,
                )
            )
            return
        execute_plan(a.output, jobs, plan, a.workers, a.resume, a.baseline)
    except ValueError as error:
        p.error(str(error))


if __name__ == "__main__":
    main()
