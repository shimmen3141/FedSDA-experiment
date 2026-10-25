"""MNIST4 i100/500のWだけを追加し、保存済み本規模D/Bに対応させる。"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys

from tools.research_s3_fullscale import run as full
from tools.research_s3_fullscale.baseline import environment
from tools.research_s3_validation.run import execute as execute_existing
from tools.research_s3_warm_control.trace import PredictionTrace, signature


def resolve_reference(path):
    if (path / "complete.json").exists():
        return path
    if path.is_dir():
        children = [
            p for p in path.iterdir() if p.is_dir() and (p / "complete.json").exists()
        ]
        if len(children) == 1:
            return children[0]
    raise ValueError(
        "完了した本規模結果を一意に選べません。--referenceを具体的な保存先へ指定してください"
    )


def jobs_for(seeds, intervals):
    if (
        len(seeds) < 4
        or len(set(seeds)) != len(seeds)
        or min(seeds) < 0
        or not intervals
        or not set(intervals) <= {100, 500}
        or len(set(intervals)) != len(intervals)
    ):
        raise ValueError("異なる4seed以上・間隔100/500が必要です")
    return [
        dict(
            dataset="mnist4",
            fixed_concept=None,
            interval=i,
            sample_count=5000,
            seed=s,
            arm="warm-matched",
        )
        for i in intervals
        for s in seeds
    ]


def source_hashes():
    hashes = full.source_hashes()
    for name in ("__init__.py", "run.py", "trace.py"):
        p = "tools/research_s3_warm_control/" + name
        hashes[p] = full.sha(full.ROOT / p)
    return hashes


def load_reference(path, jobs, check_environment=False):
    path = resolve_reference(path)
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    rows = json.loads((path / "results.json").read_text(encoding="utf-8"))
    complete = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    canonical = full.make_jobs(**meta["plan"])
    if (
        Counter(map(full.job_key, canonical))
        != Counter(map(full.job_key, meta["jobs"]))
        or meta["source_hashes"] != full.source_hashes()
        or json.loads((path / "failures.json").read_text(encoding="utf-8"))
        or complete
        != dict(
            run_count=len(rows),
            source_hashes_verified=True,
            results_sha256=full.sha(path / "results.json"),
        )
    ):
        raise ValueError("本規模参照が未完了・不整合、または数値ソースが異なります")
    full.validate_rows(rows, canonical)
    full.validate_saved_baselines(path, canonical, meta)
    checkpoints = full.load_checkpoints(path, canonical)
    indexed = {full.job_key(r["job"]): r for r in rows}
    if indexed != {full.job_key(r["job"]): r for r in checkpoints}:
        raise ValueError("参照のcheckpointと原結果が一致しません")
    for j in jobs:
        for arm in full.ARMS:
            if full.job_key(j)[:-1] + (arm,) not in indexed:
                raise ValueError("対応するD/B参照がありません")
    if check_environment and meta["execution_environment"] != environment():
        raise ValueError(
            "Wと本規模D/Bの実行環境が異なります。同じuranium環境で実行してください"
        )
    return path, meta, indexed


def execute(job, expected_signature, output):
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    sys.path.insert(0, str(full.ROOT / "src"))
    tracker = PredictionTrace()
    with tracker.activated():
        row = execute_existing(job)
    inputs, correct = signature(tracker.predictions, job["sample_count"])
    if (
        inputs != expected_signature
        or inputs != row["input_signature"]
        or correct != row["metrics"]["prediction_record_digest"]
    ):
        raise ValueError("Wの入力列または正誤digestが参照/実行指標と一致しません")
    filename = "predictions/" + full.job_id(job) + ".json"
    target = output / filename
    temp = target.with_suffix(".tmp")
    temp.write_text(
        json.dumps(dict(predictions=tracker.predictions), separators=(",", ":")),
        encoding="utf-8",
    )
    temp.replace(target)
    row.update(
        trace_file=filename, trace_sha256=full.sha(target), extra_model_forward_count=0
    )
    return row


def validate_trace(row, output, refs):
    j = row["job"]
    full.validate_one(row, j)
    expected = refs[full.job_key(j)[:-1] + ("full-score-head-train",)][
        "input_signature"
    ]
    if (
        row["input_signature"] != expected
        or row["trace_file"] != "predictions/" + full.job_id(j) + ".json"
        or row["extra_model_forward_count"] != 0
    ):
        raise ValueError("Wの参照対応・系列ファイル・計算数が不正です")
    target = output / row["trace_file"]
    if full.sha(target) != row["trace_sha256"]:
        raise ValueError("予測系列が変化しています")
    data = json.loads(target.read_text(encoding="utf-8"))["predictions"]
    inputs, correct = signature(data, j["sample_count"])
    if (
        inputs != expected
        or correct != row["metrics"]["prediction_record_digest"]
        or abs(sum(p["correct"] for p in data) / len(data) - row["metrics"]["accuracy"])
        > 1e-12
    ):
        raise ValueError("系列・入力・精度が一致しません")


def execute_plan(reference, output, jobs, workers, resume):
    reference, refmeta, refs = load_reference(reference, jobs, True)
    binding = dict(
        reference_results_sha256=full.sha(reference / "results.json"),
        reference_metadata_sha256=full.sha(reference / "metadata.json"),
    )
    hashes = source_hashes()
    if output.exists():
        if not resume:
            raise ValueError("既存出力です。再開時だけ--resumeを指定してください")
        meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        if (
            meta["jobs"] != jobs
            or meta["source_hashes"] != hashes
            or meta["execution_environment"] != environment()
            or any(meta[k] != v for k, v in binding.items())
        ):
            raise ValueError("再開時の設定・参照・環境・ソースが異なります")
    else:
        if resume:
            raise ValueError("再開先がありません")
        output.mkdir(parents=True)
        (output / "runs").mkdir()
        (output / "predictions").mkdir()
        meta = dict(
            jobs=jobs,
            source_hashes=hashes,
            execution_environment=refmeta["execution_environment"],
            reference=str(reference),
            **binding,
        )
        full.atomic_json(output / "metadata.json", meta)
    rows = full.load_checkpoints(output, jobs)
    for r in rows:
        validate_trace(r, output, refs)
    done = {full.job_key(r["job"]) for r in rows}
    pending = [j for j in jobs if full.job_key(j) not in done]
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
    print(f"START W={len(rows)}/{len(jobs)} reused_D_B={2 * len(jobs)}", flush=True)
    failures = []
    if pending:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(pending)), mp_context=get_context("spawn")
        ) as pool:
            tasks = {
                pool.submit(
                    execute,
                    j,
                    refs[full.job_key(j)[:-1] + ("full-score-head-train",)][
                        "input_signature"
                    ],
                    output,
                ): j
                for j in pending
            }
            for future in as_completed(tasks):
                try:
                    r = future.result()
                    full.validate_one(r, tasks[future])
                    validate_trace(r, output, refs)
                except Exception as error:
                    failures.append(dict(job=tasks[future], error=repr(error)))
                    full.atomic_json(output / "failures.json", failures)
                    with (output / "attempts.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(failures[-1]) + "\n")
                    print(
                        f"FAILED W i{tasks[future]['interval']} s{tasks[future]['seed']}: {error!r}",
                        flush=True,
                    )
                    continue
                full.atomic_json(output / "runs" / (full.job_id(r["job"]) + ".json"), r)
                rows.append(r)
                print(
                    f"W {len(rows)}/{len(jobs)} i{r['job']['interval']} s{r['job']['seed']} acc={r['metrics']['accuracy']:.6f}",
                    flush=True,
                )
    full.atomic_json(output / "failures.json", failures)
    if failures:
        raise RuntimeError("失敗runがあります。同じ条件で--resumeしてください")
    if source_hashes() != hashes or any(
        full.sha(
            reference
            / ("results.json" if k.startswith("reference_results") else "metadata.json")
        )
        != v
        for k, v in binding.items()
    ):
        raise ValueError("実行中にソース・参照が変化しました")
    if {full.job_key(r["job"]) for r in rows} != {full.job_key(j) for j in jobs} or len(
        rows
    ) != len(jobs):
        raise ValueError("Wの条件が揃っていません")
    full.atomic_json(
        output / "results.json", sorted(rows, key=lambda r: full.job_key(r["job"]))
    )
    full.atomic_json(
        output / "complete.json",
        dict(
            run_count=len(rows),
            source_hashes_verified=True,
            results_sha256=full.sha(output / "results.json"),
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--intervals", type=int, nargs="+", default=[100, 500])
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--print-plan", action="store_true")
    a = p.parse_args()
    try:
        jobs = jobs_for(a.seeds, a.intervals)
        if a.workers < 1 or (a.output.exists() and not a.resume):
            raise ValueError("正のworkers・未使用出力先（再開時は--resume）が必要です")
        reference, refmeta, refs = load_reference(a.reference, jobs, not a.print_plan)
        if a.print_plan:
            completed = 0
            if a.resume:
                meta = json.loads(
                    (a.output / "metadata.json").read_text(encoding="utf-8")
                )
                if (
                    meta["jobs"] != jobs
                    or meta["source_hashes"] != source_hashes()
                    or meta["execution_environment"] != refmeta["execution_environment"]
                    or meta["reference_results_sha256"]
                    != full.sha(reference / "results.json")
                    or meta["reference_metadata_sha256"]
                    != full.sha(reference / "metadata.json")
                ):
                    raise ValueError("再開planの条件・ソース・参照が異なります")
                completed_rows = full.load_checkpoints(a.output, jobs)
                for row in completed_rows:
                    validate_trace(row, a.output, refs)
                completed = len(completed_rows)
            print(
                json.dumps(
                    dict(
                        new_W_runs=len(jobs),
                        completed_W_runs=completed,
                        pending_W_runs=len(jobs) - completed,
                        reused_D_B=2 * len(jobs),
                        reference=str(reference),
                        workers=a.workers,
                        environment_check="required_at_execution",
                    ),
                    indent=2,
                )
            )
            return
        execute_plan(reference, a.output, jobs, a.workers, a.resume)
    except ValueError as error:
        p.error(str(error))


if __name__ == "__main__":
    main()
