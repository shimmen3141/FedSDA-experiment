"""Bだけを再現し、時間以外の全指標が元のBと一致する場合だけ保存する。"""

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

from tools.research_s3_fullscale.baseline import environment
from tools.research_s3_warm_control import run as warm

full = warm.full
ARM = "full-score-head-train"


def jobs_for(seeds, intervals):
    return [dict(j, arm=ARM) for j in warm.jobs_for(seeds, intervals)]


def source_hashes():
    hashes = warm.source_hashes()
    for name in ("__init__.py", "run.py"):
        path = "tools/research_s3_b_replay/" + name
        hashes[path] = full.sha(full.ROOT / path)
    return dict(sorted(hashes.items()))


def require_replay_match(row, original):
    """秒数だけを除外し、未知の追加指標も含めて完全照合する。"""

    def comparable(value):
        return {k: v for k, v in value["metrics"].items() if k != "seconds"}

    if (
        row["job"] != original["job"]
        or row["input_signature"] != original["input_signature"]
        or comparable(row) != comparable(original)
    ):
        raise ValueError("計測付きBが原Bの入力・秒以外の全指標/イベントと一致しません")


def load_context(reference, warm_output, jobs, check_environment=False):
    reference, refmeta, refs = warm.load_reference(reference, jobs, check_environment)
    meta = json.loads((warm_output / "metadata.json").read_text(encoding="utf-8"))
    rows = json.loads((warm_output / "results.json").read_text(encoding="utf-8"))
    complete = json.loads((warm_output / "complete.json").read_text(encoding="utf-8"))
    canonical = warm.jobs_for(
        sorted({j["seed"] for j in meta["jobs"]}),
        sorted({j["interval"] for j in meta["jobs"]}),
    )
    if (
        Counter(map(full.job_key, canonical))
        != Counter(map(full.job_key, meta["jobs"]))
        or Counter(full.job_key(r["job"]) for r in rows)
        != Counter(map(full.job_key, canonical))
        or meta["source_hashes"] != warm.source_hashes()
        or meta["execution_environment"] != refmeta["execution_environment"]
        or meta["reference_results_sha256"] != full.sha(reference / "results.json")
        or meta["reference_metadata_sha256"] != full.sha(reference / "metadata.json")
        or json.loads((warm_output / "failures.json").read_text(encoding="utf-8"))
        or complete
        != dict(
            run_count=len(rows),
            source_hashes_verified=True,
            results_sha256=full.sha(warm_output / "results.json"),
        )
    ):
        raise ValueError("保存済みWの計画・完了・ソース・環境・参照が不整合です")
    indexed = {full.job_key(r["job"]): r for r in rows}
    if indexed != {
        full.job_key(r["job"]): r for r in full.load_checkpoints(warm_output, canonical)
    }:
        raise ValueError("Wのcheckpointと結果が異なります")
    for r in rows:
        warm.validate_trace(r, warm_output, refs)
    for j in jobs:
        if full.job_key(j)[:-1] + ("warm-matched",) not in indexed:
            raise ValueError("対応するWがありません")
    binding = {
        name + "_" + kind + "_sha256": full.sha(path / (kind + ".json"))
        for name, path in (("reference", reference), ("warm", warm_output))
        for kind in ("results", "metadata")
    }
    return reference, refmeta, refs, indexed, binding


def execute(job, original, output):
    row = warm.execute(job, original["input_signature"], output)
    require_replay_match(row, original)
    row["original_B_metrics_verified"] = True
    return row


def validate_row(row, output, refs):
    warm.validate_trace(row, output, refs)
    require_replay_match(row, refs[full.job_key(row["job"])])
    if row["job"]["arm"] != ARM or row.get("original_B_metrics_verified") is not True:
        raise ValueError("B再現の検証記録がありません")


def run(reference, warm_output, output, jobs, workers, resume, print_plan=False):
    reference, refmeta, refs, _, binding = load_context(
        reference, warm_output, jobs, not print_plan
    )
    hashes = source_hashes()
    expected = dict(
        jobs=jobs,
        source_hashes=hashes,
        execution_environment=refmeta["execution_environment"],
        **binding,
    )
    rows = []
    if output.exists():
        if not resume:
            raise ValueError("既存出力です。再開時だけ--resumeを指定してください")
        meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        if any(meta[k] != v for k, v in expected.items()):
            raise ValueError("再開時の計画・ソース・環境・参照が異なります")
        rows = full.load_checkpoints(output, jobs)
        for row in rows:
            validate_row(row, output, refs)
    elif resume:
        raise ValueError("再開先がありません")
    if print_plan:
        print(
            json.dumps(
                dict(
                    new_B_runs=len(jobs),
                    completed=len(rows),
                    pending=len(jobs) - len(rows),
                    reused_W=len(jobs),
                    workers=workers,
                    environment_check="required_at_execution",
                ),
                indent=2,
            )
        )
        return
    if not output.exists():
        output.mkdir(parents=True)
        (output / "runs").mkdir()
        (output / "predictions").mkdir()
        full.atomic_json(
            output / "metadata.json",
            dict(expected, reference=str(reference), warm_output=str(warm_output)),
        )
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
    print(f"START B={len(rows)}/{len(jobs)} reused_W={len(jobs)}", flush=True)
    failures = []
    if pending:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(pending)), mp_context=get_context("spawn")
        ) as pool:
            tasks = {
                pool.submit(execute, j, refs[full.job_key(j)], output): j
                for j in pending
            }
            for future in as_completed(tasks):
                j = tasks[future]
                try:
                    row = future.result()
                    validate_row(row, output, refs)
                except Exception as error:
                    failures.append(dict(job=j, error=repr(error)))
                    full.atomic_json(output / "failures.json", failures)
                    print(
                        f"FAILED B i{j['interval']} s{j['seed']}: {error!r}", flush=True
                    )
                    continue
                full.atomic_json(output / "runs" / (full.job_id(j) + ".json"), row)
                rows.append(row)
                print(
                    f"B {len(rows)}/{len(jobs)} i{j['interval']} s{j['seed']} acc={row['metrics']['accuracy']:.6f} original_match=PASS",
                    flush=True,
                )
    full.atomic_json(output / "failures.json", failures)
    if failures:
        raise RuntimeError(
            "失敗runがあります。原因を確認して同条件で--resumeしてください"
        )
    _, _, _, _, final_binding = load_context(reference, warm_output, jobs, True)
    if (
        source_hashes() != hashes
        or final_binding != binding
        or environment() != refmeta["execution_environment"]
    ):
        raise ValueError("実行中にソース・参照・環境が変化しました")
    full.validate_rows(rows, jobs)
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
    p.add_argument("--warm-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 3, 7, 8])
    p.add_argument("--intervals", type=int, nargs="+", default=[100, 500])
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--print-plan", action="store_true")
    a = p.parse_args()
    try:
        if a.workers < 1:
            raise ValueError("workersは正数が必要です")
        run(
            a.reference,
            a.warm_output,
            a.output,
            jobs_for(a.seeds, a.intervals),
            a.workers,
            a.resume,
            a.print_plan,
        )
    except ValueError as error:
        p.error(str(error))


if __name__ == "__main__":
    main()
