"""MNIST4のS3二因子対照。2端点と既定は照合済みの保存値を再利用する。"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path

from tools.research_s3_contribution.run import ROOT, reference_rows
from tools.research_s3_factorial.policy import FACTORS

NEW_ARMS = ("head-score-full-train", "full-score-head-train")
COMPARISON_ARMS = ("product-default", "warm-matched", *NEW_ARMS, "output-layer-matched")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = list((ROOT / "src/federated_learning_experiments").rglob("*.py"))
    names = [
        "tools/research_provisional_candidate_expert/run.py",
        "tools/research_shadow_learner_diagnostic/controls.py",
        "tools/research_s3_contribution/__init__.py",
        "tools/research_s3_contribution/run.py",
    ]
    names += [
        "tools/research_output_layer_learner/" + p
        for p in ("__init__.py", "policy.py", "run.py")
    ]
    names += [
        "tools/research_s3_factorial/" + p
        for p in ("__init__.py", "policy.py", "run.py")
    ]
    paths += [ROOT / p for p in names]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def validate_references(reference, input_proof, seeds, sample_count):
    indexed = reference_rows(reference)
    proof = json.loads(input_proof.read_text(encoding="utf-8"))
    # 入力hashを、検算元の診断runと元の計測なし参照へ結び付ける。
    input_meta = json.loads(
        (input_proof.parent / "metadata.json").read_text(encoding="utf-8")
    )
    input_rows = json.loads(
        (input_proof.parent / "results.json").read_text(encoding="utf-8")
    )
    if input_meta["reference_results_sha256"] != sha(
        reference / "results.json"
    ) or proof["predictions_verified"] != sum(
        10 * r["job"]["sample_count"] for r in input_rows
    ):
        raise ValueError("入力の検算記録が比較対象の参照へ結び付いていません")
    verified_keys = set()
    for row in input_rows:
        j = row["job"]
        key = (
            j["dataset"],
            j["fixed_concept"],
            j["interval"],
            j["sample_count"],
            j["seed"],
            j["research_variant"],
        )
        if (
            key not in indexed
            or proof["raw_file_sha256"].get(row["diagnostic_file"])
            != row["diagnostic_sha256"]
            or any(
                row["metrics"][k] != v
                for k, v in indexed[key].items()
                if k != "seconds"
            )
        ):
            raise ValueError("入力の検算元と計測なし参照が一致しません")
        if key in verified_keys:
            raise ValueError("入力の検算元に重複があります")
        verified_keys.add(key)
    for seed in seeds:
        for arm in ("product-default", "warm-matched", "output-layer-matched"):
            if ("mnist4", None, 500, sample_count, seed, arm) not in indexed:
                raise ValueError("同条件の再利用runがありません")
            if ("mnist4", None, 500, sample_count, seed, arm) not in verified_keys:
                raise ValueError("入力の検算元に構成の欠落があります")
        if str(seed) not in proof["input_signatures"]:
            raise ValueError("対応する入力列の検算記録がありません")
    return indexed, proof["input_signatures"]


def execute(job, expected_input_signature):
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    # 共通runnerがsrc pathを設定してから、native APIへ差し込む。
    from tools.research_provisional_candidate_expert.run import _execute_job
    from tools.research_s3_factorial.policy import (
        capture_input_signature,
        factorial_policy,
        input_signature,
    )

    events, inputs = [], {}
    with (
        capture_input_signature(inputs),
        factorial_policy(arm=job["arm"], events=events),
    ):
        result = _execute_job(
            dict(
                job,
                variant="product-default",
                check=False,
                record_prediction_period_metrics=True,
            )
        )
    signature = input_signature(inputs, job["sample_count"])
    if signature != expected_input_signature:
        raise ValueError("既存の対照と画像・ラベル・概念の入力列が一致しません")
    result.update(
        research_variant=job["arm"], fixed_concept=None, research_events=events
    )
    return dict(
        job=job, metrics=result, input_signature=signature, factors=FACTORS[job["arm"]]
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--input-proof", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--sample-count", type=int, default=2500)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--arms", choices=tuple(FACTORS), nargs="+", default=list(NEW_ARMS))
    p.add_argument("--print-plan", action="store_true")
    a = p.parse_args()
    if (
        len(set(a.seeds)) != len(a.seeds)
        or len(a.seeds) < 4
        or min(a.seeds) < 0
        or a.sample_count < 2500
        or a.workers < 1
        or len(set(a.arms)) != len(a.arms)
        or not set(NEW_ARMS) <= set(a.arms)
    ):
        p.error("異なる4seed以上・2500標本以上・正のworkers・重複なしの構成が必要です")
    if a.output.exists():
        p.error("保存先が存在します。既存実験を確認してください")
    _, signatures = validate_references(
        a.reference, a.input_proof, a.seeds, a.sample_count
    )
    jobs = [
        dict(
            dataset="mnist4",
            fixed_concept=None,
            interval=500,
            sample_count=a.sample_count,
            seed=s,
            arm=arm,
        )
        for s in a.seeds
        for arm in a.arms
    ]
    if a.print_plan:
        print(
            json.dumps(
                dict(
                    new_runs=len(jobs),
                    reused_runs=(len(COMPARISON_ARMS) - len(a.arms)) * len(a.seeds),
                    jobs=jobs,
                ),
                indent=2,
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
        reference=str(a.reference),
        reference_sha256=sha(a.reference / "results.json"),
        input_proof=str(a.input_proof),
        input_proof_sha256=sha(a.input_proof),
    )
    (a.output / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    rows, failures = [], []
    with ProcessPoolExecutor(
        max_workers=a.workers, mp_context=get_context("spawn")
    ) as pool:
        tasks = {pool.submit(execute, j, signatures[str(j["seed"])]): j for j in jobs}
        for future in as_completed(tasks):
            try:
                row = future.result()
            except Exception as error:
                failures.append(dict(job=tasks[future], error=repr(error)))
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
    if (
        meta["source_hashes"] != source_hashes()
        or sha(a.reference / "results.json") != meta["reference_sha256"]
        or sha(a.input_proof) != meta["input_proof_sha256"]
    ):
        raise RuntimeError("実行中にソースまたは参照が変化しました")
    (a.output / "complete.json").write_text(
        json.dumps(dict(run_count=len(rows), source_hashes_verified=True)),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
