"""S3研究用CLI。製品の設定・手法登録は変更しない。"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

VARIANTS = ("product-default", "warm-matched", "output-layer-matched")


def execute(job):
    """既存の標準指標付き実行を、研究用の動的な対照の中で行う。"""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    from tools.research_provisional_candidate_expert.run import _execute_job
    from tools.research_output_layer_learner.policy import learner_policy
    from tools.research_shadow_learner_diagnostic.controls import (
        stationary_concept_policy,
    )

    events = []
    with (
        stationary_concept_policy(
            dataset=job["dataset"], concept_id=job["fixed_concept"]
        ),
        learner_policy(variant=job["research_variant"], events=events),
    ):
        result = _execute_job(
            dict(
                job,
                variant="product-default",
                check=False,
                record_prediction_period_metrics=True,
            )
        )
    result.update(
        research_variant=job["research_variant"],
        fixed_concept=job["fixed_concept"],
        research_events=events,
    )
    return result


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "src/federated_learning_experiments").rglob("*.py"))
    paths += sorted((root / "tools/research_output_layer_learner").glob("*.py"))
    paths += [
        root / "tools/research_provisional_candidate_expert/run.py",
        root / "tools/research_shadow_learner_diagnostic/controls.py",
    ]
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=("sea4", "sine2", "mnist4"), required=True
    )
    parser.add_argument("--intervals", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=2500)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--fixed-concept", type=int)
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()
    if (
        args.sample_count < 2500
        or args.workers < 1
        or min(args.intervals) < 1
        or min(args.seeds) < 0
    ):
        parser.error("2500標本以上、正のworkers/間隔、非負seedが必要です")
    if args.fixed_concept is not None and any(
        d == "mnist4" or args.fixed_concept not in range(4 if d == "sea4" else 2)
        for d in args.datasets
    ):
        parser.error("固定概念の範囲がデータと一致しません")
    jobs = [
        dict(
            dataset=d,
            interval=i,
            seed=s,
            sample_count=args.sample_count,
            research_variant=v,
            fixed_concept=args.fixed_concept,
        )
        for d in args.datasets
        for i in args.intervals
        for s in args.seeds
        for v in args.variants
    ]
    # 同一コマンド内と既存保存先の重複は、出力作成前に拒否する。
    if len({json.dumps(j, sort_keys=True) for j in jobs}) != len(jobs):
        parser.error("重複した条件があります")
    if args.output.exists():
        parser.error("保存先が存在します。既存結果を確認してください")
    if args.print_plan:
        print(json.dumps(dict(run_count=len(jobs), jobs=jobs), indent=2))
        return
    args.output.mkdir(parents=True)
    hashes = source_hashes()
    (args.output / "metadata.json").write_text(
        json.dumps(
            dict(jobs=jobs, source_hashes=hashes, workers=args.workers), indent=2
        ),
        encoding="utf-8",
    )
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn")
    ) as executor:
        futures = [executor.submit(execute, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            (args.output / "results.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )
            print(
                f"{len(results)}/{len(jobs)} {result['dataset']} seed{result['seed']} {result['research_variant']} acc={result['accuracy']:.6f}",
                flush=True,
            )
    if source_hashes() != hashes:
        raise RuntimeError("実行中に研究用/製品ソースが変わりました")
    (args.output / "complete.json").write_text(
        json.dumps(dict(run_count=len(results), source_hashes_verified=True)),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
