"""保存済みseed3のBだけを非介入追跡し、既存D/W/S3のtraceと照合する。"""

import argparse
import hashlib
import json
from pathlib import Path

from tools.research_s3_validation.run import execute, sha, source_hashes


def validate_predictions(data, metrics, sample_count):
    from tools.research_s3_contribution.analyze import validate_prediction_coverage

    ps = data["predictions"]
    validate_prediction_coverage(ps, sample_count)
    ordered = sorted(ps, key=lambda p: (p["client_id"], p["sample_index"]))
    if any(
        (
            max(range(len(p["probabilities"])), key=p["probabilities"].__getitem__)
            == p["label"]
        )
        != p["correct"]
        for p in ps
    ):
        raise ValueError("保存確率から正誤を復元できません")
    digest = hashlib.sha256(
        json.dumps([p["correct"] for p in ordered]).encode()
    ).hexdigest()
    if (
        digest != metrics["prediction_record_digest"]
        or abs(sum(p["correct"] for p in ps) / len(ps) - metrics["accuracy"]) > 1e-12
    ):
        raise ValueError("全標本の正誤と製品指標が一致しません")


def analyze(output):
    from tools.research_s3_contribution.analyze import (
        compare_traces,
        first_decision_difference,
        validate_decision,
    )
    from tools.research_s3_factorial.analyze import load_cells

    meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    reference = Path(meta["reference"])
    if (
        source_hashes() != meta["source_hashes"]
        or sha(reference / "results.json") != meta["reference_sha256"]
        or json.loads((output / "complete.json").read_text())
        != dict(run_count=1, source_hashes_verified=True)
    ):
        raise ValueError("追跡が未完了またはソース・参照が変化しています")
    refmeta, cells, _ = load_cells(reference)
    row = json.loads((output / "result.json").read_text())
    job, metrics = row["job"], row["metrics"]
    expected = dict(
        dataset="mnist4",
        fixed_concept=None,
        interval=500,
        sample_count=2500,
        seed=3,
        arm="full-score-head-train",
    )
    if job != expected or any(
        metrics[k] != v for k, v in cells[3, job["arm"]].items() if k != "seconds"
    ):
        raise ValueError("追跡と計測なしBが一致しません")
    target = output / row["diagnostic_file"]
    if sha(target) != row["diagnostic_sha256"]:
        raise ValueError("traceのhashが変化しています")
    data = json.loads(target.read_text())
    validate_predictions(data, metrics, 2500)
    for decision in data["decisions"]:
        validate_decision(decision)
    oldroot = Path(refmeta["input_proof"]).parent
    oldrows = json.loads((oldroot / "results.json").read_text())
    proof = json.loads(Path(refmeta["input_proof"]).read_text())
    comparisons = {}
    for arm in ("product-default", "warm-matched", "output-layer-matched"):
        old = next(
            r
            for r in oldrows
            if r["job"]["seed"] == 3 and r["job"]["research_variant"] == arm
        )
        path = oldroot / old["diagnostic_file"]
        if sha(path) != old["diagnostic_sha256"] or proof["raw_file_sha256"][
            path.name
        ] != sha(path):
            raise ValueError("元traceの検算hashが一致しません")
        base = json.loads(path.read_text())
        validate_predictions(base, old["metrics"], 2500)
        comparisons[arm] = dict(
            compare_traces(base["predictions"], data["predictions"]),
            first_decision=first_decision_difference(
                base["decisions"], data["decisions"]
            ),
        )
    first = next((r for r in data["candidates"] if r["is_head_only"]), None)
    if first is not None:
        first = {
            k: first[k]
            for k in (
                "token",
                "client_id",
                "sample_index",
                "prediction_order",
                "selection",
                "trained_samples",
                "update_steps",
            )
        }
    result = dict(
        metrics=metrics,
        comparisons=comparisons,
        predictions_verified=100000,
        decisions_verified=len(data["decisions"]),
        extra_model_forward_count=row["extra_model_forward_count"],
        first_head_candidate=first,
        trace_sha256=sha(target),
        analysis_source_sha256=sha(Path(__file__)),
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--print-plan", action="store_true")
    a = p.parse_args()
    if a.analyze:
        analyze(a.output)
        return
    from tools.research_s3_factorial.analyze import load_cells

    if a.reference is None or a.output.exists():
        p.error("参照と未使用の出力先が必要です")
    _, cells, _ = load_cells(a.reference)
    job = dict(
        dataset="mnist4",
        fixed_concept=None,
        interval=500,
        sample_count=2500,
        seed=3,
        arm="full-score-head-train",
    )
    ref = cells[3, job["arm"]]
    if a.print_plan:
        print(
            json.dumps(dict(run_count=1, job=job, reference=str(a.reference)), indent=2)
        )
        return
    a.output.mkdir(parents=True)
    meta = dict(
        reference=str(a.reference),
        reference_sha256=sha(a.reference / "results.json"),
        source_hashes=source_hashes(),
    )
    (a.output / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    row = execute(job, a.output / "seed3-B.json")
    if any(row["metrics"][k] != v for k, v in ref.items() if k != "seconds"):
        raise ValueError("非介入追跡と計測なしBが一致しません")
    (a.output / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    if (
        source_hashes() != meta["source_hashes"]
        or sha(a.reference / "results.json") != meta["reference_sha256"]
    ):
        raise ValueError("実行中にソース・参照が変化しました")
    (a.output / "complete.json").write_text(
        json.dumps(dict(run_count=1, source_hashes_verified=True)), encoding="utf-8"
    )
    print("seed3 B reference=exact; trace complete", flush=True)


if __name__ == "__main__":
    main()
