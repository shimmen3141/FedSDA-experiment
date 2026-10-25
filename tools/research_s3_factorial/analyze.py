"""同じ初期値・ln2の2×2対照を、既定差と区別してseedごとに集計する。"""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, stdev

from tools.research_s3_factorial.policy import FACTORS
from tools.research_s3_factorial.run import (
    COMPARISON_ARMS,
    NEW_ARMS,
    sha,
    source_hashes,
    validate_references,
)


def contrasts(accuracies):
    """精度は0〜1で受け取り、条件付き効果と交互作用をppで返す。"""
    w = accuracies["warm-matched"]
    a = accuracies["head-score-full-train"]
    b = accuracies["full-score-head-train"]
    c = accuracies["output-layer-matched"]
    return dict(
        score_effect_full_training_pp=100 * (a - w),
        score_effect_head_training_pp=100 * (c - b),
        training_effect_full_score_pp=100 * (b - w),
        training_effect_head_score_pp=100 * (c - a),
        interaction_pp=100 * (c - a - b + w),
        s3_minus_matched_pp=100 * (c - w),
        initialization_effect_pp=100 * (w - accuracies["product-default"]),
        s3_minus_default_pp=100 * (c - accuracies["product-default"]),
    )


def load_cells(output):
    meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if meta["source_hashes"] != source_hashes():
        raise ValueError("実行ソースのhashの欠落または変化があります")
    reference, proof = Path(meta["reference"]), Path(meta["input_proof"])
    if (
        sha(reference / "results.json") != meta["reference_sha256"]
        or sha(proof) != meta["input_proof_sha256"]
    ):
        raise ValueError("参照が変化しています")
    refs, signatures = validate_references(
        reference, proof, meta["seeds"], meta["sample_count"]
    )
    complete = json.loads((output / "complete.json").read_text(encoding="utf-8"))
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    keys = ("dataset", "fixed_concept", "interval", "sample_count", "seed", "arm")
    expected = Counter(tuple(j[k] for k in keys) for j in meta["jobs"])
    actual = Counter(tuple(r["job"][k] for k in keys) for r in rows)
    arms = {j["arm"] for j in meta["jobs"]}
    rectangle = Counter(
        ("mnist4", None, 500, meta["sample_count"], seed, arm)
        for seed in meta["seeds"]
        for arm in arms
    )
    if (
        (output / "failures.json").exists()
        or complete.get("source_hashes_verified") is not True
        or complete["run_count"] != len(rows)
        or actual != expected
        or expected != rectangle
        or any(n != 1 for n in actual.values())
        or not set(NEW_ARMS) <= arms
        or len(set(meta["seeds"])) < 4
    ):
        raise ValueError("2因子対照の条件に欠落・重複・未完了があります")
    cells, provenance = {}, {}
    for seed in meta["seeds"]:
        for arm in ("product-default", "warm-matched", "output-layer-matched"):
            cells[seed, arm] = refs[
                "mnist4", None, 500, meta["sample_count"], seed, arm
            ]
            provenance[seed, arm] = "reused"
    for row in rows:
        job, metrics = row["job"], row["metrics"]
        seed, arm = job["seed"], job["arm"]
        if row["input_signature"] != signatures[str(seed)] or row["factors"] != list(
            FACTORS[arm]
        ):
            raise ValueError("入力列または因子の指定が一致しません")
        if (
            any(
                metrics[k] != job[k]
                for k in ("dataset", "interval", "sample_count", "seed")
            )
            or metrics["research_variant"] != arm
            or metrics["fixed_concept"] is not None
            or metrics["check"] is not False
            or metrics["variant"] != "product-default"
        ):
            raise ValueError("run条件と指標が一致しません")
        if (seed, arm) in cells:
            ref = cells[seed, arm]
            if any(metrics[k] != v for k, v in ref.items() if k != "seconds"):
                raise ValueError("再実行した端点が既存実装と一致しません")
        cells[seed, arm], provenance[seed, arm] = metrics, "new"
    if set(cells) != {(seed, arm) for seed in meta["seeds"] for arm in COMPARISON_ARMS}:
        raise ValueError("比較する5構成が揃っていません")
    return meta, cells, provenance


def analyze(output):
    meta, cells, provenance = load_cells(output)
    per_seed, effects = [], []
    for seed in sorted(meta["seeds"]):
        base = cells[seed, "product-default"]["accuracy"]
        matched = cells[seed, "warm-matched"]["accuracy"]
        effect = dict(
            seed=seed,
            **contrasts({arm: cells[seed, arm]["accuracy"] for arm in COMPARISON_ARMS}),
        )
        if (
            abs(
                effect["initialization_effect_pp"]
                + effect["score_effect_full_training_pp"]
                + effect["training_effect_full_score_pp"]
                + effect["interaction_pp"]
                - effect["s3_minus_default_pp"]
            )
            > 1e-10
        ):
            raise ValueError("二因子の差の恒等式が一致しません")
        effects.append(effect)
        for arm in COMPARISON_ARMS:
            m = cells[seed, arm]
            per_seed.append(
                dict(
                    seed=seed,
                    arm=arm,
                    provenance=provenance[seed, arm],
                    accuracy_pct=100 * m["accuracy"],
                    default_delta_pp=100 * (m["accuracy"] - base),
                    matched_delta_pp=100 * (m["accuracy"] - matched),
                    uploaded_bytes=m["uploaded_bytes"],
                    downloaded_bytes=m["downloaded_bytes"],
                    mac=sum(
                        v
                        for k, v in m["model_computation_counts"].items()
                        if k.endswith("multiply_accumulate_count")
                    ),
                    seconds=m["seconds"],
                    concept_accuracies=m["concept_accuracies"],
                    prediction_period_accuracies=m["prediction_period_accuracies"],
                    warm_candidates=sum(
                        e["event"] == "matched_candidate" for e in m["research_events"]
                    ),
                )
            )
    summaries = []
    for arm in COMPARISON_ARMS:
        values = [r for r in per_seed if r["arm"] == arm]
        fields = (
            "accuracy_pct",
            "default_delta_pp",
            "matched_delta_pp",
            "uploaded_bytes",
            "downloaded_bytes",
            "mac",
            "seconds",
            "warm_candidates",
        )
        summaries.append(
            dict(
                arm=arm,
                n=len(values),
                **{f: mean(r[f] for r in values) for f in fields},
                default_delta_sd=stdev(r["default_delta_pp"] for r in values),
                seed_deltas=[r["default_delta_pp"] for r in values],
                positive_seeds=sum(r["default_delta_pp"] > 0 for r in values),
            )
        )
    effect_summary = {
        k: dict(mean=mean(r[k] for r in effects), sd=stdev(r[k] for r in effects))
        for k in effects[0]
        if k != "seed"
    }
    result = dict(
        cells=per_seed,
        contrasts=effects,
        summaries=summaries,
        effect_summary=effect_summary,
        new_runs=sum(p == "new" for p in provenance.values()),
        reused_runs=sum(p == "reused" for p in provenance.values()),
        input_results_sha256=sha(output / "results.json"),
        analysis_source_sha256=sha(Path(__file__)),
        reference_sha256=meta["reference_sha256"],
        input_proof_sha256=meta["input_proof_sha256"],
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            dict(summaries=summaries, effects=effects, effect_summary=effect_summary),
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    analyze(parser.parse_args().output)
