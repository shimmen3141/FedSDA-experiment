"""固定したBの未使用seed比較。標準指標と対応差を全seed表示する。"""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, stdev

from tools.research_s3_validation.run import (
    ARMS,
    make_jobs,
    sha,
    source_hashes,
    validate_rows,
)


def analyze(output):
    meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    done = json.loads((output / "complete.json").read_text(encoding="utf-8"))
    canonical = make_jobs(meta["seeds"], meta["sample_count"])

    def encode(job):
        return json.dumps(job, sort_keys=True)

    if (
        Counter(map(encode, canonical)) != Counter(map(encode, meta["jobs"]))
        or meta["source_hashes"] != source_hashes()
        or (output / "failures.json").exists()
        or done != dict(run_count=len(rows), source_hashes_verified=True)
    ):
        raise ValueError("未完了・ソース変化・計画不整合があります")
    signatures = validate_rows(rows, canonical)
    cells = {(r["job"]["seed"], r["job"]["arm"]): r["metrics"] for r in rows}
    per_seed = []
    for seed in sorted(meta["seeds"]):
        for arm in ARMS:
            m = cells[seed, arm]
            r = dict(
                seed=seed,
                arm=arm,
                accuracy_pct=100 * m["accuracy"],
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
            )
            for label, comparison in (
                ("default", ARMS[0]),
                ("matched", ARMS[1]),
                ("s3", ARMS[3]),
            ):
                r[label + "_delta_pp"] = 100 * (
                    m["accuracy"] - cells[seed, comparison]["accuracy"]
                )
            per_seed.append(r)
    fields = (
        "accuracy_pct",
        "default_delta_pp",
        "matched_delta_pp",
        "s3_delta_pp",
        "uploaded_bytes",
        "downloaded_bytes",
        "mac",
        "seconds",
    )
    summaries = []
    for arm in ARMS:
        rs = [r for r in per_seed if r["arm"] == arm]
        summaries.append(
            dict(
                arm=arm,
                **{k: mean(r[k] for r in rs) for k in fields},
                default_delta_sd=stdev(r["default_delta_pp"] for r in rs),
                positive_seeds=sum(r["default_delta_pp"] > 0 for r in rs),
            )
        )
    result = dict(
        cells=per_seed,
        summaries=summaries,
        input_signatures=signatures,
        results_sha256=sha(output / "results.json"),
        analysis_source_sha256=sha(Path(__file__)),
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    analyze(p.parse_args().output)
