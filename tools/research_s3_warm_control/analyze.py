"""限定W対照を既存D/Bに対応させ、初期値差と候補学習範囲差を分ける。"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from tools.research_s3_warm_control.run import (
    full,
    jobs_for,
    load_reference,
    source_hashes,
    validate_trace,
)


def analyze(output, reference=None):
    meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    jobs = meta["jobs"]
    canonical = jobs_for(
        sorted({j["seed"] for j in jobs}), sorted({j["interval"] for j in jobs})
    )
    if (
        Counter(map(full.job_key, jobs)) != Counter(map(full.job_key, canonical))
        or meta["source_hashes"] != source_hashes()
    ):
        raise ValueError("Wの計画・数値ソースに不整合があります")
    reference, refmeta, refs = load_reference(
        reference or Path(meta["reference"]), jobs
    )
    if (
        full.sha(reference / "results.json") != meta["reference_results_sha256"]
        or full.sha(reference / "metadata.json") != meta["reference_metadata_sha256"]
        or meta["execution_environment"] != refmeta["execution_environment"]
    ):
        raise ValueError("Wと参照の結び付け・環境が一致しません")
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    complete = json.loads((output / "complete.json").read_text(encoding="utf-8"))
    if (
        json.loads((output / "failures.json").read_text(encoding="utf-8"))
        or complete
        != dict(
            run_count=len(rows),
            source_hashes_verified=True,
            results_sha256=full.sha(output / "results.json"),
        )
        or Counter(full.job_key(r["job"]) for r in rows)
        != Counter(map(full.job_key, jobs))
    ):
        raise ValueError("Wの欠落・失敗・重複・未完了があります")
    checkpoints = full.load_checkpoints(output, jobs)
    if {full.job_key(r["job"]): r for r in rows} != {
        full.job_key(r["job"]): r for r in checkpoints
    }:
        raise ValueError("Wのcheckpointと最終結果が一致しません")
    cells, contrasts = [], []
    for row in sorted(rows, key=lambda r: full.job_key(r["job"])):
        validate_trace(row, output, refs)
        j = row["job"]
        d = refs[full.job_key(j)[:-1] + ("product-default",)]["metrics"]
        b = refs[full.job_key(j)[:-1] + ("full-score-head-train",)]["metrics"]
        w = row["metrics"]
        initial = 100 * (w["accuracy"] - d["accuracy"])
        training = 100 * (b["accuracy"] - w["accuracy"])
        overall = 100 * (b["accuracy"] - d["accuracy"])
        if abs(initial + training - overall) > 1e-10:
            raise ValueError("差の分解が一致しません")
        contrasts.append(
            dict(
                interval=j["interval"],
                seed=j["seed"],
                initialization_pp=initial,
                training_scope_pp=training,
                B_minus_default_pp=overall,
            )
        )
        for arm, m in (
            ("product-default", d),
            ("warm-matched", w),
            ("full-score-head-train", b),
        ):
            cells.append(
                dict(
                    interval=j["interval"],
                    seed=j["seed"],
                    arm=arm,
                    accuracy_pct=100 * m["accuracy"],
                    default_delta_pp=100 * (m["accuracy"] - d["accuracy"]),
                    warm_delta_pp=100 * (m["accuracy"] - w["accuracy"]),
                    uploaded_bytes=m["uploaded_bytes"],
                    downloaded_bytes=m["downloaded_bytes"],
                    mac=sum(
                        v
                        for k, v in m["model_computation_counts"].items()
                        if k.endswith("multiply_accumulate_count")
                    ),
                    seconds=m["seconds"],
                )
            )
    summaries = []
    fields = (
        "accuracy_pct",
        "default_delta_pp",
        "warm_delta_pp",
        "uploaded_bytes",
        "downloaded_bytes",
        "mac",
        "seconds",
    )
    for interval in sorted({j["interval"] for j in jobs}):
        for arm in ("product-default", "warm-matched", "full-score-head-train"):
            cs = [c for c in cells if (c["interval"], c["arm"]) == (interval, arm)]
            summaries.append(
                dict(
                    interval=interval,
                    arm=arm,
                    seeds=len(cs),
                    **{
                        k: mean(c[k] for c in cs)
                        if all(c[k] is not None for c in cs)
                        else None
                        for k in fields
                    },
                    default_delta_sd=stdev(c["default_delta_pp"] for c in cs),
                    warm_delta_sd=stdev(c["warm_delta_pp"] for c in cs),
                )
            )
    result = dict(
        cells=cells,
        contrasts=contrasts,
        summaries=summaries,
        new_W_runs=len(rows),
        reused_D_B_runs=2 * len(rows),
        input_pairs_verified=len(rows),
        predictions_verified=sum(j["sample_count"] * 10 for j in jobs),
        results_sha256=full.sha(output / "results.json"),
        reference_results_sha256=meta["reference_results_sha256"],
        analysis_source_sha256=full.sha(Path(__file__)),
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output / "standard_metrics_by_seed.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    table = [
        "# 限定W対照の標準指標",
        "",
        "時間は負荷・計測処理が異なるため速度比較には使わない。",
        "",
        "| 間隔 | 構成 | 精度% | 既定差pp | W差pp | 上りMB | 下りMB | GMAC | 秒 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        seconds = "N/A" if s["seconds"] is None else f"{s['seconds']:.1f}"
        table.append(
            f"| {s['interval']} | {s['arm']} | {s['accuracy_pct']:.4f} | {s['default_delta_pp']:+.4f} | {s['warm_delta_pp']:+.4f} | {s['uploaded_bytes'] / 1e6:.3f} | {s['downloaded_bytes'] / 1e6:.3f} | {s['mac'] / 1e9:.3f} | {seconds} |"
        )
    (output / "analysis_standard_metrics.md").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                new_W_runs=len(rows), reused_D_B_runs=2 * len(rows), summaries=summaries
            ),
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--reference", type=Path)
    a = p.parse_args()
    analyze(a.output, a.reference)
