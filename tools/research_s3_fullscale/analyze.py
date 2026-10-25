"""本規模の条件別・seed別標準指標を、既定からの対応差とともに保存する。"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from tools.research_s3_fullscale.run import (
    ARMS,
    job_key,
    load_checkpoints,
    make_jobs,
    sha,
    source_hashes,
    validate_rows,
    validate_saved_baselines,
)


def analyze(output):
    meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    plan = meta["plan"]
    jobs = make_jobs(**plan)
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    done = json.loads((output / "complete.json").read_text(encoding="utf-8"))
    failures = json.loads((output / "failures.json").read_text(encoding="utf-8"))
    if (
        Counter(map(job_key, jobs)) != Counter(map(job_key, meta["jobs"]))
        or meta["source_hashes"] != source_hashes()
        or failures
        or done
        != dict(
            run_count=len(rows),
            source_hashes_verified=True,
            results_sha256=sha(output / "results.json"),
        )
    ):
        raise ValueError("未完了・失敗・条件不整合・ソース/結果の変化があります")
    signatures = validate_rows(rows, jobs)
    validate_saved_baselines(output, jobs, meta)
    checkpoints = load_checkpoints(output, jobs)
    if {job_key(r["job"]): r for r in rows} != {
        job_key(r["job"]): r for r in checkpoints
    }:
        raise ValueError("全checkpointと最終結果が一致しません")
    indexed = {job_key(r["job"]): r["metrics"] for r in rows}
    cells = []
    for row in rows:
        j, m = row["job"], row["metrics"]
        key = job_key(j)[:-1]
        c = dict(
            dataset=j["dataset"],
            interval=j["interval"],
            sample_count=j["sample_count"],
            seed=j["seed"],
            arm=j["arm"],
            provenance="reused_default" if row.get("reused_default") else "new",
            accuracy_pct=100 * m["accuracy"],
            uploaded_bytes=m["uploaded_bytes"],
            downloaded_bytes=m["downloaded_bytes"],
            mac=sum(
                v
                for k, v in m["model_computation_counts"].items()
                if k.endswith("multiply_accumulate_count")
            ),
            preparation_mac=None
            if m["preparation_model_computation_counts"] is None
            else sum(
                v
                for k, v in m["preparation_model_computation_counts"].items()
                if k.endswith("multiply_accumulate_count")
            ),
            seconds=m["seconds"],
            concept_accuracies=m["concept_accuracies"],
            prediction_period_accuracies=m["prediction_period_accuracies"],
        )
        for name, arm in (("default", ARMS[0]),):
            c[name + "_delta_pp"] = 100 * (
                m["accuracy"] - indexed[key + (arm,)]["accuracy"]
            )
        cells.append(c)
    summaries = []
    fields = (
        "accuracy_pct",
        "default_delta_pp",
        "uploaded_bytes",
        "downloaded_bytes",
        "mac",
        "preparation_mac",
        "seconds",
    )
    for dataset in plan["datasets"]:
        for interval in plan["intervals"]:
            for arm in ARMS:
                cs = [
                    c
                    for c in cells
                    if (c["dataset"], c["interval"], c["arm"])
                    == (dataset, interval, arm)
                ]
                summaries.append(
                    dict(
                        dataset=dataset,
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
                        positive_seeds=sum(c["default_delta_pp"] > 0 for c in cs),
                        negative_seeds=sum(c["default_delta_pp"] < 0 for c in cs),
                        worst_default_delta_pp=min(c["default_delta_pp"] for c in cs),
                    )
                )
    result = dict(
        cells=cells,
        summaries=summaries,
        input_signatures_recorded=len(signatures),
        direct_input_pair_groups_verified=len(signatures)
        - len(meta["baseline_record_hashes"]),
        reused_default_count=len(meta["baseline_record_hashes"]),
        results_sha256=sha(output / "results.json"),
        analysis_source_sha256=sha(Path(__file__)),
    )
    # 全検証の後だけ解析成果物を書き出す。異なるデータ/間隔を先に混ぜない。
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    csv_fields = [
        k
        for k in cells[0]
        if k not in ("concept_accuracies", "prediction_period_accuracies")
    ]
    with (output / "standard_metrics_by_seed.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cells)
    table = [
        "# B本規模の条件別標準指標",
        "",
        "各条件のseed等重み平均。時間は負荷/再開条件が異なるため方式固有の速度比較には使わない。",
        "",
        "| データ | 間隔 | 構成 | 精度% | 既定差pp（SD） | 正のseed | 上りMB | 下りMB | GMAC | 秒 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        seconds = "N/A" if s["seconds"] is None else f"{s['seconds']:.1f}"
        table.append(
            f"| {s['dataset']} | {s['interval']} | {s['arm']} | {s['accuracy_pct']:.3f} | {s['default_delta_pp']:+.3f}（{s['default_delta_sd']:.3f}） | {s['positive_seeds']}/{s['seeds']} | {s['uploaded_bytes'] / 1e6:.3f} | {s['downloaded_bytes'] / 1e6:.3f} | {s['mac'] / 1e9:.3f} | {seconds} |"
        )
    (output / "analysis_standard_metrics.md").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                run_count=len(rows),
                condition_count=len(plan["datasets"]) * len(plan["intervals"]),
                input_signatures_recorded=len(signatures),
                reused_default_count=len(meta["baseline_record_hashes"]),
                output=str(output),
            ),
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    analyze(p.parse_args().output)
