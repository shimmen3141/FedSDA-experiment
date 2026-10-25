"""原Bへの再現一致とW/B系列を検査して、累積差・区間差・概念別差を出す。"""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from tools.research_s3_b_replay.run import (
    full,
    jobs_for,
    load_context,
    source_hashes,
    validate_row,
)
from tools.research_s3_warm_control.cumulative import compare


def paired_details(base, candidate):
    """厳格な入力対応検査後の系列から、救済・悪化と概念別対応差を数える。"""
    bm = {(p["client_id"], p["sample_index"]): p for p in base}
    cm = {(p["client_id"], p["sample_index"]): p for p in candidate}
    groups = []
    for concept in sorted({p["concept"] for p in base}):
        keys = [k for k, p in bm.items() if p["concept"] == concept]
        rescued = sum(not bm[k]["correct"] and cm[k]["correct"] for k in keys)
        harmed = sum(bm[k]["correct"] and not cm[k]["correct"] for k in keys)
        groups.append(
            dict(
                concept=concept,
                observations=len(keys),
                rescued=rescued,
                harmed=harmed,
                delta_pp=100 * (rescued - harmed) / len(keys),
            )
        )
    return groups


def standard_cell(job, arm, metrics, default, w):
    return dict(
        interval=job["interval"],
        seed=job["seed"],
        arm=arm,
        accuracy_pct=100 * metrics["accuracy"],
        default_delta_pp=100 * (metrics["accuracy"] - default["accuracy"]),
        warm_delta_pp=100 * (metrics["accuracy"] - w["accuracy"]),
        uploaded_bytes=metrics["uploaded_bytes"],
        downloaded_bytes=metrics["downloaded_bytes"],
        mac=sum(
            v
            for k, v in metrics["model_computation_counts"].items()
            if k.endswith("multiply_accumulate_count")
        ),
        seconds=metrics["seconds"],
    )


def analyze(output, reference=None, warm_output=None, window_size=100):
    if window_size < 1:
        raise ValueError("window-sizeは正数が必要です")
    meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    jobs = meta["jobs"]
    canonical = jobs_for(
        sorted({j["seed"] for j in jobs}), sorted({j["interval"] for j in jobs})
    )
    reference = reference or Path(meta["reference"])
    warm_output = warm_output or Path(meta["warm_output"])
    _, refmeta, refs, ws, binding = load_context(reference, warm_output, jobs)
    if (
        Counter(map(full.job_key, canonical)) != Counter(map(full.job_key, jobs))
        or meta["source_hashes"] != source_hashes()
        or meta["execution_environment"] != refmeta["execution_environment"]
        or any(meta[k] != v for k, v in binding.items())
    ):
        raise ValueError("B再現の計画・ソース・環境・参照が不整合です")
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    full.validate_rows(rows, canonical)
    if (
        json.loads((output / "failures.json").read_text(encoding="utf-8"))
        or json.loads((output / "complete.json").read_text(encoding="utf-8"))
        != dict(
            run_count=len(rows),
            source_hashes_verified=True,
            results_sha256=full.sha(output / "results.json"),
        )
        or {full.job_key(r["job"]): r for r in rows}
        != {full.job_key(r["job"]): r for r in full.load_checkpoints(output, jobs)}
    ):
        raise ValueError("B再現が未完了、またはcheckpointと不一致です")
    pairs, cells = [], []
    for row in sorted(rows, key=lambda r: full.job_key(r["job"])):
        validate_row(row, output, refs)
        j = row["job"]
        wrow = ws[full.job_key(j)[:-1] + ("warm-matched",)]
        b = json.loads((output / row["trace_file"]).read_text(encoding="utf-8"))[
            "predictions"
        ]
        w = json.loads((warm_output / wrow["trace_file"]).read_text(encoding="utf-8"))[
            "predictions"
        ]
        result = compare(w, b, j["sample_count"], window_size)
        if (
            result["base_prediction_digest"]
            != wrow["metrics"]["prediction_record_digest"]
            or result["candidate_prediction_digest"]
            != row["metrics"]["prediction_record_digest"]
            or abs(
                result["final_delta_pp"]
                - 100 * (row["metrics"]["accuracy"] - wrow["metrics"]["accuracy"])
            )
            > 1e-10
        ):
            raise ValueError("系列の累積差/digestと全体指標が一致しません")
        pairs.append(
            dict(
                interval=j["interval"],
                seed=j["seed"],
                **result,
                concepts=paired_details(w, b),
                B_trace_sha256=row["trace_sha256"],
                W_trace_sha256=wrow["trace_sha256"],
            )
        )
        d = refs[full.job_key(j)[:-1] + ("product-default",)]["metrics"]
        for arm, m in (("D", d), ("W", wrow["metrics"]), ("B", row["metrics"])):
            cells.append(standard_cell(j, arm, m, d, wrow["metrics"]))
    summaries, windows, concepts = [], [], []
    for interval in sorted({j["interval"] for j in jobs}):
        ps = [p for p in pairs if p["interval"] == interval]
        for arm in ("D", "W", "B"):
            cs = [c for c in cells if c["interval"] == interval and c["arm"] == arm]
            summaries.append(
                dict(
                    interval=interval,
                    arm=arm,
                    seeds=len(cs),
                    **{
                        k: None
                        if any(c[k] is None for c in cs)
                        else mean(c[k] for c in cs)
                        for k in (
                            "accuracy_pct",
                            "default_delta_pp",
                            "warm_delta_pp",
                            "uploaded_bytes",
                            "downloaded_bytes",
                            "mac",
                            "seconds",
                        )
                    },
                )
            )
        for index in range(len(ps[0]["blocks"])):
            blocks = [p["blocks"][index] for p in ps]
            windows.append(
                dict(
                    interval=interval,
                    prefix=blocks[0]["end_sample_index_exclusive"],
                    cumulative_delta_pp=mean(b["cumulative_delta_pp"] for b in blocks),
                    window_delta_pp=mean(b["window_delta_pp"] for b in blocks),
                )
            )
        for concept in sorted({c["concept"] for p in ps for c in p["concepts"]}):
            cs = [c for p in ps for c in p["concepts"] if c["concept"] == concept]
            if len(cs) != len(ps):
                raise ValueError("概念別のseed被覆が不完全です")
            concepts.append(
                dict(
                    interval=interval,
                    concept=concept,
                    delta_pp=mean(c["delta_pp"] for c in cs),
                    rescued=sum(c["rescued"] for c in cs),
                    harmed=sum(c["harmed"] for c in cs),
                )
            )
    result = dict(
        pairs=pairs,
        standard_metrics_by_seed=cells,
        summaries=summaries,
        cumulative_summaries=windows,
        concept_summaries=concepts,
        analysis_source_sha256=full.sha(Path(__file__)),
        cumulative_source_sha256=full.sha(
            full.ROOT / "tools/research_s3_warm_control/cumulative.py"
        ),
        results_sha256=full.sha(output / "results.json"),
        **binding,
    )
    full.atomic_json(output / "analysis_comparison.json", result)
    lines = [
        "# B計測付き再現と保存済みWの比較",
        "",
        "全Bが元Bの秒以外の全指標/イベントと一致。全W/B入力・正誤digestを検査。",
        "結果を見て選んだ4seedの原因診断であり、新規の性能検証ではない。",
        "時間は負荷・計測処理が異なるため速度比較には使わない。",
        "",
        "|間隔|構成|精度%|既定差pp|W差pp|上りMB|下りMB|GMAC|秒|",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        seconds = "N/A" if s["seconds"] is None else f"{s['seconds']:.1f}"
        lines.append(
            f"|{s['interval']}|{s['arm']}|{s['accuracy_pct']:.4f}|{s['default_delta_pp']:+.4f}|{s['warm_delta_pp']:+.4f}|{s['uploaded_bytes'] / 1e6:.3f}|{s['downloaded_bytes'] / 1e6:.3f}|{s['mac'] / 1e9:.3f}|{seconds}|"
        )
    lines += [
        "",
        "## B−Wの平均累積差・区間差",
        "",
        "各seed等重み。prefixはclientごとの標本数。",
        "",
        "|間隔|prefix|累積差pp|区間差pp|",
        "|---:|---:|---:|---:|",
    ]
    lines += [
        f"|{s['interval']}|{s['prefix']}|{s['cumulative_delta_pp']:+.4f}|{s['window_delta_pp']:+.4f}|"
        for s in windows
    ]
    lines += [
        "",
        "## 概念別B−W",
        "",
        "概念別差はseed等重み。救済/悪化は全seedの標本数合計。真の概念は診断専用。",
        "",
        "|間隔|概念|差pp|W誤→B正|W正→B誤|",
        "|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"|{s['interval']}|{s['concept']}|{s['delta_pp']:+.4f}|{s['rescued']}|{s['harmed']}|"
        for s in concepts
    ]
    (output / "analysis_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            dict(replayed_B=len(rows), reused_W=len(rows), summaries=summaries),
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--warm-output", type=Path)
    parser.add_argument("--window-size", type=int, default=100)
    args = parser.parse_args()
    analyze(args.output, args.reference, args.warm_output, args.window_size)
