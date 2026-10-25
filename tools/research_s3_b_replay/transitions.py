"""保存済みW/Bをclient・真の概念episode・変化後の経過標本数で分ける。"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tools.research_s3_b_replay.analyze import analyze
from tools.research_s3_b_replay.run import full, load_context
from tools.research_s3_warm_control.cumulative import compare

AGE_BOUNDS = (0, 50, 100, 250, 500, 1000)


def counts(base, candidate):
    rescued = sum(
        not w["correct"] and b["correct"] for w, b in zip(base, candidate, strict=True)
    )
    harmed = sum(
        w["correct"] and not b["correct"] for w, b in zip(base, candidate, strict=True)
    )
    return dict(
        observations=len(base),
        rescued=rescued,
        harmed=harmed,
        net_correct=rescued - harmed,
        W_correct=sum(p["correct"] for p in base),
        B_correct=sum(p["correct"] for p in candidate),
    )


def partition(base, candidate, sample_count):
    """各標本を一度だけ所属させる。初期episodeを変化直後として扱わない。"""
    compare(base, candidate, sample_count, sample_count)
    wm = {(p["client_id"], p["sample_index"]): p for p in base}
    bm = {(p["client_id"], p["sample_index"]): p for p in candidate}
    segments, clients = [], []
    for client in range(10):
        w = [wm[client, s] for s in range(sample_count)]
        b = [bm[client, s] for s in range(sample_count)]
        starts = [0] + [
            s for s in range(1, sample_count) if w[s]["concept"] != w[s - 1]["concept"]
        ]
        seen = set()
        for episode, (start, end) in enumerate(
            zip(starts, starts[1:] + [sample_count], strict=True)
        ):
            concept = w[start]["concept"]
            phase = (
                "initial"
                if episode == 0
                else "return"
                if concept in seen
                else "first_visit"
            )
            spans = (
                [(start, end, "initial", None, None)]
                if episode == 0
                else [
                    (
                        start + lo,
                        min(end, start + hi) if hi is not None else end,
                        f"{lo}:{hi}" if hi is not None else f"{lo}:+",
                        lo,
                        hi,
                    )
                    for lo, hi in zip(AGE_BOUNDS, AGE_BOUNDS[1:] + (None,), strict=True)
                    if start + lo < end
                ]
            )
            for lo, hi, age_bin, age_start, age_end in spans:
                segments.append(
                    dict(
                        client=client,
                        episode=episode,
                        concept=concept,
                        phase=phase,
                        previous_concept=w[start - 1]["concept"] if start else None,
                        episode_start=start,
                        episode_end=end,
                        start=lo,
                        end=hi,
                        age_bin=age_bin,
                        age_start=age_start,
                        age_end_exclusive=age_end,
                        **counts(w[lo:hi], b[lo:hi]),
                    )
                )
            seen.add(concept)
        client_counts = counts(w, b)
        own = [s for s in segments if s["client"] == client]
        for field in client_counts:
            if sum(s[field] for s in own) != client_counts[field]:
                raise ValueError("episode分割とclient集計が一致しません")
        clients.append(dict(client=client, **client_counts))
    return segments, clients


def group(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []

    def order(item):
        return tuple(
            (-1 if value == "initial" else int(value.split(":")[0]))
            if name == "age_bin"
            else value
            for name, value in zip(keys, item[0], strict=True)
        )

    for key, values in sorted(groups.items(), key=order):
        values_sum = {
            field: sum(r[field] for r in values)
            for field in (
                "observations",
                "rescued",
                "harmed",
                "net_correct",
                "W_correct",
                "B_correct",
            )
        }
        output.append(
            dict(
                zip(keys, key, strict=True),
                **values_sum,
                pooled_delta_pp=100
                * values_sum["net_correct"]
                / values_sum["observations"],
                seeds=len({r["seed"] for r in values}),
                client_runs=len({(r["seed"], r["client"]) for r in values}),
                episodes=len({(r["seed"], r["client"], r["episode"]) for r in values}),
            )
        )
    return output


def run(output, reference=None, warm_output=None):
    # 完了・原B再現・全W/B入力とdigestの検査は既存入口をそのまま実行する。
    parent = analyze(output, reference, warm_output)
    meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    reference = reference or Path(meta["reference"])
    warm_output = warm_output or Path(meta["warm_output"])
    _, _, _, ws, _ = load_context(reference, warm_output, meta["jobs"])
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    all_segments, all_clients, trace_hashes = [], [], []
    analysis_source = full.sha(Path(__file__))
    for row in rows:
        job = row["job"]
        wrow = ws[full.job_key(job)[:-1] + ("warm-matched",)]
        traces = []
        for directory, saved in ((warm_output, wrow), (output, row)):
            path = directory / saved["trace_file"]
            if full.sha(path) != saved["trace_sha256"]:
                raise ValueError("系列hashが変化しています")
            trace_hashes.append(
                dict(
                    interval=job["interval"],
                    seed=job["seed"],
                    arm=saved["job"]["arm"],
                    sha256=saved["trace_sha256"],
                )
            )
            traces.append(json.loads(path.read_text(encoding="utf-8"))["predictions"])
        segments, clients = partition(*traces, job["sample_count"])
        original = next(
            p
            for p in parent["pairs"]
            if p["interval"] == job["interval"] and p["seed"] == job["seed"]
        )
        if (
            sum(s["observations"] for s in segments) != 10 * job["sample_count"]
            or abs(
                100
                * sum(s["net_correct"] for s in segments)
                / (10 * job["sample_count"])
                - original["final_delta_pp"]
            )
            > 1e-10
        ):
            raise ValueError("分割から元の全体差を復元できません")
        all_segments.extend(
            dict(interval=job["interval"], seed=job["seed"], **s) for s in segments
        )
        all_clients.extend(
            dict(interval=job["interval"], seed=job["seed"], **s) for s in clients
        )
    post_change = [s for s in all_segments if s["phase"] != "initial"]
    result = dict(
        segments=all_segments,
        clients=all_clients,
        overall=group(all_segments, ("interval",)),
        phase=group(all_segments, ("interval", "phase")),
        post_change_age=group(post_change, ("interval", "age_bin")),
        concept_age_phase=group(
            all_segments, ("interval", "concept", "phase", "age_bin")
        ),
        concept_phase=group(all_segments, ("interval", "concept", "phase")),
        client_concept=group(all_segments, ("interval", "seed", "client", "concept")),
        standard_summaries=parent["summaries"],
        trace_hashes=trace_hashes,
        results_sha256=full.sha(output / "results.json"),
        parent_analysis_sha256=full.sha(output / "analysis_comparison.json"),
        analysis_source_sha256=analysis_source,
        cumulative_source_sha256=parent["cumulative_source_sha256"],
    )
    if full.sha(Path(__file__)) != analysis_source:
        raise ValueError("解析中にソースが変化しました")
    full.atomic_json(output / "analysis_transitions.json", result)
    lines = [
        "# client・真の概念変化後の経過標本数によるW/B対応差",
        "",
        "初期episodeは変化後に含めない。経過0は新概念の最初の標本。区間は左閉右開。",
        "first_visitは初めて現れた概念への変化、returnは過去に出た概念への再訪。",
        "以下の差は標本数で重み付けしたpooled差。欠けた区間を0埋めせず、seed数も明示する。",
        "以前の概念別seed等重み平均とは集計方法が異なる。因果・採用規則には使わない。",
        "",
    ]
    for name in ("overall", "phase", "post_change_age", "concept_phase"):
        keys = {
            "overall": ("interval",),
            "phase": ("interval", "phase"),
            "post_change_age": ("interval", "age_bin"),
            "concept_phase": ("interval", "concept", "phase"),
        }[name]
        lines += [
            "## " + name,
            "",
            "|" + "|".join(keys) + "|標本数|純増正解|差pp|seed数|",
            "|" + "|".join(["---"] * (len(keys) + 4)) + "|",
        ]
        for r in result[name]:
            lines.append(
                "|"
                + "|".join(str(r[k]) for k in keys)
                + f"|{r['observations']}|{r['net_correct']:+d}|{r['pooled_delta_pp']:+.4f}|{r['seeds']}|"
            )
        lines.append("")
    lines += [
        "## client別",
        "",
        "|間隔|seed|client|純増正解|差pp|",
        "|---|---|---|---|---|",
    ]
    for r in all_clients:
        lines.append(
            f"|{r['interval']}|{r['seed']}|{r['client']}|{r['net_correct']:+d}|{100 * r['net_correct'] / r['observations']:+.4f}|"
        )
    (output / "analysis_transitions.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                overall=result["overall"],
                phase=result["phase"],
                age=result["post_change_age"],
            ),
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--warm-output", type=Path)
    args = p.parse_args()
    run(args.output, args.reference, args.warm_output)
