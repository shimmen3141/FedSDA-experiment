"""optimizer対照の保存トレースを再検算し、同時交差と再反転を概念別に集計する。"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from tools.research_shadow_learner_diagnostic.summarize_standard_metrics import (
    extract_pair,
)


def summarize_episode(episode, trace):
    """真の概念は集計の打切りだけに用い、交差の条件には使わない。"""
    fields = (
        ("score", "current_log_loss", "first_crossing"),
        ("control_score", "online_current_log_loss", "control_first_crossing"),
        (
            "reset_control_score",
            "reset_current_log_loss",
            "reset_control_first_crossing",
        ),
    )
    scores = {f: episode["initial_score"] for f, _, _ in fields}
    first = {}
    same = True
    retained = []
    if len(trace) != episode["observed_sample_count"]:
        raise ValueError("episodeの観測数が一致しません")
    final_index = episode["alarm_sample_index"] + len(trace)
    if (
        episode["last_sample_index"] != final_index
        or episode["end_sample_index"] != final_index
        or episode["end_reason"]
        not in {
            "horizon",
            "run_end",
            "alarm_current_model",
            "alarm_new_model",
            "alarm_held_model",
        }
    ):
        raise ValueError("episodeの終端が一致しません")
    for age, t in enumerate(trace, 1):
        if (
            t["age"] != age
            or t["sample_index"] != episode["alarm_sample_index"] + age
            or t["client_id"] != episode["client_id"]
        ):
            raise ValueError("観測の順序・clientが一致しません")
        same = same and t["observed_concept_id"] == episode["alarm_concept_id"]
        if t["same_alarm_concept_so_far"] is not same:
            raise ValueError("概念維持のフラグが一致しません")
        for field, loss, _ in fields:
            scores[field] += t[loss] - t["shadow_log_loss"]
            if not math.isfinite(scores[field]) or not math.isclose(
                scores[field], t[field], abs_tol=1e-9, rel_tol=1e-12
            ):
                raise ValueError(f"累積差が一致しません: {field}")
            if field not in first and scores[field] > 0:
                first[field] = (age, same)
        if same:
            retained.append(t)
    for field, _, prefix in fields:
        crossing = first.get(field)
        if episode[prefix + "_age"] != (crossing[0] if crossing else None):
            raise ValueError("初回交差の年齢が一致しません")
        if episode[prefix + "_same_alarm_concept"] != (
            crossing[1] if crossing else None
        ):
            raise ValueError("初回交差の概念フラグが一致しません")
    for field, final in (
        ("score", "last_score"),
        ("control_score", "control_score"),
        ("reset_control_score", "reset_control_score"),
    ):
        if not math.isclose(scores[field], episode[final], abs_tol=1e-9, rel_tol=1e-12):
            raise ValueError("最終差が一致しません")
    out = dict(
        episode_id=episode["episode_id"],
        client_id=episode["client_id"],
        concept=episode["alarm_concept_id"],
        learner=episode["learner_name"],
        pending_is_single_concept=episode["pending_is_single_concept"],
        same_concept_observations=len(retained),
        end_reason=episode["end_reason"],
    )
    for name, required in (
        ("d", ("score",)),
        ("keep", ("control_score",)),
        ("reset", ("reset_control_score",)),
        ("joint_keep", ("score", "control_score")),
        ("joint_reset", ("score", "reset_control_score")),
    ):
        flags = [all(t[f] > 0 for f in required) for t in retained]
        start = next((i for i, flag in enumerate(flags) if flag), None)
        out[name + "_age"] = start + 1 if start is not None else None
        out[name + "_reversed"] = (
            any(not flag for flag in flags[start + 1 :]) if start is not None else None
        )
        out[name + "_observed_after_crossing"] = (
            len(flags) - start - 1 if start is not None else None
        )
    return out


def validate_plan(jobs):
    """同じ意味の条件の二重計上と、比較群間のseed・規模の混在を拒否する。"""
    profiles, groups, seen = set(), defaultdict(set), set()
    for job in jobs:
        key = tuple(job[k] for k in ("dataset", "stream", "interval", "seed"))
        if key in seen:
            raise ValueError("同じ条件・seedの二重計上です")
        seen.add(key)
        groups[key[:3]].add(job["seed"])
        profiles.add(
            (job["sample_count"], job["horizon_samples"], job["task"], job["disabled"])
        )
    if len(profiles) != 1 or len({tuple(sorted(s)) for s in groups.values()}) != 1:
        raise ValueError("比較群の規模・seed集合が一致しません")


def analyze(output):
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads((output / "research-source.json").read_text(encoding="utf-8"))
    for name, expected in metadata["sha256"].items():
        source = root / name
        if (
            not source.is_file()
            or hashlib.sha256(source.read_bytes()).hexdigest() != expected
        ):
            snapshot = output / metadata.get("source_snapshots", {}).get(
                name, "source_snapshot/" + name.replace("\\", "/")
            )
            if (
                not snapshot.is_file()
                or hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected
            ):
                raise ValueError(f"実行時ソースと一致しません: {name}")
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))

    def canonical(job):
        return json.dumps(job, sort_keys=True)

    expected_jobs = Counter(canonical(j) for j in metadata["jobs"])
    if (
        (output / "failures.json").exists()
        or any(n != 1 for n in expected_jobs.values())
        or Counter(canonical(r["job"]) for r in rows) != expected_jobs
    ):
        raise ValueError("未完了・重複・失敗のある実験です")
    validate_plan(metadata["jobs"])
    episodes, standards = [], []
    trace_hashes = {}
    for row in rows:
        if row["job"]["task"] != "optimizer":
            raise ValueError("optimizer対照だけが対象です")
        # extract_pair自身が共通保存値と追加分を引いた計数の一致を全て再検査する。
        standards.append(extract_pair(dict(row, baseline_parity=True)))
        d = row["shadow_diagnostic"]
        job = row["job"]
        expected_trace = (
            f"{job['dataset']}_{job['stream']}_i{job['interval']}_s{job['seed']}.jsonl"
        )
        if d["trace_file"] != expected_trace or expected_trace in trace_hashes:
            raise ValueError("トレースとjobの対応が一致しません")
        trace_bytes = (output / expected_trace).read_bytes()
        trace_hashes[expected_trace] = hashlib.sha256(trace_bytes).hexdigest()
        by_episode = defaultdict(list)
        for line in trace_bytes.decode("utf-8").splitlines():
            t = json.loads(line)
            by_episode[t["episode_id"]].append(t)
        ids = [e["episode_id"] for e in d["episodes"]]
        if (
            len(set(ids)) != len(ids)
            or set(by_episode) - set(ids)
            or sum(map(len, by_episode.values())) != d["trace_row_count"]
        ):
            raise ValueError("トレースのepisode・行数が一致しません")
        for e in d["episodes"]:
            if e["observed_sample_count"] > job["horizon_samples"] or (
                e["end_reason"] == "horizon"
                and e["observed_sample_count"] != job["horizon_samples"]
            ):
                raise ValueError("上限による打切りと観測数が一致しません")
            episodes.append(
                dict(
                    **{
                        k: row["job"][k]
                        for k in ("dataset", "stream", "interval", "seed")
                    },
                    **summarize_episode(e, by_episode[e["episode_id"]]),
                )
            )
    groups = defaultdict(list)
    for e in episodes:
        groups[(e["dataset"], e["stream"], e["interval"], e["concept"])].append(e)
    lines = [
        "# optimizer対照: 概念別の交差と安定性",
        "",
        "全episodeを分母にする。交差・再反転は警報時から真の概念が維持される区間内だけ。",
        "D/対照の同時交差は同じ観測位置で両差が正になること。真の概念は事後集計だけに使う。",
        "再反転なしには次の警報・上限・run末による打切りを含み、安定した優位の証明ではない。",
        "診断の標準指標はanalysis_followup.mdも参照。精度改善や採用後の効果は測っていない。",
        "",
        "| データ | 入力 | 間隔 | 概念 | 影 | D | C継続 | Cリセット | D/C継続 | D/Cリセット | D後の再反転 | 同時継続後の再反転 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, es in sorted(groups.items()):
        counts = [
            sum(e[n + "_age"] is not None for e in es)
            for n in ("d", "keep", "reset", "joint_keep", "joint_reset")
        ]
        reversals = [
            sum(e[n + "_reversed"] is True for e in es) for n in ("d", "joint_keep")
        ]
        lines.append(
            "| " + " | ".join(map(str, [*key, len(es), *counts, *reversals])) + " |"
        )
    lines.extend(
        [
            "",
            "## 標準指標（条件内のseed平均）",
            "",
            "精度差はpp、通信MB=10^6byte、G MAC=10^9。秒は並列負荷・実行順序の影響を含む。",
            "",
            "| データ | 入力 | 間隔 | 精度% | 差pp | 上りMB | 下りMB | G MAC 既定→診断 | 秒 既定→診断 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    sg = defaultdict(list)
    for r in standards:
        sg[(r["dataset"], r["stream"], r["interval"])].append(r)
    for key, rs in sorted(sg.items()):

        def avg(k):
            return mean(r[k] for r in rs)

        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {100 * avg('measured_accuracy'):.3f} | {avg('accuracy_delta_pp'):+.3f} | {avg('measured_uploaded_bytes') / 1e6:.3f} | {avg('measured_downloaded_bytes') / 1e6:.3f} | {avg('baseline_mac') / 1e9:.3f}→{avg('measured_mac') / 1e9:.3f} | {avg('baseline_seconds'):.2f}→{avg('measured_seconds'):.2f} |"
        )
    (output / "analysis_optimizer.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "summary_optimizer.json").write_text(
        json.dumps(
            dict(
                episodes=episodes,
                standard_metrics=standards,
                input_sha256={
                    name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                    for name in ("research-source.json", "results.json")
                },
                trace_sha256=trace_hashes,
                analysis_source_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    analyze(parser.parse_args().output)
