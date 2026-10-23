"""保存済みの影の軌跡だけで、逆転後の再反転と警報による打切りを調べる。"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from math import isfinite
from pathlib import Path
from statistics import mean


def summarize_episode(episode, trace, score_field="score"):
    """開始概念の継続中だけを見る。逆転直後の打切りを持続成功とは数えない。"""
    scores, same = [], True
    for age, row in enumerate(trace, 1):
        if not isfinite(row[score_field]):
            raise ValueError("非有限の符号長差です")
        if (
            row["age"] != age
            or row["sample_index"] != episode["alarm_sample_index"] + age
        ):
            raise ValueError("軌跡の位置が連続していません")
        if row["client_id"] != episode["client_id"]:
            raise ValueError("軌跡のclientが一致しません")
        same = same and row["observed_concept_id"] == episode["alarm_concept_id"]
        if row["same_alarm_concept_so_far"] != same:
            raise ValueError("開始概念の継続フラグが不正です")
        if same:
            scores.append(row[score_field])
    first = next((i for i, score in enumerate(scores) if score > 0), None)
    later = [] if first is None else scores[first + 1 :]
    reversal = next((i for i, score in enumerate(later, 1) if score <= 0), None)
    prefix = "control_" if score_field == "control_score" else ""
    expected_first = (
        episode[prefix + "first_crossing_age"]
        if episode[prefix + "first_crossing_same_alarm_concept"] is True
        else None
    )
    if expected_first != (None if first is None else first + 1):
        raise ValueError("保存済みの初回交差と軌跡が一致しません")
    return dict(
        episode_id=episode["episode_id"],
        client_id=episode["client_id"],
        origin_concept=episode["alarm_concept_id"],
        first_crossing_age=None if first is None else first + 1,
        followup_count=len(later),
        first_reversal_after_crossing=reversal,
        last_same_concept_score=scores[-1] if scores else episode["initial_score"],
        positive_fraction_of_followup=(
            sum(x > 0 for x in later) / len(later) if later else None
        ),
        same_concept_observation_count=len(scores),
        observation_count=len(trace),
        end_reason=episode["end_reason"],
        same_concept_end_reason="concept_change"
        if len(scores) < len(trace)
        else episode["end_reason"],
    )


def aggregate(episodes):
    """観測件数を返す。右打切りを補正した確率は推定しない。"""
    crossed = [e for e in episodes if e["first_crossing_age"] is not None]
    followed = [e for e in crossed if e["followup_count"] > 0]
    return dict(
        episodes=len(episodes),
        crossed=len(crossed),
        crossed_with_followup=len(followed),
        crossed_without_followup=len(crossed) - len(followed),
        reversed=sum(e["first_reversal_after_crossing"] is not None for e in crossed),
        crossed_last_nonpositive=sum(
            e["last_same_concept_score"] <= 0 for e in crossed
        ),
        first_crossing_after_120=sum(e["first_crossing_age"] > 120 for e in crossed),
        replaced_before_crossing=sum(
            e["same_concept_end_reason"] == "alarm_current_model"
            and e["first_crossing_age"] is None
            for e in episodes
        ),
        same_concept_end_reasons=dict(
            Counter(e["same_concept_end_reason"] for e in episodes)
        ),
    )


def analyze(output):
    """旧診断と新対照の両形式を読み、元の実験ファイルは書き換えない。"""
    input_hashes = {}

    def read(path):
        data = path.read_bytes()
        input_hashes[path.name] = hashlib.sha256(data).hexdigest()
        return data.decode("utf-8")

    if (output / "failures.json").exists():
        raise ValueError("失敗を含む実験は最終集計しません")
    metadata = json.loads(read(output / "research-source.json"))
    rows = json.loads(read(output / "results.json"))

    def key(job):
        return job["dataset"], job.get("stream", "drift"), job["interval"], job["seed"]

    jobs = [r.get("job", r) for r in rows]
    if Counter(map(key, jobs)) != Counter(map(key, metadata["jobs"])):
        raise ValueError("予定条件と完了条件が一致しません")
    records, run_strata = [], []
    for row, job in zip(rows, jobs, strict=True):
        diag = row["shadow_diagnostic"]
        trace = [
            json.loads(line) for line in read(output / diag["trace_file"]).splitlines()
        ]
        if len(trace) != diag["trace_row_count"]:
            raise ValueError("軌跡の行数が不一致です")
        by_episode = defaultdict(list)
        ids = {e["episode_id"] for e in diag["episodes"]}
        if len(ids) != len(diag["episodes"]) or any(
            t["episode_id"] not in ids for t in trace
        ):
            raise ValueError("重複または未知のepisode IDです")
        for t in trace:
            by_episode[t["episode_id"]].append(t)
        metrics = ["D", "C"] if "job" in row else ["D"]
        concept_count = {"sea4": 4, "sine2": 2}[job["dataset"]]
        # 影0件のseed/概念も行を作る。分母から落とさない。
        for metric in metrics:
            this_run = []
            for episode in diag["episodes"]:
                trajectory = by_episode[episode["episode_id"]]
                if len(trajectory) != episode["observed_sample_count"]:
                    raise ValueError("episodeの観測数が不一致です")
                d, c = episode["initial_score"], episode["initial_score"]
                if not isfinite(d) or episode["end_reason"] is None:
                    raise ValueError("未完了または非有限のepisodeです")
                if (
                    d != episode["current_code_length"] - episode["new_code_length"]
                    or d > 0
                ):
                    raise ValueError("初期差が不正です")
                for t in trajectory:
                    if not all(
                        isfinite(t[k])
                        for k in ("current_log_loss", "shadow_log_loss", "score")
                    ):
                        raise ValueError("非有限の損失またはDです")
                    d += t["current_log_loss"] - t["shadow_log_loss"]
                    if abs(d - t["score"]) > 1e-10:
                        raise ValueError("Dの累積値が不一致です")
                    if metric == "C":
                        if not all(
                            isfinite(t[k])
                            for k in ("online_current_log_loss", "control_score")
                        ):
                            raise ValueError("非有限の対照損失またはCです")
                        c += t["online_current_log_loss"] - t["shadow_log_loss"]
                        if abs(c - t["control_score"]) > 1e-10:
                            raise ValueError("Cの累積値が不一致です")
                detail = summarize_episode(
                    episode, trajectory, "score" if metric == "D" else "control_score"
                )
                detail.update(
                    dataset=job["dataset"],
                    stream=job.get("stream", "drift"),
                    interval=job["interval"],
                    seed=job["seed"],
                    metric=metric,
                )
                this_run.append(detail)
                records.append(detail)
            for concept in range(concept_count):
                es = [e for e in this_run if e["origin_concept"] == concept]
                run_strata.append(
                    dict(
                        dataset=job["dataset"],
                        stream=job.get("stream", "drift"),
                        interval=job["interval"],
                        seed=job["seed"],
                        metric=metric,
                        concept=concept,
                        **aggregate(es),
                    )
                )
    strata = defaultdict(list)
    for r in run_strata:
        strata[
            (r["dataset"], r["stream"], r["interval"], r["metric"], r["concept"])
        ].append(r)
    summary = []
    lines = [
        "# 影の逆転後の安定性と打切り",
        "",
        "保存済みの軌跡だけを使う事後分析。再学習・採用・元ファイルの変更は行わない。",
        "",
        "| データ | 入力 | 間隔 | 指標 | 開始概念 | 影 | 交差 | 追跡あり | 再反転 | 最後≤0 | 120超で交差 | 交差前に置換 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, seeds in sorted(strata.items()):
        dataset, stream, interval, metric, concept = group
        es = [
            e
            for e in records
            if (
                e["dataset"],
                e["stream"],
                e["interval"],
                e["metric"],
                e["origin_concept"],
            )
            == group
        ]
        stats = aggregate(es)
        stats.update(
            dataset=dataset,
            stream=stream,
            interval=interval,
            metric=metric,
            concept=concept,
            seed_count=len(seeds),
            seeds_with_crossings=sum(s["crossed"] > 0 for s in seeds),
            mean_crossed_episodes_per_seed=mean(s["crossed"] for s in seeds),
        )
        summary.append(stats)
        lines.append(
            f"| {dataset} | {stream} | {interval} | {metric} | {concept} | "
            + " | ".join(
                str(stats[k])
                for k in (
                    "episodes",
                    "crossed",
                    "crossed_with_followup",
                    "reversed",
                    "crossed_last_nonpositive",
                    "first_crossing_after_120",
                    "replaced_before_crossing",
                )
            )
            + " |"
        )
    lines += [
        "",
        "開始概念が続いている範囲だけを追う。一度でも別概念になった後は再来しても含めない。再反転＝初回>0の後に一度でも≤0。最後≤0は概念変化直前または打切り時の値で、run末までの安定性とは限らない。",
        "",
        "追跡あり＝初回交差の後に同じ概念の標本が少なくとも1件ある。追跡なしを安定成功と数えない。再反転率の母数には追跡期間差・右打切りがあり、表は記述統計だけ。警報で置換した影を残した場合の結果は未観測。",
        "",
        "120超は実際に120より後で初めて交差した影であり、上限を変えた実験の予測ではない。D/Cとも採用後の精度を示さない。",
        "",
        "## seed別集計（影0件を含む）",
        "",
        "```json",
        json.dumps(run_strata, indent=2),
        "```",
    ]
    (output / "analysis_stability.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    source = Path(__file__).resolve()
    (output / "summary_stability.json").write_text(
        json.dumps(
            dict(
                input_sha256=input_hashes,
                analysis_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                groups=summary,
                runs=run_strata,
                episodes=records,
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
