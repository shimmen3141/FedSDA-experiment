"""混合利得の保存結果をseed内で集計し、候補平均・標本平均・seed平均を分ける。"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean

SCORES = ("current", "mixture_alone", "add_0.1", "add_0.01", "add_0.5")
QUADRANTS = tuple(
    f"current_{a}:add_{b}"
    for a, b in (
        ("positive", "positive"),
        ("positive", "nonpositive"),
        ("nonpositive", "positive"),
        ("nonpositive", "nonpositive"),
    )
)


def summarize_seed(records):
    """採用まで完了した窓を主解析とし、中止・打切りを別計数する。"""
    complete = [r for r in records if r["end_reason"] == "adopted"]
    samples = [s for r in complete for s in r["mixture_samples"]]
    scores = [
        {name: sum(s["gains"][name] for s in r["mixture_samples"]) for name in SCORES}
        for r in complete
    ]
    quadrants = Counter(
        f"current_{'positive' if s['current'] > 0 else 'nonpositive'}:add_{'positive' if s['add_0.1'] > 0 else 'nonpositive'}"
        for s in scores
    )
    return dict(
        started=len(records),
        completed=len(complete),
        sample_count=len(samples),
        partial_window_count=len(records) - len(complete),
        partial_sample_count=sum(
            len(r["mixture_samples"]) for r in records if r["end_reason"] != "adopted"
        ),
        outcomes=dict(Counter(r["end_reason"] for r in records)),
        quadrants={q: quadrants[q] for q in QUADRANTS},
        quadrant_fractions={q: quadrants[q] / len(complete) for q in QUADRANTS}
        if complete
        else None,
        strict_zero_window_scores={
            name: sum(s[name] == 0.0 for s in scores) for name in SCORES
        },
        candidate_mean_cumulative_gains={
            name: mean(s[name] for s in scores) for name in SCORES
        }
        if complete
        else None,
        sample_mean_gains={
            name: mean(s["gains"][name] for s in samples) for name in SCORES
        }
        if samples
        else None,
        virtual_accuracy={
            f"add_{alpha}": dict(
                helped=sum(
                    not s["statistics"]["mixture"]["correct"]
                    and s["statistics"][f"add_{alpha}"]["correct"]
                    for s in samples
                ),
                harmed=sum(
                    s["statistics"]["mixture"]["correct"]
                    and not s["statistics"][f"add_{alpha}"]["correct"]
                    for s in samples
                ),
                delta_pp=100
                * mean(
                    int(s["statistics"][f"add_{alpha}"]["correct"])
                    - int(s["statistics"]["mixture"]["correct"])
                    for s in samples
                )
                if samples
                else None,
            )
            for alpha in (0.1, 0.01, 0.5)
        },
    )


def summarize_groups(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r["job"][k] for k in ("dataset", "stream", "interval"))].append(r)
    result = []
    for key, rs in sorted(groups.items()):
        seeds = [
            dict(seed=r["job"]["seed"], **summarize_seed(r["wait"]["holdouts"]))
            for r in sorted(rs, key=lambda r: r["job"]["seed"])
        ]
        eligible = [s for s in seeds if s["completed"]]
        balanced = dict(
            quadrant_fractions={
                q: mean(s["quadrant_fractions"][q] for s in eligible) for q in QUADRANTS
            }
            if eligible
            else None,
            candidate_mean_cumulative_gains={
                name: mean(s["candidate_mean_cumulative_gains"][name] for s in eligible)
                for name in SCORES
            }
            if eligible
            else None,
            sample_mean_gains={
                name: mean(s["sample_mean_gains"][name] for s in eligible)
                for name in SCORES
            }
            if eligible
            else None,
            virtual_accuracy_delta_pp={
                f"add_{alpha}": mean(
                    s["virtual_accuracy"][f"add_{alpha}"]["delta_pp"] for s in eligible
                )
                for alpha in (0.1, 0.01, 0.5)
            }
            if eligible
            else None,
        )
        result.append(
            dict(
                dataset=key[0],
                stream=key[1],
                interval=key[2],
                seeds=seeds,
                eligible_seed_count=len(eligible),
                all_seed_count=len(seeds),
                seed_balanced=balanced,
                strict_zero_window_scores={
                    name: sum(s["strict_zero_window_scores"][name] for s in seeds)
                    for name in SCORES
                },
            )
        )
    return result


def analyze(output):
    from tools.research_current_mixture_benefit import analyze as verify_and_analyze

    # 先に実行ソース・全job・標本対応・累積値を既存解析で検証する。
    verify_and_analyze(output)
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    groups = summarize_groups(rows)
    lines = [
        "# 混合追加利得: seedを均等に扱う集計",
        "",
        "完了候補があるseed内で比率/平均を計算し、そのseedを等重みで平均する。候補0件seedは診断平均の対象なしとし、除外数を明示する。標準精度は全seed平均。partialは主解析から分離。",
        "",
        "| データ | 入力 | 間隔 | 対象seed/全seed | 現行+/追加+ | 現行+/追加≤0 | 現行≤0/追加+ | 現行≤0/追加≤0 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g in groups:
        fractions = g["seed_balanced"]["quadrant_fractions"]
        cells = (
            [f"{100 * fractions[q]:.2f}%" for q in QUADRANTS]
            if fractions
            else ["対象なし"] * 4
        )
        lines.append(
            f"| {g['dataset']} | {g['stream']} | {g['interval']} | {g['eligible_seed_count']}/{g['all_seed_count']} | "
            + " | ".join(cells)
            + " |"
        )
    lines.extend(
        [
            "",
            "## seed別の分母と厳密0",
            "",
            "| データ | 入力 | seed | 開始/完了 | 完了標本 | 中止/打切り標本 | 現行0 | 追加0(alpha=0.1) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for g in groups:
        for s in g["seeds"]:
            lines.append(
                f"| {g['dataset']} | {g['stream']} | {s['seed']} | {s['started']}/{s['completed']} | {s['sample_count']} | {s['partial_sample_count']} | {s['strict_zero_window_scores']['current']} | {s['strict_zero_window_scores']['add_0.1']} |"
            )
    (output / "analysis_mixture_by_seed.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "summary_mixture_by_seed.json").write_text(
        json.dumps(
            dict(
                groups=groups,
                input_sha256={
                    n: hashlib.sha256((output / n).read_bytes()).hexdigest()
                    for n in ("results.json", "research-source.json")
                },
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
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args().output)
