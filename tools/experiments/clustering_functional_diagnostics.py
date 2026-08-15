"""クラスタリング結果とモデル対の予測相補性をraw NPZから集計する。"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


REQUIRED_ARRAYS = (
    "clustering_pair_rounds",
    "clustering_pair_left_model_ids",
    "clustering_pair_right_model_ids",
    "clustering_pair_distances",
    "clustering_pair_decision_scores",
    "clustering_pair_same_cluster",
    "cross_evaluation_round_index",
    "cross_evaluation_candidate_model_id",
    "cross_evaluation_target_model_id",
    "cross_evaluation_n",
    "cross_evaluation_candidate_only_correct",
    "cross_evaluation_target_only_correct",
    "cross_evaluation_both_correct",
)


def _scalar(arrays, key, default):
    if key not in arrays:
        return default
    return arrays[key].item()


def summarize_raw(path):
    """1実験runの統合対・非統合対を機能的診断値へ変換する。"""
    with np.load(path, allow_pickle=False) as arrays:
        missing = [key for key in REQUIRED_ARRAYS if key not in arrays]
        if missing:
            raise ValueError(
                f"{path}: 新しいクラスタリング対診断がありません: "
                + ", ".join(missing)
            )

        cross = defaultdict(lambda: {
            "n": 0,
            "disagreement": 0,
            "oracle_gain": 0,
            "both_correct": 0,
        })
        cross_arrays = {
            key: arrays[f"cross_evaluation_{key}"]
            for key in (
                "round_index", "candidate_model_id", "target_model_id", "n",
                "candidate_only_correct", "target_only_correct", "both_correct",
            )
        }
        for values in zip(*cross_arrays.values()):
            record = dict(zip(cross_arrays, values))
            candidate = int(record["candidate_model_id"])
            target = int(record["target_model_id"])
            if candidate == target or int(record["candidate_only_correct"]) < 0:
                continue
            key = (
                int(record["round_index"]),
                min(candidate, target),
                max(candidate, target),
            )
            candidate_only = int(record["candidate_only_correct"])
            target_only = int(record["target_only_correct"])
            cross[key]["n"] += int(record["n"])
            cross[key]["disagreement"] += candidate_only + target_only
            cross[key]["oracle_gain"] += min(candidate_only, target_only)
            cross[key]["both_correct"] += int(record["both_correct"])

        rows = []
        pair_arrays = {
            key: arrays[f"clustering_pair_{key}"]
            for key in (
                "rounds", "left_model_ids", "right_model_ids", "distances",
                "decision_scores", "same_cluster",
            )
        }
        for values in zip(*pair_arrays.values()):
            record = dict(zip(pair_arrays, values))
            key = (
                int(record["rounds"]),
                int(record["left_model_ids"]),
                int(record["right_model_ids"]),
            )
            counts = cross.get(key)
            if counts is None or counts["n"] <= 0:
                continue
            rows.append({
                "dataset": str(_scalar(arrays, "dataset", "")),
                "mode": str(_scalar(arrays, "mode", "")),
                "seed": int(_scalar(arrays, "seed", -1)),
                "aggregation_interval": int(
                    _scalar(arrays, "aggregation_interval", -1)
                ),
                "clustering_decision": str(
                    _scalar(arrays, "clustering_decision", "")
                ),
                "same_cluster": bool(record["same_cluster"]),
                "distance": float(record["distances"]),
                "decision_score": float(record["decision_scores"]),
                **counts,
            })
        return rows


def aggregate(rows):
    """データセット・判定方式・集約間隔・統合結果ごとに集約する。"""
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"], row["clustering_decision"],
            row["aggregation_interval"], row["same_cluster"],
        )
        groups[key].append(row)

    summaries = []
    for key, items in sorted(groups.items()):
        sample_count = sum(item["n"] for item in items)
        summaries.append({
            "dataset": key[0],
            "clustering_decision": key[1],
            "aggregation_interval": key[2],
            "outcome": "merged" if key[3] else "retained",
            "pair_observation_count": len(items),
            "sample_count": sample_count,
            "correctness_disagreement_rate": (
                sum(item["disagreement"] for item in items) / sample_count
            ),
            "oracle_gain_rate": (
                sum(item["oracle_gain"] for item in items) / sample_count
            ),
            "both_correct_rate": (
                sum(item["both_correct"] for item in items) / sample_count
            ),
            "distance_mean": float(np.mean([
                item["distance"] for item in items
            ])),
            "decision_score_mean": float(np.mean([
                item["decision_score"] for item in items
            ])),
        })
    return summaries


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="raw NPZから統合対と非統合対の予測相補性を比較する"
    )
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = sorted(args.result_root.rglob("*.npz"))
    rows = []
    skipped = []
    for path in paths:
        try:
            rows.extend(summarize_raw(path))
        except ValueError as error:
            skipped.append(str(error))
    summaries = aggregate(rows)
    if not summaries:
        detail = f"\n最初の欠損: {skipped[0]}" if skipped else ""
        raise SystemExit(f"集計可能なモデル対診断がありません。{detail}")

    if args.output:
        _write_csv(args.output, summaries)
        print(f"CSV saved: {args.output}")
    for row in summaries:
        print(
            f"{row['dataset']}/{row['clustering_decision']}/"
            f"A={row['aggregation_interval']}/{row['outcome']}: "
            f"pairs={row['pair_observation_count']} "
            f"disagreement={row['correctness_disagreement_rate']:.4f} "
            f"oracle_gain={row['oracle_gain_rate']:.4f}"
        )
    if skipped:
        print(f"旧スキーマのため除外: {len(skipped)} files")


if __name__ == "__main__":
    main()
