"""割り当てを推定する版（estimated.py）を掃引し、runごと・方式と設定ごとの要約をCSVへ書く。

使い方:
    python -m tools.research_linear_concept_pool.run_estimated --output results/linear-concept-pool/<名前> --workers 14

軸: 概念の大きさs × 間隔 × 窓w × 帰属の保留（0、w/2）× 共有あり・なし（10 client）。
対照として、割り当てを真の概念にしたoracle（窓・保留なし）を s × 間隔 × 共有あり・なし で流す。
学習率（3値）ごとに別のrunとして流す（割り当てが学習率で変わるため）。混合の設定は1 runで全部を同時に計算する。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tools.research_linear_concept_pool.estimated import EstimatedCell, simulate_estimated, threshold
from tools.research_linear_concept_pool.run import EVALUATION_SEEDS, TUNING_SEEDS, _summaries
from tools.research_linear_concept_pool.simulate import SEGMENT_TYPE_NAMES, TIME_BIN_EDGES, TIME_BIN_NAMES

SCALES = (0.3, 0.5, 1.0, 3.0)
INTERVALS = (50, 200, 1000)
WINDOWS = (10, 30, 100, 300)
SHARINGS = ("shared", "local")
LEARNING_RATES = (0.01, 0.02, 0.05)


def planned_cells() -> list[EstimatedCell]:
    cells = []
    for scale, interval, window, pending_half, sharing in itertools.product(
        SCALES, INTERVALS, WINDOWS, (False, True), SHARINGS
    ):
        cells.append(EstimatedCell(scale, interval, window, 10, sharing, pending=window // 2 if pending_half else 0))
    for scale, interval, sharing in itertools.product(SCALES, INTERVALS, SHARINGS):
        cells.append(EstimatedCell(scale, interval, 30, 10, sharing, mode="oracle"))
    return cells


def change_delays(concepts: np.ndarray, assigned: np.ndarray) -> np.ndarray:
    """最初の区間を除く各区間で、学習の割り当てが真の概念に合うまでの標本数（合わなければ区間の長さ）。"""
    delays = []
    for m in range(concepts.shape[0]):
        starts = np.concatenate([[0], np.flatnonzero(np.diff(concepts[m])) + 1, [concepts.shape[1]]])
        for start, end in zip(starts[1:-1], starts[2:], strict=True):
            hits = np.flatnonzero(assigned[m, start:end] == concepts[m, start])
            delays.append(hits[0] if len(hits) else end - start)
    return np.array(delays, dtype=float)


def run_one(arguments: tuple[EstimatedCell, int, float]) -> list[dict]:
    cell, seed, learning_rate = arguments
    started = time.perf_counter()
    result = simulate_estimated(cell, seed, learning_rate)
    seconds = time.perf_counter() - started
    types = result["types"]
    time_bins = np.digitize(result["since"], TIME_BIN_EDGES[1:-1])
    concepts, assigned = result["concepts"], result["assigned_concept"]
    delays = change_delays(concepts, assigned)
    d = cell.dimension
    base = {
        **{f"cell_{key}": value for key, value in asdict(cell).items()},
        "threshold": threshold(cell),
        "seed": seed,
        "learning_rate": learning_rate,
        "seconds_all_methods": seconds,
        "misassigned_training_fraction": float((assigned != concepts).mean()),
        "mean_assignment_delay": float(delays.mean()) if len(delays) else float("nan"),
        "median_assignment_delay": float(np.median(delays)) if len(delays) else float("nan"),
        "models_created": float(result["models_created"]),
        "merges": float(result["merges"]),
        "distinct_models_used": float(result["distinct_models"]),
        "mean_active_models": float(result["active_models"].mean()),
        "decision_mac_per_sample": float(result["decision_mac_per_sample"]),
    }
    rows = []

    def add(method: str, hyper: dict, losses: np.ndarray, communication: tuple[float, float], prediction_mac: float):
        row = dict(base, method=method, **hyper)
        row["upload_parameters_per_client"] = communication[0]
        row["download_parameters_per_client"] = communication[1]
        row["prediction_mac_per_sample"] = prediction_mac
        row["training_mac_per_sample"] = 2.0 * d
        values = _summaries(losses, types, time_bins)
        row["loss"] = values[0]
        for name, value in zip(SEGMENT_TYPE_NAMES, values[1:5], strict=True):
            row[f"loss_type_{name}"] = value
        for name, value in zip(TIME_BIN_NAMES, values[5:], strict=True):
            row[f"loss_since_{name}"] = value
        rows.append(row)

    pool_communication = (float(result["pool_uploaded_parameters"]), float(result["pool_downloaded_parameters"]))
    single_value = float(result["single_parameters_each_way"])
    add("hard", {}, result["hard"], pool_communication, float(d))
    for f, (mix_rate, share) in enumerate(zip(result["mix_rates"], result["share_rates"], strict=True)):
        add(
            "mix",
            {"mixture_learning_rate": mix_rate, "share_rate": share},
            result["mix"][f],
            pool_communication,
            float(result["active_models"].mean() * d),
        )
    add("single", {}, result["single"], (single_value, single_value), float(d))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="小さい計画（3 cell×2 seed×1学習率）だけ流す")
    arguments = parser.parse_args(argv)
    output: Path = arguments.output
    if output.exists() and any(output.iterdir()):
        print(f"出力先が空ではない: {output}", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    cells = planned_cells()
    seeds = TUNING_SEEDS + EVALUATION_SEEDS
    rates = LEARNING_RATES
    if arguments.smoke:
        cells = [
            EstimatedCell(1.0, 200, 30, 10, "shared", pending=15),
            EstimatedCell(1.0, 200, 30, 10, "local"),
            EstimatedCell(1.0, 200, 30, 10, "shared", mode="oracle"),
        ]
        seeds = (100, 0)
        rates = (0.02,)
    jobs = [(cell, seed, rate) for cell in cells for seed in seeds for rate in rates]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "tools/research_linear_concept_pool"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    record = {
        "commit": commit,
        "tool_has_uncommitted_changes": bool(dirty),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cells": len(cells),
        "seeds": list(seeds),
        "tuning_seeds": [s for s in seeds if s in TUNING_SEEDS],
        "evaluation_seeds": [s for s in seeds if s in EVALUATION_SEEDS],
        "learning_rates": list(rates),
        "runs": len(jobs),
        "workers": arguments.workers,
    }
    started = time.time()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        all_rows = [row for rows in executor.map(run_one, jobs, chunksize=4) for row in rows]
    record["wall_seconds"] = time.time() - started
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in all_rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    (output / "run_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(jobs)} runs, {len(all_rows)} rows, {record['wall_seconds']:.0f}s -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
