"""線形回帰の概念の切り替えのシミュレーションを掃引し、runごと・方式と設定ごとの要約をCSVへ書く。

使い方:
    python -m tools.research_linear_concept_pool.run --output results/linear-concept-pool/<名前> --workers 14

調整用のseed（既定100〜104）と評価用のseed（既定0〜9）を両方流す。方式ごとの設定の選択は analyze で、
調整用のseedだけで行う。
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

from tools.research_linear_concept_pool.simulate import (
    SEGMENT_TYPE_NAMES,
    TIME_BIN_EDGES,
    TIME_BIN_NAMES,
    Cell,
    simulate,
)

SCALES = (0.3, 1.0, 3.0)
INTERVALS = (50, 200, 1000)
DELAYS = (0, 5, 20, 50)
MISASSIGNMENT_PROBABILITIES = (0.0, 0.2)
CLIENT_SETTINGS = ((10, "shared"), (10, "local"), (1, "local"))
TUNING_SEEDS = tuple(range(100, 105))
EVALUATION_SEEDS = tuple(range(10))


def planned_cells() -> list[Cell]:
    cells = []
    for scale, interval, delay, p, (clients, sharing) in itertools.product(
        SCALES, INTERVALS, DELAYS, MISASSIGNMENT_PROBABILITIES, CLIENT_SETTINGS
    ):
        if delay >= interval:
            continue
        cells.append(Cell(scale, interval, delay, p, clients, sharing))
    return cells


def _summaries(losses: np.ndarray, types: np.ndarray, time_bins: np.ndarray) -> list[float]:
    """全体、区間の種類ごと、変化からの経過ごとの平均（対象が無ければNaN）。"""
    values = [float(losses.mean())]
    for kind in range(len(SEGMENT_TYPE_NAMES)):
        selected = types == kind
        values.append(float(losses[selected].mean()) if selected.any() else float("nan"))
    for index in range(len(TIME_BIN_NAMES)):
        selected = time_bins == index
        values.append(float(losses[selected].mean()) if selected.any() else float("nan"))
    return values


def run_one(arguments: tuple[Cell, int]) -> list[dict]:
    cell, seed = arguments
    started = time.perf_counter()
    result = simulate(cell, seed)
    seconds = time.perf_counter() - started
    types = result["types"]
    time_bins = np.digitize(result["since"], TIME_BIN_EDGES[1:-1])
    concepts, assignment = result["concepts"], result["assignment"]
    base = {
        **{f"cell_{key}": value for key, value in asdict(cell).items()},
        "seed": seed,
        "seconds_all_methods": seconds,
        "switches_per_client": float((np.diff(concepts, axis=1) != 0).sum(axis=1).mean()),
        "misassigned_training_fraction": float((assignment != concepts).mean()),
        "mean_active_models": float(result["active_models"].mean()),
    }
    d = cell.dimension
    rows = []

    def add(method: str, hyper: dict, losses: np.ndarray, communication: tuple[float, float], prediction_mac: float):
        row = dict(base, method=method, **hyper)
        row["upload_parameters_per_client"] = communication[0]
        row["download_parameters_per_client"] = communication[1]
        row["prediction_mac_per_sample"] = prediction_mac
        row["training_mac_per_sample"] = 2.0 * d
        values = _summaries(losses, types, time_bins)
        row["loss"] = values[0]
        for name, value in zip(SEGMENT_TYPE_NAMES, values[1:5]):
            row[f"loss_type_{name}"] = value
        for name, value in zip(TIME_BIN_NAMES, values[5:]):
            row[f"loss_since_{name}"] = value
        rows.append(row)

    pool_communication = (float(result["pool_uploaded_parameters"]), float(result["pool_downloaded_parameters"]))
    single_value = float(result["single_parameters_each_way"])
    for e, rate in enumerate(result["pool_rates"]):
        add("hard", {"learning_rate": rate}, result["hard"][e], pool_communication, float(d))
        add("oracle", {"learning_rate": rate}, result["oracle"][e], pool_communication, float(d))
        for f, (mix_rate, share) in enumerate(zip(result["mix_rates"], result["share_rates"])):
            add(
                "mix",
                {"learning_rate": rate, "mixture_learning_rate": mix_rate, "share_rate": share},
                result["mix"][e, f],
                pool_communication,
                float(result["active_models"].mean() * d),
            )
    for s, rate in enumerate(result["single_rates"]):
        add("single", {"learning_rate": rate}, result["single"][s], (single_value, single_value), float(d))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="小さい計画（2 cell×2 seed）だけ流す")
    arguments = parser.parse_args(argv)
    output: Path = arguments.output
    if output.exists() and any(output.iterdir()):
        print(f"出力先が空ではない: {output}", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    cells = planned_cells()
    seeds = TUNING_SEEDS + EVALUATION_SEEDS
    if arguments.smoke:
        cells = [Cell(1.0, 200, 20, 0.2, 10, "shared"), Cell(1.0, 200, 0, 0.0, 1, "local")]
        seeds = (100, 0)
    jobs = [(cell, seed) for cell in cells for seed in seeds]
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
