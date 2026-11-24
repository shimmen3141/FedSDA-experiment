"""版の混合の仕組み（同期の版が良いのは、client間のばらつきが平均で消えるからか）を直接測る。

使い方:
    python -m tools.research_version_mixing.run_variance --output results/version-mixing/<名前> --workers 12

run.py と同じ掃引（モデル×client数×同期の間隔×学習率×seed 0〜9）を、5件ごとの計測つきで流す。
固定した1,000点の試験点で、確率予測のベイズ最適からのずれ（Brierの超過）を測り、
手元の版のずれを「clientの予測の平均のずれ」と「client間のばらつき」に分ける（恒等式）。

判断の材料（実験の前に記録した予測）:
- 分散で説明できるなら、(手元の版のずれ − 同期の版のずれ) の大部分を、client間のばらつきが占める。
- 残り（clientの予測の平均のずれ − 同期の版のずれ）は、同期の版が古いことと、パラメータの平均と予測の平均の違いの分。
- client間のばらつきは、同期からの件数・学習率とともに増える。
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

from tools.research_version_mixing.run import CLIENTS, INTERVALS, LEARNING_RATES, MODELS, SEEDS
from tools.research_version_mixing.simulate import VersionCell, simulate_versions

MEASURE_EVERY = 5
MEASURES = ("local_excess", "sync_excess", "local_variance", "local_mean_bias")


def run_one(arguments: tuple[VersionCell, int]) -> list[dict]:
    cell, seed = arguments
    started = time.perf_counter()
    records = simulate_versions(cell, seed, measure_every=MEASURE_EVERY)
    seconds = time.perf_counter() - started
    rows = []
    interval = cell.synchronization_interval
    since = records["measure_steps_since_sync"]
    # 同期の周期の中の位置を3つに分ける（前半の3分の1、中ほど、後ろ）と、全体。
    phases = {
        "all": np.ones_like(since, dtype=bool),
        "early": since <= interval / 3,
        "middle": (since > interval / 3) & (since <= 2 * interval / 3),
        "late": since > 2 * interval / 3,
    }
    for phase, selected in phases.items():
        row = {**asdict(cell), "seed": seed, "phase": phase, "seconds": seconds, "measurements": int(selected.sum())}
        for name in MEASURES:
            values = records[f"measure_{name}"][selected]
            row[name] = float(values.mean()) if len(values) else float("nan")
        rows.append(row)
    return rows


def render(rows: list[dict], record: dict) -> str:
    lines = ["# 版の混合の仕組みの計測（研究ツール `research_version_mixing` の `run_variance`）", ""]
    lines.append(
        f"- 入力: runs_variance.csv（commit `{record['commit']}`、ツールの未commitの変更 {record['tool_has_uncommitted_changes']}、"
        f"{record['runs']} run、{record['workers']} workers、壁時計 {record['wall_seconds']:.0f} 秒）。"
        f"5件ごとに、固定した1,000点の試験点で計測（学習の後・同期の前）。値はseed 0〜9と計測の時刻の平均。"
    )
    lines.append(
        "- 量（確率予測のベイズ最適からの2乗のずれ、×1000）: 手元＝手元の版、同期＝同期の版、ばらつき＝client間の分散、"
        "平均＝clientの予測の平均のずれ（手元＝平均＋ばらつき）。差＝手元−同期。ばらつきの割合＝ばらつき÷差。"
    )
    lines.append("")
    lines.append(
        "| モデル | client数 | 同期の間隔 | 学習率 | 手元 | 同期 | 差 | ばらつき | 平均−同期 | ばらつきの割合 | ばらつき 前・中・後 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    groups = {}
    for row in rows:
        key = (row["model"], int(row["clients"]), int(row["synchronization_interval"]), float(row["learning_rate"]))
        groups.setdefault(key, {}).setdefault(row["phase"], []).append(row)
    for key in sorted(groups):
        phase_rows = groups[key]

        def mean(phase: str, name: str, phase_rows: dict = phase_rows) -> float:
            return float(np.mean([r[name] for r in phase_rows[phase]])) * 1000

        local, sync = mean("all", "local_excess"), mean("all", "sync_excess")
        variance, bias = mean("all", "local_variance"), mean("all", "local_mean_bias")
        gap = local - sync
        share = f"{variance / gap:.2f}" if gap > 1e-9 else "—（差が0以下）"
        by_phase = " / ".join(f"{mean(phase, 'local_variance'):.2f}" for phase in ("early", "middle", "late"))
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]:g} | {local:.2f} | {sync:.2f} | {gap:+.2f} | {variance:.2f} | "
            f"{bias - sync:+.2f} | {share} | {by_phase} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args(argv)
    output: Path = arguments.output
    if output.exists() and any(output.iterdir()):
        print(f"出力先が空ではない: {output}", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    cells = [
        VersionCell(model, interval, rate, clients=clients)
        for model, clients, interval, rate in itertools.product(MODELS, CLIENTS, INTERVALS, LEARNING_RATES)
    ]
    jobs = [(cell, seed) for cell in cells for seed in SEEDS]
    record = {
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "tool_has_uncommitted_changes": bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", "tools/research_version_mixing"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        ),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "runs": len(jobs),
        "workers": arguments.workers,
    }
    started = time.time()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        rows = [row for result in executor.map(run_one, jobs) for row in result]
    record["wall_seconds"] = time.time() - started
    with (output / "runs_variance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "run_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "analysis_version_mixing_variance.md").write_text(render(rows, record), encoding="utf-8", newline="\n")
    print(f"{len(jobs)} runs, {record['wall_seconds']:.0f}s -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
