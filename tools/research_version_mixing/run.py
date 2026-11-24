"""版の混合のシミュレーションを掃引し、runごとの要約をCSVへ、集計を analysis_version_mixing.md へ書く。

使い方:
    python -m tools.research_version_mixing.run --output results/version-mixing/<名前> --workers 12

軸: モデル（線形、MLP）× client数（2、10）× 同期の間隔（10、50、200）× 学習率（0.001、0.01）× seed 0〜9。
全標本（3000件/client）と、事前学習の直後の100件を除いた標本の、正解率と対数損失を出す。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tools.research_version_mixing.simulate import VersionCell, simulate_versions

MODELS = ("linear", "mlp")
CLIENTS = (2, 10)
INTERVALS = (10, 50, 200)
LEARNING_RATES = (0.001, 0.01)
SEEDS = tuple(range(10))
VERSIONS = ("sync", "local", "mix", "average")
T_975_DF9 = 2.262


def run_one(arguments: tuple[VersionCell, int]) -> dict:
    cell, seed = arguments
    started = time.perf_counter()
    records = simulate_versions(cell, seed)
    row = {**asdict(cell), "seed": seed, "seconds": time.perf_counter() - started}
    for version in VERSIONS:
        row[f"accuracy_{version}"] = float(records[f"correct_{version}"].mean() * 100)
        row[f"log_loss_{version}"] = float(records[f"loss_{version}"].mean())
        row[f"accuracy_after100_{version}"] = float(records[f"correct_{version}"][:, 100:].mean() * 100)
    row["mean_weight_local"] = float(records["weight_local"].mean())
    return row


def paired(a: np.ndarray, b: np.ndarray) -> str:
    difference = a - b
    half = T_975_DF9 * difference.std(ddof=1) / math.sqrt(len(difference))
    return f"{difference.mean():+.2f} [{difference.mean() - half:+.2f}, {difference.mean() + half:+.2f}]"


def render(rows: list[dict], record: dict) -> str:
    lines = ["# 同期の版と手元の版の混合（研究ツール `research_version_mixing`）", ""]
    lines.append(
        f"- 入力: runs.csv（commit `{record['commit']}`、ツールの未commitの変更 {record['tool_has_uncommitted_changes']}、"
        f"{record['runs']} run、{record['workers']} workers、壁時計 {record['wall_seconds']:.0f} 秒、numpy {record['numpy']}）。"
    )
    lines.append("- 値はseed 0〜9の平均。差は対にした差と95%区間（多重比較の補正なし）。正解率は%、差はpp。")
    lines.append("")
    lines.append(
        "| モデル | client数 | 同期の間隔 | 学習率 | 同期の版 | 手元の版 | 混合 | 固定平均 | 混合−手元 | 混合−同期 | 同期−手元 | 手元の重み |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    groups = {}
    for row in rows:
        key = (row["model"], int(row["clients"]), int(row["synchronization_interval"]), float(row["learning_rate"]))
        groups.setdefault(key, []).append(row)
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: r["seed"])
        value = {v: np.array([r[f"accuracy_{v}"] for r in group]) for v in VERSIONS}
        weight = np.mean([r["mean_weight_local"] for r in group])
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]:g} | {value['sync'].mean():.2f} | {value['local'].mean():.2f} | "
            f"{value['mix'].mean():.2f} | {value['average'].mean():.2f} | {paired(value['mix'], value['local'])} | "
            f"{paired(value['mix'], value['sync'])} | {paired(value['sync'], value['local'])} | {weight:.2f} |"
        )
    lines.append("")
    lines.append("## 対数損失（同じ順、平均）")
    lines.append("")
    lines.append("| モデル | client数 | 同期の間隔 | 学習率 | 同期の版 | 手元の版 | 混合 | 固定平均 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for key in sorted(groups):
        group = groups[key]
        value = {v: np.mean([r[f"log_loss_{v}"] for r in group]) for v in VERSIONS}
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]:g} | {value['sync']:.4f} | {value['local']:.4f} | "
            f"{value['mix']:.4f} | {value['average']:.4f} |"
        )
    lines.append("")
    seconds = {m: np.mean([r["seconds"] for r in rows if r["model"] == m]) for m in MODELS}
    lines.append(
        "- 標準指標: 通信は同期1回につき1 clientあたりパラメータ1組の上り・下り（線形22、MLP 1,474浮動小数点数）で、"
        "予測の版の選び方によらず同じ。予測のMAC/標本は、混合と固定平均が2つの版の計算で、1つの版だけの2倍。"
        f"経過時間（1 runの平均、全方式を同時に計算）: 線形 {seconds['linear']:.1f} 秒、MLP {seconds['mlp']:.1f} 秒。"
    )
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
        rows = list(executor.map(run_one, jobs))
    record["wall_seconds"] = time.time() - started
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "run_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "analysis_version_mixing.md").write_text(render(rows, record), encoding="utf-8", newline="\n")
    print(f"{len(jobs)} runs, {record['wall_seconds']:.0f}s -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
