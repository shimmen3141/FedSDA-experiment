"""run_estimated.py の runs.csv を集計し、analysis_estimated_assignment.md を入力のディレクトリへ書く。

設定の選び方: cellと方式ごとに、調整用のseedの平均損失が最小の設定（学習率、混合は混合の学習率・共有率も）を選び、
評価用のseedの値を使う。理論の予測（P1〜P4）は、実験の前に研究記録へ書いたもの。

使い方:
    python -m tools.research_linear_concept_pool.analyze_estimated results/linear-concept-pool/<名前>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.research_linear_concept_pool.simulate import SEGMENT_TYPE_NAMES

T_975_DF9 = 2.262
CELL_KEYS = ("cell_mode", "cell_scale", "cell_interval", "cell_window", "cell_pending", "cell_sharing")
TEXT_KEYS = ("method", "cell_mode", "cell_sharing")
SAMPLES_PER_CLIENT = 3000.0
THRESHOLD_Z = 3.0


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if key in TEXT_KEYS:
                continue
            row[key] = float(value) if value not in ("", None) else float("nan")
    return rows


def _hyper(row: dict) -> tuple:
    def value(key):
        v = row.get(key, float("nan"))
        return None if isinstance(v, float) and math.isnan(v) else v

    return (row["learning_rate"], value("mixture_learning_rate"), value("share_rate"))


class Table:
    def __init__(self, rows: list[dict], tuning: list[int], evaluation: list[int]):
        self.rows = {}
        self.hypers = defaultdict(set)
        for row in rows:
            cell = tuple(row[k] for k in CELL_KEYS)
            hyper = _hyper(row)
            self.rows[(cell, row["method"], hyper, int(row["seed"]))] = row
            self.hypers[(cell, row["method"])].add(hyper)
        self.tuning = tuning
        self.evaluation_seeds = evaluation
        self._selected = {}

    def select(self, cell, method):
        key = (cell, method)
        if key not in self._selected:
            self._selected[key] = min(
                self.hypers[key],
                key=lambda h: np.mean([self.rows[(cell, method, h, s)]["loss"] for s in self.tuning]),
            )
        return self._selected[key]

    def has(self, cell, method="hard") -> bool:
        return (cell, method) in self.hypers

    def values(self, cell, method, column="loss") -> np.ndarray:
        hyper = self.select(cell, method)
        return np.array([self.rows[(cell, method, hyper, s)][column] for s in self.evaluation_seeds])


def estimated(scale, interval, window, pending, sharing="shared"):
    return ("estimated", float(scale), float(interval), float(window), float(pending), sharing)


def oracle(scale, interval, sharing="shared"):
    return ("oracle", float(scale), float(interval), 30.0, 0.0, sharing)


def paired(a: np.ndarray, b: np.ndarray) -> str:
    difference = a - b
    half = T_975_DF9 * difference.std(ddof=1) / math.sqrt(len(difference))
    return f"{difference.mean():+.3f} [{difference.mean() - half:+.3f}, {difference.mean() + half:+.3f}]"


def predicted_window(scale: float) -> float:
    """P1: 概念を分けられる窓の下限 w* = 2 z² σ⁴ / Δ²（Δ = 2s²、σ = 1）。"""
    return 2 * THRESHOLD_Z**2 / (2 * scale**2) ** 2


def predicted_recurring_delay(scale: float, window: float, pending: float) -> float:
    """P2: 再来の概念の保有モデルへの切り替えの遅れ max(w/2, w − 1.5√(2w)/s²) − P（0未満は0）。"""
    switch = max(window / 2, window - THRESHOLD_Z * math.sqrt(2 * window) / (2 * scale**2))
    return max(switch - pending, 0.0)


def render(table: Table, record: dict) -> str:
    scales = (0.3, 0.5, 1.0, 3.0)
    intervals = (50, 200, 1000)
    windows = (10, 30, 100, 300)
    lines = ["# 割り当てを推定する版の結果（研究ツール `research_linear_concept_pool`）", ""]
    lines.append(
        f"- 入力: runs.csv（commit `{record.get('commit')}`、ツールの未commitの変更 {record.get('tool_has_uncommitted_changes')}、"
        f"{record.get('runs')} run、{record.get('workers')} workers、壁時計 {record.get('wall_seconds', 0):.0f} 秒、numpy {record.get('numpy')}）。"
    )
    lines.append(
        f"- 設定（学習率、混合の学習率・共有率）は調整用のseed {table.tuning} でcellと方式ごとに選び、評価は {table.evaluation_seeds}。"
        "値は評価用seedの平均。損失は二乗誤差（雑音の分散1が下限）。差は対にした差と95%区間（多重比較の補正なし）。"
    )
    lines.append("- 保留Pは0とw/2（表では「保留」）。10 client。")
    lines.append("")

    # P1
    lines.append("## P1. 概念を分けられる窓（共有あり、間隔1000、保留w/2）")
    lines.append("")
    lines.append(
        "予測: w > w* = 4.5/s⁴ のときだけ4つの概念を分けられる。分けられないと、誤った学習の割合は約0.75（全部が1つ）へ近づく。"
    )
    lines.append("")
    lines.append("| s | w* | " + " | ".join(f"w={w}: 誤り / 使ったモデル数" for w in windows) + " |")
    lines.append("|---|---|" + "---|" * len(windows))
    for scale in scales:
        cells = []
        for window in windows:
            cell = estimated(scale, 1000, window, window // 2)
            mis = table.values(cell, "hard", "misassigned_training_fraction").mean()
            used = table.values(cell, "hard", "distinct_models_used").mean()
            cells.append(f"{mis:.3f} / {used:.1f}")
        lines.append(f"| {scale:g} | {predicted_window(scale):.1f} | " + " | ".join(cells) + " |")
    lines.append("")

    # P2
    lines.append("## P2. 切り替えの遅れ（共有あり、間隔1000。実測は全ての変化の平均、予測は再来の概念への切り替え）")
    lines.append("")
    lines.append("| s | 保留 | " + " | ".join(f"w={w}: 実測 / 予測" for w in windows) + " |")
    lines.append("|---|---|" + "---|" * len(windows))
    for scale in scales:
        for half in (False, True):
            cells = []
            for window in windows:
                pending = window // 2 if half else 0
                cell = estimated(scale, 1000, window, pending)
                observed = table.values(cell, "hard", "median_assignment_delay").mean()
                cells.append(f"{observed:.1f} / {predicted_recurring_delay(scale, window, pending):.1f}")
            lines.append(f"| {scale:g} | {'w/2' if half else '0'} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "- 実測は、各変化からの、学習の割り当てが真の概念に合うまでの標本数の中央値（seedの平均）。新しい概念（モデルを作る）も含む。"
    )
    lines.append("")

    # P3
    lines.append("## P3. 混合−固い予測（共有あり）と、遅れの費用・混合の費用の比較")
    lines.append("")
    lines.append(
        "C＝oracle（真の概念で割り当て、遅れ0）の（混合−固い予測）×（標本数÷切り替えの回数）。遅れの費用＝実測の平均の遅れ×2s²。"
        "予測: 遅れの費用＞Cなら混合が良い（誤りの分を除いた粗い予測）。"
    )
    lines.append("")
    lines.append("| s | 間隔 | C | w | 保留 | 平均の遅れ | 遅れの費用 | 予測 | 混合−固い予測 | 合うか |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    agreements = []
    for scale in scales:
        for interval in intervals:
            base = oracle(scale, interval)
            switches = SAMPLES_PER_CLIENT / interval
            overhead = (table.values(base, "mix") - table.values(base, "hard")).mean() * SAMPLES_PER_CLIENT / switches
            for window in windows:
                for half in (False, True):
                    pending = window // 2 if half else 0
                    cell = estimated(scale, interval, window, pending)
                    delay = table.values(cell, "hard", "mean_assignment_delay").mean()
                    cost = delay * 2 * scale**2
                    prediction = "混合" if cost > overhead else "固い"
                    difference = table.values(cell, "mix") - table.values(cell, "hard")
                    half_width = T_975_DF9 * difference.std(ddof=1) / math.sqrt(len(difference))
                    if difference.mean() + half_width < 0:
                        observed = "混合"
                    elif difference.mean() - half_width > 0:
                        observed = "固い"
                    else:
                        observed = "差なし"
                    agree = "—" if observed == "差なし" else ("合う" if observed == prediction else "合わない")
                    agreements.append(agree)
                    lines.append(
                        f"| {scale:g} | {interval} | {overhead:.1f} | {window} | {'w/2' if half else '0'} | {delay:.1f} | {cost:.1f} | "
                        f"{prediction} | {paired(table.values(cell, 'mix'), table.values(cell, 'hard'))} | {agree} |"
                    )
    lines.append("")
    lines.append(
        f"- 合う {agreements.count('合う')}、合わない {agreements.count('合わない')}、差なし（区間が0をまたぐ） {agreements.count('—')}。"
    )
    lines.append("")

    # P4
    lines.append("## P4. 最適な窓（共有あり。評価用seedの平均損失が最小のw）")
    lines.append("")
    lines.append("予測: 混合の最適な窓は、固い予測の最適な窓以上（遅れの費用が安く、長い窓で誤りを減らせるため）。")
    lines.append("")
    lines.append(
        "| s | 間隔 | 保留 | 固い予測の最適w（損失） | 混合の最適w（損失） | oracle 固い / 混合 | 単一モデル |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for scale in scales:
        for interval in intervals:
            for half in (False, True):
                best = {}
                for method in ("hard", "mix"):
                    losses = {
                        w: table.values(estimated(scale, interval, w, w // 2 if half else 0), method).mean()
                        for w in windows
                    }
                    best_window = min(losses, key=losses.get)
                    best[method] = (best_window, losses[best_window])
                base = oracle(scale, interval)
                single = table.values(estimated(scale, interval, 30, 15 if half else 0), "single").mean()
                lines.append(
                    f"| {scale:g} | {interval} | {'w/2' if half else '0'} | {best['hard'][0]}（{best['hard'][1]:.3f}） | "
                    f"{best['mix'][0]}（{best['mix'][1]:.3f}） | {table.values(base, 'hard').mean():.3f} / "
                    f"{table.values(base, 'mix').mean():.3f} | {single:.3f} |"
                )
    lines.append("")

    # 共有
    lines.append("## 共有の効果（共有あり−共有なし、w=30・保留15、区間の種類ごと）")
    lines.append("")
    lines.append("| s | 間隔 | 方式 | " + " | ".join(SEGMENT_TYPE_NAMES) + " | 全体 |")
    lines.append("|" + "---|" * (4 + len(SEGMENT_TYPE_NAMES)))
    for scale in scales:
        for interval in intervals:
            shared = estimated(scale, interval, 30, 15, "shared")
            local = estimated(scale, interval, 30, 15, "local")
            for method in ("hard", "mix"):
                cells = []
                for name in (*SEGMENT_TYPE_NAMES, ""):
                    column = f"loss_type_{name}" if name else "loss"
                    a, b = table.values(shared, method, column), table.values(local, method, column)
                    cells.append("—" if np.isnan(a).any() or np.isnan(b).any() else f"{(a - b).mean():+.3f}")
                lines.append(f"| {scale:g} | {interval} | {method} | " + " | ".join(cells) + " |")
    lines.append("")

    # 標準指標
    lines.append("## 標準指標（s=1・間隔200・w=30・保留15、1 clientあたり）")
    lines.append("")
    lines.append("| 構成 | 方式 | 損失 | 上り | 下り | 予測のMAC/標本 | 割り当ての判断のMAC/標本 | 学習のMAC/標本 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for label, cell in (
        ("推定・共有あり", estimated(1, 200, 30, 15, "shared")),
        ("推定・共有なし", estimated(1, 200, 30, 15, "local")),
        ("oracle・共有あり", oracle(1, 200, "shared")),
    ):
        for method in ("hard", "mix", "single"):

            def get(column: str, cell: tuple = cell, method: str = method) -> float:
                return float(table.values(cell, method, column).mean())

            decision = get("decision_mac_per_sample") if method != "single" else 0.0
            lines.append(
                f"| {label} | {method} | {get('loss'):.3f} | {get('upload_parameters_per_client'):.0f} | "
                f"{get('download_parameters_per_client'):.0f} | {get('prediction_mac_per_sample'):.1f} | {decision:.0f} | "
                f"{get('training_mac_per_sample'):.0f} |"
            )
    seconds = np.mean([row["seconds_all_methods"] for row in table.rows.values()])
    lines.append("")
    lines.append(
        f"- 経過時間: 1 runで固い予測・混合・単一モデルを同時に計算するので、方式ごとには分けられない（1 runの平均 {seconds:.2f} 秒）。"
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args(argv)
    record = json.loads((arguments.directory / "run_record.json").read_text(encoding="utf-8"))
    table = Table(load(arguments.directory / "runs.csv"), record["tuning_seeds"], record["evaluation_seeds"])
    output = arguments.directory / "analysis_estimated_assignment.md"
    output.write_text(render(table, record), encoding="utf-8", newline="\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
