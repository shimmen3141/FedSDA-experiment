"""run.py の runs.csv を集計し、analysis_linear_concept_pool.md を入力のディレクトリへ書く。

設定の選び方: cellと方式ごとに、調整用のseedの平均損失が最小の設定（学習率、混合の学習率・共有率）を選び、
評価用のseedの値を使う。理論の照合の表では、遅れ・誤りの費用を同じ学習の集まりで見るため、
固い予測が選んだ学習率で、真の概念のモデルと混合（混合の設定は同じ学習率の中で選ぶ）を揃える。

使い方:
    python -m tools.research_linear_concept_pool.analyze results/linear-concept-pool/<名前>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.research_linear_concept_pool.simulate import SEGMENT_TYPE_NAMES, TIME_BIN_NAMES

CELL_KEYS = (
    "cell_scale",
    "cell_interval",
    "cell_delay",
    "cell_misassignment_probability",
    "cell_clients",
    "cell_sharing",
)
HYPER_KEYS = ("learning_rate", "mixture_learning_rate", "share_rate")
T_975_DF9 = 2.262


def _number(value: str) -> float:
    return float(value) if value not in ("", None) else float("nan")


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if key in ("method", "cell_sharing"):
                continue
            row[key] = _number(value)
    return rows


def cell_of(row: dict) -> tuple:
    return tuple(row[key] for key in CELL_KEYS)


def hyper_of(row: dict) -> tuple:
    return tuple(row.get(key, float("nan")) for key in HYPER_KEYS)


class Table:
    """(cell, method, hyper, seed) -> row。"""

    def __init__(self, rows: list[dict], tuning_seeds: set[int], evaluation_seeds: set[int]):
        self.rows = {}
        for row in rows:
            self.rows[(cell_of(row), row["method"], _key(hyper_of(row)), int(row["seed"]))] = row
        self.tuning_seeds = sorted(tuning_seeds)
        self.evaluation_seeds = sorted(evaluation_seeds)
        self.hypers = defaultdict(set)
        for cell, method, hyper, _ in self.rows:
            self.hypers[(cell, method)].add(hyper)
        self.cells = sorted({key[0] for key in self.rows}, key=lambda c: tuple(str(v) for v in c))

    def values(self, cell, method, hyper, seeds, column="loss") -> np.ndarray:
        return np.array([self.rows[(cell, method, hyper, s)][column] for s in seeds])

    def select(self, cell, method, learning_rate: float | None = None) -> tuple:
        candidates = self.hypers[(cell, method)]
        if learning_rate is not None:
            candidates = {h for h in candidates if h[0] == learning_rate}
        return min(candidates, key=lambda h: self.values(cell, method, h, self.tuning_seeds).mean())

    def evaluation(self, cell, method, hyper, column="loss") -> np.ndarray:
        return self.values(cell, method, hyper, self.evaluation_seeds, column)


def _key(hyper: tuple) -> tuple:
    return tuple(None if (isinstance(v, float) and math.isnan(v)) else v for v in hyper)


def paired(a: np.ndarray, b: np.ndarray) -> str:
    difference = a - b
    half = T_975_DF9 * difference.std(ddof=1) / math.sqrt(len(difference))
    wins = int((difference < 0).sum())
    return f"{difference.mean():+.3f} [{difference.mean() - half:+.3f}, {difference.mean() + half:+.3f}]（{wins}/{len(difference)}）"


def _cell(scale, interval, delay, p, clients, sharing):
    return (float(scale), float(interval), float(delay), float(p), float(clients), sharing)


def render(table: Table, record: dict) -> str:
    lines = ["# 線形回帰の概念の切り替えでの、予測の方式と共有の比較（研究ツール `research_linear_concept_pool`）", ""]
    lines.append(
        f"- 入力: runs.csv（commit `{record.get('commit', '?')}`、ツールの未commitの変更 {record.get('tool_has_uncommitted_changes')}、"
        f"{record.get('runs')} run、{record.get('workers')} workers、壁時計 {record.get('wall_seconds', 0):.0f} 秒、numpy {record.get('numpy')}）。"
    )
    lines.append(
        f"- 設定の選択は調整用のseed {table.tuning_seeds}、評価は {table.evaluation_seeds}。値は評価用のseedの平均損失（二乗誤差。"
        "雑音の分散1が下限）。差は対にした差の平均と95%区間（t分布、自由度9。多重比較の補正なし）、括弧は左が小さいseedの数。"
    )
    lines.append("- 分類ではないので精度は無い。損失で比べる。")
    lines.append("")

    scales = sorted({c[0] for c in table.cells})
    intervals = sorted({c[1] for c in table.cells})
    delays = sorted({c[2] for c in table.cells})
    probabilities = sorted({c[3] for c in table.cells})

    # 1. 主な比較（10 client、共有あり）
    lines.append("## 1. 方式の比較（10 client、モデルを共有、方式ごとに設定を選ぶ）")
    lines.append("")
    lines.append(
        "| s | 間隔 | 遅れ | 誤り率 | 固い予測 | 混合 | 真の概念のモデル | 単一モデル | 混合−固い予測 | 単一−固い予測 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for scale in scales:
        for interval in intervals:
            for delay in delays:
                for p in probabilities:
                    cell = _cell(scale, interval, delay, p, 10, "shared")
                    if (cell, "hard") not in table.hypers:
                        continue
                    values = {
                        m: table.evaluation(cell, m, table.select(cell, m)) for m in ("hard", "mix", "oracle", "single")
                    }
                    lines.append(
                        f"| {scale:g} | {interval:g} | {delay:g} | {p:g} | {values['hard'].mean():.3f} | {values['mix'].mean():.3f} | "
                        f"{values['oracle'].mean():.3f} | {values['single'].mean():.3f} | {paired(values['mix'], values['hard'])} | "
                        f"{paired(values['single'], values['hard'])} |"
                    )
    lines.append("")

    # 2. 理論の照合（遅れ・誤りの費用）
    lines.append("## 2. 遅れ・誤りの費用（同じ学習の集まり。固い予測の学習率に揃える）")
    lines.append("")
    lines.append(
        "予測: 固い予測−真の概念のモデル ≈ 2s²×（遅れの割合＋誤った割り当ての割合）。混合−真の概念のモデルは、"
        "切り替え1回あたりの定数（Fixed-Shareの保証の形）で、遅れ・誤りに依らない。"
    )
    lines.append("")
    lines.append(
        "| s | 間隔 | 遅れ | 誤り率 | 誤った学習の割合 | 予測 2s²×割合 | 固い−真 | 混合−真 | 混合−真（遅れ0・誤り0） | 真の概念のモデル−（遅れ0・誤り0） |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for scale in scales:
        for interval in intervals:
            base_cell = _cell(scale, interval, 0, 0, 10, "shared")
            base_rate = table.select(base_cell, "hard")[0]
            base_oracle = table.evaluation(base_cell, "oracle", (base_rate, None, None))
            base_mix = table.evaluation(base_cell, "mix", table.select(base_cell, "mix", base_rate))
            for delay in delays:
                for p in probabilities:
                    cell = _cell(scale, interval, delay, p, 10, "shared")
                    if (cell, "hard") not in table.hypers:
                        continue
                    rate = table.select(cell, "hard")[0]
                    hard = table.evaluation(cell, "hard", (rate, None, None))
                    oracle = table.evaluation(cell, "oracle", (rate, None, None))
                    mix = table.evaluation(cell, "mix", table.select(cell, "mix", rate))
                    fraction = table.evaluation(
                        cell, "hard", (rate, None, None), "misassigned_training_fraction"
                    ).mean()
                    lines.append(
                        f"| {scale:g} | {interval:g} | {delay:g} | {p:g} | {fraction:.3f} | {2 * scale**2 * fraction:.3f} | "
                        f"{(hard - oracle).mean():+.3f} | {(mix - oracle).mean():+.3f} | {(base_mix - base_oracle).mean():+.3f} | "
                        f"{(oracle - base_oracle).mean():+.3f} |"
                    )
    lines.append("")

    # 2b. 入れ替わりの予測
    lines.append("## 2b. 混合が固い予測を上回る遅れの予測と実際（誤り0）")
    lines.append("")
    lines.append(
        "C＝遅れ0・誤り0での（混合−真の概念のモデル）×（標本数÷clientあたりの切り替えの回数）。切り替え1回あたりの混合の費用の実測。"
        "補題A・Bから、遅れdの1回あたりの固い予測の費用は d×2s² なので、d* ＝ C／(2s²) を超える遅れで混合が上回ると予測する。"
        "実際は、方式ごとに設定を選んだ表1で、混合−固い予測が負（区間の上端が0未満）になる最小の遅れ。"
    )
    lines.append("")
    lines.append("| s | 間隔 | C | 予測 d* | 遅れ0 | 遅れ5 | 遅れ20 | 遅れ50 | 実際に上回る最小の遅れ |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for scale in scales:
        for interval in intervals:
            base_cell = _cell(scale, interval, 0, 0, 10, "shared")
            base_rate = table.select(base_cell, "hard")[0]
            oracle = table.evaluation(base_cell, "oracle", (base_rate, None, None))
            mix = table.evaluation(base_cell, "mix", table.select(base_cell, "mix", base_rate))
            switches = table.evaluation(base_cell, "oracle", (base_rate, None, None), "switches_per_client").mean()
            samples = 3000.0
            overhead = (mix - oracle).mean() * samples / switches
            predicted = overhead / (2 * scale**2)
            cells, first = [], "—"
            for delay in delays:
                cell = _cell(scale, interval, delay, 0, 10, "shared")
                if (cell, "hard") not in table.hypers:
                    cells.append("—")
                    continue
                difference = table.evaluation(cell, "mix", table.select(cell, "mix")) - table.evaluation(
                    cell, "hard", table.select(cell, "hard")
                )
                half = T_975_DF9 * difference.std(ddof=1) / math.sqrt(len(difference))
                cells.append(f"{difference.mean():+.3f}")
                if first == "—" and difference.mean() + half < 0:
                    first = f"{delay:g}"
            lines.append(
                f"| {scale:g} | {interval:g} | {overhead:.1f} | {predicted:.1f} | "
                + " | ".join(cells)
                + f" | {first} |"
            )
    lines.append("")
    single_worse = []
    for cell in table.cells:
        if (cell, "mix") not in table.hypers:
            continue
        difference = table.evaluation(cell, "single", table.select(cell, "single")) - table.evaluation(
            cell, "mix", table.select(cell, "mix")
        )
        single_worse.append(difference.mean() > 0)
    lines.append(
        f"- 単一モデルが混合より損失が大きいcell: {sum(single_worse)}/{len(single_worse)}（全ての設定、評価用seedの平均）。"
    )
    lines.append("")

    # 3. 共有の効果（区間の種類ごと）
    lines.append("## 3. モデルの共有の効果（10 client、共有あり−共有なし。区間の種類ごと）")
    lines.append("")
    lines.append(
        "予測: 共有の利得は、そのclientには初めてで他のclientが経験済みの区間（seen_by_others）に集中する。"
        "全体で初めて（globally_new）の区間では小さい。"
    )
    lines.append("")
    header = "| s | 間隔 | 遅れ | 誤り率 | 方式 | " + " | ".join(SEGMENT_TYPE_NAMES) + " | 全体 |"
    lines.append(header)
    lines.append("|" + "---|" * (6 + len(SEGMENT_TYPE_NAMES)))
    for scale in scales:
        for interval in intervals:
            for delay, p in ((0.0, 0.0), (20.0, 0.2)):
                shared = _cell(scale, interval, delay, p, 10, "shared")
                local = _cell(scale, interval, delay, p, 10, "local")
                if (shared, "hard") not in table.hypers:
                    continue
                for method in ("hard", "mix"):
                    hs, hl = table.select(shared, method), table.select(local, method)
                    cells = []
                    for name in SEGMENT_TYPE_NAMES + ("",):
                        column = f"loss_type_{name}" if name else "loss"
                        a = table.evaluation(shared, method, hs, column)
                        b = table.evaluation(local, method, hl, column)
                        cells.append("—" if np.isnan(a).any() or np.isnan(b).any() else f"{(a - b).mean():+.3f}")
                    lines.append(
                        f"| {scale:g} | {interval:g} | {delay:g} | {p:g} | {method} | " + " | ".join(cells) + " |"
                    )
    lines.append("")

    # 4. 変化からの経過
    lines.append("## 4. 変化からの経過ごとの損失（s=1、間隔200、10 client、共有あり）")
    lines.append("")
    lines.append("| 遅れ | 誤り率 | 方式 | " + " | ".join(TIME_BIN_NAMES) + " |")
    lines.append("|" + "---|" * (3 + len(TIME_BIN_NAMES)))
    for delay in delays:
        for p in probabilities:
            cell = _cell(1.0, 200, delay, p, 10, "shared")
            if (cell, "hard") not in table.hypers:
                continue
            for method in ("hard", "mix", "oracle", "single"):
                hyper = table.select(cell, method)
                values = [table.evaluation(cell, method, hyper, f"loss_since_{name}").mean() for name in TIME_BIN_NAMES]
                lines.append(f"| {delay:g} | {p:g} | {method} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    lines.append("")

    # 5. 標準指標
    lines.append("## 5. 標準指標（10 client・共有あり・s=1・間隔200・遅れ0・誤り0、1 clientあたり）")
    lines.append("")
    cell = _cell(1.0, 200, 0, 0, 10, "shared")
    lines.append("| 方式 | 損失 | 上り（浮動小数点数） | 下り | 予測のMAC/標本 | 学習のMAC/標本 |")
    lines.append("|---|---|---|---|---|---|")
    for method in ("hard", "mix", "single"):
        hyper = table.select(cell, method)

        def get(column: str, method: str = method, hyper: tuple = hyper) -> float:
            return float(table.evaluation(cell, method, hyper, column).mean())

        lines.append(
            f"| {method} | {get('loss'):.3f} | {get('upload_parameters_per_client'):.0f} | {get('download_parameters_per_client'):.0f} | "
            f"{get('prediction_mac_per_sample'):.1f} | {get('training_mac_per_sample'):.0f} |"
        )
    seconds = np.mean([row["seconds_all_methods"] for row in table.rows.values()])
    lines.append("")
    lines.append(
        f"- 経過時間: 1 runで全方式・全設定を同時に計算するので、方式ごとには分けられない（1 runの平均 {seconds:.2f} 秒）。"
        "MACは線形モデルの内積（d=10）で数え、混合は保有モデル全部の予測を数える。"
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args(argv)
    record = json.loads((arguments.directory / "run_record.json").read_text(encoding="utf-8"))
    rows = load(arguments.directory / "runs.csv")
    table = Table(rows, set(record["tuning_seeds"]), set(record["evaluation_seeds"]))
    text = render(table, record)
    (arguments.directory / "analysis_linear_concept_pool.md").write_text(text, encoding="utf-8", newline="\n")
    print(f"書いた: {arguments.directory / 'analysis_linear_concept_pool.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
