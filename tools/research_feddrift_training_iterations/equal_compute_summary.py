"""uraniumの本規模（13データ×間隔50・500×seed 0〜9）で、回数ごとの効果と、同じ回数どうしのFedHASP−FedDriftを集計する。

使い方: python -m tools.research_feddrift_training_iterations.equal_compute_summary <FedHASPの出力> <FedDriftの出力>
出力はFedHASPの出力の`analysis_equal_compute_all.md`。精度は全体（%）、差はpp・seedで対。
"""

import statistics
import sys
from pathlib import Path

from tools.research_false_alarm_control_alpha.analyze import read_configuration

DATASETS = (
    "sine2 circle2 sea2 sea4 mnist2 mnist4 fmnist2 fmnist4 rotmnist2 rotmnist4 stagger3 agrawal4 randomrbf4".split()
)
SEEDS = range(10)


def main():
    h, d = map(Path, sys.argv[1:3])
    sources = {
        "H1": read_configuration(h / "product-default/fedsda", []),
        "H2": read_configuration(h / "iterations-2/fedsda", []),
        "F1": read_configuration(d / "product-default/feddrift", []),
        "F2": read_configuration(d / "updates-2/feddrift", []),
    }
    lines = [
        "| データ | 間隔 | FedHASP 回数2−1 | FedDrift 回数2−1 | FedHASP−FedDrift（回数1） | FedHASP−FedDrift（回数2） |",
        "|---|---|---|---|---|---|",
    ]
    means = {50: [], 500: []}
    for interval in (50, 500):
        for dataset in DATASETS:
            cells, values = [], []
            for a, b in (("H2", "H1"), ("F2", "F1"), ("H1", "F1"), ("H2", "F2")):
                diff = [
                    sources[a][(dataset, interval, s)]["overall"] - sources[b][(dataset, interval, s)]["overall"]
                    for s in SEEDS
                ]
                values.append(statistics.mean(diff))
                cells.append(f"{statistics.mean(diff):+.2f}（{sum(x > 0 for x in diff)}/10）")
            means[interval].append(values)
            lines.append(f"| {dataset} | {interval} | " + " | ".join(cells) + " |")
        average = [statistics.mean(v[i] for v in means[interval]) for i in range(4)]
        lines.append(f"| 13データの平均 | {interval} | " + " | ".join(f"{x:+.2f}" for x in average) + " |")
    text = (
        "全体の精度の差（pp、seed 0〜9で対、括弧は前者が上回ったseedの数）。\n\n" + "\n".join(lines) + "\n"
    )
    (h / "analysis_equal_compute_all.md").write_text(text, encoding="utf-8", newline="\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
