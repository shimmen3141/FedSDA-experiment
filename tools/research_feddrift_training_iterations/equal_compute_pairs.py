"""同じ回数どうしのFedHASP−FedDriftを、seedで対にして集計する（3データ×間隔50×seed 0〜4）。

使い方: python -m tools.research_feddrift_training_iterations.equal_compute_pairs <FedHASPの既定の出力> <FedHASPの回数の出力> <FedDriftの回数の出力>
"""

import sys
from pathlib import Path
import statistics
from tools.research_false_alarm_control_alpha.analyze import read_configuration
B, T, D = map(Path, sys.argv[1:4])
pairs = {
    1: (B / "product-default/fedsda", D / "product-default/feddrift"),
    2: (T / "iterations-2/fedsda", D / "updates-2/feddrift"),
    4: (T / "iterations-4/fedsda", D / "updates-4/feddrift"),
}
lines = ["| データ | 回数 | 全体 FedHASP−FedDrift | 直後 | 安定 |", "|---|---|---|---|---|"]
for dataset in ("agrawal4", "randomrbf4", "stagger3"):
    for k, (h, f) in pairs.items():
        H, F = read_configuration(h, []), read_configuration(f, [])
        cells = []
        for key in ("overall", "post", "stable"):
            d = [H[(dataset, 50, s)][key] - F[(dataset, 50, s)][key] for s in range(5)]
            cells.append(f"{statistics.mean(d):+.2f}（{sum(x > 0 for x in d)}/5）")
        lines.append(f"| {dataset} | {k} | " + " | ".join(cells) + " |")
text = "\n".join(lines) + "\n"
(D / "analysis_equal_compute_pairs.md").write_text("同じ回数どうしのFedHASP−FedDrift（pp、seed 0〜4で対、括弧はFedHASPが上回ったseedの数）。FedDriftは手元で流したもの。\n\n" + text, encoding="utf-8", newline="\n")
sys.stdout.reconfigure(encoding="utf-8")
print(text)
