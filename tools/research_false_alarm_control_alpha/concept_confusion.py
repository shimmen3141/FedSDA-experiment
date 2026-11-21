"""真の概念ごとに、学習先の系統の概念の割合を数える（agrawal4・間隔50・seed 0〜4）。

使い方: python -m tools.research_false_alarm_control_alpha.concept_confusion <名前> <構成のフォルダ[@列=値]> [...]
"""

import collections
import sys
from pathlib import Path

from tools.research_false_alarm_control_alpha.assignment_destinations import _runs, analyze_run

for name, spec in zip(sys.argv[1::2], sys.argv[2::2]):
    directory, _, condition = spec.partition("@")
    row_filter = tuple(condition.split("=", 1)) if condition else None
    total = collections.Counter()
    for key, a in _runs(Path(directory), ["agrawal4"], [50], range(5), row_filter):
        total.update(analyze_run(a, 50)["confusion"])
    per_true = collections.Counter()
    for (t, l), v in total.items():
        per_true[t] += v
    print(name, "真の概念→学習先の概念（%、真の概念ごと）")
    for t in sorted(per_true):
        print("  ", t, {l: round(100 * total[(t, l)] / per_true[t], 1) for l in sorted({l for (tt, l) in total if tt == t})})
