"""既存の run の記録（標本ごとの正誤）から、重みが最大の1モデルの予測と、混合の予測を、概念の変化からの経過で分けて比べる
（再実験なし。基準版16の既定の結果を読む）。

比べる予測（どれも同じ run の同じ標本、同じ同期の版のモデル）:
- `max_weight`: 予測の重み（既定では Fixed Share）が最大の1モデル（`maximum_weight_model_prediction_is_correct`）。
- `fixed_share`: Fixed Share の重みでの同期の版の混合（`fixed_share_weighted_prediction_is_correct`）。
- `adahedge`: AdaHedge の重みでの同期の版の混合（診断）。
- `product`: 実際の予測（同期の版と手元の版を混ぜたもの、`combined_prediction_is_correct`）。
期間: 各clientで最初の概念の変化の前（初期）、変化からの経過 [0,50)・[50,100)・[100,200)・[200,500)・500以上。
変化が続くときは、直前の変化からの経過で数える。保有モデルが2つ以上の標本だけの集計も併記する（1つなら全部の予測が同じ）。
"""

import argparse
import io
import re
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

PREDICTIONS = {
    "max_weight": "sample_prediction.maximum_weight_model_prediction_is_correct",
    "fixed_share": "sample_prediction.fixed_share_weighted_prediction_is_correct",
    "adahedge": "sample_prediction.adahedge_weighted_prediction_is_correct",
    "product": "sample_prediction.combined_prediction_is_correct",
}
PERIOD_EDGES = (0, 50, 100, 200, 500)
PERIOD_NAMES = ("初期", "0-50", "50-100", "100-200", "200-500", "500以上")


def _period_index(table):
    shape = table[PREDICTIONS["product"]].shape
    since = np.full(shape, -1, dtype=np.int64)
    sample_indices = np.arange(shape[1])
    for client_id, start in sorted(
        zip(table["concept_change.client_id"], table["concept_change.first_new_concept_sample_index"], strict=True),
        key=lambda pair: pair[1],
    ):
        after = sample_indices >= int(start)
        since[int(client_id), after] = sample_indices[after] - int(start)
    period = np.zeros(shape, dtype=np.int64)
    for offset, edge in enumerate(PERIOD_EDGES):
        period[since >= edge] = offset + 1
    return period


def _iterate_tables(archive):
    with zipfile.ZipFile(archive) as packed:
        for name in sorted(packed.namelist()):
            if name.endswith(".npz"):
                yield Path(name).stem, dict(np.load(io.BytesIO(packed.read(name))))


def analyze(archives, output):
    # counts[(dataset, interval)][subset][period][prediction] = [正解数, 標本数] を seed ごとに
    per_seed = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    run_count = 0
    for archive in archives:
        for run_identifier, table in _iterate_tables(archive):
            found = re.match(r"(.+)_interval(\d+)_seed(\d+)_", run_identifier)
            condition = (found.group(1), int(found.group(2)))
            period = _period_index(table)
            multiple = table["sample_prediction.held_model_count_at_prediction"] >= 2
            run_count += 1
            for subset, subset_mask in (("全標本", np.ones_like(multiple)), ("保有2以上", multiple)):
                for offset, name in enumerate(PERIOD_NAMES):
                    mask = (period == offset) & subset_mask
                    count = int(mask.sum())
                    for prediction, column in PREDICTIONS.items():
                        per_seed[condition][subset][name][prediction].append(
                            (int(table[column][mask].sum()), count)
                        )
    lines = [
        "# 重みが最大の1モデルと混合の、概念の変化からの経過で分けた比較（既存の結果、再実験なし）",
        "",
        f"- 入力（{run_count} run）: " + "、".join(f"`{archive}`" for archive in archives),
        "- 差は pp（予測A − 予測B の正解率）。seed ごとの標本の正解率の差を seed で平均し、条件の等重みで平均する。"
        "期間に標本の無い seed は除く。標本の割合は、その期間の標本数の全標本に対する割合（条件の等重みの平均）。",
        "- 予測はどれも同期の版のモデル（`product` だけ手元の版を混ぜる）。",
        "",
    ]
    comparisons = (("fixed_share", "max_weight"), ("adahedge", "fixed_share"), ("product", "fixed_share"))
    for subset in ("全標本", "保有2以上"):
        lines += [f"## {subset}", ""]
        header = "| 条件 | 期間 | 標本の割合 | 混合FS の正解率 | FS − 最大重み | AdaHedge − FS | 実際 − FS |"
        lines += [header, "|---|---|---|---|---|---|---|"]
        overall = defaultdict(lambda: defaultdict(list))
        for condition in sorted(per_seed):
            total = sum(
                count for name in PERIOD_NAMES for _, count in per_seed[condition]["全標本"][name]["product"]
            )
            for name in PERIOD_NAMES:
                records = per_seed[condition][subset][name]
                counts = [count for _, count in records["product"]]
                if sum(counts) == 0:
                    continue
                share = 100 * sum(counts) / total
                accuracy = statistics.mean(
                    100 * correct / count for correct, count in records["fixed_share"] if count
                )
                cells = []
                for left, right in comparisons:
                    differences = [
                        100 * (a / n - b / n)
                        for (a, n), (b, _) in zip(records[left], records[right], strict=True)
                        if n
                    ]
                    value = statistics.mean(differences)
                    cells.append(f"{value:+.3f}")
                    overall[name][(left, right)].append(value)
                overall[name]["share"].append(share)
                overall[name]["accuracy"].append(accuracy)
                lines.append(
                    f"| {condition[0]} i{condition[1]} | {name} | {share:.1f}% | {accuracy:.2f} | " + " | ".join(cells) + " |"
                )
        lines += ["", f"### {subset}: 条件の等重みの平均", ""]
        lines += ["| 期間 | 条件の数 | 標本の割合 | 混合FS の正解率 | FS − 最大重み | AdaHedge − FS | 実際 − FS |", "|---|---|---|---|---|---|---|"]
        for name in PERIOD_NAMES:
            if not overall[name]["share"]:
                continue
            cells = [f"{statistics.mean(overall[name][pair]):+.3f}" for pair in comparisons]
            lines.append(
                f"| {name} | {len(overall[name]['share'])} | {statistics.mean(overall[name]['share']):.1f}% | "
                f"{statistics.mean(overall[name]['accuracy']):.2f} | " + " | ".join(cells) + " |"
            )
        lines.append("")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path, help="構成の runs.zip")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(command_arguments)
    print(analyze([path.resolve() for path in arguments.archives], arguments.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
