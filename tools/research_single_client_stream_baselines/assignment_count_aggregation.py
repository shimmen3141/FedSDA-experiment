"""柔らかい学習への介入の集計: 介入の構成を、固い学習・柔らかい学習と比べる（`--intervention`で構成を選ぶ）。

介入の構成（既定は1つ目）:
- `fedsda-soft-training-assignment-count-aggregation-weights`（仮説2(a)）: 局所は柔らかい学習、集約の重みと参加の判定は件数。
- `fedsda-soft-training-own-assigned-models`（E1）: 責任度の対象を、そのclientが自分で固く割り当てた標本を持つモデルと、
  いま割り当てる先に限る。

3つの構成（連合学習、同じデータ・間隔・seed）を対にする。
- 固い学習: `fedsda-training-mass-diagnostics`（`fedsda`と同じ学習・予測に、学習量の記録を足した構成）
- 柔らかい学習: `fedsda-soft-training`
- 介入: `fedsda-soft-training-assignment-count-aggregation-weights`（局所は柔らかい学習、集約の重みと参加の判定は件数）

読み方: 「固い−介入」が0に近づけば（「固い−柔らかい」のうち介入で戻った割合が大きければ）、柔らかい学習の精度の
低下は集約を通る混ざりから来る。戻らなければ、手元の柔らかい学習の分散が原因。

照合（どれかが合わなければ止める）:
- 3構成でrunの組（データ・間隔・seed）が同じで、各組に1つずつ。基準版が同じ。
- 介入のrunの設定が、同じ組の柔らかい学習のrunの設定と、介入の構成の設定の項目（下の`INTERVENTIONS`）だけ違う。
- 3構成のrunのラベルと真のコンセプトの系列が全標本で一致する。標本ごとの正誤の平均が要約の精度と一致する。

出力（`--output`。既定は介入の結果のディレクトリ）: `analysis_assignment_count_aggregation.md`と`assignment_count_aggregation_runs.csv`。
差はpp。セル（データ×間隔）ごとにseedの平均をとり、セルを等しい重みで平均。95%区間はseedごとのセルの平均の差のt区間。
区間の種類は`recurrence_breakdown.segment_categories`と同じ分け方。純度は`aggregated_purity.run_purities`と同じ定義。

使い方（プロジェクトのvenv、リポジトリのルートかworktreeで）:
    python -m tools.research_single_client_stream_baselines.assignment_count_aggregation \\
        --existing <固い・柔らかいの結果のディレクトリ> --new <介入の結果のディレクトリ> \\
        [--intervention <介入の構成>] [--logs <掃引のログ> ...]
"""

import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.research_single_client_stream_baselines.aggregated_purity import run_purities
from tools.research_single_client_stream_baselines.analyze import read_elapsed_seconds
from tools.research_single_client_stream_baselines.recurrence_breakdown import (
    CATEGORIES,
    CATEGORY_LABELS,
    paired_interval,
    segment_categories,
)

csv.field_size_limit(10**9)

HARD = "fedsda-training-mass-diagnostics"
SOFT = "fedsda-soft-training"
# 介入の構成ごとの、表の名前・題・柔らかい学習との違いの設定の項目（clientの設定の中）と値。
INTERVENTIONS = {
    "fedsda-soft-training-assignment-count-aggregation-weights": dict(
        label="介入（柔らかい学習＋件数で集約）",
        title="柔らかい学習の集約の重みを件数にした介入（仮説2(a)）",
        setting="model_aggregation_weight_source",
        value="single_model_assignment_sample_counts",
        setting_label="集約の重みの元",
    ),
    "fedsda-soft-training-own-assigned-models": dict(
        label="介入（柔らかい学習＋自分で割り当てたモデルだけ）",
        title="柔らかい学習の責任度を自分で固く割り当てたモデルに限る介入（E1）",
        setting="training_responsibility_model_scope",
        value="models_with_own_assigned_samples",
        setting_label="責任度の対象",
    ),
}
NEW = "fedsda-soft-training-assignment-count-aggregation-weights"
LABELS = {}
COMPARISONS = ()
SETTING_PATH = ()


def select_intervention(configuration):
    """介入の構成を選び、表の名前・比較の組・照合する設定の項目を決める。"""
    global NEW, LABELS, COMPARISONS, SETTING_PATH
    if configuration not in INTERVENTIONS:
        raise ValueError(f"介入の構成ではありません: {configuration}")
    NEW = configuration
    LABELS = {HARD: "固い学習", SOFT: "柔らかい学習", NEW: INTERVENTIONS[NEW]["label"]}
    COMPARISONS = (
        ("固い−柔らかい（柔らかい学習による低下）", HARD, SOFT),
        ("固い−介入（介入の後に残る低下）", HARD, NEW),
        ("介入−柔らかい（介入で戻った分）", NEW, SOFT),
    )
    SETTING_PATH = ("run_participant_settings", "run_client_settings", INTERVENTIONS[NEW]["setting"])


select_intervention(NEW)
MAC_COLUMN = (
    "computation_cost_summary.client_forward_and_backward_multiply_accumulate_count_per_processed_sample"
)


def load_configuration(directory, configuration):
    """構成のディレクトリの`run_summary.csv`を読み、(データ, 間隔, seed)ごとのrunを返す。"""
    runs = {}
    with (directory / configuration / "run_summary.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["configuration_name"] != configuration:
                raise ValueError(f"構成の名前が違います: {row['configuration_name']} != {configuration}")
            key = (
                row["dataset_name"],
                int(row["server_aggregation_interval_per_client_samples"]),
                int(row["random_seed"]),
            )
            if key in runs:
                raise ValueError(f"同じ組のrunが2つあります: {configuration} {key}")
            npz_path = directory / configuration / "runs" / f"{row['run_identifier']}.npz"
            runs[key] = dict(
                run_identifier=row["run_identifier"],
                npz_path=npz_path,
                json_path=npz_path.with_suffix(".json"),
                baseline=row["behavior_baseline_version"],
                accuracy=float(row["prediction_accuracy"]),
                uploaded_bytes=float(row["communication_volume.uploaded_byte_count"]),
                downloaded_bytes=float(row["communication_volume.downloaded_byte_count"]),
                uploaded_models=float(row["communication_volume.uploaded_model_count"]),
                mac_per_sample=float(row[MAC_COLUMN]),
                final_model_count=float(row["final_global_model_count"]),
            )
    if not runs:
        raise ValueError(f"runがありません: {directory / configuration}")
    return runs


def without_weight_source(settings):
    """runの設定から、集約の重みの元（無ければ既定）を除いた写しを返す。"""
    copied = json.loads(json.dumps(settings))
    node = copied
    for name in SETTING_PATH[:-1]:
        node = node[name]
    node.pop(SETTING_PATH[-1], None)
    return copied


def check_settings(new_run, soft_run):
    new_settings = json.loads(new_run["json_path"].read_text(encoding="utf-8"))["run_settings"]
    soft_settings = json.loads(soft_run["json_path"].read_text(encoding="utf-8"))["run_settings"]
    node = new_settings
    for name in SETTING_PATH[:-1]:
        node = node[name]
    node = node.get(SETTING_PATH[-1])
    if node != INTERVENTIONS[NEW]["value"]:
        raise ValueError(
            f"介入のrunの{INTERVENTIONS[NEW]['setting_label']}が違います: {new_run['run_identifier']} {node}"
        )
    if without_weight_source(new_settings) != without_weight_source(soft_settings):
        raise ValueError(
            f"介入と柔らかい学習の設定が、{INTERVENTIONS[NEW]['setting_label']}以外でも違います: "
            f"{new_run['run_identifier']}"
        )


def read_run(run, window):
    """標本ごとの正誤から、区間の種類ごとの精度を求め、入力の系列と純度を返す。"""
    with np.load(run["npz_path"]) as npz:
        correct = npz["sample_prediction.combined_prediction_is_correct"].astype(bool)
        labels = npz["sample_prediction.observed_class_id"]
        concepts = npz["sample_prediction.observed_concept_id"]
    if abs(correct.mean() - run["accuracy"]) > 1e-9:
        raise ValueError(f"標本ごとの正誤の平均が精度と違います: {run['npz_path']}")
    category, position = segment_categories(concepts)
    accuracy = {}
    counts = {}
    for kind in CATEGORIES:
        selected = np.ones(correct.shape, dtype=bool) if kind == "all" else (category == kind)
        for scope, mask in (("all", selected), ("first", selected & (position < window))):
            count = int(mask.sum())
            accuracy[(kind, scope)] = float(correct[mask].mean()) if count else None
            counts[(kind, scope)] = count
    local_purity, aggregated_purity, excluded = run_purities(run["npz_path"])
    return dict(
        accuracy_by_category=accuracy,
        sample_counts=counts,
        labels=labels,
        concepts=concepts,
        local_purity=local_purity,
        aggregated_purity=aggregated_purity,
        excluded_mass_fraction=excluded,
    )


def summarize(values_by_key, keys, intervals):
    """(データ, 間隔, seed)ごとの値から、セルの等重みの平均・seedのt区間・上のseed・セルの符号を求める。"""
    cells = defaultdict(list)
    for key in keys:
        if key[1] in intervals and values_by_key.get(key) is not None:
            cells[key[:2]].append((key[2], values_by_key[key]))
    if not cells:
        return None
    cell_means = {cell: st.mean(v for _, v in values) for cell, values in cells.items()}
    by_seed = defaultdict(list)
    for values in cells.values():
        for seed, value in values:
            by_seed[seed].append(value)
    seed_means = [st.mean(by_seed[s]) for s in sorted(by_seed)]
    _, low, high = paired_interval(seed_means)
    cell_intervals = {}
    for cell, values in cells.items():
        _, cell_low, cell_high = paired_interval([v for _, v in values])
        cell_intervals[cell] = (cell_low, cell_high)
    return dict(
        mean=st.mean(cell_means.values()),
        low=low,
        high=high,
        positive_seeds=sum(m > 0 for m in seed_means),
        seed_count=len(seed_means),
        cells_above=sum(lo is not None and lo > 0 for lo, _ in cell_intervals.values()),
        cells_below=sum(hi is not None and hi < 0 for _, hi in cell_intervals.values()),
        cell_count=len(cell_means),
        cell_means=cell_means,
    )


def fmt(value, digits=2):
    return "—" if value is None else f"{value:+.{digits}f}"


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--existing", type=Path, required=True, help="固い学習・柔らかい学習の結果のディレクトリ")
    parser.add_argument("--new", type=Path, required=True, help="介入の構成の結果のディレクトリ")
    parser.add_argument("--intervention", choices=sorted(INTERVENTIONS), default=NEW)
    parser.add_argument("--logs", type=Path, nargs="*", default=[], help="掃引のログ（経過時間）")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--window", type=int, default=200)
    arguments = parser.parse_args(command_arguments)
    select_intervention(arguments.intervention)
    output = arguments.output or arguments.new
    runs = {
        HARD: load_configuration(arguments.existing, HARD),
        SOFT: load_configuration(arguments.existing, SOFT),
        NEW: load_configuration(arguments.new, NEW),
    }
    keys = sorted(runs[HARD])
    for configuration, configuration_runs in runs.items():
        if sorted(configuration_runs) != keys:
            raise ValueError(f"{configuration}: runの組が固い学習とそろっていません")
    baselines = {run["baseline"] for configuration_runs in runs.values() for run in configuration_runs.values()}
    if len(baselines) != 1:
        raise ValueError(f"基準版がそろっていません: {sorted(baselines)}")
    for key in keys:
        check_settings(runs[NEW][key], runs[SOFT][key])
    details = {}
    for configuration, configuration_runs in runs.items():
        for key, run in configuration_runs.items():
            details[(configuration, key)] = read_run(run, arguments.window)
    for key in keys:
        reference = details[(HARD, key)]
        for configuration in (SOFT, NEW):
            other = details[(configuration, key)]
            if not (
                np.array_equal(reference["labels"], other["labels"])
                and np.array_equal(reference["concepts"], other["concepts"])
            ):
                raise ValueError(f"{configuration} {key}: ラベルか真のコンセプトの系列が固い学習と違います")
    elapsed = read_elapsed_seconds(arguments.logs)
    datasets = sorted({k[0] for k in keys})
    intervals = sorted({k[1] for k in keys})
    seeds = sorted({k[2] for k in keys})
    interval_sets = [("全体", tuple(intervals))] + [(f"間隔{i}", (i,)) for i in intervals]

    lines = [f"# {INTERVENTIONS[NEW]['title']}", ""]
    lines.append(
        f"- 連合学習、データ{len(datasets)}、間隔{intervals}、seed {seeds[0]}〜{seeds[-1]}（{len(seeds)}）、基準版{baselines.pop()}。"
        "差はpp、セル（データ×間隔）ごとにseedの平均をとり、セルを等しい重みで平均。95%区間はseedごとのセルの平均の差のt区間"
        "（多重比較の補正なし）。「区間が0より上／下のセル」は、セルの中のseedのt区間で数えた。"
    )
    lines.append(
        f"- 照合: 3構成でrunの組が{len(keys)}組ずつそろい、介入の設定が柔らかい学習と{INTERVENTIONS[NEW]['setting_label']}だけ違い、"
        f"ラベルと真のコンセプトの系列が全標本で一致し（{2 * len(keys)}組）、標本ごとの正誤の平均が精度と一致した。"
    )
    lines += ["", "## 精度の差", ""]
    lines.append("| 比較 | 範囲 | 平均 | 95%区間 | 上のseed | 区間が0より上／下のセル |")
    lines.append("|---|---|---|---|---|---|")
    accuracy_by = {
        configuration: {key: 100 * run["accuracy"] for key, run in configuration_runs.items()}
        for configuration, configuration_runs in runs.items()
    }
    overall = {}
    for label, first, second in COMPARISONS:
        differences = {key: accuracy_by[first][key] - accuracy_by[second][key] for key in keys}
        for range_label, selected in interval_sets:
            summary = summarize(differences, keys, selected)
            overall[(label, range_label)] = summary
            lines.append(
                f"| {label} | {range_label} | {fmt(summary['mean'])} | [{fmt(summary['low'])}, {fmt(summary['high'])}] | "
                f"{summary['positive_seeds']}/{summary['seed_count']} | {summary['cells_above']}／{summary['cells_below']}"
                f"（{summary['cell_count']}セル） |"
            )
    lines += ["", "- 介入で戻った割合（「介入−柔らかい」÷「固い−柔らかい」、平均どうしの比。記述値）: "]
    for range_label, _ in interval_sets:
        drop = overall[(COMPARISONS[0][0], range_label)]["mean"]
        recovered = overall[(COMPARISONS[2][0], range_label)]["mean"]
        ratio = "—（低下が0以下）" if drop <= 0 else f"{100 * recovered / drop:.0f}%"
        lines[-1] += f"{range_label} {ratio}（{fmt(recovered)}／{fmt(drop)}）、"
    lines[-1] = lines[-1].rstrip("、")

    lines += ["", "## データ別の精度（%、seedの平均）", ""]
    lines.append(
        "| データ | 間隔 | 固い学習 | 柔らかい学習 | 介入 | 固い−柔らかい | 固い−介入 | 介入−柔らかい |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for dataset in datasets:
        for interval in intervals:
            cell_keys = [k for k in keys if k[:2] == (dataset, interval)]
            means = {c: st.mean(accuracy_by[c][k] for k in cell_keys) for c in runs}
            lines.append(
                f"| {dataset} | {interval} | {means[HARD]:.2f} | {means[SOFT]:.2f} | {means[NEW]:.2f} | "
                f"{fmt(means[HARD] - means[SOFT])} | {fmt(means[HARD] - means[NEW])} | {fmt(means[NEW] - means[SOFT])} |"
            )

    lines += ["", f"## 区間の種類ごとの差（区間の全標本／区間の最初の{arguments.window}標本）", ""]
    total = defaultdict(int)
    for key in keys:
        for kind in CATEGORIES[1:]:
            total[kind] += details[(HARD, key)]["sample_counts"][(kind, "all")]
    grand = sum(total.values())
    lines.append(
        "- 標本の割合: " + "、".join(f"{CATEGORY_LABELS[k]} {100 * total[k] / grand:.1f}%" for k in CATEGORIES[1:])
    )
    lines.append("")
    lines.append("| 比較 | 区間の種類 | 範囲 | 全標本: 平均 [95%区間] | 最初の標本: 平均 [95%区間] |")
    lines.append("|---|---|---|---|---|")
    for label, first, second in COMPARISONS:
        for kind in CATEGORIES:
            for range_label, selected in interval_sets:
                cells = []
                for scope in ("all", "first"):
                    differences = {}
                    for key in keys:
                        a = details[(first, key)]["accuracy_by_category"][(kind, scope)]
                        b = details[(second, key)]["accuracy_by_category"][(kind, scope)]
                        differences[key] = None if a is None or b is None else 100 * (a - b)
                    summary = summarize(differences, keys, selected)
                    cells.append(
                        "—"
                        if summary is None
                        else f"{fmt(summary['mean'])} [{fmt(summary['low'])}, {fmt(summary['high'])}]"
                    )
                lines.append(f"| {label} | {CATEGORY_LABELS[kind]} | {range_label} | {cells[0]} | {cells[1]} |")

    lines += ["", "## 標準指標と診断（runの平均）", ""]
    lines.append(
        "| 構成 | 間隔 | 精度（%） | 上り（MB） | 下り（MB） | 上りのモデル数 | 標本あたりのMAC（順＋逆） | 最後のモデルの数 | "
        "clientの中の純度 | 集約した後の純度 | 1 runの経過時間（秒、ログにある分） |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for configuration, configuration_runs in runs.items():
        for interval in intervals:
            selected = [(k, r) for k, r in configuration_runs.items() if k[1] == interval]
            seconds = [elapsed.get((configuration, r["run_identifier"])) for _, r in selected]
            seconds = [s for s in seconds if s is not None]
            seconds_cell = f"{st.mean(seconds):.0f}（{len(seconds)} run）" if seconds else "ログなし"
            purities = [details[(configuration, k)] for k, _ in selected]
            local = [p["local_purity"] for p in purities if p["local_purity"] is not None]
            aggregated = [p["aggregated_purity"] for p in purities if p["aggregated_purity"] is not None]
            lines.append(
                f"| {LABELS[configuration]} | {interval} | {st.mean(100 * r['accuracy'] for _, r in selected):.2f} | "
                f"{st.mean(r['uploaded_bytes'] for _, r in selected) / 1e6:.1f} | "
                f"{st.mean(r['downloaded_bytes'] for _, r in selected) / 1e6:.1f} | "
                f"{st.mean(r['uploaded_models'] for _, r in selected):.0f} | "
                f"{st.mean(r['mac_per_sample'] for _, r in selected):.3g} | "
                f"{st.mean(r['final_model_count'] for _, r in selected):.2f} | "
                f"{st.mean(local):.3f} | {st.mean(aggregated):.3f} | {seconds_cell} |"
            )
    lines.append("")
    lines.append(
        "- 上り・下りはrunの全clientの合計。純度は`aggregated_purity`と同じ定義（集約した後の純度は、負のmodel IDの学習量を除く）。"
    )

    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis_assignment_count_aggregation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_rows = []
    for configuration, configuration_runs in runs.items():
        for key in keys:
            run = configuration_runs[key]
            detail = details[(configuration, key)]
            run_rows.append(
                dict(
                    configuration=configuration,
                    dataset=key[0],
                    interval=key[1],
                    seed=key[2],
                    run_identifier=run["run_identifier"],
                    accuracy=run["accuracy"],
                    uploaded_bytes=run["uploaded_bytes"],
                    downloaded_bytes=run["downloaded_bytes"],
                    mac_per_sample=run["mac_per_sample"],
                    final_model_count=run["final_model_count"],
                    local_purity=detail["local_purity"],
                    aggregated_purity=detail["aggregated_purity"],
                    elapsed_seconds=elapsed.get((configuration, run["run_identifier"])),
                    **{
                        f"accuracy_{kind}_{scope}": detail["accuracy_by_category"][(kind, scope)]
                        for kind in CATEGORIES
                        for scope in ("all", "first")
                    },
                )
            )
    with (output / "assignment_count_aggregation_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)
    print(f"wrote {output / 'analysis_assignment_count_aggregation.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
