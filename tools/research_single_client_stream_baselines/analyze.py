"""1クライアントの予備実験（docs/research/fedhasp-single-client-pilot-v16.md）と、Riverの比較手法の結果を集計する。

入力:
- --pilot: 製品の掃引の結果（構成ごとのディレクトリに`run_summary.csv`と`runs/*.npz`）。
- --oracle: 研究ツール`research_oracle_concept_assignment`の出力先（省略可）。
- --river: river_baselines.pyの出力先（省略可）。--streamsと組で使う。
- --streams: stream_export.pyの書き出し先（--riverと組で指定する。系列だけの照合はしない）。
- --logs: 掃引・Oracle・Riverの標準出力を保存したログ（省略可）。runごとの経過時間は記録ファイルに無く、
  掃引の出力の行（`[i/N] <構成>/<runの識別子>: completed (<秒>s)`）にだけ出るので、ここから読む。

照合（どれかが合わなければ止める）:
- 各構成で、10データ×間隔×seedの組がすべてそろい、1 client・同じ標本数であること。
- Riverの結果の系列のhashが、書き出しの記録と一致すること。
- 書き出した系列のラベルと真のコンセプトが、同じデータ・seedの全構成（Oracleを含む）・両方の間隔のrunの記録と
  全標本で一致すること。

出力（--output）: analysis_single_client.md、cells.csv（セルごとの差）、accuracy.csv（手法ごとの精度）。
差の95%区間は、seedごとに対にした差のt区間。全体の値は、セルの平均を等しい重みで平均し、区間はseedごとの
「セルの平均の差」の平均から求める（データをまたいで、あるデータの悪化を別のデータの改善で打ち消さないよう、
セルごとの値も必ず併記する）。
"""

import argparse
import csv
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

csv.field_size_limit(10**9)

HARD_MIX = "fedsda-training-mass-diagnostics"
SOFT_MIX = "fedsda-soft-training"
HARD_SINGLE = "fedsda-single-model-prediction"
SOFT_SINGLE = "fedsda-soft-training-single-model-prediction"
HARD_MIX_LOCAL = "fedsda-local-parameter-prediction"
SINGLE_MODEL = "fedavg"
ORACLE = "oracle"
PILOT_CONFIGURATIONS = (HARD_MIX, SOFT_MIX, HARD_SINGLE, SOFT_SINGLE, HARD_MIX_LOCAL, SINGLE_MODEL)
ORACLE_VARIANT_DIRECTORY = Path("product-default-oracle-concept-assignment") / "fedsda"

COMPARISONS = (
    ("学習を固くする効果（混合の予測）", HARD_MIX, SOFT_MIX),
    ("学習を固くする効果（現在のモデルだけの予測）", HARD_SINGLE, SOFT_SINGLE),
    ("予測を混ぜる効果（固い学習）", HARD_MIX, HARD_SINGLE),
    ("予測を混ぜる効果（柔らかい学習）", SOFT_MIX, SOFT_SINGLE),
    ("2つの版を混ぜる効果（FedHASP−局所のパラメータだけ）", HARD_MIX, HARD_MIX_LOCAL),
    ("Oracle−FedHASP", ORACLE, HARD_MIX),
    ("FedHASP−単一のモデル", HARD_MIX, SINGLE_MODEL),
)


def purity(npz):
    """学習の量のコンセプト別の記録から、純度と、全部を1モデルに入れた基準を求める（1 client）。

    定義は`results/fedsda-soft-training-v16/aggregate_soft_training.py`の`purity`と同じ。
    """
    names = (
        "training_mass_by_concept.model_id",
        "training_mass_by_concept.observed_concept_id",
        "training_mass_by_concept.training_mass",
    )
    if names[0] not in npz.files:
        return None, None
    table = defaultdict(float)
    for model_id, concept_id, mass in zip(*(npz[name] for name in names)):
        table[(int(model_id), int(concept_id))] += float(mass)
    total = sum(table.values())
    if total == 0:
        return None, None
    models = {model_id for model_id, _ in table}
    concepts = {concept_id for _, concept_id in table}
    value = sum(max(table.get((m, k), 0.0) for k in concepts) for m in models) / total
    base = max(sum(table.get((m, k), 0.0) for m in models) for k in concepts) / total
    return value, base


def load_sweep_configuration(directory, configuration_label):
    """1つの構成の`run_summary.csv`を読み、(データ, 間隔, seed)ごとの値を返す。"""
    summary_path = directory / "run_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"結果がありません: {summary_path}")
    runs = {}
    with summary_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["client_count"]) != 1:
                raise ValueError(f"1 clientでないrunがあります: {row['run_identifier']}")
            key = (
                row["dataset_name"],
                int(row["server_aggregation_interval_per_client_samples"]),
                int(row["random_seed"]),
            )
            if key in runs:
                raise ValueError(f"同じ組のrunが2つあります: {configuration_label} {key}")
            npz_path = directory / "runs" / f"{row['run_identifier']}.npz"
            json_path = npz_path.with_suffix(".json")
            change_response = (
                json.loads(json_path.read_text(encoding="utf-8"))
                .get("run_metrics", {})
                .get("change_response_metrics")
                or {}
            )
            alarm_metrics = change_response.get("loss_change_alarm_detection_metrics") or {}
            action_metrics = change_response.get("adaptation_action_metrics") or {}
            runs[key] = dict(
                run_identifier=row["run_identifier"],
                alarm_count=alarm_metrics.get("detection_count"),
                new_model_adoption_count=action_metrics.get("adopt_candidate_model_count"),
                accuracy=float(row["prediction_accuracy"]),
                sample_count=int(row["per_client_sample_count"]),
                final_model_count=float(row["final_global_model_count"]),
                mac_per_sample=float(
                    row[
                        "computation_cost_summary.client_forward_and_backward_multiply_accumulate_count_per_processed_sample"
                    ]
                ),
                npz_path=npz_path,
            )
    return runs


_SWEEP_LOG_LINE = re.compile(r"\] (?P<configuration>[^/\s]+)/(?P<run>\S+): completed \((?P<seconds>\d+)s\)")


def read_elapsed_seconds(log_paths):
    """掃引の出力のログから、(構成, runの識別子)ごとの経過時間（秒）を読む。"""
    elapsed = {}
    for path in log_paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _SWEEP_LOG_LINE.search(line)
            if match:
                elapsed[(match["configuration"], match["run"])] = int(match["seconds"])
    return elapsed


def mean_or_none(values):
    values = [value for value in values if value is not None]
    return st.mean(values) if values else None


def check_complete(runs_by_method, datasets, intervals, seeds):
    for method, runs in runs_by_method.items():
        method_intervals = intervals if method in runs_by_method.interval_methods else (None,)
        method_expected = {(d, i, s) for d in datasets for i in method_intervals for s in seeds}
        missing = sorted(method_expected - set(runs))
        extra = sorted(set(runs) - method_expected)
        if missing or extra:
            raise ValueError(
                f"{method}: 欠け{len(missing)}件（例 {missing[:3]}）、余分{len(extra)}件（例 {extra[:3]}）"
            )


class RunsByMethod(dict):
    """手法ごとのrun。interval_methodsは集約の間隔ごとにrunがある手法（FedHASPの構成など）。"""

    def __init__(self):
        super().__init__()
        self.interval_methods = set()


def load_river(river_directory, streams_directory):
    runs_by_method = defaultdict(dict)
    stream_digests = {}
    for path in sorted(streams_directory.glob("*.json")):
        if path.name.startswith("export_provenance") or path.name.endswith(".tmp.json"):
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        stream_digests[
            (metadata["dataset_name"], metadata["random_seed"], metadata["sample_count"])
        ] = (
            metadata["array_sha256"],
            path.with_suffix(".npz"),
        )
    for path in sorted((river_directory / "runs").glob("*.json")):
        if path.name.endswith(".tmp.json"):
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        key = (result["dataset_name"], result["random_seed"], result["sample_count"])
        if key not in stream_digests or stream_digests[key][0] != result["stream_array_sha256"]:
            raise ValueError(f"Riverの結果の系列が書き出しと一致しません: {path.name}")
        runs_by_method[f"river:{result['method_name']}"][(key[0], None, key[1])] = dict(
            accuracy=result["prediction_accuracy"],
            sample_count=result["sample_count"],
            elapsed_seconds=result["elapsed_seconds"],
        )
    return runs_by_method, stream_digests


def check_stream_inputs(stream_digests, runs_by_configuration, datasets, intervals, seeds, sample_count):
    """書き出した系列のラベルと真のコンセプトが、全構成の全runの記録と一致するかを確かめる。"""
    checked = 0
    for dataset in datasets:
        for seed in seeds:
            if (dataset, seed, sample_count) not in stream_digests:
                raise ValueError(f"書き出した系列がありません: {dataset} seed{seed} n{sample_count}")
            with np.load(stream_digests[(dataset, seed, sample_count)][1], allow_pickle=False) as stream:
                labels, concepts = stream["labels"], stream["concept_ids"]
            for configuration, runs in runs_by_configuration.items():
                for interval in intervals:
                    with np.load(runs[(dataset, interval, seed)]["npz_path"]) as run:
                        run_labels = run["sample_prediction.observed_class_id"][0]
                        run_concepts = run["sample_prediction.observed_concept_id"][0]
                    if not (
                        np.array_equal(labels, run_labels) and np.array_equal(concepts, run_concepts)
                    ):
                        raise ValueError(
                            f"系列とrunの入力が一致しません: {configuration} {dataset} {interval} {seed}"
                        )
                    checked += 1
    return checked


def paired_summary(differences):
    values = np.asarray(differences, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, None, None
    half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def compare(runs_by_method, first, second, datasets, intervals, seeds):
    """セルごと（データ×間隔）の対にした差（pp）と、全体・間隔ごとの値を返す。"""

    def value(method, dataset, interval, seed):
        key_interval = interval if method in runs_by_method.interval_methods else None
        return runs_by_method[method][(dataset, key_interval, seed)]["accuracy"] * 100

    cells = []
    per_seed = defaultdict(list)
    for dataset in datasets:
        for interval in intervals:
            differences = [
                value(first, dataset, interval, seed) - value(second, dataset, interval, seed)
                for seed in seeds
            ]
            mean, low, high = paired_summary(differences)
            cells.append(
                dict(
                    dataset=dataset,
                    interval=interval,
                    mean=mean,
                    low=low,
                    high=high,
                    wins=sum(d > 0 for d in differences),
                    losses=sum(d < 0 for d in differences),
                )
            )
            for seed, difference in zip(seeds, differences):
                per_seed[(interval, seed)].append(difference)
    overall = {}
    for label, selected in [("全体", intervals), *[(f"間隔{i}", (i,)) for i in intervals]]:
        seed_means = [
            st.mean(d for i in selected for d in per_seed[(i, seed)]) for seed in seeds
        ]
        cell_means = [c["mean"] for c in cells if c["interval"] in selected]
        mean, low, high = paired_summary(seed_means)
        overall[label] = dict(
            mean=st.mean(cell_means),
            low=low,
            high=high,
            seeds_up=sum(m > 0 for m in seed_means),
            cells_up=sum(c["low"] is not None and c["low"] > 0 for c in cells if c["interval"] in selected),
            cells_down=sum(c["high"] is not None and c["high"] < 0 for c in cells if c["interval"] in selected),
            cell_count=len(cell_means),
            cell_min=min(cell_means),
            cell_max=max(cell_means),
        )
    return cells, overall


def fmt_plain(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def elapsed_cell(configuration, selected, elapsed_seconds):
    # Oracleは研究ツールの中で製品の構成`fedsda`として走るので、ログの構成の名前は`fedsda`。
    log_name = "fedsda" if configuration == ORACLE else configuration
    values = [elapsed_seconds.get((log_name, r["run_identifier"])) for r in selected]
    found = [v for v in values if v is not None]
    if not found:
        return "ログなし"
    return f"{st.mean(found):.0f}（{len(found)}/{len(values)} run）"


def fmt(value, digits=2):
    return "—" if value is None else f"{value:+.{digits}f}"


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--river", type=Path)
    parser.add_argument("--streams", type=Path)
    parser.add_argument("--logs", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(command_arguments)
    elapsed_seconds = read_elapsed_seconds(arguments.logs)
    if (arguments.river is None) != (arguments.streams is None):
        raise ValueError("--riverと--streamsは組で指定してください")

    runs_by_method = RunsByMethod()
    for configuration in PILOT_CONFIGURATIONS:
        runs_by_method[configuration] = load_sweep_configuration(
            arguments.pilot / configuration, configuration
        )
        runs_by_method.interval_methods.add(configuration)
    if arguments.oracle is not None:
        runs_by_method[ORACLE] = load_sweep_configuration(
            arguments.oracle / ORACLE_VARIANT_DIRECTORY, ORACLE
        )
        runs_by_method.interval_methods.add(ORACLE)
    reference = runs_by_method[HARD_MIX]
    datasets = sorted({d for d, _, _ in reference})
    intervals = sorted({i for _, i, _ in reference})
    seeds = sorted({s for _, _, s in reference})
    sample_counts = {r["sample_count"] for runs in runs_by_method.values() for r in runs.values()}
    if len(sample_counts) != 1:
        raise ValueError(f"標本数がそろっていません: {sorted(sample_counts)}")
    (sample_count,) = sample_counts

    check_complete(runs_by_method, datasets, intervals, seeds)
    stream_check_count = 0
    if arguments.river is not None:
        river_runs, stream_digests = load_river(arguments.river, arguments.streams)
        for method, runs in river_runs.items():
            if {r["sample_count"] for r in runs.values()} != {sample_count}:
                raise ValueError(f"{method}の標本数が予備実験と違います")
            runs_by_method[method] = runs
        stream_check_count = check_stream_inputs(
            stream_digests,
            {m: runs_by_method[m] for m in runs_by_method.interval_methods},
            datasets,
            intervals,
            seeds,
            sample_count,
        )
        check_complete(runs_by_method, datasets, intervals, seeds)

    comparisons = [c for c in COMPARISONS if c[1] in runs_by_method and c[2] in runs_by_method]
    comparisons += [
        (f"FedHASP−{method.split(':', 1)[1]}", HARD_MIX, method)
        for method in sorted(m for m in runs_by_method if m.startswith("river:"))
    ]
    arguments.output.mkdir(parents=True, exist_ok=True)
    cell_rows, lines = [], []
    lines.append("# 1クライアントの予備実験の集計")
    lines.append("")
    lines.append(
        f"- データ{len(datasets)}、間隔{intervals}、seed {seeds[0]}〜{seeds[-1]}（{len(seeds)}）、"
        f"1 client×{sample_count}標本。差はpp、95%区間はseedで対にしたt区間（セルごとの区間で、"
        "多重比較の補正はしていない）。"
    )
    lines.append(
        "- 入力の照合: 全構成で組の欠け・余分なし。"
        + (
            f"書き出した系列と全構成のrunのラベル・真のコンセプトが{stream_check_count}組で一致。"
            if stream_check_count
            else "Riverの結果は含めていない。"
        )
    )
    lines.append("")
    lines.append("## 差の全体の値")
    lines.append("")
    lines.append("| 比較 | 範囲 | 平均 | 95%区間 | 上のseed | 区間が0より上／下のセル | セルの範囲 |")
    lines.append("|---|---|---|---|---|---|---|")
    for label, first, second in comparisons:
        cells, overall = compare(runs_by_method, first, second, datasets, intervals, seeds)
        for cell in cells:
            cell_rows.append(dict(comparison=label, **cell))
        for scope, o in overall.items():
            lines.append(
                f"| {label} | {scope} | {fmt(o['mean'])} | [{fmt(o['low'])}, {fmt(o['high'])}] | "
                f"{o['seeds_up']}/{len(seeds)} | {o['cells_up']}／{o['cells_down']}（{o['cell_count']}セル） | "
                f"{fmt(o['cell_min'])}〜{fmt(o['cell_max'])} |"
            )
    lines.append("")
    lines.append("## 手法ごとの精度（%、seedの平均）")
    lines.append("")
    methods = list(runs_by_method)
    lines.append("| データ | 間隔 | " + " | ".join(methods) + " |")
    lines.append("|---|---|" + "---|" * len(methods))
    accuracy_rows = []
    for dataset in datasets:
        for interval in intervals:
            row = dict(dataset=dataset, interval=interval)
            for method in methods:
                key_interval = interval if method in runs_by_method.interval_methods else None
                row[method] = st.mean(
                    runs_by_method[method][(dataset, key_interval, s)]["accuracy"] * 100 for s in seeds
                )
            accuracy_rows.append(row)
            lines.append(
                f"| {dataset} | {interval} | " + " | ".join(f"{row[m]:.2f}" for m in methods) + " |"
            )
    lines.append("")
    lines.append("## 診断（seedの平均）")
    lines.append("")
    lines.append(
        "| 構成 | 間隔 | 純度 | 全部を1モデルに入れた基準 | 最後のモデルの数 | 警報の回数 | 新しいモデルの採用 | "
        "標本あたりのMAC（順＋逆） | 1 runの経過時間（秒、ログにある分） |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for configuration in (*PILOT_CONFIGURATIONS, *( [ORACLE] if ORACLE in runs_by_method else [])):
        for interval in intervals:
            selected = [runs_by_method[configuration][(d, interval, s)] for d in datasets for s in seeds]
            purities, bases = [], []
            for run in selected:
                with np.load(run["npz_path"]) as npz:
                    value, base = purity(npz)
                if value is not None:
                    purities.append(value)
                    bases.append(base)
            lines.append(
                f"| {configuration} | {interval} | "
                + (f"{st.mean(purities):.3f}（{len(purities)} run）" if purities else "記録なし")
                + " | "
                + (f"{st.mean(bases):.3f}" if bases else "—")
                + f" | {st.mean(r['final_model_count'] for r in selected):.2f}"
                + f" | {fmt_plain(mean_or_none(r['alarm_count'] for r in selected))}"
                + f" | {fmt_plain(mean_or_none(r['new_model_adoption_count'] for r in selected))}"
                + f" | {st.mean(r['mac_per_sample'] for r in selected):.3g}"
                + f" | {elapsed_cell(configuration, selected, elapsed_seconds)} |"
            )
    river_methods = [m for m in methods if m.startswith("river:")]
    if river_methods:
        lines.append("")
        lines.append("| Riverの手法 | 1 runの経過時間の平均（秒） |")
        lines.append("|---|---|")
        for method in river_methods:
            lines.append(
                f"| {method} | {st.mean(r['elapsed_seconds'] for r in runs_by_method[method].values()):.1f} |"
            )
    (arguments.output / "analysis_single_client.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, rows in (("cells.csv", cell_rows), ("accuracy.csv", accuracy_rows)):
        with (arguments.output / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {arguments.output / 'analysis_single_client.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
