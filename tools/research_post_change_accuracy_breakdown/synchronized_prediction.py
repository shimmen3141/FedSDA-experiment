"""予測の版の比較（同期の版だけ・手元の版だけ・2つの版の混合＝FedHASPの既定）を、変化の直後とそれ以外に分けて並べる。

追加の実験なし。10 client×5000標本、10データ×間隔50・100・200・500×seed 0〜9の既存の結果:
- mix: FedHASPの既定`fedsda`（基準版16）。6データは`fedsda-component-ablation-v16`。新しい4データは、基準版16の既定と
  同じ値の、基準版15のときの構成`fedsda-every-sample-change-interval-search`の結果（`docs/research/fedsda-component-ablation-v16-20261030.md`
  の一致の確認）。これが基準版16の既定と同じ値であることを、間隔50・500で`fedhasp-oracle-v16`の`product-default`と照合する。
  （基準版15の`fedsda`＝`new-datasets-v15`は、基準版16では別の構成に当たるので使わない。）
- sync: 予測を同期の時点のモデルだけで行う`fedsda-synchronized-model-prediction`（`fedsda-synchronized-prediction-v16`）。
- local: 予測を手元の版だけで行う`fedsda-local-parameter-prediction`（6データは`fedsda-component-ablation-v16`、
  新しい4データは`fedsda-component-ablation-new-datasets-v16`）。
比べる組（データ×間隔×seed）ごとに、3つの版の上りの通信量が一致する（予測だけが違い、学習が同じ）ことを確かめる。
集計は10データ×間隔50・100・200・500。
各runで標本ごとの正誤（`sample_prediction.combined_prediction_is_correct`）の平均が要約の精度と一致することを確かめ、
「変化の直後」（各clientの真の概念の変化から集約の間隔I件）とそれ以外に分ける。標準指標（上り・下りの通信、clientのMAC）は
要約の表から読む。経過時間はrunごとに保存されていない。runのNPZは`runs/`か、研究ツール`research_results_archive`で
まとめた`runs.zip`から読む。

使い方（プロジェクトのvenv、worktreeのルートで）:
    python -m tools.research_post_change_accuracy_breakdown.synchronized_prediction \\
        --results ../../results --output ../../results/fedsda-synchronized-prediction-v16/analysis --workers 12
"""

import argparse
import csv
import io
import statistics as st
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

from tools.research_post_change_accuracy_breakdown.analyze import post_change_mask

DATASETS = ["sine2", "circle2", "sea2", "sea4", "mnist2", "mnist4", "fmnist2", "fmnist4", "rotmnist2", "rotmnist4"]
INTERVALS = [50, 100, 200, 500]
SEEDS = list(range(10))
VARIANTS = {
    "mix": [
        "fedsda-component-ablation-v16/20261029_231413_core-components-v16-seeds0-9/fedsda",
        "fedsda-every-sample-change-interval-search/20261029_172159_new-datasets-seeds0-9/fedsda-every-sample-change-interval-search",
    ],
    "sync": [
        "fedsda-synchronized-prediction-v16/20261124_165503_all-datasets-seeds0-9/fedsda-synchronized-model-prediction"
    ],
    "local": [
        "fedsda-component-ablation-v16/20261029_231413_core-components-v16-seeds0-9/fedsda-local-parameter-prediction",
        "fedsda-component-ablation-v16/20261030_181023_other-components-seeds0-9/fedsda-local-parameter-prediction",
    ],
}
V16_DEFAULT_CHECK = "fedhasp-oracle-v16/20261102_035503_oracle-seeds0-9/product-default/fedsda"
LABELS = {"mix": "混合（FedHASP）", "sync": "同期の版だけ", "local": "手元の版だけ"}
RESOURCE_COLUMNS = {
    "upload_mb": "communication_volume.uploaded_byte_count",
    "download_mb": "communication_volume.downloaded_byte_count",
    "client_forward_mac": "computation_cost_summary.client_forward_multiply_accumulate_count",
    "client_backward_mac": "computation_cost_summary.client_estimated_backward_multiply_accumulate_count",
}


def _key(row):
    return (row["dataset_name"], int(row["server_aggregation_interval_per_client_samples"]), int(row["random_seed"]))


def collect_tasks(results):
    tasks, resources = [], {}
    for variant, directories in VARIANTS.items():
        seen = set()
        for directory in directories:
            for row in csv.DictReader(open(Path(results) / directory / "run_summary.csv", encoding="utf-8")):
                key = _key(row)
                if key[0] not in DATASETS or key[1] not in INTERVALS or key[2] not in SEEDS or key in seen:
                    continue
                seen.add(key)
                tasks.append(
                    (
                        variant,
                        *key,
                        str(Path(results) / directory),
                        row["run_identifier"],
                        float(row["prediction_accuracy"]),
                    )
                )
                resources[(variant, *key)] = {
                    name: float(row[column]) if row.get(column) not in (None, "") else float("nan")
                    for name, column in RESOURCE_COLUMNS.items()
                }
    return tasks, resources


SCOPES = {"10データ×4間隔": (DATASETS, INTERVALS)}


def check_new_dataset_defaults(results, resources_accuracy):
    """新しい4データの既定（基準版15のときの同じ値の構成）が、基準版16の既定と同じ精度か（間隔50・500）。"""
    compared, mismatched = 0, 0
    for row in csv.DictReader(open(Path(results) / V16_DEFAULT_CHECK / "run_summary.csv", encoding="utf-8")):
        key = _key(row)
        if key[0] in DATASETS[6:] and key in resources_accuracy:
            compared += 1
            mismatched += abs(float(row["prediction_accuracy"]) - resources_accuracy[key]) > 1e-12
    return f"新しい4データの既定と基準版16の既定（`{V16_DEFAULT_CHECK}`）の精度の照合: 間隔50・500の{compared} runのうち不一致 {mismatched}"


def check_scopes(resources):
    """各範囲で3つの版がそろい、上りの通信量が一致することを確かめる。"""
    messages = []
    for name, (datasets, intervals) in SCOPES.items():
        keys = [(d, i, s) for d in datasets for i in intervals for s in SEEDS]
        missing = [k for k in keys for v in VARIANTS if (v, *k) not in resources]
        if missing:
            raise ValueError(f"{name}: runが欠けています {missing[:5]}")
        same = sum(
            resources[("mix", *k)]["upload_mb"]
            == resources[("sync", *k)]["upload_mb"]
            == resources[("local", *k)]["upload_mb"]
            for k in keys
        )
        messages.append(f"{name}: {len(keys)}組のうち、3つの版の上りの通信量が一致 {same}")
    return messages


def summarize(task):
    variant, dataset, interval, seed, directory, run_identifier, summary_accuracy = task
    member = f"runs/{run_identifier}.npz"
    archive_path = Path(directory) / "runs.zip"
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            data = archive.read(member)
    else:
        data = (Path(directory) / member).read_bytes()
    npz = f"{directory}/{member}"
    with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
        correct = loaded["sample_prediction.combined_prediction_is_correct"].astype(bool)
        concepts = loaded["sample_prediction.observed_concept_id"]
    if abs(correct.mean() - summary_accuracy) > 1e-9:
        raise ValueError(f"正誤の平均が要約の精度と違います: {npz}")
    mask = post_change_mask(concepts, interval)
    return (variant, dataset, interval, seed), dict(
        overall=100 * correct.mean(), post=100 * correct[mask].mean(), rest=100 * correct[~mask].mean()
    )


def paired(a, b):
    differences = np.array(a) - np.array(b)
    half = stats.t.ppf(0.975, len(differences) - 1) * differences.std(ddof=1) / np.sqrt(len(differences))
    wins = int((differences > 0).sum())
    return f"{differences.mean():+.2f} [{differences.mean() - half:+.2f}, {differences.mean() + half:+.2f}]（{wins}/{len(differences)}）"


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args(command_arguments)
    tasks, resources = collect_tasks(arguments.results)
    scope_messages = check_scopes(resources)
    mix_accuracy = {(t[1], t[2], t[3]): t[6] for t in tasks if t[0] == "mix"}
    scope_messages.append(check_new_dataset_defaults(arguments.results, mix_accuracy))
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        values = dict(executor.map(summarize, tasks, chunksize=8))
    arguments.output.mkdir(parents=True, exist_ok=True)

    def per_seed(variant, interval, scope, datasets):
        return [st.mean(values[(variant, d, interval, s)][scope] for d in datasets) for s in SEEDS]

    lines = [
        "# 予測の版の比較: 同期の版だけ・手元の版だけ・混合（連合学習、基準版16）",
        "",
        "- 10 client×5000標本、seed 0〜9。精度（%）は各seedでデータを等しい重みで平均し、seedの平均 ± 標準偏差。",
        "- 差は対にした差（seed）の平均と95%区間、括弧は左が上回ったseedの数。各runで標本ごとの正誤の平均が要約の精度と一致した。",
        *[f"- 確認: {m}" for m in scope_messages],
        "- 変化の直後は、各clientの真の概念の変化から集約の間隔I件。",
        "",
    ]
    for scope_name, (datasets, intervals) in SCOPES.items():
        lines += [f"# 範囲: {scope_name}", ""]
        for scope, title in (("overall", "全標本"), ("post", "変化の直後"), ("rest", "それ以外")):
            lines += [
                f"## {title}（{scope_name}）",
                "",
                "| 間隔 | 混合 | 同期の版だけ | 手元の版だけ | 混合−同期 | 混合−手元 | 同期−手元 |",
                "|---|---|---|---|---|---|---|",
            ]
            for interval in intervals:
                m, s_, l_ = (per_seed(v, interval, scope, datasets) for v in ("mix", "sync", "local"))
                lines.append(
                    f"| {interval} | {st.mean(m):.2f} ± {st.stdev(m):.2f} | {st.mean(s_):.2f} ± {st.stdev(s_):.2f} | "
                    f"{st.mean(l_):.2f} ± {st.stdev(l_):.2f} | {paired(m, s_)} | {paired(m, l_)} | {paired(s_, l_)} |"
                )
            lines.append("")
    lines += [
        "## データ別（全標本、混合−同期 / 混合−手元、pp）",
        "",
        "| データ | " + " | ".join(f"間隔{i}" for i in INTERVALS) + " |",
        "|---|" + "---|" * len(INTERVALS),
    ]
    for dataset in DATASETS:
        cells = []
        for interval in INTERVALS:
            if ("mix", dataset, interval, 0) not in resources:
                cells.append("—")
                continue
            m, s_, l_ = (per_seed(v, interval, "overall", [dataset]) for v in ("mix", "sync", "local"))
            cells.append(f"{st.mean(m) - st.mean(s_):+.2f} / {st.mean(m) - st.mean(l_):+.2f}")
        lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 標準指標（10データ、seedの平均、1 runあたり）",
        "",
        "| 間隔 | 版 | 上り MB | 下り MB | clientの前向きMAC | clientの逆向きMAC（推定） |",
        "|---|---|---|---|---|---|",
    ]
    for interval in INTERVALS:
        for variant in ("mix", "sync", "local"):

            def mean_of(name, variant=variant, interval=interval):
                return st.mean(resources[(variant, d, interval, s)][name] for d in DATASETS for s in SEEDS)

            lines.append(
                f"| {interval} | {LABELS[variant]} | {mean_of('upload_mb') / 1e6:.1f} | {mean_of('download_mb') / 1e6:.1f} | "
                f"{mean_of('client_forward_mac'):.3e} | {mean_of('client_backward_mac'):.3e} |"
            )
    lines += ["", "- 経過時間はrunごとに保存されていない（同期の版だけの掃引は uranium 120 workers で1時間53分）。"]
    (arguments.output / "analysis_synchronized_prediction.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (arguments.output / "synchronized_prediction_cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "dataset", "interval", "seed", "overall", "post_change", "rest"])
        for (variant, dataset, interval, seed), v in sorted(values.items()):
            writer.writerow([variant, dataset, interval, seed, v["overall"], v["post"], v["rest"]])
    print(f"wrote {arguments.output / 'analysis_synchronized_prediction.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
