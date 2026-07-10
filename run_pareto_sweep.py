"""精度–通信量トレードオフの掃引実験(複数シード × 複数データセット)。

各データセット・各シードで以下を実行し、結果を CSV と散布図に出力する:
- FedSDA    : 提案手法(検出バッチ非依存の1点)
- Oblivious : 単一モデル・無適応(基準線)
- FedDrift  : 検出バッチサイズ(--batches)を掃引した曲線

散布図は横軸=通信量(モデル転送数, 対数)、縦軸=paper_accuracy。FedSDA が
FedDrift の曲線の左上(高精度・低通信)を取れば「パレート支配」を示せる。

例:
    python run_pareto_sweep.py --quick                       # 動作確認(小規模)
    python run_pareto_sweep.py --datasets blobs sea --seeds 0 1 2
    python run_pareto_sweep.py                               # 既定(全4データセット × 5シード)
"""
import argparse
import csv
import os
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from FedSDA import config, run_random_drift_experiment

METRIC_KEYS = [
    "paper_accuracy", "paper_accuracy_all", "accuracy",
    "comm_upload", "comm_download", "comm_total",
    "final_model_count", "precision", "recall", "f1", "avg_delay", "total_detect",
]


def _run(mode, dataset, seed, feddrift_batch=None):
    config.DATASET = dataset
    if feddrift_batch is not None:
        config.FEDDRIFT_DETECT_BATCH = feddrift_batch
    r = run_random_drift_experiment(mode=mode, random_seed=seed, verbose=False, show_plot=False)
    row = {"mode": mode, "dataset": dataset, "seed": seed,
           "feddrift_batch": feddrift_batch if feddrift_batch is not None else ""}
    for k in METRIC_KEYS:
        row[k] = r.get(k)
    return row


def run_sweep(datasets, seeds, batches):
    rows = []
    total = len(datasets) * len(seeds) * (2 + len(batches))
    done = 0
    t0 = time.perf_counter()
    for dataset in datasets:
        for seed in seeds:
            jobs = [("FedSDA", None), ("Oblivious", None)]
            jobs += [("FedDrift", b) for b in batches]
            for mode, b in jobs:
                done += 1
                tag = f"{dataset}/{mode}" + (f"/batch{b}" if b is not None else "") + f"/seed{seed}"
                try:
                    row = _run(mode, dataset, seed, feddrift_batch=b)
                    rows.append(row)
                    print(f"[{done}/{total}] {tag}: paper_acc={row['paper_accuracy']:.4f} "
                          f"comm={row['comm_total']} models={row['final_model_count']} "
                          f"({time.perf_counter()-t0:.0f}s)")
                except Exception:
                    print(f"[{done}/{total}] {tag}: FAILED")
                    traceback.print_exc()
    return rows


def write_csv(rows, path):
    fields = ["mode", "dataset", "seed", "feddrift_batch"] + METRIC_KEYS
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"CSV saved: {path}")


def _agg(rows, x_key="comm_total", y_key="paper_accuracy"):
    """(x_mean, x_std, y_mean, y_std) を返す。空なら None。"""
    if not rows:
        return None
    xs = np.array([r[x_key] for r in rows], dtype=float)
    ys = np.array([r[y_key] for r in rows], dtype=float)
    return xs.mean(), xs.std(), ys.mean(), ys.std()


def plot_pareto(rows, datasets, batches, path):
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        ds_rows = [r for r in rows if r["dataset"] == ds]

        # FedDrift 曲線(バッチごとにシード集約)
        fx, fy, fxe, fye, labels = [], [], [], [], []
        for b in batches:
            sub = [r for r in ds_rows if r["mode"] == "FedDrift" and r["feddrift_batch"] == b]
            a = _agg(sub)
            if a:
                fx.append(a[0]); fxe.append(a[1]); fy.append(a[2]); fye.append(a[3]); labels.append(b)
        if fx:
            # 掃引パラメータ(バッチサイズ)順に線を結ぶ(通信量は batch に単調でないため)
            order = np.argsort(labels)
            fx = np.array(fx)[order]; fy = np.array(fy)[order]
            fxe = np.array(fxe)[order]; fye = np.array(fye)[order]
            labels = [labels[i] for i in order]
            ax.errorbar(fx, fy, xerr=fxe, yerr=fye, marker="o", color="tab:red",
                        capsize=3, label="FedDrift (batch sweep)", zorder=2)
            for b, x, y in zip(labels, fx, fy):
                ax.annotate(f"b={b}", (x, y), fontsize=8, xytext=(4, 4),
                            textcoords="offset points")

        # FedSDA(1点)
        a = _agg([r for r in ds_rows if r["mode"] == "FedSDA"])
        if a:
            ax.errorbar([a[0]], [a[2]], xerr=[a[1]], yerr=[a[3]], marker="*", markersize=18,
                        color="tab:blue", capsize=3, label="FedSDA", zorder=3)

        # Oblivious(基準線)
        a = _agg([r for r in ds_rows if r["mode"] == "Oblivious"])
        if a:
            ax.axhline(a[2], color="gray", linestyle="--", label="Oblivious", zorder=1)

        ax.set_xscale("log")
        ax.set_title(ds)
        ax.set_xlabel("Communication (model transfers, log)")
        ax.set_ylabel("paper_accuracy (omit-drift)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")

    fig.suptitle("Accuracy vs Communication (FedSDA dominates FedDrift's tradeoff curve)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="FedSDA accuracy-communication Pareto sweep")
    all_datasets = list(config._FEATURE_DIMS)
    parser.add_argument("--datasets", nargs="+", choices=all_datasets, default=all_datasets)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--batches", nargs="+", type=int, default=[10, 25, 50, 100, 200, 500],
                        help="FedDrift の検出バッチ掃引値")
    parser.add_argument("--total-data", type=int, default=None,
                        help="TOTAL_DATA_POINTS の上書き(短時間実行用)")
    parser.add_argument("--out-dir", default="results/pareto")
    parser.add_argument("--quick", action="store_true",
                        help="動作確認用の小規模設定(blobs, seed0, batch{50,500}, 総数600)")
    args = parser.parse_args()

    if args.quick:
        args.datasets = ["blobs"]
        args.seeds = [0]
        args.batches = [50, 500]
        config.TOTAL_DATA_POINTS = 600
    elif args.total_data is not None:
        config.TOTAL_DATA_POINTS = args.total_data

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Datasets={args.datasets} seeds={args.seeds} batches={args.batches} "
          f"TOTAL_DATA_POINTS={config.TOTAL_DATA_POINTS}")

    rows = run_sweep(args.datasets, args.seeds, args.batches)
    write_csv(rows, os.path.join(args.out_dir, "pareto_results.csv"))
    plot_pareto(rows, args.datasets, args.batches,
                os.path.join(args.out_dir, "pareto_accuracy_vs_comm.png"))
    print("Done.")


if __name__ == "__main__":
    main()
