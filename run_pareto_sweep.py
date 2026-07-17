"""精度–通信量トレードオフの掃引実験(複数シード × 複数データセット)。

各データセット・各シードで以下を実行し、結果を CSV と散布図に出力する:
- FedSDA v1/v2/v3: δ_adwin と AGG_INTERVAL の掃引
- FedDrift v1/v2 : 検出バッチと検出閾値 δ の掃引
- FedSDA_without_server / Oblivious: 単一点の基準線

FedDrift の各掃引では、固定した側のパラメータを凡例に明示する
(例: "FedDrift batch sweep (δ=0.1)", "FedDrift δ sweep (batch=50)")。
散布図は横軸=通信量(モデル転送数, 対数)、縦軸=stable_accuracy(定常精度)。FedSDA が
FedDrift の各曲線の左上(高精度・低通信)を取れば「パレート支配」を示せる。

例:
    python run_pareto_sweep.py --quick                       # 動作確認(小規模)
    python run_pareto_sweep.py --datasets sea sine --seeds 0 1 2
    python run_pareto_sweep.py                               # 既定(全4データセット × 5シード)※長時間
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

from federated_drift_experiment import config, run_random_drift_experiment

# この実行を一意に識別するタイムスタンプ。--out-dir / --raw-dir を明示しない場合、
# 既定の出力先は results/results_<YYYYMMDD_HHMMSS>/... となり実行ごとに別ディレクトリへ分かれる。
_RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")
_DEFAULT_RUN_DIR = f"results/results_{_RUN_STAMP}"

METRIC_KEYS = [
    "stable_accuracy", "accuracy",
    "comm_models_up", "comm_models_down", "comm_models_total",
    "comm_messages_up", "comm_messages_down", "comm_messages_total",
    "final_model_count", "precision", "recall", "f1", "avg_delay", "total_detect",
]
ROW_KEYS = ["mode", "dataset", "seed", "series", "sweep_value",
            "feddrift_batch", "agg_interval", "distance_threshold", "adwin_delta"] + METRIC_KEYS

FEDSDA_SWEEP_MODES = ("FedSDA", "FedSDA_v2", "FedSDA_v3")
FEDDRIFT_SWEEP_MODES = ("FedDrift", "FedDrift_v2")
BASELINE_MODES = ("FedSDA_without_server", "Oblivious")


def _slug(text):
    """ファイル名に使える形へ簡易サニタイズ(英数以外は _ にまとめる)。"""
    import re
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", str(text))).strip("_")


def _run(mode, dataset, seed, series, sweep_value,
         feddrift_batch=None, distance_threshold=None, adwin_delta=None, agg_interval=None,
         raw_dir=None):
    config.DATASET = dataset
    if feddrift_batch is not None:
        config.FEDDRIFT_DETECT_BATCH = feddrift_batch
    if adwin_delta is not None:
        config.ADWIN_DELTA = adwin_delta
    if agg_interval is not None:
        config.AGG_INTERVAL = agg_interval

    # 回復曲線分析は label 単位でグループ化する(seed をまたいで平均)。掃引値が異なれば
    # 別ハイパーパラメータ設定なので、label に掃引値を含めて別系列として扱う。
    raw_path = None
    raw_label = series
    if raw_dir is not None:
        sv = "na" if sweep_value in (None, "", "None") else f"{sweep_value:g}"
        fname = f"{_slug(series)}_{dataset}_seed{seed}_sv{sv}.npz"
        raw_path = os.path.join(raw_dir, fname)
        if sweep_value not in (None, "", "None"):
            raw_label = f"{series} [{sweep_value:g}]"

    r = run_random_drift_experiment(mode=mode, distance_threshold=distance_threshold,
                                    random_seed=seed, verbose=False, show_plot=False,
                                    raw_path=raw_path, raw_label=raw_label)
    row = {
        "mode": mode, "dataset": dataset, "seed": seed, "series": series, "sweep_value": sweep_value,
        "feddrift_batch": config.FEDDRIFT_DETECT_BATCH,
        "agg_interval": config.AGG_INTERVAL,
        "distance_threshold": distance_threshold if distance_threshold is not None else config.DISTANCE_THRESHOLD,
        "adwin_delta": config.ADWIN_DELTA,
    }
    for k in METRIC_KEYS:
        row[k] = r.get(k)
    return row


def run_sweep(datasets, seeds, batches, deltas, adwin_deltas, fixed_delta, fixed_batch, fixed_gamma,
              agg_sweep=(), fixed_adwin=None, raw_dir=None,
              fedsda_modes=FEDSDA_SWEEP_MODES, feddrift_modes=FEDDRIFT_SWEEP_MODES,
              baseline_modes=BASELINE_MODES):
    default_adwin = config.ADWIN_DELTA
    default_agg = config.AGG_INTERVAL
    if fixed_adwin is None:
        fixed_adwin = default_adwin
    rows = []
    jobs_per = (len(fedsda_modes) * (len(adwin_deltas) + len(agg_sweep))
                + len(feddrift_modes) * (len(batches) + len(deltas))
                + len(baseline_modes))
    total = len(datasets) * len(seeds) * jobs_per
    done = 0
    t0 = time.perf_counter()

    def do(tag, **kw):
        nonlocal done
        done += 1
        try:
            row = _run(raw_dir=raw_dir, **kw)
            rows.append(row)
            print(f"[{done}/{total}] {tag}: stable_acc={row['stable_accuracy']:.4f} "
                  f"comm={row['comm_models_total']} models={row['final_model_count']} "
                  f"({time.perf_counter()-t0:.0f}s)")
        except Exception:
            print(f"[{done}/{total}] {tag}: FAILED")
            traceback.print_exc()

    for dataset in datasets:
        for seed in seeds:
            for mode in fedsda_modes:
                delta_series = f"{mode} δ_adwin sweep (γ={fixed_gamma})"
                agg_series = f"{mode} AGG_INTERVAL sweep (δ_adwin={fixed_adwin})"
                for adwin_delta in adwin_deltas:
                    do(f"{dataset}/{mode}/da={adwin_delta}/s{seed}",
                       mode=mode, dataset=dataset, seed=seed, series=delta_series,
                       sweep_value=adwin_delta, distance_threshold=fixed_gamma,
                       adwin_delta=adwin_delta, agg_interval=default_agg)
                for agg_interval in agg_sweep:
                    do(f"{dataset}/{mode}/agg={agg_interval}/s{seed}",
                       mode=mode, dataset=dataset, seed=seed, series=agg_series,
                       sweep_value=agg_interval, distance_threshold=fixed_gamma,
                       adwin_delta=fixed_adwin, agg_interval=agg_interval)
                config.AGG_INTERVAL = default_agg
                config.ADWIN_DELTA = default_adwin

            for mode in baseline_modes:
                do(f"{dataset}/{mode}/s{seed}", mode=mode, dataset=dataset, seed=seed,
                   series=mode, sweep_value=None, adwin_delta=default_adwin,
                   agg_interval=default_agg)

            for mode in feddrift_modes:
                batch_series = f"{mode} batch sweep (δ={fixed_delta})"
                delta_series = f"{mode} δ sweep (batch={fixed_batch})"
                for batch in batches:
                    do(f"{dataset}/{mode}/batch{batch}/s{seed}",
                       mode=mode, dataset=dataset, seed=seed, series=batch_series,
                       sweep_value=batch, feddrift_batch=batch,
                       distance_threshold=fixed_delta)
                for delta in deltas:
                    do(f"{dataset}/{mode}/delta{delta}/s{seed}",
                       mode=mode, dataset=dataset, seed=seed, series=delta_series,
                       sweep_value=delta, feddrift_batch=fixed_batch,
                       distance_threshold=delta)

    return rows


def _experiment_slug(datasets, seeds, total_data, tag=None):
    """実験内容がわかる出力ファイル名(拡張子なし)を組み立てる。

    例: pareto_sea-circle-sine_seed0_n5000 / pareto_sea_seeds0-2_n5000_myrun
    """
    ds = "-".join(datasets)
    if len(seeds) == 1:
        sd = f"seed{seeds[0]}"
    elif seeds == list(range(seeds[0], seeds[-1] + 1)):
        sd = f"seeds{seeds[0]}-{seeds[-1]}"
    else:
        sd = "seeds" + "-".join(str(s) for s in seeds)
    parts = [f"pareto_{ds}", sd, f"n{total_data}"]
    if tag:
        parts.append(tag)
    return "_".join(parts)


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROW_KEYS)
        w.writeheader()
        w.writerows(rows)
    print(f"CSV saved: {path}")


def _load_csv(path):
    """write_csv が出力した結果CSVを読み込み、数値型に復元した行リストを返す。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row = dict(r)
            row["seed"] = int(float(row["seed"]))
            row["sweep_value"] = (float(row["sweep_value"])
                                  if row["sweep_value"] not in ("", "None") else None)
            row["feddrift_batch"] = (int(float(row["feddrift_batch"]))
                                     if row["feddrift_batch"] not in ("", "None") else "")
            agg_interval = row.get("agg_interval")
            row["agg_interval"] = (int(float(agg_interval))
                                   if agg_interval not in (None, "", "None") else "")
            for k in ["distance_threshold", "adwin_delta"] + METRIC_KEYS:
                v = row.get(k)
                row[k] = float(v) if v not in (None, "", "None") else float("nan")
            rows.append(row)
    return rows


def write_markdown_table(rows, path):
    """(データセット, 系列, 掃引値)ごとにシード平均した Markdown 表を書き出す。"""
    from collections import defaultdict
    canon = list(config._FEATURE_DIMS)
    datasets = [d for d in canon if any(r["dataset"] == d for r in rows)]

    def order_key(item):
        series, sv = item
        base = (0 if series in BASELINE_MODES else
                (1 if "FedSDA" in series else (2 if "batch" in series else 3)))
        try:
            v = float(sv) if sv not in (None, "", "None") else -1.0
        except (TypeError, ValueError):
            v = -1.0
        return (base, v)

    lines = []
    for ds in datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        groups = defaultdict(list)
        for r in ds_rows:
            groups[(r["series"], r["sweep_value"])].append(r)

        lines.append(f"### {ds}")
        lines.append("")
        lines.append("| Method | sweep | Accuracy (overall) | Accuracy (stable) | Comm (transfers) | Models |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for (series, sv) in sorted(groups.keys(), key=order_key):
            rs = groups[(series, sv)]
            overall_acc = np.array([float(x["accuracy"]) for x in rs])
            stable_acc = np.array([float(x["stable_accuracy"]) for x in rs])
            comm = np.array([float(x["comm_models_total"]) for x in rs])
            models = np.array([float(x["final_model_count"]) for x in rs])
            svtxt = "–" if sv in (None, "", "None") else f"{float(sv):g}"
            lines.append(f"| {series} | {svtxt} | "
                         f"{overall_acc.mean():.4f} ± {overall_acc.std():.4f} | "
                         f"{stable_acc.mean():.4f} ± {stable_acc.std():.4f} | "
                         f"{comm.mean():,.0f} | {models.mean():.1f} |")
        lines.append("")

    n_seeds = len(set(r["seed"] for r in rows))
    lines.append(f"*{n_seeds} シード平均。overall は全期間の prequential 精度、stable は回復窓 "
                 f"W={config.STABLE_WINDOW} を除外した定常精度で、いずれも平均±標準偏差。"
                 f"Comm はモデル転送数。*")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Table saved: {path}")


def combine_and_plot(patterns, out_dir, tag=None, plot_metric="stable_accuracy"):
    """複数の結果CSV(glob可)を読み込み、シード平均の散布図を描画する。"""
    import glob
    paths = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)))
    paths = sorted(set(paths))
    if not paths:
        print("No CSV matched the given pattern(s).")
        return
    rows = []
    for p in paths:
        rows.extend(_load_csv(p))
        print(f"loaded: {p}")

    canon = list(config._FEATURE_DIMS)
    datasets = [d for d in canon if any(r["dataset"] == d for r in rows)]
    seeds = sorted(set(r["seed"] for r in rows))
    ds_slug = "-".join(datasets)
    sd = f"seed{seeds[0]}" if len(seeds) == 1 else "seeds" + "-".join(str(s) for s in seeds)
    name = f"pareto_combined_{ds_slug}_{sd}" + (f"_{tag}" if tag else "")

    os.makedirs(out_dir, exist_ok=True)
    print(f"Combining {len(paths)} CSV(s), datasets={datasets}, seeds={seeds} "
          f"(誤差棒/± = シード間の標準偏差)")
    plot_pareto(rows, datasets, os.path.join(out_dir, f"{name}.png"), y_key=plot_metric)
    write_markdown_table(rows, os.path.join(out_dir, f"{name}.md"))


def _agg(rows, x_key="comm_models_total", y_key="stable_accuracy"):
    if not rows:
        return None
    xs = np.array([r[x_key] for r in rows], dtype=float)
    ys = np.array([r[y_key] for r in rows], dtype=float)
    return xs.mean(), xs.std(), ys.mean(), ys.std()


def _fixed_parameter_label(rows, key, display_name):
    """基準方式で固定したパラメータを凡例用に整形する。"""
    values = []
    for row in rows:
        value = row.get(key)
        if value in (None, "", "None"):
            continue
        numeric = float(value)
        if np.isfinite(numeric) and numeric not in values:
            values.append(numeric)
    if len(values) == 1:
        return f"{display_name}={values[0]:g}"
    if len(values) > 1:
        return f"{display_name}=varied"
    return f"{display_name}=?"


def _series_style(series):
    """手法を色、掃引対象をマーカーと線種で表し、系列の見分けを保つ。"""
    method_colors = {
        "FedSDA": "tab:green",
        "FedSDA_v2": "tab:blue",
        "FedSDA_v3": "tab:orange",
        "FedDrift": "tab:purple",
        "FedDrift_v2": "tab:red",
    }
    method = series.split(maxsplit=1)[0]
    color = method_colors.get(method, "tab:brown")
    if "AGG_INTERVAL sweep" in series:
        return color, "s", "--"
    if "batch sweep" in series:
        return color, "^", "-."
    if "δ_adwin sweep" in series:
        return color, "o", "-"
    if "δ sweep" in series:
        return color, "D", ":"
    return color, "X", "-"


def plot_pareto(rows, datasets, path, y_key="stable_accuracy"):
    # 色は手法、マーカーと線種は掃引対象を表す。
    sweep_series = []
    for r in rows:
        if r["series"] not in BASELINE_MODES and r["series"] not in sweep_series:
            sweep_series.append(r["series"])
    style = {series: _series_style(series) for series in sweep_series}

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        ds_rows = [r for r in rows if r["dataset"] == ds]

        # 系列ごとにラベルのオフセット方向を変えて系列間の重なりを軽減
        label_offsets = [(6, 8), (7, -13), (-18, 8), (-18, -13), (6, 18)]
        for si, s in enumerate(sweep_series):
            srows = [r for r in ds_rows if r["series"] == s]
            vals = sorted(set(r["sweep_value"] for r in srows))
            xs, ys, xe, ye = [], [], [], []
            for v in vals:
                a = _agg([r for r in srows if r["sweep_value"] == v], y_key=y_key)
                if a:
                    xs.append(a[0]); xe.append(a[1]); ys.append(a[2]); ye.append(a[3])
            if not xs:
                continue
            color, marker, linestyle = style[s]
            ax.errorbar(xs, ys, xerr=xe, yerr=ye, marker=marker, color=color, markersize=8,
                        linestyle=linestyle, capsize=3, label=s, zorder=2, alpha=0.85,
                        markeredgecolor="white", markeredgewidth=0.6)
            ox, oy = label_offsets[si % len(label_offsets)]
            box = dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.55)
            # 点が密集している系列はラベルを値域1つにまとめる(掃引に不感=ロバストの意)
            clustered = (len(xs) > 1 and min(xs) > 0 and max(xs) / min(xs) < 1.6
                         and (max(ys) - min(ys)) < 0.02)
            if clustered:
                cx = float(np.exp(np.mean(np.log(xs))))
                cy = float(np.mean(ys))
                ax.annotate(f"{min(vals):g}–{max(vals):g}", (cx, cy), fontsize=7, color=color,
                            xytext=(ox, oy), textcoords="offset points", bbox=box)
            else:
                for v, x, y in zip(vals, xs, ys):
                    ax.annotate(f"{v:g}", (x, y), fontsize=7, color=color,
                                xytext=(ox, oy), textcoords="offset points", bbox=box)

        baseline_styles = {
            "Oblivious": ("gray", "--"),
            "FedSDA_without_server": ("black", ":"),
        }
        for baseline in BASELINE_MODES:
            baseline_rows = [r for r in ds_rows if r["series"] == baseline]
            a = _agg(baseline_rows, y_key=y_key)
            if not a:
                continue
            color, linestyle = baseline_styles[baseline]
            if baseline == "Oblivious":
                parameter = _fixed_parameter_label(
                    baseline_rows, "agg_interval", "AGG_INTERVAL")
            else:
                parameter = _fixed_parameter_label(
                    baseline_rows, "adwin_delta", "δ_adwin")
            # 線はシード平均、半透明帯はシード間の±1標準偏差を表す。
            ax.axhline(a[2], color=color, linestyle=linestyle,
                       label=f"{baseline} ({parameter}, mean±std)", zorder=1)
            if a[3] > 0:
                ax.axhspan(a[2] - a[3], a[2] + a[3], color=color, alpha=0.10, zorder=0)

        ax.set_xscale("log")
        ax.set_title(ds)
        ax.set_xlabel("Communication (model transfers, log)")
        ylabel = ("stable_accuracy (omit-recovery)" if y_key == "stable_accuracy"
                  else "accuracy (overall prequential)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")

    fig.suptitle("Accuracy vs Communication: FedSDA vs FedDrift (batch / δ sweeps)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="FedSDA accuracy-communication Pareto sweep")
    all_datasets = list(config._FEATURE_DIMS)
    parser.add_argument("--datasets", nargs="+", choices=all_datasets, default=all_datasets)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--fedsda-modes", nargs="*", choices=FEDSDA_SWEEP_MODES,
                        default=list(FEDSDA_SWEEP_MODES),
                        help="δ_adwin・AGG_INTERVALを掃引するFedSDAモード")
    parser.add_argument("--feddrift-modes", nargs="*", choices=FEDDRIFT_SWEEP_MODES,
                        default=list(FEDDRIFT_SWEEP_MODES),
                        help="検出バッチ・距離閾値を掃引するFedDriftモード")
    parser.add_argument("--baseline-modes", nargs="*", choices=BASELINE_MODES,
                        default=list(BASELINE_MODES),
                        help="単一点の平均線・標準偏差帯として描画する基準モード")
    parser.add_argument("--batches", nargs="*", type=int, default=[25, 50, 100, 200, 500],
                        help="FedDrift 検出バッチ掃引値(空指定で無効化)")
    parser.add_argument("--deltas", nargs="*", type=float, default=[0.05, 0.1, 0.15, 0.2],
                        help="FedDrift 検出閾値 δ 掃引値(空指定で無効化)")
    parser.add_argument("--adwin-deltas", nargs="*", type=float, default=[0.01, 0.05, 0.1, 0.2, 0.3],
                        help="FedSDA δ_adwin 掃引値(空指定で無効化)")
    parser.add_argument("--agg-sweep", nargs="*", type=int, default=[25, 50, 100, 200, 500],
                        help="FedSDA の AGG_INTERVAL(集約間隔)掃引値(空指定で無効化)")
    parser.add_argument("--fixed-adwin", type=float, default=None,
                        help="AGG_INTERVAL 掃引時に固定する δ_adwin(既定 config.ADWIN_DELTA)")
    parser.add_argument("--fixed-delta", type=float, default=None,
                        help="バッチ掃引時に固定する FedDrift δ(既定 config.DISTANCE_THRESHOLD)")
    parser.add_argument("--fixed-batch", type=int, default=None,
                        help="δ 掃引時に固定する FedDrift 検出バッチ(既定 config.FEDDRIFT_DETECT_BATCH)")
    parser.add_argument("--fixed-gamma", type=float, default=None,
                        help="FedSDA で固定する γ_dist(既定 config.DISTANCE_THRESHOLD)")
    parser.add_argument("--total-data", type=int, default=None, help="TOTAL_DATA_POINTS 上書き")
    parser.add_argument("--out-dir", default=f"{_DEFAULT_RUN_DIR}/pareto",
                        help="結果CSV・図の出力先(既定: results/results_<実行時刻>/pareto)")
    parser.add_argument("--raw-dir", default=f"{_DEFAULT_RUN_DIR}/raw",
                        help="各実験の生データ(.npz)の保存先。回復曲線の事後分析用"
                             "(既定: results/results_<実行時刻>/raw)")
    parser.add_argument("--no-recovery", action="store_true",
                        help="掃引後の回復図・表の自動生成を抑止する"
                             "(後から recovery_analysis.py で個別に実行できる)")
    parser.add_argument("--tag", default=None, help="出力ファイル名に付ける任意の識別子")
    parser.add_argument("--plot-csvs", nargs="+", default=None,
                        help="実験は行わず、指定した結果CSV(glob可)を読み込みシード平均で再描画する")
    parser.add_argument("--plot-metric", choices=["stable_accuracy", "accuracy"],
                        default="stable_accuracy",
                        help="Pareto図の縦軸に使う精度指標(既定: stable_accuracy)")
    parser.add_argument("--quick", action="store_true", help="動作確認用の小規模設定")
    args = parser.parse_args()

    # 集約プロットモード: 既存CSVを読み込みシード平均で描画して終了
    if args.plot_csvs:
        combine_and_plot(args.plot_csvs, args.out_dir, args.tag, args.plot_metric)
        return

    if args.quick:
        args.datasets = ["blobs"]
        args.seeds = [0]
        args.batches = [50, 500]
        args.deltas = [0.1, 0.2]
        args.adwin_deltas = [0.05, 0.3]
        args.agg_sweep = [50, 500]
        config.N_CLIENTS = 4
        config.TOTAL_DATA_POINTS = 600
        config.PRETRAIN_SAMPLES = 100
        config.PRETRAIN_EPOCHS = 5
    elif args.total_data is not None:
        config.TOTAL_DATA_POINTS = args.total_data

    fixed_delta = args.fixed_delta if args.fixed_delta is not None else config.DISTANCE_THRESHOLD
    fixed_batch = args.fixed_batch if args.fixed_batch is not None else config.FEDDRIFT_DETECT_BATCH
    fixed_gamma = args.fixed_gamma if args.fixed_gamma is not None else config.DISTANCE_THRESHOLD
    fixed_adwin = args.fixed_adwin if args.fixed_adwin is not None else config.ADWIN_DELTA

    os.makedirs(args.out_dir, exist_ok=True)
    slug = _experiment_slug(args.datasets, args.seeds, config.TOTAL_DATA_POINTS, args.tag)
    n_runs = len(args.datasets) * len(args.seeds) * (
        len(args.fedsda_modes) * (len(args.adwin_deltas) + len(args.agg_sweep))
        + len(args.feddrift_modes) * (len(args.batches) + len(args.deltas))
        + len(args.baseline_modes))
    print(f"Experiment: {slug}")
    print(f"Datasets={args.datasets} seeds={args.seeds} TOTAL_DATA_POINTS={config.TOTAL_DATA_POINTS}")
    print(f"batches={args.batches} deltas={args.deltas} adwin_deltas={args.adwin_deltas} "
          f"agg_sweep={args.agg_sweep}")
    print(f"modes: FedSDA={args.fedsda_modes} FedDrift={args.feddrift_modes} "
          f"baselines={args.baseline_modes}")
    print(f"fixed: delta={fixed_delta} batch={fixed_batch} gamma={fixed_gamma} adwin={fixed_adwin}")
    print(f"Total runs = {n_runs}  (フルスケールでは1実験~60-90秒。長時間になり得ます)")

    if args.raw_dir:
        os.makedirs(args.raw_dir, exist_ok=True)
        print(f"Raw per-run data (.npz) -> {args.raw_dir}")

    rows = run_sweep(args.datasets, args.seeds, args.batches, args.deltas, args.adwin_deltas,
                     fixed_delta, fixed_batch, fixed_gamma,
                     agg_sweep=args.agg_sweep, fixed_adwin=fixed_adwin, raw_dir=args.raw_dir,
                     fedsda_modes=args.fedsda_modes, feddrift_modes=args.feddrift_modes,
                     baseline_modes=args.baseline_modes)
    write_csv(rows, os.path.join(args.out_dir, f"{slug}.csv"))
    plot_pareto(rows, args.datasets, os.path.join(args.out_dir, f"{slug}.png"),
                y_key=args.plot_metric)

    # 掃引で保存した生データ(.npz)から回復図・表を自動生成する。
    # recovery は軽い事後分析なので、パラメータを変えて後から recovery_analysis.py 単体で
    # 何度でも回せる(--no-recovery でこの自動実行を抑止)。
    if not args.no_recovery and args.raw_dir:
        import glob
        from recovery_analysis import load_npz, generate_recovery_outputs, infer_out_dir
        npz_paths = sorted(glob.glob(os.path.join(args.raw_dir, "*.npz")))
        if npz_paths:
            rec_dir = infer_out_dir(npz_paths)
            print(f"回復分析: {len(npz_paths)} npz -> {rec_dir}")
            recs = [load_npz(p) for p in npz_paths]
            generate_recovery_outputs(recs, rec_dir, tag=args.tag)
        else:
            print("回復分析: raw npz が見つからないためスキップ")
    print("Done.")


if __name__ == "__main__":
    main()
