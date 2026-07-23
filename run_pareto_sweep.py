"""精度–通信量トレードオフの掃引実験(複数シード × 複数データセット)。

各データセット・各シードで以下を実行し、結果を CSV と散布図に出力する:
- FedSDA Cached/NoCached: 検出器パラメータと AGG_INTERVAL の掃引
- FedDrift: 検出バッチと検出閾値 δ の掃引
- FedSDA_without_server / Oblivious: 単一点の基準線

FedDrift の各掃引では、固定した側のパラメータを凡例に明示する
(例: "FedDrift B_detect sweep (δ_FedDrift=0.1)", "FedDrift δ_FedDrift sweep (B_detect=50)")。
散布図は横軸=通信量(モデル転送数, 対数)、縦軸=stable_accuracy(定常精度)。FedSDA が
FedDrift の各曲線の左上(高精度・低通信)を取れば「パレート支配」を示せる。

例:
    python run_pareto_sweep.py --quick                       # 動作確認(小規模)
    python run_pareto_sweep.py --datasets sea4 sine2 --seeds 0 1 2
    python run_pareto_sweep.py                               # 既定(全4データセット × 5シード)※長時間
"""
import argparse
import csv
import os
import time
import traceback

from experiment_runtime import configure_torch_threads

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from federated_drift_experiment import config, run_random_drift_experiment
from federated_drift_experiment.data import dataset_cli_choices, normalize_dataset_name
from federated_drift_experiment.mode_names import (
    BASELINE_MODES,
    FEDDRIFT_MODES,
    FEDSDA_MODES,
    is_adwin_mode,
    is_esr_mode,
    is_hddm_mode,
    normalize_legacy_mode,
    normalize_legacy_series,
    normalize_series_notation,
)

# この実行を一意に識別するタイムスタンプ。--out-dir / --raw-dir を明示しない場合、
# 既定の出力先は results/results_<YYYYMMDD_HHMMSS>/... となり実行ごとに別ディレクトリへ分かれる。
_RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")
_DEFAULT_RUN_DIR = f"results/results_{_RUN_STAMP}"

METRIC_KEYS = [
    "stable_accuracy", "accuracy",
    "comm_models_up", "comm_models_down", "comm_models_total",
    "comm_messages_up", "comm_messages_down", "comm_messages_total",
    "final_model_count", "precision", "recall", "f1", "avg_delay", "total_detect",
    "change_point_mae", "change_point_bias", "change_point_estimate_count",
    "alarm_precision", "alarm_recall", "alarm_f1", "alarm_total",
    "switch_fp_early", "switch_fp_late", "switch_fp_duplicate", "switch_fp_isolated",
    "adaptation_reuse_count", "adaptation_reuse_precision",
    "adaptation_create_count", "adaptation_create_precision",
    "adaptation_create_rejected_count",
    "provisional_proposal_count", "provisional_acceptance_rate",
    "provisional_matched_true_count", "provisional_accepted_matched_true_count",
    "provisional_rejected_matched_true_count", "provisional_accepted_precision",
    "provisional_interval_count_mean", "provisional_training_count_mean",
    "provisional_validation_count_mean",
    "provisional_accepted_full_margin_mean",
    "provisional_accepted_recent_margin_mean",
    "provisional_rejected_full_margin_mean",
    "provisional_rejected_recent_margin_mean",
    "provisional_reject_insufficient_data_count",
    "provisional_reject_full_interval_count",
    "provisional_reject_recent_interval_count",
    "provisional_reject_full_and_recent_count",
    "adaptation_maintain_count", "adaptation_episode_suppressed_count",
    "server_mapping_change_count",
    "runtime_seconds", "client_compute_seconds_sum", "client_compute_seconds_max",
    "compute_inference_examples_total", "compute_training_examples_total",
    "compute_model_examples_total", "compute_optimizer_steps_total",
    "compute_drift_detector_updates_total", "compute_drift_detector_hypotheses_total",
    "mean_model_count", "max_model_count", "model_count_auc",
]
ROW_KEYS = ["mode", "dataset", "concept_schedule", "seed", "series", "sweep_value",
            "feddrift_batch", "agg_interval", "clustering_policy", "detection_episodes",
            "new_model_creation_policy",
            "fifo_size", "new_model_validation_fraction",
            "distance_threshold", "adwin_delta",
            ] + METRIC_KEYS

FEDSDA_SWEEP_MODES = FEDSDA_MODES
FEDDRIFT_SWEEP_MODES = FEDDRIFT_MODES

PLOT_X_LABELS = {
    "comm_models_total": "Communication (model transfers, log)",
    "compute_model_examples_total": "Model-processed examples (log)",
    "compute_optimizer_steps_total": "Optimizer steps (log)",
    "client_compute_seconds_sum": "Client compute time (seconds, log)",
    "runtime_seconds": "Runtime (seconds, log)",
}


def _slug(text):
    """ファイル名に使える形へ簡易サニタイズ(英数以外は _ にまとめる)。"""
    import re
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", str(text))).strip("_")


def _run(mode, dataset, seed, series, sweep_value,
         feddrift_batch=None, distance_threshold=None, adwin_delta=None, agg_interval=None,
         raw_dir=None, concept_schedule=None):
    config.DATASET = dataset
    if concept_schedule is None:
        concept_schedule = config.CONCEPT_SCHEDULE
    config.CONCEPT_SCHEDULE = concept_schedule
    if feddrift_batch is not None:
        config.FEDDRIFT_DETECT_BATCH = feddrift_batch
    if adwin_delta is not None:
        config.ADWIN_DELTA = adwin_delta
    if agg_interval is not None:
        config.AGG_INTERVAL = agg_interval

    # 回復曲線分析は label 単位でグループ化する(seed をまたいで平均)。掃引値が異なれば
    # 別ハイパーパラメータ設定なので、label に掃引値を含めて別系列として扱う。
    raw_path = None
    display_series = (series if concept_schedule == "random"
                      else f"{series} [{concept_schedule}]")
    if "FedSDA" in mode and config.FEDSDA_CLUSTERING_POLICY != "on_new_model":
        display_series = (
            f"{display_series} [cluster={config.FEDSDA_CLUSTERING_POLICY}]"
        )
    if "FedSDA" in mode and config.FEDSDA_DETECTION_EPISODES_ENABLED:
        display_series = f"{display_series} [episodes]"
    if "FedSDA" in mode and config.NEW_MODEL_CREATION_POLICY != "immediate":
        display_series = (
            f"{display_series} [creation={config.NEW_MODEL_CREATION_POLICY}]"
        )
    if "FedSDA" in mode:
        display_series = f"{display_series} [N_FIFO={config.FIFO_BUFFER_SIZE}]"
    if (
        "FedSDA" in mode
        and config.NEW_MODEL_CREATION_POLICY == "validated"
    ):
        display_series = (
            f"{display_series} [validation={config.NEW_MODEL_VALIDATION_FRACTION:g}]"
        )
    raw_label = display_series
    if raw_dir is not None:
        sv = "na" if sweep_value in (None, "", "None") else f"{sweep_value:g}"
        fname = f"{_slug(display_series)}_{dataset}_seed{seed}_sv{sv}.npz"
        raw_path = os.path.join(raw_dir, fname)
        if sweep_value not in (None, "", "None"):
            raw_label = f"{display_series} [{sweep_value:g}]"

    r = run_random_drift_experiment(mode=mode, distance_threshold=distance_threshold,
                                    random_seed=seed, verbose=False, show_plot=False,
                                    raw_path=raw_path, raw_label=raw_label)
    row = {
        "mode": mode, "dataset": dataset, "concept_schedule": concept_schedule,
        "seed": seed, "series": display_series, "sweep_value": sweep_value,
        "feddrift_batch": config.FEDDRIFT_DETECT_BATCH,
        "agg_interval": config.AGG_INTERVAL,
        "clustering_policy": config.FEDSDA_CLUSTERING_POLICY,
        "detection_episodes": config.FEDSDA_DETECTION_EPISODES_ENABLED,
        "new_model_creation_policy": config.NEW_MODEL_CREATION_POLICY,
        "fifo_size": config.FIFO_BUFFER_SIZE,
        "new_model_validation_fraction": config.NEW_MODEL_VALIDATION_FRACTION,
        "distance_threshold": distance_threshold if distance_threshold is not None else config.DISTANCE_THRESHOLD,
        "adwin_delta": config.ADWIN_DELTA,
    }
    for k in METRIC_KEYS:
        row[k] = r.get(k)
    return row


def run_sweep(datasets, seeds, batches, deltas, adwin_deltas, fixed_delta, fixed_batch, fixed_gamma,
              agg_sweep=(), fixed_adwin=None, fixed_agg=None, raw_dir=None,
              fedsda_modes=FEDSDA_SWEEP_MODES, feddrift_modes=FEDDRIFT_SWEEP_MODES,
              baseline_modes=BASELINE_MODES, concept_schedule=None):
    if concept_schedule is None:
        concept_schedule = config.CONCEPT_SCHEDULE
    default_adwin = config.ADWIN_DELTA
    default_agg = config.AGG_INTERVAL
    if fixed_adwin is None:
        fixed_adwin = default_adwin
    if fixed_agg is None:
        fixed_agg = default_agg
    rows = []
    adwin_mode_count = sum(is_adwin_mode(mode) for mode in fedsda_modes)
    jobs_per = (adwin_mode_count * len(adwin_deltas)
                + len(fedsda_modes) * len(agg_sweep)
                + len(feddrift_modes) * (len(batches) + len(deltas))
                + len(baseline_modes))
    total = len(datasets) * len(seeds) * jobs_per
    done = 0
    t0 = time.perf_counter()

    def do(tag, **kw):
        nonlocal done
        done += 1
        try:
            row = _run(raw_dir=raw_dir, concept_schedule=concept_schedule, **kw)
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
                is_e_detector = is_esr_mode(mode)
                is_hddm = is_hddm_mode(mode)
                if is_e_detector:
                    fixed_detector = f"alpha_e={config.E_DETECTOR_ALPHA}"
                elif is_hddm:
                    fixed_detector = f"confidence={config.HDDM_DRIFT_CONFIDENCE}"
                else:
                    fixed_detector = f"δ_ADWIN={fixed_adwin}"
                agg_series = f"{mode} A sweep ({fixed_detector}, γ={fixed_gamma})"
                if is_adwin_mode(mode):
                    delta_series = f"{mode} δ_ADWIN sweep (A={fixed_agg}, γ={fixed_gamma})"
                    for adwin_delta in adwin_deltas:
                        do(f"{dataset}/{mode}/da={adwin_delta}/s{seed}",
                           mode=mode, dataset=dataset, seed=seed, series=delta_series,
                           sweep_value=adwin_delta, distance_threshold=fixed_gamma,
                           adwin_delta=adwin_delta, agg_interval=fixed_agg)
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
                batch_series = f"{mode} B_detect sweep (δ_FedDrift={fixed_delta})"
                delta_series = f"{mode} δ_FedDrift sweep (B_detect={fixed_batch})"
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


def _experiment_slug(datasets, seeds, total_data, tag=None, concept_schedule="random"):
    """実験内容がわかる出力ファイル名(拡張子なし)を組み立てる。

    例: pareto_sea4-circle2-sine2_seed0_n5000 / pareto_sea4_seeds0-2_n5000_myrun
    """
    ds = "-".join(datasets)
    if len(seeds) == 1:
        sd = f"seed{seeds[0]}"
    elif seeds == list(range(seeds[0], seeds[-1] + 1)):
        sd = f"seeds{seeds[0]}-{seeds[-1]}"
    else:
        sd = "seeds" + "-".join(str(s) for s in seeds)
    parts = [f"pareto_{ds}", sd, f"n{total_data}"]
    if concept_schedule != "random":
        parts.append(concept_schedule)
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
            old_mode = row["mode"]
            row["mode"] = normalize_legacy_mode(old_mode)
            row["series"] = normalize_legacy_series(
                row.get("series", ""), old_mode, row["mode"]
            )
            row["series"] = normalize_series_notation(row["series"])
            row.setdefault("concept_schedule", "random")
            row["dataset"] = normalize_dataset_name(row["dataset"])
            row["seed"] = int(float(row["seed"]))
            row["sweep_value"] = (float(row["sweep_value"])
                                  if row["sweep_value"] not in ("", "None") else None)
            feddrift_batch = row.get("feddrift_batch", row.get("b_detect"))
            row["feddrift_batch"] = (
                int(float(feddrift_batch))
                if feddrift_batch not in (None, "", "None") else ""
            )
            if "distance_threshold" not in row and "delta_feddrift" in row:
                row["distance_threshold"] = row["delta_feddrift"]
            agg_interval = row.get("agg_interval")
            row["agg_interval"] = (int(float(agg_interval))
                                   if agg_interval not in (None, "", "None") else "")
            row.setdefault("clustering_policy", "on_new_model")
            row.setdefault("detection_episodes", "False")
            row.setdefault("new_model_creation_policy", "immediate")
            row.setdefault("fifo_size", str(config.FIFO_BUFFER_SIZE))
            row.setdefault(
                "new_model_validation_fraction",
                str(config.NEW_MODEL_VALIDATION_FRACTION),
            )
            for k in ["distance_threshold", "adwin_delta"] + METRIC_KEYS:
                v = row.get(k)
                row[k] = float(v) if v not in (None, "", "None") else float("nan")
            rows.append(row)
    return rows


def write_markdown_table(rows, path, x_key="comm_models_total"):
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
        x_label = PLOT_X_LABELS.get(x_key, x_key).replace(" (log)", "")
        lines.append(f"| Method | sweep | Accuracy (overall) | Accuracy (stable) | {x_label} | Models |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for (series, sv) in sorted(groups.keys(), key=order_key):
            rs = groups[(series, sv)]
            overall_acc = np.array([float(x["accuracy"]) for x in rs])
            stable_acc = np.array([float(x["stable_accuracy"]) for x in rs])
            x_values = np.array([float(x[x_key]) for x in rs])
            models = np.array([float(x["final_model_count"]) for x in rs])
            svtxt = "–" if sv in (None, "", "None") else f"{float(sv):g}"
            lines.append(f"| {series} | {svtxt} | "
                         f"{overall_acc.mean():.4f} ± {overall_acc.std():.4f} | "
                         f"{stable_acc.mean():.4f} ± {stable_acc.std():.4f} | "
                         f"{x_values.mean():,.2f} | {models.mean():.1f} |")
        lines.append("")

    n_seeds = len(set(r["seed"] for r in rows))
    lines.append(f"*{n_seeds} シード平均。overall は全期間の prequential 精度、stable は回復窓 "
                 f"W={config.STABLE_WINDOW} を除外した定常精度で、いずれも平均±標準偏差。"
                 f"横軸集計値は `{x_key}`。*")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Table saved: {path}")


def _filter_replot_rows(rows, modes=None, sweep_kind="all"):
    """既存CSVから再描画対象の手法・掃引系列だけを選ぶ。"""
    if modes:
        rows = [row for row in rows if row["mode"] in modes]
    if sweep_kind == "interval":
        rows = [
            row for row in rows
            if ((row["mode"].startswith("FedSDA")
                 and "A sweep" in row["series"])
                or (row["mode"].startswith("FedDrift")
                    and "B_detect sweep" in row["series"]))
        ]
    return rows


def combine_and_plot(patterns, out_dir, tag=None, plot_metric="stable_accuracy",
                     plot_x_metric="comm_models_total", modes=None,
                     sweep_kind="all"):
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
    rows = _filter_replot_rows(rows, modes=modes, sweep_kind=sweep_kind)
    if not rows:
        print("No rows remained after applying plot filters.")
        return

    canon = list(config._FEATURE_DIMS)
    datasets = [d for d in canon if any(r["dataset"] == d for r in rows)]
    seeds = sorted(set(r["seed"] for r in rows))
    ds_slug = "-".join(datasets)
    sd = f"seed{seeds[0]}" if len(seeds) == 1 else "seeds" + "-".join(str(s) for s in seeds)
    name = f"pareto_combined_{ds_slug}_{sd}" + (f"_{tag}" if tag else "")

    os.makedirs(out_dir, exist_ok=True)
    print(f"Combining {len(paths)} CSV(s), datasets={datasets}, seeds={seeds} "
          f"(誤差棒/± = シード間の標準偏差)")
    plot_pareto(
        rows, datasets, os.path.join(out_dir, f"{name}.png"),
        y_key=plot_metric, x_key=plot_x_metric,
    )
    write_markdown_table(
        rows, os.path.join(out_dir, f"{name}.md"), x_key=plot_x_metric
    )


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
        "FedSDA_NoCached_ADWIN": "tab:blue",
        "FedSDA_NoCached_ClassADWIN": "tab:cyan",
        "FedSDA_NoCached_ESR": "deepskyblue",
        "FedSDA_NoCached_HDDMA": "cornflowerblue",
        "FedSDA_NoCached_ClassHDDMA": "mediumslateblue",
        "FedSDA_NoCached_HDDMW": "slateblue",
        "FedSDA_NoCached_ClassESR": "dodgerblue",
        "FedSDA_Cached_ADWIN": "tab:orange",
        "FedSDA_Cached_ClassADWIN": "tab:pink",
        "FedSDA_Cached_ESR": "tab:olive",
        "FedSDA_Cached_HDDMA": "goldenrod",
        "FedSDA_Cached_ClassHDDMA": "peru",
        "FedSDA_Cached_HDDMW": "darkorange",
        "FedSDA_Cached_ClassESR": "darkolivegreen",
        "FedSDA_Legacy": "tab:green",
        "FedDrift": "tab:red",
    }
    method = series.split(maxsplit=1)[0]
    color = method_colors.get(method, "tab:brown")
    if "A sweep" in series:
        return color, "s", "--"
    if "B_detect sweep" in series:
        return color, "^", "-."
    if "δ_ADWIN sweep" in series:
        return color, "o", "-"
    if "δ_FedDrift sweep" in series:
        return color, "D", ":"
    return color, "X", "-"


def plot_pareto(rows, datasets, path, y_key="stable_accuracy",
                x_key="comm_models_total"):
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
                a = _agg(
                    [r for r in srows if r["sweep_value"] == v],
                    x_key=x_key, y_key=y_key,
                )
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
            a = _agg(baseline_rows, x_key=x_key, y_key=y_key)
            if not a:
                continue
            color, linestyle = baseline_styles[baseline]
            if baseline == "Oblivious":
                parameter = _fixed_parameter_label(
                    baseline_rows, "agg_interval", "A")
            else:
                parameter = _fixed_parameter_label(
                    baseline_rows, "adwin_delta", "δ_ADWIN")
            # 線はシード平均、半透明帯はシード間の±1標準偏差を表す。
            ax.axhline(a[2], color=color, linestyle=linestyle,
                       label=f"{baseline} ({parameter}, mean±std)", zorder=1)
            if a[3] > 0:
                ax.axhspan(a[2] - a[3], a[2] + a[3], color=color, alpha=0.10, zorder=0)

        ax.set_xscale("log")
        ax.set_title(ds)
        ax.set_xlabel(PLOT_X_LABELS.get(x_key, x_key))
        ylabel = ("stable_accuracy (omit-recovery)" if y_key == "stable_accuracy"
                  else "accuracy (overall prequential)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")

    methods = []
    for row in rows:
        method = row["mode"]
        if method not in methods:
            methods.append(method)
    fig.suptitle(f"{y_key} vs {x_key} by Method\n" + ", ".join(methods))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {path}")


def build_parser():
    """関連するオプションを手法・用途ごとにまとめたCLIパーサーを返す。"""
    parser = argparse.ArgumentParser(
        description="FedSDA accuracy-communication Pareto sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""指定例:
  FedSDAだけ実行:
    --feddrift-modes --baseline-modes
  FedSDAのAGG_INTERVAL掃引だけ実行:
    --adwin-deltas --agg-sweep 25 50 100
  FedDriftの距離閾値δ掃引だけ実行:
    --batches --deltas 0.05 0.1 0.2
  既存CSVだけ再描画:
    --plot-csvs results/pareto/*.csv --plot-metric accuracy

値を取らない空指定の例は「--batches」のように、次のオプションを直後に置く。
""",
    )
    all_datasets = dataset_cli_choices(config._FEATURE_DIMS)
    # blobs・固定系列・MNISTは計算量や実験上の位置づけが異なるため、明示指定時だけ実行する。
    default_datasets = ["sea4", "circle2", "sine2"]
    scope = parser.add_argument_group("実験対象・規模")
    scope.add_argument("--datasets", nargs="+", choices=all_datasets, default=default_datasets,
                       help="対象データセット(既定: sea4 circle2 sine2。blobs・MNIST等は明示指定)")
    scope.add_argument("--concept-schedule", choices=config.CONCEPT_SCHEDULES,
                       default=config.CONCEPT_SCHEDULE,
                       help=f"全データセットに適用する概念切替方式(既定: {config.CONCEPT_SCHEDULE})")
    scope.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                       help="乱数シード(既定: 0 1 2 3 4)")
    scope.add_argument("--total-data", type=int, default=None,
                       help="TOTAL_DATA_POINTSを上書き")
    scope.add_argument("--quick", action="store_true",
                       help="データセット・シード・掃引値・データ量を小規模設定で上書き")

    fedsda = parser.add_argument_group("FedSDAの手法・掃引")
    fedsda.add_argument("--fedsda-modes", nargs="*", choices=FEDSDA_SWEEP_MODES,
                        default=list(FEDSDA_SWEEP_MODES),
                        help="対象モード。空指定でFedSDAをすべて無効化")
    fedsda.add_argument("--adwin-deltas", nargs="*", type=float,
                        default=[0.01, 0.05, 0.1, 0.2, 0.3],
                        help="δ_ADWIN掃引値。空指定でこの掃引を無効化")
    fedsda.add_argument("--agg-sweep", nargs="*", type=int,
                        default=[50, 100, 200, 500],
                        help="集約間隔A（AGG_INTERVAL）の掃引値。空指定で無効化")
    fedsda.add_argument("--fixed-adwin", type=float, default=None,
                        help="A掃引中の固定δ_ADWIN。--agg-sweepが空なら未使用")
    fedsda.add_argument("--fixed-agg", type=int, default=None,
                        help="δ_ADWIN掃引中の固定A。--adwin-deltasが空なら未使用")
    fedsda.add_argument(
        "--clustering-policy",
        choices=config.FEDSDA_CLUSTERING_POLICIES,
        default=config.FEDSDA_CLUSTERING_POLICY,
        help="FedSDAのクラスタリング頻度",
    )
    fedsda.add_argument(
        "--detection-episodes",
        action=argparse.BooleanOptionalAction,
        default=config.FEDSDA_DETECTION_EPISODES_ENABLED,
        help="近接した検出をN_FIFO幅の一つの適応エピソードへ統合する",
    )
    fedsda.add_argument(
        "--new-model-creation-policy",
        choices=config.NEW_MODEL_CREATION_POLICIES,
        default=config.NEW_MODEL_CREATION_POLICY,
        help="FedSDAの新規モデル作成方針（immediate / validated）",
    )
    fedsda.add_argument(
        "--fifo-size", type=int, default=config.FIFO_BUFFER_SIZE,
        help="FedSDAのFIFOバッファ長 N_FIFO",
    )
    fedsda.add_argument(
        "--new-model-validation-fraction", type=float,
        default=config.NEW_MODEL_VALIDATION_FRACTION,
        help="検証付き仮モデルで末尾から検証用に確保する割合",
    )
    fedsda.add_argument("--fixed-gamma", type=float, default=None,
                        help="FedSDAの固定γ_dist。FedSDA掃引がすべて空なら未使用")

    feddrift = parser.add_argument_group("FedDriftの手法・掃引")
    feddrift.add_argument("--feddrift-modes", nargs="*", choices=FEDDRIFT_SWEEP_MODES,
                          default=list(FEDDRIFT_SWEEP_MODES),
                          help="対象モード。空指定でFedDriftをすべて無効化")
    feddrift.add_argument("--batches", nargs="*", type=int,
                          default=[50, 100, 200, 500],
                          help="検出バッチB_detectの掃引値。空指定でこの掃引を無効化")
    feddrift.add_argument("--fixed-delta", type=float, default=None,
                          help="B_detect掃引中の固定δ_FedDrift。--batchesが空なら未使用")
    feddrift.add_argument("--deltas", nargs="*", type=float,
                          default=[0.05, 0.1, 0.15, 0.2],
                          help="δ_FedDriftの掃引値。空指定でこの掃引を無効化")
    feddrift.add_argument("--fixed-batch", type=int, default=None,
                          help="δ_FedDrift掃引中の固定B_detect。--deltasが空なら未使用")

    baselines = parser.add_argument_group("基準手法")
    baselines.add_argument("--baseline-modes", nargs="*", choices=BASELINE_MODES,
                           default=list(BASELINE_MODES),
                           help="単一点の基準線。空指定で基準手法をすべて無効化")

    output = parser.add_argument_group("新規実験の出力・回復分析")
    output.add_argument("--out-dir", default=f"{_DEFAULT_RUN_DIR}/pareto",
                        help="結果CSV・図の出力先(既定: results/results_<実行時刻>/pareto)")
    output.add_argument("--raw-dir", default=f"{_DEFAULT_RUN_DIR}/raw",
                        help="生データ(.npz)の保存先。空文字なら保存と回復分析を無効化")
    output.add_argument("--no-recovery", action="store_true",
                        help="生データは保存するが、掃引後の回復図・表の自動生成を抑止")
    output.add_argument("--tag", default=None, help="出力ファイル名に付ける識別子")

    replot = parser.add_argument_group("既存CSVの再描画")
    replot.add_argument("--plot-csvs", nargs="+", default=None,
                        help="実験を行わず指定CSV(glob可)を再描画。他の実験設定は無視")
    replot.add_argument("--plot-metric", choices=["stable_accuracy", "accuracy"],
                        default="stable_accuracy",
                        help="Pareto図の縦軸。新規実験と再描画の両方に適用")
    replot.add_argument("--plot-x-metric", choices=list(PLOT_X_LABELS),
                        default="comm_models_total",
                        help="再描画時の横軸。計算量・実行時間も選択可能")
    replot.add_argument("--plot-modes", nargs="*",
                        choices=FEDSDA_SWEEP_MODES + FEDDRIFT_SWEEP_MODES + BASELINE_MODES,
                        default=None,
                        help="再描画対象の手法。未指定ならCSV内の全手法")
    replot.add_argument("--plot-sweep-kind", choices=("all", "interval"), default="all",
                        help="intervalでFedSDAのAとFedDriftのB_detect掃引だけを描画")
    return parser


def main():
    configure_torch_threads()
    parser = build_parser()
    args = parser.parse_args()
    args.datasets = [normalize_dataset_name(dataset) for dataset in args.datasets]

    if args.fifo_size < 1:
        parser.error("--fifo-size must be at least 1")
    if not 0.0 < args.new_model_validation_fraction < 1.0:
        parser.error("--new-model-validation-fraction must be between 0 and 1")

    # 集約プロットモード: 既存CSVを読み込みシード平均で描画して終了
    if args.plot_csvs:
        combine_and_plot(
            args.plot_csvs, args.out_dir, args.tag, args.plot_metric,
            plot_x_metric=args.plot_x_metric, modes=args.plot_modes,
            sweep_kind=args.plot_sweep_kind,
        )
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
    config.CONCEPT_SCHEDULE = args.concept_schedule
    config.FEDSDA_CLUSTERING_POLICY = args.clustering_policy
    config.FEDSDA_DETECTION_EPISODES_ENABLED = args.detection_episodes
    config.NEW_MODEL_CREATION_POLICY = args.new_model_creation_policy
    config.FIFO_BUFFER_SIZE = args.fifo_size
    config.NEW_MODEL_VALIDATION_FRACTION = args.new_model_validation_fraction

    fixed_delta = args.fixed_delta if args.fixed_delta is not None else config.DISTANCE_THRESHOLD
    fixed_batch = args.fixed_batch if args.fixed_batch is not None else config.FEDDRIFT_DETECT_BATCH
    fixed_gamma = args.fixed_gamma if args.fixed_gamma is not None else config.DISTANCE_THRESHOLD
    fixed_adwin = args.fixed_adwin if args.fixed_adwin is not None else config.ADWIN_DELTA
    fixed_agg = args.fixed_agg if args.fixed_agg is not None else config.AGG_INTERVAL

    os.makedirs(args.out_dir, exist_ok=True)
    slug = _experiment_slug(
        args.datasets, args.seeds, config.TOTAL_DATA_POINTS, args.tag,
        concept_schedule=args.concept_schedule,
    )
    n_runs = len(args.datasets) * len(args.seeds) * (
        len(args.fedsda_modes) * (len(args.adwin_deltas) + len(args.agg_sweep))
        + len(args.feddrift_modes) * (len(args.batches) + len(args.deltas))
        + len(args.baseline_modes))
    print(f"Experiment: {slug}")
    print(f"Datasets={args.datasets} schedule={args.concept_schedule} "
          f"seeds={args.seeds} TOTAL_DATA_POINTS={config.TOTAL_DATA_POINTS}")
    print(f"batches={args.batches} deltas={args.deltas} adwin_deltas={args.adwin_deltas} "
          f"agg_sweep={args.agg_sweep}")
    print(f"modes: FedSDA={args.fedsda_modes} FedDrift={args.feddrift_modes} "
          f"baselines={args.baseline_modes}")
    print(f"fixed: delta={fixed_delta} batch={fixed_batch} gamma={fixed_gamma} "
          f"adwin={fixed_adwin} agg={fixed_agg}")
    print(f"Total runs = {n_runs}  (フルスケールでは1実験~60-90秒。長時間になり得ます)")

    if args.raw_dir:
        os.makedirs(args.raw_dir, exist_ok=True)
        print(f"Raw per-run data (.npz) -> {args.raw_dir}")

    rows = run_sweep(args.datasets, args.seeds, args.batches, args.deltas, args.adwin_deltas,
                     fixed_delta, fixed_batch, fixed_gamma,
                     agg_sweep=args.agg_sweep, fixed_adwin=fixed_adwin,
                     fixed_agg=fixed_agg, raw_dir=args.raw_dir,
                     fedsda_modes=args.fedsda_modes, feddrift_modes=args.feddrift_modes,
                     baseline_modes=args.baseline_modes,
                     concept_schedule=args.concept_schedule)
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
