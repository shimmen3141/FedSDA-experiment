"""ドリフト後の回復曲線 acc(Δ) を生データ(.npz)から集計・比較する。

run_experiment.py / run_pareto_sweep.py の --raw-dir で保存した .npz を読み込み、
各ドリフト発生点を Δ=0 に揃えて「ドリフト後 Δ サンプル時点の平均精度 acc(Δ)」を求める。
これは検出遅延(手法ごとに定義が揺れる)に代わる outcome ベースの適応速度指標:
実際にモデルが誤り続けた度合いを直接測る。

出力:
- 図: データセット別に acc(Δ) を手法(label)ごとに重ね描き(シード間 std を帯で表示)
- 表(Markdown): 固定オフセット精度 acc@Δ(既定 20/50/100)と、
  固定窓 [0, W) の平均精度(= 適応リグレットの符号反転。W は次ドリフトを跨がない長さ)

Δ 上限・窓幅は MIN_STABLE_PERIOD 未満に取るため、集計窓が必ず単一の安定区間に収まる
(次ドリフトの混入がない)。

例:
    python recovery_analysis.py --npz "results/raw/*.npz"
    python recovery_analysis.py --npz "results/raw/*sine*.npz" --window 200 --checkpoints 20 50 100
"""
import argparse
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# データセットの正準表示順(存在するものだけ使う)
_CANON_DATASETS = ["blobs", "sea", "circle", "sine"]


def infer_out_dir(npz_paths):
    """分析対象 .npz の場所から、紐づく results ディレクトリ内の recovery/ を推定する。

    典型レイアウト results_<stamp>/raw/*.npz に対し results_<stamp>/recovery を返す
    (npz の共通親ディレクトリが raw なら、その親=実験ディレクトリの直下に置く)。
    複数実験にまたがる glob の場合は共通祖先ディレクトリ直下の recovery/ に集約する。
    """
    dirs = [os.path.dirname(p) or "." for p in npz_paths]
    common = os.path.commonpath(dirs)
    if os.path.basename(common) == "raw":
        common = os.path.dirname(common) or "."
    return os.path.join(common, "recovery")


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    return {
        "history": d["history_accuracy"],        # (N_CLIENTS, N_SAMPLES) int8 の 0/1
        "d_cids": d["drift_client_ids"],
        "d_pos": d["drift_positions"],
        "dataset": str(d["dataset"]),
        "label": str(d["label"]),
        "seed": int(d["seed"]),
        "min_stable": int(d["min_stable"]),
    }


def curve_sums(rec, max_delta):
    """1 ファイル(=1 シード)について、Δ=0..max_delta の精度合計と件数を返す。

    各ドリフト (client ci, 位置 p) について history[ci, p:p+Δ+1] を Δ=0 起点で足し込む。
    ストリーム終端で切れる分は、その Δ の件数に計上しない(短い窓のバイアスを避ける)。
    """
    hist = rec["history"]
    n_samples = hist.shape[1]
    sums = np.zeros(max_delta + 1)
    counts = np.zeros(max_delta + 1)
    for ci, p in zip(rec["d_cids"], rec["d_pos"]):
        end = min(int(p) + max_delta + 1, n_samples)
        seg = hist[int(ci), int(p):end].astype(float)
        L = len(seg)
        if L == 0:
            continue
        sums[:L] += seg
        counts[:L] += 1
    return sums, counts


def _acc(sums, counts):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def aggregate(recs, max_delta):
    """(dataset, label) ごとに、プールした acc(Δ) とシード間 std、件数を集計する。"""
    groups = defaultdict(list)  # (dataset, label) -> [(sums, counts), ...]  (ファイル=シード単位)
    for r in recs:
        groups[(r["dataset"], r["label"])].append(curve_sums(r, max_delta))

    agg = {}
    for key, per_seed in groups.items():
        pooled_s = np.sum([s for s, _ in per_seed], axis=0)
        pooled_c = np.sum([c for _, c in per_seed], axis=0)
        mean_curve = _acc(pooled_s, pooled_c)
        # シード間ばらつき(各シードの acc(Δ) の std)。1 シードなら 0。
        seed_curves = np.array([_acc(s, c) for s, c in per_seed])
        with np.errstate(invalid="ignore"):
            std_curve = np.nanstd(seed_curves, axis=0) if len(seed_curves) > 1 else np.zeros_like(mean_curve)
        n_drifts = int(pooled_c[0]) if len(pooled_c) else 0
        agg[key] = {"mean": mean_curve, "std": std_curve,
                    "n_drifts": n_drifts, "n_seeds": len(per_seed)}
    return agg


def _ordered_datasets(keys):
    present = {ds for ds, _ in keys}
    return [d for d in _CANON_DATASETS if d in present] + \
           [d for d in sorted(present) if d not in _CANON_DATASETS]


def _sorted_labels(labels):
    """凡例順: Oblivious → FedSDA → FedDrift → その他。"""
    def rank(lab):
        if "Oblivious" in lab:
            return (0, lab)
        if "FedSDA" in lab:
            return (1, lab)
        if "FedDrift" in lab:
            return (2, lab)
        return (3, lab)
    return sorted(labels, key=rank)


def plot_recovery(agg, max_delta, out_path, window):
    datasets = _ordered_datasets(agg.keys())
    deltas = np.arange(max_delta + 1)
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 4.8), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        labels = _sorted_labels([lab for (d, lab) in agg if d == ds])
        for lab in labels:
            info = agg[(ds, lab)]
            mean = info["mean"]
            std = info["std"]
            line, = ax.plot(deltas, mean, label=lab, linewidth=1.8)
            if np.any(std > 0):
                ax.fill_between(deltas, mean - std, mean + std,
                                color=line.get_color(), alpha=0.15)
        ax.axvline(window, color="gray", linestyle=":", linewidth=1,
                   label=f"window W={window}")
        ax.set_title(ds)
        ax.set_xlabel("Δ (samples since drift)")
        ax.set_ylabel("accuracy acc(Δ)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")
    fig.suptitle("Recovery curves: accuracy vs samples since drift")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def write_table(agg, checkpoints, window, out_path):
    datasets = _ordered_datasets(agg.keys())
    cp_cols = " ".join(f"acc@{c}" for c in checkpoints)
    header = "| Method | " + " | ".join(f"acc@{c}" for c in checkpoints) + \
             f" | mean acc[0,{window}) | n_drifts | seeds |"
    sep = "|---|" + "---:|" * (len(checkpoints) + 1) + "---:|---:|"

    lines = []
    for ds in datasets:
        labels = _sorted_labels([lab for (d, lab) in agg if d == ds])
        lines.append(f"### {ds}")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for lab in labels:
            info = agg[(ds, lab)]
            mean = info["mean"]
            cps = []
            for c in checkpoints:
                v = mean[c] if c < len(mean) else np.nan
                cps.append("–" if np.isnan(v) else f"{v:.3f}")
            win = mean[:window]
            win_mean = np.nanmean(win) if np.any(~np.isnan(win)) else np.nan
            win_txt = "–" if np.isnan(win_mean) else f"{win_mean:.3f}"
            lines.append(f"| {lab} | " + " | ".join(cps) +
                         f" | {win_txt} | {info['n_drifts']} | {info['n_seeds']} |")
        lines.append("")

    lines.append(f"*acc@Δ = ドリフト発生から Δ サンプル後の平均精度(全ドリフト・全クライアント・"
                 f"全シードでプール)。mean acc[0,{window}) = 固定窓の平均精度で、値が高いほど"
                 f"適応が速い(適応リグレット = 基準精度 − この値)。窓幅 W={window} は "
                 f"MIN_STABLE_PERIOD 未満で、次ドリフトを跨がない。*")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Table saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Recovery-curve analysis from raw .npz runs")
    parser.add_argument("--npz", nargs="+", required=True,
                        help="生データ .npz のパス(glob 可)")
    parser.add_argument("--out-dir", default=None,
                        help="図・表の出力先。未指定なら --npz が属する results ディレクトリ内の "
                             "recovery/(例: results_<stamp>/raw/*.npz → results_<stamp>/recovery)")
    parser.add_argument("--max-delta", type=int, default=250,
                        help="回復曲線の Δ 上限(MIN_STABLE_PERIOD 未満に。default: 250)")
    parser.add_argument("--window", type=int, default=200,
                        help="固定窓リグレットの窓幅 W(MIN_STABLE_PERIOD 未満に。default: 200)")
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[20, 50, 100],
                        help="表に出す固定オフセット Δ(default: 20 50 100)")
    parser.add_argument("--tag", default=None, help="出力ファイル名に付ける識別子")
    args = parser.parse_args()

    paths = []
    for pat in args.npz:
        paths.extend(sorted(glob.glob(pat)))
    paths = sorted(set(paths))
    if not paths:
        print("No .npz matched the given pattern(s).")
        return
    recs = [load_npz(p) for p in paths]
    print(f"Loaded {len(recs)} npz file(s).")

    # --out-dir 未指定なら、分析対象 .npz が属する results ディレクトリ内の recovery/ へ出す
    if args.out_dir is None:
        args.out_dir = infer_out_dir(paths)
        print(f"[out-dir] 未指定のため推定: {args.out_dir}")

    # 窓が単一安定区間に収まるか軽く検証(MIN_STABLE_PERIOD 未満であること)
    min_stable = min(r["min_stable"] for r in recs)
    if args.max_delta >= min_stable or args.window > min_stable:
        print(f"[warn] max_delta({args.max_delta})/window({args.window}) が MIN_STABLE_PERIOD"
              f"({min_stable}) 以上です。次ドリフトが混入する可能性があります。")

    agg = aggregate(recs, args.max_delta)

    os.makedirs(args.out_dir, exist_ok=True)
    datasets = _ordered_datasets(agg.keys())
    seeds = sorted(set(r["seed"] for r in recs))
    sd = f"seed{seeds[0]}" if len(seeds) == 1 else "seeds" + "-".join(str(s) for s in seeds)
    name = f"recovery_{'-'.join(datasets)}_{sd}" + (f"_{args.tag}" if args.tag else "")

    plot_recovery(agg, args.max_delta, os.path.join(args.out_dir, f"{name}.png"), args.window)
    write_table(agg, args.checkpoints, args.window, os.path.join(args.out_dir, f"{name}.md"))
    print("Done.")


if __name__ == "__main__":
    main()
