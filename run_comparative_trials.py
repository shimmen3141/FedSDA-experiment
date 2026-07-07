"""複数シード比較試行のCLIエントリポイント。

例:
    python run_comparative_trials.py --n-trials 10 --plot-dir results
    python run_comparative_trials.py --modes FedSDA FedDrift --n-trials 3
"""
import argparse

from fedsda import config, run_comparative_trials
from fedsda.experiment import MODE_SPECS

MODES = list(MODE_SPECS)


def main():
    parser = argparse.ArgumentParser(description="FedSDA comparative trials runner")
    parser.add_argument("--n-trials", type=int, default=10, help="モードごとの試行回数 (default: 10)")
    parser.add_argument("--threshold", type=float, default=config.DISTANCE_THRESHOLD,
                        help=f"距離閾値 gamma_dist (default: {config.DISTANCE_THRESHOLD})")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES,
                        help=f"実行するモード (default: 全モード)")
    parser.add_argument("--start-seed", type=int, default=0, help="開始シード (default: 0)")
    parser.add_argument("--no-plot", action="store_true", help="最終試行のプロットも生成しない")
    parser.add_argument("--plot-dir", default=None,
                        help="図の保存先ディレクトリ。未指定なら画面表示 (plt.show)")
    parser.add_argument("--verbose-per-trial", action="store_true",
                        help="最終試行の詳細ログを表示する")
    args = parser.parse_args()

    summary = run_comparative_trials(
        n_trials=args.n_trials,
        threshold=args.threshold,
        modes=args.modes,
        start_seed=args.start_seed,
        show_plot_last=not args.no_plot,
        verbose_per_trial=args.verbose_per_trial,
        plot_dir=args.plot_dir,
    )
    print("Summary:", summary)


if __name__ == "__main__":
    main()
