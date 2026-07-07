"""単発実験のCLIエントリポイント。

例:
    python run_experiment.py --mode FedSDA --seed 0
    python run_experiment.py --mode FedDrift --seed 0 --plot-dir results
"""
import argparse

from FedSDA import config, run_random_drift_experiment
from FedSDA.experiment import MODE_SPECS

MODES = list(MODE_SPECS)


def main():
    parser = argparse.ArgumentParser(description="FedSDA single experiment runner")
    parser.add_argument("--mode", choices=MODES, default='FedSDA',
                        help="実験モード (default: FedSDA)")
    parser.add_argument("--threshold", type=float, default=config.DISTANCE_THRESHOLD,
                        help=f"距離閾値 gamma_dist (default: {config.DISTANCE_THRESHOLD})")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード (default: 0)")
    parser.add_argument("--quiet", action="store_true", help="詳細ログを抑制する")
    parser.add_argument("--no-plot", action="store_true", help="プロットを生成しない")
    parser.add_argument("--plot-dir", default=None,
                        help="図の保存先ディレクトリ。未指定なら画面表示 (plt.show)")
    args = parser.parse_args()

    results = run_random_drift_experiment(
        mode=args.mode,
        distance_threshold=args.threshold,
        random_seed=args.seed,
        verbose=not args.quiet,
        show_plot=not args.no_plot,
        plot_dir=args.plot_dir,
    )
    print("\nFinal results:", results)


if __name__ == "__main__":
    main()
