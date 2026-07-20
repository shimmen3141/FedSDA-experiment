"""単発実験のCLIエントリポイント。

例:
    python run_experiment.py --mode FedSDA_NoCached_ADWIN --seed 0
    python run_experiment.py --mode FedDrift --seed 0 --plot-dir results
"""
import argparse
import os

from experiment_runtime import configure_torch_threads

from federated_drift_experiment import config, run_random_drift_experiment
from federated_drift_experiment.experiment import MODE_SPECS

MODES = list(MODE_SPECS)


def main():
    configure_torch_threads()
    parser = argparse.ArgumentParser(description="FedSDA single experiment runner")
    parser.add_argument("--mode", choices=MODES, default='FedSDA_NoCached_ADWIN',
                        help="実験モード (default: FedSDA_NoCached_ADWIN)")
    parser.add_argument("--threshold", type=float, default=config.DISTANCE_THRESHOLD,
                        help=f"距離閾値 gamma_dist (default: {config.DISTANCE_THRESHOLD})")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード (default: 0)")
    parser.add_argument("--dataset", choices=list(config._FEATURE_DIMS), default=config.DATASET,
                        help=f"データセット (default: {config.DATASET})")
    parser.add_argument("--concept-schedule", choices=config.CONCEPT_SCHEDULES,
                        default=config.CONCEPT_SCHEDULE,
                        help=f"概念切替方式 (default: {config.CONCEPT_SCHEDULE})")
    parser.add_argument("--feddrift-batch", type=int, default=config.FEDDRIFT_DETECT_BATCH,
                        help=f"FedDrift の検出バッチサイズ (default: {config.FEDDRIFT_DETECT_BATCH})")
    parser.add_argument("--cluster-linkage", choices=("complete", "connected"),
                        default=config.CLUSTER_LINKAGE,
                        help=f"共通クラスタリング戦略 (default: {config.CLUSTER_LINKAGE})")
    parser.add_argument("--feddrift-isolation", type=int,
                        default=config.FEDDRIFT_ISOLATION_TIMESTEPS,
                        help="FedDriftの新規モデル隔離時刻数 W (default: 1)")
    parser.add_argument("--quiet", action="store_true", help="詳細ログを抑制する")
    parser.add_argument("--no-plot", action="store_true", help="プロットを生成しない")
    parser.add_argument("--plot-dir", default=None,
                        help="図の保存先ディレクトリ。未指定なら画面表示 (plt.show)")
    parser.add_argument("--raw-dir", default=None,
                        help="生データ(.npz)の保存先ディレクトリ。回復曲線などの事後分析用")
    args = parser.parse_args()

    config.DATASET = args.dataset
    config.CONCEPT_SCHEDULE = args.concept_schedule
    config.FEDDRIFT_DETECT_BATCH = args.feddrift_batch
    config.CLUSTER_LINKAGE = args.cluster_linkage
    config.FEDDRIFT_ISOLATION_TIMESTEPS = args.feddrift_isolation

    raw_path = None
    if args.raw_dir is not None:
        raw_path = os.path.join(
            args.raw_dir,
            f"{args.mode}_{args.dataset}_{args.concept_schedule}_seed{args.seed}.npz",
        )

    results = run_random_drift_experiment(
        mode=args.mode,
        distance_threshold=args.threshold,
        random_seed=args.seed,
        verbose=not args.quiet,
        show_plot=not args.no_plot,
        plot_dir=args.plot_dir,
        raw_path=raw_path,
        raw_label=args.mode,
    )
    print("\nFinal results:", results)


if __name__ == "__main__":
    main()
