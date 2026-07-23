"""単発実験のCLIエントリポイント。

例:
    python run_experiment.py --mode FedSDA_NoCached_ADWIN --seed 0
    python run_experiment.py --mode FedDrift --seed 0 --plot-dir results
"""
import argparse
import os

from experiment_runtime import configure_torch_threads

from federated_drift_experiment import config, run_random_drift_experiment
from federated_drift_experiment.data import dataset_cli_choices, normalize_dataset_name
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
    parser.add_argument("--dataset", choices=dataset_cli_choices(config._FEATURE_DIMS),
                        default=config.DATASET,
                        help=f"データセット (default: {config.DATASET})")
    parser.add_argument("--concept-schedule", choices=config.CONCEPT_SCHEDULES,
                        default=config.CONCEPT_SCHEDULE,
                        help=f"概念切替方式 (default: {config.CONCEPT_SCHEDULE})")
    parser.add_argument("--total-data", type=int, default=None,
                        help="クライアントごとの総サンプル数を一時的に上書きする")
    parser.add_argument("--n-clients", type=int, default=None,
                        help="クライアント数を一時的に上書きする")
    parser.add_argument(
        "--new-model-training",
        choices=("fixed", "none", "early_stopping"),
        default=config.NEW_MODEL_TRAINING,
        help="FedSDAの新規モデル初期学習戦略",
    )
    parser.add_argument(
        "--new-model-epochs", type=int, default=config.NEW_MODEL_EPOCHS,
        help="fixedのエポック数、またはearly_stoppingの最大エポック数",
    )
    parser.add_argument(
        "--new-model-initialization",
        choices=("current", "best_candidate", "average"),
        default=config.NEW_MODEL_INITIALIZATION,
        help="FedSDAの新規モデルを初期化する既存モデルの選択方法",
    )
    parser.add_argument("--feddrift-batch", type=int, default=config.FEDDRIFT_DETECT_BATCH,
                        help=f"FedDrift の検出バッチサイズ (default: {config.FEDDRIFT_DETECT_BATCH})")
    parser.add_argument("--cluster-linkage", choices=("complete", "connected"),
                        default=config.CLUSTER_LINKAGE,
                        help=f"共通クラスタリング戦略 (default: {config.CLUSTER_LINKAGE})")
    parser.add_argument(
        "--clustering-policy",
        choices=config.FEDSDA_CLUSTERING_POLICIES,
        default=config.FEDSDA_CLUSTERING_POLICY,
        help="FedSDAのクラスタリング頻度",
    )
    parser.add_argument(
        "--detection-episodes",
        action=argparse.BooleanOptionalAction,
        default=config.FEDSDA_DETECTION_EPISODES_ENABLED,
        help="近接した検出をN_FIFO幅の一つの適応エピソードへ統合する",
    )
    parser.add_argument(
        "--new-model-creation-policy",
        choices=config.NEW_MODEL_CREATION_POLICIES,
        default=config.NEW_MODEL_CREATION_POLICY,
        help="FedSDAの新規モデル作成方針（immediate / validated）",
    )
    parser.add_argument(
        "--fifo-size", type=int, default=config.FIFO_BUFFER_SIZE,
        help="FedSDAのFIFOバッファ長 N_FIFO",
    )
    parser.add_argument(
        "--new-model-validation-fraction", type=float,
        default=config.NEW_MODEL_VALIDATION_FRACTION,
        help="検証付き仮モデルで末尾から検証用に確保する割合",
    )
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

    if args.fifo_size < 1:
        parser.error("--fifo-size must be at least 1")
    if not 0.0 < args.new_model_validation_fraction < 1.0:
        parser.error("--new-model-validation-fraction must be between 0 and 1")

    args.dataset = normalize_dataset_name(args.dataset)
    config.DATASET = args.dataset
    config.CONCEPT_SCHEDULE = args.concept_schedule
    if args.total_data is not None:
        config.TOTAL_DATA_POINTS = args.total_data
    if args.n_clients is not None:
        config.N_CLIENTS = args.n_clients
    config.NEW_MODEL_TRAINING = args.new_model_training
    config.NEW_MODEL_EPOCHS = args.new_model_epochs
    config.NEW_MODEL_INITIALIZATION = args.new_model_initialization
    config.FEDDRIFT_DETECT_BATCH = args.feddrift_batch
    config.CLUSTER_LINKAGE = args.cluster_linkage
    config.FEDSDA_CLUSTERING_POLICY = args.clustering_policy
    config.FEDSDA_DETECTION_EPISODES_ENABLED = args.detection_episodes
    config.NEW_MODEL_CREATION_POLICY = args.new_model_creation_policy
    config.FIFO_BUFFER_SIZE = args.fifo_size
    config.NEW_MODEL_VALIDATION_FRACTION = args.new_model_validation_fraction
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
