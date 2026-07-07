"""複数シードでの比較試行と集計。"""
import time
import traceback
from collections import defaultdict

import numpy as np

from .experiment import run_random_drift_experiment


def run_comparative_trials(
    n_trials=10,
    threshold=0.1,
    modes=None,
    start_seed=0,
    show_plot_last=True,
    verbose_per_trial=False,
    plot_dir=None,
):
    """実験を複数回回すユーティリティ。各試行ごとに res を print する。"""
    if modes is None:
        modes = ['FedSDA', 'FedDrift']

    overall_summary = {}

    for mode in modes:
        metrics_list = defaultdict(list)
        print("\n" + "=" * 60)
        print(f"Running {n_trials} trials for MODE: {mode} (Thr={threshold})")
        print("=" * 60)

        for i in range(n_trials):
            trial_seed = start_seed + i
            print(f"\n--- Trial {i+1}/{n_trials} (seed={trial_seed}) ---")
            show_plot_flag = (show_plot_last and (i == n_trials - 1))
            trial_verbose = verbose_per_trial and (i == n_trials - 1)

            try:
                t0 = time.perf_counter()
                res = run_random_drift_experiment(
                    mode=mode,
                    distance_threshold=threshold,
                    random_seed=trial_seed,
                    verbose=trial_verbose,
                    show_plot=show_plot_flag,
                    plot_dir=plot_dir,
                )
                t1 = time.perf_counter()
                if isinstance(res, dict):
                    if 'runtime_seconds' not in res:
                        res['runtime_seconds'] = t1 - t0
                else:
                    raise ValueError("run_random_drift_experiment did not return a dict.")

                # 各試行結果をそのまま表示する
                print(f"Result (mode={mode}, trial={i+1}): {res}")

                # collect numeric scalars
                for k, v in res.items():
                    if isinstance(v, (int, float, np.number)):
                        metrics_list[k].append(float(v))

            except Exception as e:
                print("  !!! Trial failed with exception:")
                traceback.print_exc()
                metrics_list['_errors_'].append(str(e))

        # Aggregation & print (summary)
        print(f"\n=== Summary for {mode} (N={n_trials}) ===")
        display_keys = [
            'accuracy', 'recall', 'precision', 'f1', 'miss_rate', 'fdr', 'avg_delay',
            'final_model_count', 'runtime_seconds',
            'total_true', 'total_detect', 'tp', 'fp', 'fn'
        ]
        extra_keys = sorted([k for k in metrics_list.keys() if k not in display_keys])
        header_keys = [k for k in display_keys if k in metrics_list] + extra_keys

        if not header_keys:
            print("No scalar metrics were collected for this mode.")
            overall_summary[mode] = {}
            continue

        print(f"{'Metric':<22} | {'Mean':>10} | {'Std':>10} | {'N':>3}")
        print("-" * 52)
        summary_for_mode = {}
        for k in header_keys:
            vals = metrics_list[k]
            if k == '_errors_':
                print(f"{k:<22} | {'-':>10} | {'-':>10} | {len(vals):3d}")
                summary_for_mode[k] = vals
                continue
            arr = np.array(vals, dtype=float)
            mean_val = float(np.mean(arr)) if len(arr) > 0 else float('nan')
            std_val = float(np.std(arr, ddof=0)) if len(arr) > 0 else float('nan')
            n_val = len(arr)
            print(f"{k:<22} | {mean_val:10.4f} | {std_val:10.4f} | {n_val:3d}")
            summary_for_mode[k] = {'mean': mean_val, 'std': std_val, 'n': n_val}

        overall_summary[mode] = summary_for_mode

    print("\nAll modes finished.")
    return overall_summary
