"""ランダムドリフト実験の本体。

mode:
- 'FedSDA'                : 提案手法(ADWIN逐次検出 + サーバ集約)
- 'FedDrift'              : ベースライン(固定バッチ検出 + サーバ集約)
- 'FedSDA_without_server' : 提案手法のローカルのみ版(サーバ集約なし)
"""
import random
import time

import numpy as np
import torch

from . import config
from .clients import AdwinClient, PeriodicClient
from .data import build_data_streams, extract_true_drift_events, generate_data, make_concept_schedules
from .metrics import compute_metrics
from .models import SimpleMLP
from .plotting import plot_client_details, plot_system_overview
from .server import Server


def _pretrain_initial_model():
    """concept 0 のデータでモデル0を事前学習し、ベースライン統計も算出する。"""
    n_samples = config.PRETRAIN_SAMPLES
    n_epochs = config.PRETRAIN_EPOCHS
    batch_size = config.PRETRAIN_BATCH_SIZE

    model0 = SimpleMLP()
    stats_0 = {'n': 0, 'mean': 0.0, 'M2': 0.0}

    replay_buf = []
    for _ in range(n_samples):
        x_in, y_in = generate_data(0)
        replay_buf.append((x_in, y_in))

    for _ in range(n_epochs):
        random.shuffle(replay_buf)
        for i in range(0, len(replay_buf), batch_size):
            batch = replay_buf[i:i + batch_size]
            bx = torch.stack([d[0] for d in batch])
            by = torch.stack([d[1] for d in batch])
            model0.update(bx, by)

    for x, y in replay_buf:
        loss_val = model0.get_absolute_error(x.unsqueeze(0), y.unsqueeze(0))
        stats_0['n'] += 1
        delta = loss_val - stats_0['mean']
        stats_0['mean'] += delta / stats_0['n']
        delta2 = loss_val - stats_0['mean']
        stats_0['M2'] += delta * delta2

    return model0, stats_0


def run_random_drift_experiment(mode='FedDrift', distance_threshold=None,
                                random_seed=None, verbose=True, show_plot=True, plot_dir=None):
    """1回分の実験を実行し、メトリクスの dict を返す。

    plot_dir を指定すると図をそのディレクトリに保存し、None なら画面表示する
    (show_plot=False なら描画自体を行わない)。
    実験規模などのハイパーパラメータは fedsda/config.py で管理する。
    """
    if distance_threshold is None:
        distance_threshold = config.DISTANCE_THRESHOLD

    n_clients = config.N_CLIENTS
    r_rounds = config.R_ROUNDS
    k_steps = config.K_STEPS
    total_data_points = config.TOTAL_DATA_POINTS

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    if mode not in ('FedSDA', 'FedDrift', 'FedSDA_without_server'):
        raise ValueError(f"Unknown mode: {mode!r}")

    no_federated = False
    if mode == 'FedSDA_without_server':
        is_proposed = True
        no_federated = True
    else:
        is_proposed = (mode == 'FedSDA')

    print(f"=== System Experiment: {mode} (Threshold={distance_threshold}) ===")

    # --- 初期モデルとサーバ・クライアントのセットアップ ---
    model0, stats_0 = _pretrain_initial_model()
    initial_models = {0: model0}
    initial_stats = {0: stats_0}
    init_params = model0.get_params()

    server = Server(distance_threshold=distance_threshold, verbose=verbose)
    server.register_model_params(0, init_params)
    server.register_model_stats(0, stats_0)

    clients = []
    ClientClass = AdwinClient if is_proposed else PeriodicClient

    for i in range(n_clients):
        c = ClientClass(
            client_id=i,
            server=server,
            initial_models=initial_models,
            initial_stats=initial_stats,
            distance_threshold=distance_threshold,
            verbose=verbose
        )
        # サーバを使うモードのみ register する（NoFedでは登録しない）
        if not no_federated:
            server.register_client(c)
        clients.append(c)

    if verbose:
        print("Clients initialized. All holding Model 0.")

    # --- ドリフトスケジュールとデータストリームの生成 ---
    data_per_time = r_rounds * k_steps
    t_steps = total_data_points // data_per_time

    client_concept_schedule = make_concept_schedules(n_clients, total_data_points)
    true_drift_events = extract_true_drift_events(client_concept_schedule)
    all_client_data = build_data_streams(client_concept_schedule)

    if verbose:
        print(f"Simulation Start (Total Data={total_data_points}, Mode={mode})...")

    global_data_idx = 0

    # --- measure wall-clock runtime for the whole experiment ---
    exp_start = time.perf_counter()

    for t in range(t_steps):
        if verbose and t % 5 == 0:
            print(f"--- Time {t} (Data Index {global_data_idx}) ---")

        start_idx = t * data_per_time
        end_idx = (t + 1) * data_per_time

        current_time_data = [all_client_data[i][start_idx:end_idx] for i in range(n_clients)]
        current_time_concepts = [client_concept_schedule[i][start_idx:end_idx] for i in range(n_clients)]

        if is_proposed:
            chunk_size = k_steps
            for r in range(r_rounds):
                r_offset = r * chunk_size
                for k in range(k_steps):
                    k_idx = r_offset + k
                    if k_idx >= len(current_time_data[0]):
                        break
                    for i, c in enumerate(clients):
                        x_in, y_in = current_time_data[i][k_idx]
                        con = current_time_concepts[i][k_idx]
                        c.process_one_step(x_in, y_in, con)

                # Proposed: サーバ同期は no_federated フラグで制御
                if not no_federated:
                    has_new = any(c.has_pending_model() for c in clients)
                    server.run_aggregation_and_merge(t, clustering_enabled=has_new)
                    # aggregation 後に "pending -> ready" を行い、次ラウンドで回収されるようにする
                    for c in clients:
                        c.promote_pending_to_ready()
                else:
                    # NoFed: ローカル処理のみ、サーバは呼ばない
                    if verbose and random.random() < 0.01:
                        print(f"  [NoFed] t={t}, r={r}: skipped server aggregation (local-only).")
        else:
            # FedDrift baseline behavior (unchanged)
            for i, c in enumerate(clients):
                batch_data = current_time_data[i]
                last_con = current_time_concepts[i][-1]
                c.phase1_detect(batch_data, t, last_con)
            # baseline always performs server cluster/aggregate here (we keep same behavior)
            server.run_aggregation_and_merge(t, clustering_enabled=True)
            for c in clients:
                c.promote_pending_to_ready()

            for r in range(r_rounds):
                for c in clients:
                    c.phase2_train(k_steps=k_steps)
                # This aggregation only updates models (no clustering)
                server.run_aggregation_and_merge(t, clustering_enabled=False)
                for c in clients:
                    c.promote_pending_to_ready()

        global_data_idx += data_per_time

    exp_end = time.perf_counter()
    runtime_seconds = exp_end - exp_start

    if verbose:
        print("Simulation Finished.")
        print(f"  Experiment runtime: {runtime_seconds:.3f} sec")

    # --- メトリクス計算 ---
    results = compute_metrics(clients, true_drift_events)
    results["final_model_count"] = len(server.global_models)
    results["runtime_seconds"] = runtime_seconds

    if verbose:
        print("\n=== Experiment Metrics ===")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Recall (TP Rate): {results['recall']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Avg Delay: {results['avg_delay']:.1f} steps")
        print(f"  Final Global Models: {results['final_model_count']}")
        print(f"  Total Local Switches (total_detect): {results['total_detect']}")
        print(f"  TP / FP / FN: {results['tp']} / {results['fp']} / {results['fn']}")
        print(f"  Runtime: {runtime_seconds:.3f} sec")

    # --- 可視化 ---
    if show_plot:
        if plot_dir is not None:
            import os
            os.makedirs(plot_dir, exist_ok=True)
            seed_tag = f"seed{random_seed}" if random_seed is not None else "noseed"
            overview_path = os.path.join(plot_dir, f"{mode}_{seed_tag}_overview.png")
            details_path = os.path.join(plot_dir, f"{mode}_{seed_tag}_clients.png")
        else:
            overview_path = None
            details_path = None

        plot_system_overview(clients, mode, results["accuracy"], save_path=overview_path)
        plot_client_details(clients, save_path=details_path)

    return results
