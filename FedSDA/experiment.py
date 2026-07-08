"""ランダムドリフト実験の本体。

実験モードは MODE_SPECS で定義する:
- 'FedSDA'                : 提案手法(ADWIN逐次検出 + サーバ集約)
- 'FedDrift'              : ベースライン(固定バッチ検出 + サーバ集約)
- 'FedSDA_without_server' : 提案手法のローカルのみ版(サーバ集約なし)
- 'Oblivious'            : ベースライン(単一モデル・FedAvg・無適応)

比較手法を追加する場合は、クライアントクラス(clients.py)を実装して
MODE_SPECS にエントリを足す。処理の流れが既存2種と異なる場合は、
タイムステップ実行関数(_run_*_timestep)も追加する。
"""
import os
import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from . import config
from .clients import AdwinClient, ObliviousClient, PeriodicClient
from .data import build_data_streams, extract_true_drift_events, generate_data, make_concept_schedules
from .metrics import compute_metrics
from .models import SimpleMLP
from .plotting import plot_client_details, plot_system_overview
from .server import BaseServer, ClusteringServer


# ==========================================
# タイムステップ実行(処理スタイルごと)
# ==========================================
def _run_per_sample_timestep(clients, server, data, concepts, t, use_server, verbose):
    """1サンプルずつ逐次処理するスタイル(ADWIN系)の1タイムステップ。

    K ステップの逐次処理を r_rounds 回行い、各ラウンド末にサーバ同期する。
    """
    for r in range(config.R_ROUNDS):
        r_offset = r * config.K_STEPS
        for k in range(config.K_STEPS):
            k_idx = r_offset + k
            if k_idx >= len(data[0]):
                break
            for i, c in enumerate(clients):
                x_in, y_in = data[i][k_idx]
                c.process_one_step(x_in, y_in, concepts[i][k_idx])

        if use_server:
            # 新規モデルがあるときだけクラスタリングを行う
            has_new = any(c.has_pending_model() for c in clients)
            server.run_aggregation_and_merge(t, clustering_enabled=has_new)
            # aggregation 後に pending -> ready を行い、次ラウンドで回収されるようにする
            for c in clients:
                c.promote_pending_to_ready()
        else:
            if verbose and random.random() < 0.01:
                print(f"  [without_server] t={t}, r={r}: skipped server aggregation (local-only).")


def _run_batch_timestep(clients, server, data, concepts, t, use_server, verbose):
    """バッチ一括処理するスタイル(FedDrift系)の1タイムステップ。

    検出フェーズ(クラスタリングあり)の後、学習フェーズ(集約のみ)を行う。
    """
    for i, c in enumerate(clients):
        c.phase1_detect(data[i], t, concepts[i][-1])
    server.run_aggregation_and_merge(t, clustering_enabled=True)
    for c in clients:
        c.promote_pending_to_ready()

    for _ in range(config.R_ROUNDS):
        for c in clients:
            c.phase2_train(k_steps=config.K_STEPS)
        server.run_aggregation_and_merge(t, clustering_enabled=False)
        for c in clients:
            c.promote_pending_to_ready()


# ==========================================
# モード定義
# ==========================================
@dataclass(frozen=True)
class ModeSpec:
    """実験モードの定義。比較手法の追加はここにエントリを足す。"""
    client_cls: type
    run_timestep: Callable
    use_server: bool = True
    server_cls: type = ClusteringServer


MODE_SPECS = {
    'FedSDA': ModeSpec(AdwinClient, _run_per_sample_timestep, server_cls=ClusteringServer),
    'FedDrift': ModeSpec(PeriodicClient, _run_batch_timestep, server_cls=ClusteringServer),
    'FedSDA_without_server': ModeSpec(AdwinClient, _run_per_sample_timestep, use_server=False),
    'Oblivious': ModeSpec(ObliviousClient, _run_per_sample_timestep, server_cls=BaseServer),
}


# ==========================================
# セットアップ
# ==========================================
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


def _setup_server_and_clients(spec, distance_threshold, verbose):
    """初期モデルの事前学習、サーバ登録、クライアント生成を行う。"""
    model0, stats_0 = _pretrain_initial_model()

    server = spec.server_cls(distance_threshold=distance_threshold, verbose=verbose)
    server.register_model_params(0, model0.get_params())
    server.register_model_stats(0, stats_0)

    clients = []
    for i in range(config.N_CLIENTS):
        c = spec.client_cls(
            client_id=i,
            initial_models={0: model0},
            initial_stats={0: stats_0},
            distance_threshold=distance_threshold,
            verbose=verbose
        )
        # サーバを使うモードのみ register する
        if spec.use_server:
            server.register_client(c)
        clients.append(c)

    return server, clients


# ==========================================
# 実験本体
# ==========================================
def run_random_drift_experiment(mode='FedDrift', distance_threshold=None,
                                random_seed=None, verbose=True, show_plot=True, plot_dir=None):
    """1回分の実験を実行し、メトリクスの dict を返す。

    plot_dir を指定すると図をそのディレクトリに保存し、None なら画面表示する
    (show_plot=False なら描画自体を行わない)。
    実験規模などのハイパーパラメータは FedSDA/config.py で管理する。
    """
    try:
        spec = MODE_SPECS[mode]
    except KeyError:
        raise ValueError(f"Unknown mode: {mode!r} (choose from {sorted(MODE_SPECS)})") from None

    if distance_threshold is None:
        distance_threshold = config.DISTANCE_THRESHOLD

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    print(f"=== System Experiment: {mode} (Threshold={distance_threshold}) ===")

    server, clients = _setup_server_and_clients(spec, distance_threshold, verbose)

    if verbose:
        print("Clients initialized. All holding Model 0.")

    # --- ドリフトスケジュールとデータストリームの生成 ---
    data_per_time = config.R_ROUNDS * config.K_STEPS
    t_steps = config.TOTAL_DATA_POINTS // data_per_time

    client_concept_schedule = make_concept_schedules(config.N_CLIENTS, config.TOTAL_DATA_POINTS)
    true_drift_events = extract_true_drift_events(client_concept_schedule)
    all_client_data = build_data_streams(client_concept_schedule)

    if verbose:
        print(f"Simulation Start (Total Data={config.TOTAL_DATA_POINTS}, Mode={mode})...")

    # --- シミュレーションループ ---
    exp_start = time.perf_counter()

    for t in range(t_steps):
        start_idx = t * data_per_time
        end_idx = start_idx + data_per_time

        if verbose and t % 5 == 0:
            print(f"--- Time {t} (Data Index {start_idx}) ---")

        current_time_data = [stream[start_idx:end_idx] for stream in all_client_data]
        current_time_concepts = [sched[start_idx:end_idx] for sched in client_concept_schedule]

        spec.run_timestep(clients, server, current_time_data, current_time_concepts,
                          t, spec.use_server, verbose)

    runtime_seconds = time.perf_counter() - exp_start

    if verbose:
        print("Simulation Finished.")
        print(f"  Experiment runtime: {runtime_seconds:.3f} sec")

    # --- メトリクス計算 ---
    results = compute_metrics(clients, true_drift_events)
    if spec.use_server:
        results["final_model_count"] = len(server.global_models)
    else:
        # サーバ集約がないため、クライアントが保持するローカルモデル数の平均を報告する
        results["final_model_count"] = float(np.mean([len(c.models) for c in clients]))
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
