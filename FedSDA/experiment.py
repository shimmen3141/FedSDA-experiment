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
    """1サンプルずつ逐次処理するスタイル(ADWIN系)の1ラウンド。

    1ラウンド = AGG_INTERVAL サンプル(=data のチャンク長)を逐次処理し、末尾でサーバ同期する。
    """
    for k in range(len(data[0])):
        for i, c in enumerate(clients):
            x_in, y_in = data[i][k]
            c.process_one_step(x_in, y_in, concepts[i][k])

    if use_server:
        # 新規モデルがあるときだけクラスタリングを行う
        has_new = any(c.has_pending_model() for c in clients)
        server.run_round(t, clustering_enabled=has_new)
        # aggregation 後に pending -> ready を行い、次ラウンドで回収されるようにする
        for c in clients:
            c.promote_pending_to_ready()
    else:
        if verbose and random.random() < 0.01:
            print(f"  [without_server] t={t}: skipped server aggregation (local-only).")


def _run_batch_timestep(clients, server, data, concepts, t, use_server, verbose):
    """バッチ処理するスタイル(FedDrift系)の1検出バッチ。

    1 ラウンドで処理するサンプル数 ＝ 検出バッチ(FEDDRIFT_DETECT_BATCH 件)なので、1 回の呼び出しで
    必ず検出バッチが完成し、検出+割り当て+通信を行う(通信量は検出バッチサイズに反比例)。完了後は論文の R
    ラウンドに倣い、{配布 → ローカル学習 → 集約} を config.FEDDRIFT_ROUNDS 回繰り返す(既定 1)。
    1 ラウンドのローカル学習量は FEDDRIFT_DETECT_BATCH × UPDATES_PER_SAMPLE 更新(= FedSDA の
    同区間の予算と一致)なので、R=1 のとき総ローカル更新数は FedSDA と等しい(公平比較)。R>1 は
    論文忠実(バッチ収束学習)だが更新数・通信量が R 倍になる。
    """
    # 全クライアントで検出バッチを処理(process_batch 内で必ず1回発火する)
    for i, c in enumerate(clients):
        c.process_batch(data[i], concepts[i])

    # 検出バッチ完了: クラスタリング付き集約(モデル併合/割当)
    server.run_round(t, clustering_enabled=True)
    for c in clients:
        c.promote_pending_to_ready()

    # 論文 R に対応: {配布 → ローカル学習 → 集約} を FEDDRIFT_ROUNDS 回。
    # 1 ラウンドの学習量 = FEDDRIFT_DETECT_BATCH × UPDATES_PER_SAMPLE 更新(=1バッチ分の予算)。
    for _ in range(config.FEDDRIFT_ROUNDS):
        for c in clients:
            c.local_train(k_steps=config.FEDDRIFT_DETECT_BATCH)
        server.run_round(t, clustering_enabled=False)
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
    # 1ラウンドで処理するサンプル数を持つ config 属性名。逐次系は集約間隔 AGG_INTERVAL、
    # FedDrift は検出バッチ FEDDRIFT_DETECT_BATCH(処理=検出=通信の単位)。
    chunk_attr: str = 'AGG_INTERVAL'


MODE_SPECS = {
    'FedSDA': ModeSpec(AdwinClient, _run_per_sample_timestep, server_cls=ClusteringServer),
    'FedDrift': ModeSpec(PeriodicClient, _run_batch_timestep, server_cls=ClusteringServer,
                         chunk_attr='FEDDRIFT_DETECT_BATCH'),
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


def _mode_param_summary(mode, distance_threshold):
    """ログ表示用に、手法ごとの関連ハイパーパラメータを1行にまとめる。"""
    if mode in ('FedSDA', 'FedSDA_without_server'):
        return (f"gamma_dist={distance_threshold}, delta_adwin={config.ADWIN_DELTA}, "
                f"N_FIFO={config.FIFO_BUFFER_SIZE}")
    if mode == 'FedDrift':
        return (f"detect_delta={distance_threshold}, detect_batch={config.FEDDRIFT_DETECT_BATCH}, "
                f"rounds={config.FEDDRIFT_ROUNDS}")
    if mode == 'Oblivious':
        return "single model, no adaptation"
    return f"threshold={distance_threshold}"


# ==========================================
# 実験本体
# ==========================================
def _save_raw_run(raw_path, clients, true_drift_events, mode, label, seed):
    """per-sample の生データを 1 つの .npz にまとめて保存する(gitignore 前提の軽量形式)。

    - history_accuracy: (N_CLIENTS, N_SAMPLES) の int8 (各サンプルの当否 0/1)
    - drift_client_ids / drift_positions: 真のドリフトを (クライアントid, サンプルindex) の
      並列配列で平坦化(可変長を object 配列にせず保持)
    - history_model_id: (N_CLIENTS, N_SAMPLES) の int32 (各サンプルで選択中のモデルID)
    - switch_client_ids / switch_positions: ローカルで実際にモデル切替が起きた位置を
      (クライアントid, サンプルindex) の並列配列で平坦化
    - dataset/mode/label/seed/min_stable/agg_interval: 分析時のグループ化・Δ上限用メタデータ

    注: history_model_id / switch_* は後から追加した純増キー。これらを持たない旧 .npz
    でも読み手が `key in npz` で存在判定すれば従来どおり読める(形式は後方互換)。
    """
    out_dir = os.path.dirname(raw_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    hist = np.asarray([c.history_accuracy for c in clients], dtype=np.int8)
    model_id_hist = np.asarray([c.history_model_id for c in clients], dtype=np.int32)

    d_cids, d_pos = [], []
    for ci, positions in true_drift_events.items():
        for p in positions:
            d_cids.append(ci)
            d_pos.append(p)

    # モデル切替が実際に起きたサンプル位置(検出として数えるもの)を平坦化
    s_cids, s_pos = [], []
    for ci, c in enumerate(clients):
        for p in getattr(c, "local_switch_positions", []):
            s_cids.append(ci)
            s_pos.append(p)

    np.savez_compressed(
        raw_path,
        history_accuracy=hist,
        drift_client_ids=np.asarray(d_cids, dtype=np.int32),
        drift_positions=np.asarray(d_pos, dtype=np.int32),
        history_model_id=model_id_hist,
        switch_client_ids=np.asarray(s_cids, dtype=np.int32),
        switch_positions=np.asarray(s_pos, dtype=np.int32),
        dataset=str(config.DATASET),
        mode=str(mode),
        label=str(label),
        seed=(int(seed) if seed is not None else -1),
        min_stable=int(config.MIN_STABLE_PERIOD),
        agg_interval=int(config.AGG_INTERVAL),
        total_data=int(config.TOTAL_DATA_POINTS),
    )


def run_random_drift_experiment(mode='FedDrift', distance_threshold=None,
                                random_seed=None, verbose=True, show_plot=True, plot_dir=None,
                                raw_path=None, raw_label=None):
    """1回分の実験を実行し、メトリクスの dict を返す。

    plot_dir を指定すると図をそのディレクトリに保存し、None なら画面表示する
    (show_plot=False なら描画自体を行わない)。
    raw_path を指定すると、回復曲線 acc(Δ) 等の事後分析用に per-sample の生データ
    (クライアント別 history_accuracy と真のドリフト位置、メタデータ)を .npz に保存する。
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

    print(f"=== System Experiment: {mode} ({_mode_param_summary(mode, distance_threshold)}) ===")

    server, clients = _setup_server_and_clients(spec, distance_threshold, verbose)

    if verbose:
        print("Clients initialized. All holding Model 0.")

    # --- ドリフトスケジュールとデータストリームの生成 ---
    # 1ラウンドで処理するサンプル数は手法依存: 逐次系=AGG_INTERVAL(集約間隔)、
    # FedDrift=FEDDRIFT_DETECT_BATCH(処理=検出=通信の単位)。
    chunk = getattr(config, spec.chunk_attr)
    t_steps = config.TOTAL_DATA_POINTS // chunk

    client_concept_schedule = make_concept_schedules(config.N_CLIENTS, config.TOTAL_DATA_POINTS)
    true_drift_events = extract_true_drift_events(client_concept_schedule)
    all_client_data = build_data_streams(client_concept_schedule)

    if verbose:
        print(f"Simulation Start (Total Data={config.TOTAL_DATA_POINTS}, Mode={mode})...")

    # --- シミュレーションループ ---
    exp_start = time.perf_counter()

    for t in range(t_steps):
        start_idx = t * chunk
        end_idx = start_idx + chunk

        if verbose and t % 5 == 0:
            print(f"--- Round {t} (Data Index {start_idx}) ---")

        current_time_data = [stream[start_idx:end_idx] for stream in all_client_data]
        current_time_concepts = [sched[start_idx:end_idx] for sched in client_concept_schedule]

        spec.run_timestep(clients, server, current_time_data, current_time_concepts,
                          t, spec.use_server, verbose)

    # バッファ型手法(FedDrift)の終端フラッシュ: 未検出の残りバッチを処理し新規モデルを回収
    buffering_clients = [c for c in clients if hasattr(c, 'flush')]
    if buffering_clients:
        for c in buffering_clients:
            c.flush()
        if spec.use_server and any(c.has_pending_model() for c in clients):
            server.run_round(t, clustering_enabled=True)
            for c in clients:
                c.promote_pending_to_ready()

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

    # --- 通信量(モデル転送数。up=クライアント→サーバ, down=サーバ→クライアント)---
    results["comm_upload"] = server.comm_up
    results["comm_download"] = server.comm_down
    results["comm_total"] = server.comm_up + server.comm_down

    # 定常精度 stable_accuracy(回復窓除外)は compute_metrics で算出済み。

    # --- 生データの保存(回復曲線などの事後分析用)---
    if raw_path is not None:
        _save_raw_run(raw_path, clients, true_drift_events, mode,
                      raw_label if raw_label is not None else mode, random_seed)

    if verbose:
        print("\n=== Experiment Metrics ===")
        print(f"  Accuracy (prequential): {results['accuracy']:.4f}")
        print(f"  Accuracy (stable, omit-recovery W={config.STABLE_WINDOW}): {results['stable_accuracy']:.4f}")
        print(f"  Recall (TP Rate): {results['recall']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Avg Delay: {results['avg_delay']:.1f} steps")
        print(f"  Final Global Models: {results['final_model_count']}")
        print(f"  Total Local Switches (total_detect): {results['total_detect']}")
        print(f"  TP / FP / FN: {results['tp']} / {results['fp']} / {results['fn']}")
        print(f"  Comm (up / down / total): {results['comm_upload']} / {results['comm_download']} / {results['comm_total']}")
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
