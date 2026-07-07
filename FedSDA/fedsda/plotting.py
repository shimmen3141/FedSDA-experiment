"""実験結果の可視化。

save_path を指定するとファイルへ保存し、None なら plt.show() で表示する
(ローカルの非対話実行では保存を推奨)。
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from . import config

CONCEPT_COLORS = ['#ffcccc', '#ccffcc', '#ccccff', '#ffffcc']


def _finish(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved: {save_path}")
    else:
        plt.show()


def plot_system_overview(clients, mode, avg_accuracy, save_path=None):
    """全クライアントの精度推移と、コンセプト(背景色) vs 使用モデルID(テキスト)の帯グラフ。"""
    n_clients = len(clients)
    window = config.PLOT_SMOOTH_WINDOW

    fig = plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    avg_acc_history = []
    cmap = plt.get_cmap('tab10')

    for i, c in enumerate(clients):
        if len(c.history_accuracy) >= window:
            sm = np.convolve(c.history_accuracy, np.ones(window) / window, mode='valid')
            plt.plot(range(len(sm)), sm, alpha=0.3, linewidth=1.0, color=cmap(i % 10), label=f'C{i}')
        else:
            plt.plot(range(len(c.history_accuracy)), c.history_accuracy, alpha=0.3, linewidth=1.0,
                     color=cmap(i % 10), label=f'C{i}')

    min_len = min(len(c.history_accuracy) for c in clients)
    for idx in range(min_len):
        accs = [c.history_accuracy[idx] for c in clients]
        avg_acc_history.append(sum(accs) / len(accs))

    if len(avg_acc_history) >= window:
        smooth_acc = np.convolve(avg_acc_history, np.ones(window) / window, mode='valid')
        plt.plot(range(len(smooth_acc)), smooth_acc, color='black', linewidth=2.5, label='Avg')

    plt.title(f"System Accuracy (Avg: {avg_accuracy:.3f})")
    plt.ylim(0, 1.1)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    colors = CONCEPT_COLORS

    for i, c in enumerate(clients):
        for (start, width, con) in _concept_ranges(c):
            plt.broken_barh([(start, width)], (i - 0.4, 0.8), facecolors=colors[con], alpha=0.5)

        for start, end, mid in _model_ranges(c):
            if start > 0:
                plt.vlines(start, i - 0.4, i + 0.4, colors='black', linestyles='dotted', alpha=0.7)

            mid_str = str(mid) if mid >= 0 else "x"
            center_t = (start + end) / 2
            plt.text(center_t, i, mid_str, fontsize=9, va='center', ha='center', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.5))

        # Draw local-switch markers (red triangles) at recorded local_switch_positions
        for sw in c.local_switch_positions:
            plt.plot(sw, i, marker='^', color='red', markersize=6)

    plt.yticks(range(n_clients), [f"Client {i}" for i in range(n_clients)])
    plt.title(f"Concept (Color) vs Model ID (Text) [{mode}]")

    n_concepts = min(config.NUM_CONCEPTS, len(colors))
    patches = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(n_concepts)]
    marker_handle = Line2D([0], [0], marker='^', color='red', linestyle='None', markersize=8)
    handles = patches + [marker_handle]
    labels = [f"Concept {i}" for i in range(n_concepts)] + ["Local switch (drift detection)"]
    plt.legend(handles=handles, labels=labels, loc='upper right')

    plt.tight_layout()
    _finish(fig, save_path)


def plot_client_details(clients, save_path=None):
    """クライアントごとの精度・コンセプト背景・モデル切替の詳細プロット。"""
    n_clients = len(clients)
    window = config.PLOT_SMOOTH_WINDOW
    colors = CONCEPT_COLORS

    n_rows = math.ceil(n_clients / 2)
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_clients):
        ax = axes[i]
        c = clients[i]
        if len(c.history_accuracy) >= window:
            sm = np.convolve(c.history_accuracy, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(c.history_accuracy)), sm, color='blue', label='Accuracy')
        else:
            ax.plot(range(len(c.history_accuracy)), c.history_accuracy, color='blue', label='Accuracy')

        for (start, width, con) in _concept_ranges(c):
            ax.axvspan(start, start + width, facecolor=colors[con], alpha=0.2)

        for start, end, mid in _model_ranges(c):
            if start > 0:
                ax.axvline(x=start, color='black', linestyle=':', alpha=0.8)
            mid_str = str(mid) if mid >= 0 else "x"
            center_t = (start + end) / 2
            ax.text(center_t, 0.1, mid_str, fontsize=9, va='center', ha='center',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.8))

        # model switch markers (local)
        for sw in c.local_switch_positions:
            ax.plot(sw, 0.05, marker='^', color='red', markersize=8)

        ax.set_title(f"Client {i}")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True)
        if i >= n_clients - 2:
            ax.set_xlabel("Time Step")
        if i % 2 == 0:
            ax.set_ylabel("Accuracy")

    for j in range(n_clients, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    _finish(fig, save_path)


def _concept_ranges(client):
    """history_concept を (start, width, concept_id) の連続区間リストに変換する。"""
    ranges = []
    if len(client.history_concept) > 0:
        curr_c = client.history_concept[0]
        start_t = 0
        for t_idx, con in enumerate(client.history_concept):
            if con != curr_c:
                ranges.append((start_t, t_idx - start_t, curr_c))
                start_t = t_idx
                curr_c = con
        ranges.append((start_t, len(client.history_concept) - start_t, curr_c))
    return ranges


def _model_ranges(client):
    """history_model_id を (start, end, model_id) の連続区間リストに変換する。"""
    model_ranges = []
    if len(client.history_model_id) > 0:
        curr_m = client.history_model_id[0]
        start_t = 0
        for t_idx, mid in enumerate(client.history_model_id):
            if mid != curr_m:
                model_ranges.append((start_t, t_idx - 1, curr_m))
                start_t = t_idx
                curr_m = mid
        model_ranges.append((start_t, len(client.history_model_id) - 1, curr_m))
    return model_ranges
