"""検出性能・精度メトリクスの計算。

検出イベントは「ローカルで実際にモデル切替を実行した位置 (local_switch_positions)」
として数え、真のドリフト位置と greedy マッチングして TP/FP/FN を算出する。
"""
import bisect

from . import config


def _stable_accuracy(clients, true_drift_events, window):
    """定常精度: 各真ドリフト直後 window サンプル(回復中)を除いた prequential 精度。

    サンプル idx は、直前(idx 以下)の真ドリフト位置 p が存在し idx < p + window のとき
    「回復中」として平均から除外する。最初のドリフト前の区間は定常として含める。
    = 回復曲線 acc(Δ) の Δ≥window の裾に相当するスカラー(全クライアント・全区間プール)。
    """
    correct = 0
    total = 0
    for i, c in enumerate(clients):
        drifts = sorted(true_drift_events[i])
        for idx, acc in enumerate(c.history_accuracy):
            j = bisect.bisect_right(drifts, idx) - 1  # idx 以下で最も近い真ドリフト
            if j >= 0 and idx < drifts[j] + window:
                continue  # 回復窓 [p, p+window) 内は除外
            correct += acc
            total += 1
    return correct / total if total > 0 else float('nan')


def compute_metrics(clients, true_drift_events, delay_tolerance=None, stable_window=None):
    if delay_tolerance is None:
        delay_tolerance = config.DELAY_TOLERANCE
    if stable_window is None:
        stable_window = config.STABLE_WINDOW
    all_accs = []
    for c in clients:
        all_accs.extend(c.history_accuracy)
    avg_accuracy = sum(all_accs) / len(all_accs) if len(all_accs) > 0 else 0.0

    total_tp = 0
    total_fn = 0
    total_fp = 0
    total_true_drifts = 0
    delays = []

    total_local_switches = sum(len(c.local_switch_positions) for c in clients)

    # per-client greedy matching
    for i, c in enumerate(clients):
        true_drifts = list(true_drift_events[i])  # sample indices where concept changed
        local_sw = sorted(c.local_switch_positions)
        used = set()

        for td_time in true_drifts:
            total_true_drifts += 1
            matched = False
            for j, sw in enumerate(local_sw):
                if j in used:
                    continue
                if td_time <= sw <= td_time + delay_tolerance:
                    total_tp += 1
                    delays.append(sw - td_time)
                    used.add(j)
                    matched = True
                    break
            if not matched:
                total_fn += 1

        # remaining unused local switches count as FP
        total_fp += (len(local_sw) - len(used))

    total_detections = total_local_switches  # 検出は「ローカルで実際に切替を実行した回数」
    fn_rate = total_fn / total_true_drifts if total_true_drifts > 0 else 0.0
    fdr = total_fp / total_detections if total_detections > 0 else 0.0
    recall = total_tp / total_true_drifts if total_true_drifts > 0 else 0.0
    precision = total_tp / total_detections if total_detections > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    avg_delay = sum(delays) / len(delays) if delays else 0.0

    return {
        "accuracy": avg_accuracy,
        "stable_accuracy": _stable_accuracy(clients, true_drift_events, stable_window),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "miss_rate": fn_rate,
        "fdr": fdr,
        "avg_delay": avg_delay,
        "total_true": total_true_drifts,
        "total_detect": total_detections,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }
