"""検出性能・精度メトリクスの計算。

検出イベントは「ローカルで実際にモデル切替を実行した位置 (local_switch_positions)」
として数え、真のドリフト位置と greedy マッチングして TP/FP/FN を算出する。
"""
from . import config


def compute_metrics(clients, true_drift_events, delay_tolerance=None):
    if delay_tolerance is None:
        delay_tolerance = config.DELAY_TOLERANCE
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
