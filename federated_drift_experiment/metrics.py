"""検出性能・精度メトリクスの計算。

検出イベントは「ローカルで実際にモデル切替を実行した位置 (local_switch_positions)」
として数え、真のドリフト位置と greedy マッチングして TP/FP/FN を算出する。
"""
import bisect
from collections import Counter

from . import config


def _match_events(true_positions, event_positions, delay_tolerance):
    """既存指標と同じgreedy規則で真のドリフトとイベントを一対一対応付けする。"""
    used = set()
    matched_true = set()
    delays = []
    for true_index, true_position in enumerate(true_positions):
        for event_index, event_position in enumerate(event_positions):
            if event_index in used:
                continue
            if true_position <= event_position <= true_position + delay_tolerance:
                used.add(event_index)
                matched_true.add(true_index)
                delays.append(event_position - true_position)
                break
    return used, matched_true, delays


def _classify_unmatched_events(true_positions, event_positions, used, tolerance):
    """未対応イベントを重複・早期・遅延・単独へ排他的に分類する。

    早期は真の変化前tolerance以内、遅延は許容窓終了後tolerance以内とする。
    それ以外は単独イベントなので、区分の境界は評価の許容遅延だけで再現できる。
    """
    counts = Counter()
    for index, position in enumerate(event_positions):
        if index in used:
            continue
        if any(td <= position <= td + tolerance for td in true_positions):
            counts["duplicate"] += 1
        elif any(td - tolerance <= position < td for td in true_positions):
            counts["early"] += 1
        elif any(td + tolerance < position <= td + 2 * tolerance for td in true_positions):
            counts["late"] += 1
        else:
            counts["isolated"] += 1
    return counts


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


def _change_point_errors(clients, true_drift_events, delay_tolerance):
    """真の変化後に発火した検知と、その検出器推定変化点の誤差を対応付ける。"""
    errors = []
    for client_id, client in enumerate(clients):
        alarms = list(getattr(client, "detected_event_positions", []))
        estimates = list(getattr(client, "estimated_drift_start_positions", []))
        if len(alarms) != len(estimates):
            continue
        used = set()
        for true_position in sorted(true_drift_events[client_id]):
            for index, alarm_position in enumerate(alarms):
                if index in used:
                    continue
                if true_position <= alarm_position <= true_position + delay_tolerance:
                    errors.append(estimates[index] - true_position)
                    used.add(index)
                    break
    return errors


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
    fp_types = Counter()
    alarm_true_total = 0
    alarm_event_total = 0
    alarm_tp_total = 0
    operation_counts = Counter()
    operation_tp = Counter()
    server_mapping_changes = 0

    total_local_switches = sum(len(c.local_switch_positions) for c in clients)

    # per-client greedy matching
    for i, c in enumerate(clients):
        true_drifts = list(true_drift_events[i])  # sample indices where concept changed
        local_sw = sorted(c.local_switch_positions)
        used, matched_true, client_delays = _match_events(
            true_drifts, local_sw, delay_tolerance
        )
        delays.extend(client_delays)
        total_tp += len(used)
        total_fn += len(true_drifts) - len(matched_true)
        total_true_drifts += len(true_drifts)
        total_fp += len(local_sw) - len(used)
        fp_types.update(
            _classify_unmatched_events(
                true_drifts, local_sw, used, delay_tolerance
            )
        )

        alarms = sorted(getattr(c, "detected_event_positions", []))
        alarm_used, alarm_matched, _ = _match_events(
            true_drifts, alarms, delay_tolerance
        )
        alarm_true_total += len(true_drifts)
        alarm_event_total += len(alarms)
        alarm_tp_total += len(alarm_used)

        events = list(getattr(c, "adaptation_events", []))
        actionable = [event for event in events if event.action in ("reuse", "create")]
        actionable_positions = [event.position for event in actionable]
        action_used, _, _ = _match_events(
            true_drifts, actionable_positions, delay_tolerance
        )
        for event_index, event in enumerate(actionable):
            operation_counts[event.action] += 1
            if event_index in action_used:
                operation_tp[event.action] += 1
        operation_counts.update(
            event.action for event in events
            if event.action not in ("reuse", "create", "server_merge")
        )
        server_mapping_changes += sum(
            event.action == "server_merge" for event in events
        )

    total_detections = total_local_switches  # 検出は「ローカルで実際に切替を実行した回数」
    fn_rate = total_fn / total_true_drifts if total_true_drifts > 0 else 0.0
    fdr = total_fp / total_detections if total_detections > 0 else 0.0
    recall = total_tp / total_true_drifts if total_true_drifts > 0 else 0.0
    precision = total_tp / total_detections if total_detections > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    avg_delay = sum(delays) / len(delays) if delays else 0.0
    alarm_precision = (
        alarm_tp_total / alarm_event_total if alarm_event_total else 0.0
    )
    alarm_recall = alarm_tp_total / alarm_true_total if alarm_true_total else 0.0
    alarm_f1 = (
        2 * alarm_precision * alarm_recall / (alarm_precision + alarm_recall)
        if alarm_precision + alarm_recall > 0 else 0.0
    )
    change_point_errors = _change_point_errors(
        clients, true_drift_events, delay_tolerance
    )
    change_point_mae = (
        sum(abs(error) for error in change_point_errors) / len(change_point_errors)
        if change_point_errors else 0.0
    )
    change_point_bias = (
        sum(change_point_errors) / len(change_point_errors)
        if change_point_errors else 0.0
    )

    return {
        "accuracy": avg_accuracy,
        "stable_accuracy": _stable_accuracy(clients, true_drift_events, stable_window),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "miss_rate": fn_rate,
        "fdr": fdr,
        "avg_delay": avg_delay,
        "change_point_mae": change_point_mae,
        "change_point_bias": change_point_bias,
        "change_point_estimate_count": len(change_point_errors),
        "total_true": total_true_drifts,
        "total_detect": total_detections,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "alarm_precision": alarm_precision,
        "alarm_recall": alarm_recall,
        "alarm_f1": alarm_f1,
        "alarm_total": alarm_event_total,
        "switch_fp_early": fp_types["early"],
        "switch_fp_late": fp_types["late"],
        "switch_fp_duplicate": fp_types["duplicate"],
        "switch_fp_isolated": fp_types["isolated"],
        "adaptation_reuse_count": operation_counts["reuse"],
        "adaptation_reuse_precision": (
            operation_tp["reuse"] / operation_counts["reuse"]
            if operation_counts["reuse"] else 0.0
        ),
        "adaptation_create_count": operation_counts["create"],
        "adaptation_create_precision": (
            operation_tp["create"] / operation_counts["create"]
            if operation_counts["create"] else 0.0
        ),
        "adaptation_create_rejected_count": operation_counts["create_rejected"],
        "adaptation_maintain_count": operation_counts["maintain"],
        "adaptation_episode_suppressed_count": operation_counts["episode_suppressed"],
        "server_mapping_change_count": server_mapping_changes,
    }
