"""同じ入力の予測系列を、sample indexごとの累積差・区間差で比較する。"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from tools.research_s3_warm_control.trace import signature


def normalize(predictions):
    rows = []
    for original in predictions:
        p = dict(original)
        if not isinstance(p["correct"], bool) or not 0 <= p["label"] < 10:
            raise ValueError("MNIST4のラベル・正誤が不正です")
        if "probabilities" in p:
            probabilities = p["probabilities"]
            if (
                len(probabilities) != 10
                or any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities)
                or abs(sum(probabilities) - 1) > 1e-5
            ):
                raise ValueError("この計測はMNIST4系列を対象とします")
            predicted = max(range(10), key=probabilities.__getitem__)
            if "predicted_label" in p and p["predicted_label"] != predicted:
                raise ValueError("予測クラスと確率が一致しません")
            p["predicted_label"] = predicted
        rows.append(p)
    return rows


def compare(base, candidate, sample_count, window_size):
    if sample_count < 1 or window_size < 1:
        raise ValueError("正の標本数と窓幅が必要です")
    base, candidate = normalize(base), normalize(candidate)
    bs, bd = signature(base, sample_count)
    cs, cd = signature(candidate, sample_count)
    if bs != cs:
        raise ValueError("同じ画像・ラベル・概念の系列ではありません")
    bm = {(p["client_id"], p["sample_index"]): p for p in base}
    cm = {(p["client_id"], p["sample_index"]): p for p in candidate}
    blocks = []
    total_base = total_candidate = 0
    for start in range(0, sample_count, window_size):
        end = min(start + window_size, sample_count)
        b = sum(bm[c, s]["correct"] for c in range(10) for s in range(start, end))
        m = sum(cm[c, s]["correct"] for c in range(10) for s in range(start, end))
        total_base += b
        total_candidate += m
        blocks.append(
            dict(
                start_sample_index=start,
                end_sample_index_exclusive=end,
                window_observations=10 * (end - start),
                window_base_correct=b,
                window_candidate_correct=m,
                window_delta_pp=100 * (m - b) / (10 * (end - start)),
                cumulative_observations=10 * end,
                cumulative_base_correct=total_base,
                cumulative_candidate_correct=total_candidate,
                cumulative_delta_pp=100 * (total_candidate - total_base) / (10 * end),
            )
        )
    return dict(
        sample_count=sample_count,
        window_size=window_size,
        input_signature=bs,
        base_prediction_digest=bd,
        candidate_prediction_digest=cd,
        blocks=blocks,
        final_delta_pp=100 * (total_candidate - total_base) / (10 * sample_count),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-trace", type=Path, required=True)
    p.add_argument("--candidate-trace", type=Path, required=True)
    p.add_argument("--sample-count", type=int, required=True)
    p.add_argument("--window-size", type=int, default=500)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("既存の解析結果は上書きしません")
    b = json.loads(a.baseline_trace.read_text(encoding="utf-8"))["predictions"]
    c = json.loads(a.candidate_trace.read_text(encoding="utf-8"))["predictions"]
    result = compare(b, c, a.sample_count, a.window_size)
    result.update(
        base_trace_sha256=hashlib.sha256(a.baseline_trace.read_bytes()).hexdigest(),
        candidate_trace_sha256=hashlib.sha256(
            a.candidate_trace.read_bytes()
        ).hexdigest(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
