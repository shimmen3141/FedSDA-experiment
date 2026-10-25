"""既存の候補学習を再利用し、warm符号長の更新範囲だけを独立に選ぶ。"""

from contextlib import contextmanager, nullcontext
import hashlib
import json
from unittest.mock import patch

FACTORS = {
    "warm-matched": (False, False),
    "head-score-full-train": (True, False),
    "full-score-head-train": (False, True),
    "output-layer-matched": (True, True),
}


@contextmanager
def factorial_policy(*, arm, events):
    from tools.research_output_layer_learner.policy import (
        learner_policy,
        output_layer_training,
    )
    import federated_learning_experiments.runtime.alarm_interval_minimum_description_length_selection as selection

    score_head, train_head = FACTORS[arm]
    native_score = selection.compute_warm_started_learner_prequential_code_length
    # 実候補の初期値・作成・学習・採用は検証済みの既存対照をそのまま使う。
    existing = "output-layer-matched" if train_head else "warm-matched"
    with learner_policy(variant=existing, events=events):

        def score(**kw):
            with output_layer_training() if score_head else nullcontext():
                value = native_score(**kw)
            events.append(dict(event="warm_score", samples=len(kw["input_features"])))
            return value

        with patch.object(
            selection, "compute_warm_started_learner_prequential_code_length", score
        ):
            yield


@contextmanager
def capture_input_signature(records):
    """通常の観測処理の入口で入力をhash化する。追加のNN推論はしない。"""
    import federated_learning_experiments.runtime.fedsda_run_client as client

    original = client.process_observed_sample

    def process(**kw):
        o = kw["indexed_observation"]
        key = kw["client_id"], o.sample_index
        if key in records:
            raise ValueError("入力標本が重複しています")
        records[key] = (
            hashlib.sha256(
                o.training_sample.input_features.numpy().tobytes()
            ).hexdigest(),
            int(o.training_sample.observed_class_labels.item()),
            o.observed_concept_id,
        )
        return original(**kw)

    with patch.object(client, "process_observed_sample", process):
        yield


def input_signature(records, sample_count):
    expected = {(c, s) for c in range(10) for s in range(sample_count)}
    if set(records) != expected:
        raise ValueError("10clientsの全標本を被覆していません")
    payload = [(key, *value) for key, value in sorted(records.items())]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()
