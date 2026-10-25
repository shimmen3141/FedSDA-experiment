"""NN推論を追加せず、実予測の正誤と入力対応だけを軽量記録する。"""

from contextlib import contextmanager
import hashlib
import json
from unittest.mock import patch


def signature(predictions, sample_count):
    indexed = {(p["client_id"], p["sample_index"]): p for p in predictions}
    expected = {(c, s) for c in range(10) for s in range(sample_count)}
    if len(predictions) != len(indexed) or set(indexed) != expected:
        raise ValueError("予測系列に欠落・重複があります")
    values = []
    for key in sorted(indexed):
        p = indexed[key]
        if (
            not 0 <= p["predicted_label"] < 10
            or not 0 <= p["label"] < 10
            or not isinstance(p["correct"], bool)
            or p["correct"] != (p["predicted_label"] == p["label"])
        ):
            raise ValueError("記録した予測クラスと正誤が一致しません")
        values.append((key, p["input_hash"], p["label"], p["concept"]))
    return hashlib.sha256(json.dumps(values).encode()).hexdigest(), hashlib.sha256(
        json.dumps([indexed[k]["correct"] for k in sorted(indexed)]).encode()
    ).hexdigest()


class PredictionTrace:
    def __init__(self):
        self.predictions = []
        self.client_id = None

    @contextmanager
    def activated(self):
        import federated_learning_experiments.runtime.fedsda_run_client as client
        import federated_learning_experiments.runtime.observed_sample_processing as processing

        original_process = client.process_observed_sample
        original_predict = (
            processing.predict_observed_sample_and_update_prediction_weights
        )

        def process(**kw):
            previous = self.client_id
            self.client_id = kw["client_id"]
            try:
                return original_process(**kw)
            finally:
                self.client_id = previous

        def predict(**kw):
            result = original_predict(**kw)
            observation = kw["indexed_observation"]
            probabilities = result.combined_prediction_probabilities.detach().reshape(
                -1
            )
            if self.client_id is None or probabilities.numel() != 10:
                raise ValueError("MNIST4予測の記録文脈が不正です")
            self.predictions.append(
                dict(
                    client_id=self.client_id,
                    sample_index=observation.sample_index,
                    concept=observation.observed_concept_id,
                    label=int(observation.training_sample.observed_class_labels.item()),
                    input_hash=hashlib.sha256(
                        observation.training_sample.input_features.numpy().tobytes()
                    ).hexdigest(),
                    predicted_label=int(probabilities.argmax().item()),
                    correct=bool(
                        result.sample_prediction_record.combined_prediction_is_correct
                    ),
                )
            )
            return result

        with (
            patch.object(client, "process_observed_sample", process),
            patch.object(
                processing,
                "predict_observed_sample_and_update_prediction_weights",
                predict,
            ),
        ):
            yield
