"""FedCCFAの表現部とクラスごとの行・アンカーの集約、配布、通信の記録。"""

from typing import TYPE_CHECKING

import numpy as np
import torch

from federated_learning_experiments.runtime.fedccfa_classifier_clustering import (
    _label_groups,
    _weighted_mean,
)

if TYPE_CHECKING:
    from federated_learning_experiments.runtime.feddrift_run_server import FeddriftRunServer


@torch.no_grad()
def aggregate_fedccfa_training_round(
    *, run_server: "FeddriftRunServer", timestep_index: int
) -> None:
    server = run_server
    states = [c.stream_processor._fedccfa for c in server.run_clients]
    assert all(s is not None for s in states)
    states = [s for s in states if s is not None]
    round_index = server._current_training_round_index

    def transfer(
        component: str, direction: str, values: dict[str, torch.Tensor], count: int = 1
    ) -> None:
        server._record_parameter_transfer(direction, dict(values), count)
        server._fedccfa_communication_records.append(
            (
                timestep_index,
                round_index,
                component,
                direction,
                sum(v.numel() for v in values.values()) * count,
            )
        )

    def representation(s):
        return {
            k: p.detach()
            for k, p in s.classifier.named_parameters()
            if not k.startswith("classification_layer.")
        }

    reps = [representation(s) for s in states]
    counts = [len(s.samples) for s in states]
    averaged = {k: _weighted_mean([rep[k] for rep in reps], counts) for k in reps[0]}
    parameters = server.global_model_repository.get_global_model_parameters(model_id=0)
    parameters.update(averaged)
    # グローバルなモデルの初期の分類層は一度も更新しない。
    server.global_model_repository.set_global_model_parameters(
        model_id=0, parameter_snapshot=parameters
    )
    for s, rep in zip(states, reps):
        s.stage_trace.append("aggregate_representation")
        transfer("representation", "upload", rep)
        transfer("diagnostic_classifier", "upload", s.diagnostic_parameters)
        transfer(
            "personal_classifier", "upload", dict(s.classifier.classification_layer.state_dict())
        )
        transfer("anchor", "upload", {str(c): p for c, p in s.local_anchors.items()})
        for key, value in averaged.items():
            rep[key].copy_(value)
    transfer("representation", "download", averaged, len(states))
    server._aggregation_parameter_multiply_accumulate_count += sum(
        v.numel() for v in averaged.values()
    ) * len(states)
    eps = server.run_clients[0]._settings.fedccfa_dbscan_eps
    for label in range(states[0].classifier.class_count):
        eligible = [i for i, s in enumerate(states) if s.class_counts[label] > 0]
        vectors = np.array(
            [
                torch.cat(
                    (
                        states[i].diagnostic_parameters["weight"][label],
                        states[i].diagnostic_parameters["bias"][label : label + 1],
                    )
                ).numpy()
                for i in eligible
            ]
        )
        groups = tuple(tuple(eligible[i] for i in group) for group in _label_groups(vectors, eps))
        by_client = {i: group_index for group_index, group in enumerate(groups) for i in group}
        for i, s in enumerate(states):
            s.cluster_records.append(
                (timestep_index, round_index, label, by_client.get(i, -1), s.class_counts[label])
            )
        for group in groups:
            weights = _weighted_mean(
                [states[i].classifier.classification_layer.weight[label] for i in group],
                [1] * len(group),
            )
            bias = _weighted_mean(
                [states[i].classifier.classification_layer.bias[label] for i in group],
                [1] * len(group),
            )
            anchor = _weighted_mean(
                [states[i].local_anchors[label] for i in group], [1] * len(group)
            )
            for i in group:
                states[i].classifier.classification_layer.weight[label].copy_(weights)
                states[i].classifier.classification_layer.bias[label].copy_(bias)
                states[i].anchors[label] = anchor.clone()
            transfer(
                "personal_classifier_row", "download", {"weight": weights, "bias": bias}, len(group)
            )
            transfer("anchor", "download", {"anchor": anchor}, len(group))
            server._aggregation_parameter_multiply_accumulate_count += (
                weights.numel() + 1 + anchor.numel()
            ) * len(group)
    for s in states:
        s.stage_trace.append("cluster_and_average")
