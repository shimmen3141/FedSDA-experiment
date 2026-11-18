"""系列1本を処理し、記録・指標・資源・経過時間をまとめる。"""

import time
from dataclasses import dataclass

import numpy as np

from single_stream_hard_assignment_soft_prediction.hoeffding_tree_concept_model import (
    convert_features_to_dictionary,
)
from single_stream_hard_assignment_soft_prediction.single_stream_input import SingleStreamInput
from single_stream_hard_assignment_soft_prediction.single_stream_metrics import (
    classify_concept_identification,
    compute_prequential_metrics,
)
from single_stream_hard_assignment_soft_prediction.single_stream_processing import (
    SingleStreamHardAssignmentSoftPredictionProcessor,
)
from single_stream_hard_assignment_soft_prediction.single_stream_settings import (
    SingleStreamHardAssignmentSoftPredictionSettings,
)


@dataclass(frozen=True, kw_only=True)
class SingleStreamRunResult:
    """系列1本の結果。`record_tables`は名前付きの配列、`summary`は指標・資源・経過時間。"""

    record_tables: dict[str, np.ndarray]
    summary: dict[str, object]


def run_single_stream(
    *, stream: SingleStreamInput, settings: SingleStreamHardAssignmentSoftPredictionSettings
) -> SingleStreamRunResult:
    """全標本を順に処理する（予測→学習）。事前学習の標本は使わない。"""
    if type(stream) is not SingleStreamInput:
        raise TypeError("streamはSingleStreamInputが必要です。")
    processor = SingleStreamHardAssignmentSoftPredictionProcessor(
        settings=settings, class_count=stream.class_count
    )
    started = time.perf_counter()
    records = [
        processor.process_sample(
            sample_index=index,
            features=convert_features_to_dictionary(row=row),
            observed_class_id=int(label),
            observed_concept_id=int(concept_id),
        )
        for index, (row, label, concept_id) in enumerate(
            zip(stream.features, stream.labels, stream.concept_ids, strict=True)
        )
    ]
    elapsed_seconds = time.perf_counter() - started
    sample_count = len(records)
    assigned = np.full(sample_count, -1, dtype=np.int64)
    for sample_index, model_id in processor.assigned_model_id_by_sample_index.items():
        assigned[sample_index] = model_id
    alarms = processor.alarm_records
    record_tables = {
        "sample.predicted_class_id": np.array(
            [r.predicted_class_id for r in records], dtype=np.int64
        ),
        "sample.prediction_is_correct": np.array(
            [r.prediction_is_correct for r in records], dtype=bool
        ),
        "sample.observed_class_id": np.array(
            [r.observed_class_id for r in records], dtype=np.int64
        ),
        "sample.observed_concept_id": np.array(
            [r.observed_concept_id for r in records], dtype=np.int64
        ),
        "sample.current_model_id": np.array([r.current_model_id for r in records], dtype=np.int64),
        "sample.held_model_count": np.array([r.held_model_count for r in records], dtype=np.int64),
        "sample.monitored_log_e_value": np.array(
            [r.monitored_log_e_value for r in records], dtype=np.float64
        ),
        "sample.assigned_model_id": assigned,
        "alarm.sample_index": np.array([a.sample_index for a in alarms], dtype=np.int64),
        "alarm.selected_hypothesis": np.array([a.selected_hypothesis for a in alarms], dtype=str),
        "alarm.change_interval_sample_count": np.array(
            [a.change_interval_sample_count for a in alarms], dtype=np.int64
        ),
        "alarm.window_sample_count": np.array(
            [a.window_sample_count for a in alarms], dtype=np.int64
        ),
        "alarm.destination_model_id": np.array(
            [a.destination_model_id for a in alarms], dtype=np.int64
        ),
        "alarm.new_model_learner_name": np.array(
            [a.new_model_learner_name for a in alarms], dtype=str
        ),
        "alarm.description_length": np.array(
            [a.description_length for a in alarms], dtype=np.float64
        ),
        "alarm.alarm_raised": np.array([a.alarm_raised for a in alarms], dtype=bool),
        "alarm.adaptation_outcome": np.array([a.adaptation_outcome for a in alarms], dtype=str),
        "model.model_id": np.array(
            list(processor.model_created_sample_index_by_model_id), dtype=np.int64
        ),
        "model.created_sample_index": np.array(
            list(processor.model_created_sample_index_by_model_id.values()), dtype=np.int64
        ),
    }
    metrics = compute_prequential_metrics(
        predicted_class_ids=record_tables["sample.predicted_class_id"],
        observed_class_ids=record_tables["sample.observed_class_id"],
        concept_ids=record_tables["sample.observed_concept_id"],
        class_count=stream.class_count,
    )
    raised = [a for a in alarms if a.alarm_raised]
    identification = classify_concept_identification(
        alarms=[
            dict(
                sample_index=a.sample_index,
                adaptation_outcome=a.adaptation_outcome,
                destination_model_id=a.destination_model_id,
                tail_concept_ids=a.window_tail_concept_ids,
            )
            for a in raised
        ],
        assigned_model_ids=assigned,
        concept_ids=record_tables["sample.observed_concept_id"],
        model_created_sample_index_by_model_id=processor.model_created_sample_index_by_model_id,
    )
    summary: dict[str, object] = dict(
        dataset_name=stream.dataset_name,
        random_seed=stream.random_seed,
        sample_count=sample_count,
        class_count=stream.class_count,
        **metrics,
        alarm_count=len(raised),
        detector_signal_count=len(alarms),
        created_model_count=len(processor.model_created_sample_index_by_model_id) - 1,
        final_held_model_count=len(processor.held_models),
        final_tree_node_count=sum(model.node_count for _, model in processor.held_models),
        elapsed_seconds=elapsed_seconds,
        concept_identification=identification,
    )
    return SingleStreamRunResult(record_tables=record_tables, summary=summary)
