import csv
import json

import numpy as np

from federated_drift_experiment.experiment_spec.artifacts import (
    bounded_artifact_stem,
    raw_run_filename,
)
from tools.experiments.artifacts import rebuild_artifacts


def _raw(path, *, exact=False):
    metrics = {
        "accuracy": 0.75,
        "stable_accuracy": 0.8,
        "comm_models_total": 123,
        "final_model_count": 2,
    } if exact else {}
    np.savez_compressed(
        path,
        history_accuracy=np.asarray([[1, 0, 1, 1]], dtype=np.int8),
        drift_client_ids=np.asarray([], dtype=np.int32),
        drift_positions=np.asarray([], dtype=np.int32),
        dataset=np.asarray("sea2"), concept_schedule=np.asarray("random"),
        mode=np.asarray("FedSDA_NoCached_ClassESR"),
        label=np.asarray("FedSDA_NoCached_ClassESR A sweep [50]"),
        parameter_schema_version=np.asarray(1),
        sweep_parameter=np.asarray("aggregation_interval"),
        sweep_value=np.asarray(50.0), seed=np.asarray(0),
        aggregation_interval=np.asarray(50),
        feddrift_detection_batch_size=np.asarray(-1),
        fedsda_distance_threshold=np.asarray(0.1),
        feddrift_distance_threshold=np.asarray(np.nan),
        adwin_delta=np.asarray(np.nan), clustering_policy=np.asarray("on_new_model"),
        clustering_decision=np.asarray("distance"),
        clustering_consolidation=np.asarray("merge"),
        shared_backbone_training=np.asarray(""),
        shared_backbone_routing_recalibration=np.asarray(""),
        shared_adapter_rank=np.asarray(-1), total_data=np.asarray(4),
        result_metrics_json=np.asarray(json.dumps(metrics)),
        round_global_model_count=np.asarray([1, 2]),
    )


def test_artifact_names_are_bounded_and_identity_sensitive():
    stem = bounded_artifact_stem("x" * 300, hint="pareto-study", limit=48)
    first = raw_run_filename({"value": 1}, dataset="mnist4", seed=3)
    second = raw_run_filename({"value": 2}, dataset="mnist4", seed=3)

    assert len(stem) <= 48
    assert len(first) < 50
    assert first != second


def test_rebuild_artifacts_uses_embedded_exact_metrics(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _raw(raw / "run.npz", exact=True)

    csv_path, metadata = rebuild_artifacts(
        tmp_path, output_dir=tmp_path / "pareto", tag="test", plot=False,
    )
    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))

    assert metadata["quality"] == "exact"
    assert float(row["comm_models_total"]) == 123
    assert float(row["accuracy"]) == 0.75


def test_rebuild_artifacts_recovers_pareto_metrics_from_old_raw_and_log(tmp_path):
    raw = tmp_path / "raw"
    logs = tmp_path / "logs"
    raw.mkdir()
    logs.mkdir()
    _raw(raw / "legacy.npz", exact=False)
    (logs / "run.log").write_text(
        "[1/1] sea2/FedSDA_NoCached_ClassESR/aggregation_interval=50/s0: "
        "stable_acc=0.8000 comm=321 models=3 (1s)\n",
        encoding="utf-8",
    )

    csv_path, metadata = rebuild_artifacts(
        tmp_path, output_dir=tmp_path / "pareto", tag="legacy", plot=False,
    )
    row = next(csv.DictReader(csv_path.open(encoding="utf-8")))

    assert metadata["quality"] == "partial"
    assert float(row["accuracy"]) == 0.75
    assert float(row["comm_models_total"]) == 321
    assert float(row["final_model_count"]) == 3
