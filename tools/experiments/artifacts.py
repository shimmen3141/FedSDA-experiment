"""NPZとログから欠損した正規CSV・Pareto図を復元する。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from federated_drift_experiment.experiment_spec.artifacts import (
    bounded_artifact_stem,
)
from federated_drift_experiment.experiment_spec.manifests import write_json_atomic
from run_pareto_sweep import (
    ADWIN_DELTA,
    AGGREGATION_INTERVAL,
    FEDDRIFT_DETECTION_BATCH_SIZE,
    FEDDRIFT_DISTANCE_THRESHOLD,
    FEDSDA_DISTANCE_THRESHOLD,
    METRIC_KEYS,
    PARAMETER_SCHEMA_VERSION,
    combine_and_plot,
    write_csv,
)


_PROGRESS = re.compile(
    r"^\[\d+/\d+\] (?P<dataset>[^/]+)/(?P<mode>[^/]+)"
    r"(?:/(?P<parameter>[^/=]+)=(?P<value>[^/]+))?/s(?P<seed>\d+): "
    r"stable_acc=(?P<stable>[-+0-9.eE]+) comm=(?P<comm>\d+) "
    r"models=(?P<models>\d+)"
)


def _scalar(arrays, key, default=None):
    if key not in arrays:
        return default
    value = arrays[key]
    return value.item() if getattr(value, "shape", None) == () else value


def _optional_number(value, *, integer=False):
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number) or number < 0:
        return None
    return int(number) if integer else number


def _infer_sweep(mode, label, arrays):
    parameter = str(_scalar(arrays, "sweep_parameter", "") or "")
    value = _optional_number(_scalar(arrays, "sweep_value"))
    if not parameter:
        if mode == "FedDrift" and "B_detect sweep" in label:
            parameter = FEDDRIFT_DETECTION_BATCH_SIZE
        elif mode == "FedDrift" and "δ_FedDrift sweep" in label:
            parameter = FEDDRIFT_DISTANCE_THRESHOLD
        elif " A sweep" in label:
            parameter = AGGREGATION_INTERVAL
        elif "δ_ADWIN sweep" in label:
            parameter = ADWIN_DELTA
    if value is None:
        value = _optional_number(_scalar(arrays, parameter)) if parameter else None
    return parameter or None, value


def _series_without_value(label, sweep_value):
    if sweep_value is None:
        return label
    return re.sub(rf"\s+\[{re.escape(f'{sweep_value:g}')}\]$", "", label)


def _stable_accuracy(arrays, window=200):
    history = np.asarray(arrays["history_accuracy"])
    events = {client_id: [] for client_id in range(history.shape[0])}
    for client_id, position in zip(
        arrays["drift_client_ids"], arrays["drift_positions"]
    ):
        events[int(client_id)].append(int(position))
    values = []
    for client_id, client_history in enumerate(history):
        positions = sorted(events[client_id])
        for index, correct in enumerate(client_history):
            previous = next(
                (position for position in reversed(positions) if position <= index),
                None,
            )
            if previous is None or index >= previous + window:
                values.append(float(correct))
    return float(np.mean(values)) if values else float("nan")


def _parse_logs(result_root):
    records = {}
    for path in sorted(Path(result_root).rglob("*.log")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _PROGRESS.match(line)
            if not match:
                continue
            values = match.groupdict()
            key = (
                values["dataset"], values["mode"],
                values["parameter"] or "", values["value"] or "",
                int(values["seed"]),
            )
            records[key] = {
                "stable_accuracy": float(values["stable"]),
                "comm_models_total": int(values["comm"]),
                "final_model_count": int(values["models"]),
            }
    return records


def _metadata_row(arrays):
    mode = str(_scalar(arrays, "mode"))
    label = str(_scalar(arrays, "label", mode))
    sweep_parameter, sweep_value = _infer_sweep(mode, label, arrays)
    row = {key: float("nan") for key in METRIC_KEYS}
    row.update({
        "parameter_schema_version": int(
            _scalar(arrays, "parameter_schema_version", PARAMETER_SCHEMA_VERSION)
        ),
        "mode": mode,
        "dataset": str(_scalar(arrays, "dataset")),
        "concept_schedule": str(_scalar(arrays, "concept_schedule", "random")),
        "seed": int(_scalar(arrays, "seed")),
        "series": _series_without_value(label, sweep_value),
        "sweep_parameter": sweep_parameter,
        "sweep_value": sweep_value,
        FEDDRIFT_DETECTION_BATCH_SIZE: _optional_number(
            _scalar(arrays, FEDDRIFT_DETECTION_BATCH_SIZE), integer=True,
        ),
        AGGREGATION_INTERVAL: _optional_number(
            _scalar(arrays, AGGREGATION_INTERVAL), integer=True,
        ),
        "clustering_policy": str(_scalar(arrays, "clustering_policy", "")),
        "clustering_decision": str(_scalar(arrays, "clustering_decision", "")),
        "clustering_consolidation": str(
            _scalar(arrays, "clustering_consolidation", "merge")
        ),
        "detection_episodes": bool(_scalar(arrays, "detection_episodes", False)),
        "new_model_creation_policy": str(_scalar(
            arrays, "new_model_creation_policy",
            re.search(r"\[creation=([^]]+)\]", label).group(1)
            if re.search(r"\[creation=([^]]+)\]", label) else "immediate",
        )),
        "fifo_size": int(_scalar(
            arrays, "fifo_size",
            re.search(r"\[N_FIFO=(\d+)\]", label).group(1)
            if re.search(r"\[N_FIFO=(\d+)\]", label) else 30,
        )),
        "new_model_validation_fraction": float(_scalar(
            arrays, "new_model_validation_fraction", 0.2,
        )),
        "new_model_forward_validation_samples": int(_scalar(
            arrays, "new_model_forward_validation_samples",
            re.search(r"\[forward=(\d+)\]", label).group(1)
            if re.search(r"\[forward=(\d+)\]", label) else 10,
        )),
        "shared_backbone_training": str(_scalar(
            arrays, "shared_backbone_training", "",
        )),
        "expert_training_assignment": str(_scalar(
            arrays, "expert_training_assignment", "assigned",
        )),
        "shared_backbone_gradient_strategy": str(_scalar(
            arrays, "shared_backbone_gradient_strategy", "",
        )),
        "shared_backbone_routing_recalibration": str(_scalar(
            arrays, "shared_backbone_routing_recalibration", "",
        )),
        "shared_adapter_rank": _optional_number(
            _scalar(arrays, "shared_adapter_rank"), integer=True,
        ),
        FEDSDA_DISTANCE_THRESHOLD: _optional_number(
            _scalar(arrays, FEDSDA_DISTANCE_THRESHOLD),
        ),
        FEDDRIFT_DISTANCE_THRESHOLD: _optional_number(
            _scalar(arrays, FEDDRIFT_DISTANCE_THRESHOLD),
        ),
        ADWIN_DELTA: _optional_number(_scalar(arrays, ADWIN_DELTA)),
    })
    return row


def _row_from_raw(path, log_records):
    with np.load(path, allow_pickle=False) as arrays:
        row = _metadata_row(arrays)
        exact_metrics = json.loads(str(_scalar(arrays, "result_metrics_json", "{}")))
        row.update({key: exact_metrics.get(key, row[key]) for key in METRIC_KEYS})
        row["accuracy"] = exact_metrics.get(
            "accuracy", float(np.asarray(arrays["history_accuracy"]).mean())
        )
        row["stable_accuracy"] = exact_metrics.get(
            "stable_accuracy", _stable_accuracy(arrays)
        )
        parameter = row["sweep_parameter"] or ""
        value = "" if row["sweep_value"] is None else f"{row['sweep_value']:g}"
        log_key = (row["dataset"], row["mode"], parameter, value, row["seed"])
        row.update(log_records.get(log_key, {}))

        model_counts = np.asarray(arrays.get("round_global_model_count", ()))
        if model_counts.size:
            row["mean_model_count"] = exact_metrics.get(
                "mean_model_count", float(model_counts.mean())
            )
            row["max_model_count"] = exact_metrics.get(
                "max_model_count", float(model_counts.max())
            )
            row["model_count_auc"] = exact_metrics.get(
                "model_count_auc", float(model_counts.sum())
            )
        telemetry = {
            "compute_optimizer_steps_total": "round_client_optimizer_steps",
            "compute_backbone_optimizer_steps_total": "round_client_backbone_optimizer_steps",
            "compute_head_optimizer_steps_total": "round_client_head_optimizer_steps",
            "compute_backbone_examples_total": "round_client_backbone_examples",
            "compute_head_examples_total": "round_client_head_examples",
            "compute_drift_detector_updates_total": "round_client_drift_detector_updates",
            "compute_drift_detector_hypotheses_total": "round_client_drift_detector_hypotheses",
        }
        for metric, key in telemetry.items():
            if key in arrays and metric not in exact_metrics:
                row[metric] = float(np.asarray(arrays[key]).sum())
    return row, bool(exact_metrics)


def rebuild_artifacts(result_root, *, output_dir=None, tag="recovered", plot=True):
    result_root = Path(result_root).resolve()
    raw_paths = sorted(result_root.rglob("*.npz"))
    if not raw_paths:
        raise FileNotFoundError(f"NPZが見つかりません: {result_root}")
    output_dir = Path(output_dir or result_root / "pareto").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_records = _parse_logs(result_root)
    rows = []
    exact_count = 0
    for path in raw_paths:
        row, exact = _row_from_raw(path, log_records)
        rows.append(row)
        exact_count += int(exact)
    identity = hashlib.sha256(
        "\n".join(str(path) for path in raw_paths).encode("utf-8")
    ).hexdigest()[:12]
    stem = bounded_artifact_stem(
        f"recovered_{tag}_{identity}", hint=f"recovered_{tag}",
    )
    csv_path = output_dir / f"{stem}.csv"
    write_csv(rows, csv_path)
    if plot:
        combine_and_plot(
            [str(csv_path)], str(output_dir), f"{tag}_stable", "stable_accuracy",
        )
        combine_and_plot(
            [str(csv_path)], str(output_dir), f"{tag}_accuracy", "accuracy",
        )
    metadata = {
        "schema_version": 1,
        "source_root": str(result_root),
        "raw_file_count": len(raw_paths),
        "exact_rows": exact_count,
        "partial_rows": len(rows) - exact_count,
        "metrics_csv": str(csv_path),
        "quality": "exact" if exact_count == len(rows) else "partial",
        "note": (
            "partial行はNPZと進捗ログからPareto必須指標を復元し、"
            "復元不能な指標をNaNとした"
        ),
    }
    write_json_atomic(output_dir / f"{stem}.reconstruction.json", metadata)
    return csv_path, metadata


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tag", default="recovered")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    path, metadata = rebuild_artifacts(
        args.result_root, output_dir=args.output_dir, tag=args.tag,
        plot=not args.no_plot,
    )
    print(
        f"Rebuilt: {path} (quality={metadata['quality']}, "
        f"exact={metadata['exact_rows']}, partial={metadata['partial_rows']})"
    )


if __name__ == "__main__":
    main()
