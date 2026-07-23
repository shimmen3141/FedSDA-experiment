"""既存実験結果から、再利用可能なFedDrift固定ベースラインを作成する。"""

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from federated_drift_experiment.data import normalize_dataset_name
from federated_drift_experiment.mode_names import (
    normalize_legacy_mode,
    normalize_legacy_series,
    normalize_series_notation,
)


DEFAULT_OUTPUT = Path("results/baselines/feddrift")
DEFAULT_BATCHES = (50, 100, 200, 500)

# FedDriftでは使用しない設定値は固定ベースラインのCSVから除外する。
IRRELEVANT_COLUMNS = {
    "agg_interval",
    "adwin_delta",
    "e_detector_baseline_strategy",
    "e_detector_baseline_beta",
    "clustering_policy",
    "detection_episodes",
    "new_model_creation_policy",
    "fifo_size",
    "new_model_validation_fraction",
}

PARAMETER_COLUMN_NAMES = {
    "feddrift_batch": "b_detect",
    "distance_threshold": "delta_feddrift",
}


@dataclass(frozen=True)
class BaselineSource:
    """固定ベースラインへ取り込む1つの既存実験。"""

    root: Path
    csv_path: Path
    datasets: tuple[str, ...]
    concept_schedule: str = "random"

    @property
    def raw_dir(self):
        return self.root / "raw"


DEFAULT_SOURCES = (
    BaselineSource(
        root=Path("results/results_20260718_222845"),
        csv_path=Path(
            "results/results_20260718_222845/pareto/"
            "pareto_sea-circle-sine_seeds0-4_n5000_adwin_feddrift.csv"
        ),
        datasets=("sea", "circle", "sine"),
    ),
    BaselineSource(
        root=Path("results/results_20260719_083206"),
        csv_path=Path(
            "results/results_20260719_083206/pareto/"
            "pareto_mnist2-sea2_seeds0-4_n5000_mnist2-sea2-random-e-detectors.csv"
        ),
        datasets=("mnist2", "sea2"),
    ),
)


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_text(value):
    array = np.asarray(value)
    if array.ndim != 0:
        return None
    scalar = array.item()
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    return str(scalar)


def _normalize_series(value, old_mode):
    renamed = normalize_legacy_series(
        str(value), old_mode, normalize_legacy_mode(old_mode)
    )
    return normalize_series_notation(renamed)


def _canonical_series(value, old_mode):
    """文字化けした旧δ表記に依存せず、系列名を標準表記へそろえる。"""
    normalized = _normalize_series(value, old_mode)
    lower = normalized.lower()
    if "batch sweep" in lower or "b_detect sweep" in lower:
        match = re.search(r"\((?:[^=]+)=([0-9.]+)\)", normalized)
        fixed = match.group(1) if match else "0.1"
        return f"FedDrift B_detect sweep (delta_feddrift={fixed})"
    if "sweep" in lower:
        match = re.search(r"(?:batch|B_detect)=([0-9.]+)", normalized)
        fixed = match.group(1) if match else "50"
        return f"FedDrift delta_feddrift sweep (B_detect={fixed})"
    return normalized.replace("δ_FedDrift", "delta_feddrift")


def _sweep_parameter(series):
    return "b_detect" if "B_detect sweep" in series else "delta_feddrift"


def _canonical_fieldnames(source_fieldnames):
    fieldnames = []
    for name in source_fieldnames:
        if name in IRRELEVANT_COLUMNS:
            continue
        target = PARAMETER_COLUMN_NAMES.get(name, name)
        if target not in fieldnames:
            fieldnames.append(target)
    if "concept_schedule" not in fieldnames:
        fieldnames.insert(fieldnames.index("seed"), "concept_schedule")
    if "sweep_parameter" not in fieldnames:
        fieldnames.insert(fieldnames.index("sweep_value"), "sweep_parameter")
    return fieldnames


def _canonical_row(row, concept_schedule):
    old_mode = row.get("mode", "")
    series = _canonical_series(row.get("series", ""), old_mode)
    result = {}
    for name, value in row.items():
        if name in IRRELEVANT_COLUMNS:
            continue
        result[PARAMETER_COLUMN_NAMES.get(name, name)] = value
    result["mode"] = normalize_legacy_mode(old_mode)
    result["dataset"] = normalize_dataset_name(row["dataset"])
    result["series"] = series
    result["concept_schedule"] = row.get("concept_schedule") or concept_schedule
    result["sweep_parameter"] = _sweep_parameter(series)
    return result


def _read_source_rows(source, batches):
    with source.csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"CSVヘッダーがありません: {source.csv_path}")
        fieldnames = _canonical_fieldnames(reader.fieldnames)
        rows = []
        for row in reader:
            if normalize_legacy_mode(row.get("mode", "")) != "FedDrift":
                continue
            if row.get("dataset") not in source.datasets:
                continue
            if int(float(row["feddrift_batch"])) not in batches:
                continue
            rows.append(_canonical_row(row, source.concept_schedule))
    return fieldnames, rows


def _raw_parameters(label, filename):
    series = _canonical_series(label, "FedDrift_v2")
    sweep_parameter = _sweep_parameter(series)
    match = re.search(r"_sv(.+)\.npz$", filename)
    if not match:
        raise ValueError(f"掃引値をファイル名から取得できません: {filename}")
    sweep_text = match.group(1).replace("_", ".")
    sweep_value = float(sweep_text)
    if sweep_parameter == "b_detect":
        return series, sweep_parameter, int(sweep_value), 0.1, sweep_value
    return series, sweep_parameter, 50, sweep_value, sweep_value


def _number_slug(value):
    return f"{float(value):g}".replace(".", "p")


def _raw_target_name(seed, sweep_parameter, sweep_value):
    short = "b" if sweep_parameter == "b_detect" else "d"
    return f"{sweep_parameter}_sweep_seed{seed}_{short}{_number_slug(sweep_value)}.npz"


def _copy_raw_files(source, output_root, batches):
    copied = []
    for path in sorted(source.raw_dir.glob("FedDrift*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        source_dataset = _scalar_text(arrays.get("dataset", ""))
        if source_dataset not in source.datasets:
            continue
        dataset = normalize_dataset_name(source_dataset)
        label = _scalar_text(arrays.get("label", "")) or ""
        series, parameter, b_detect, delta, sweep_value = _raw_parameters(
            label, path.name
        )
        if b_detect not in batches:
            continue

        seed = int(np.asarray(arrays["seed"]).item())
        arrays["mode"] = np.asarray("FedDrift")
        arrays["dataset"] = np.asarray(dataset)
        arrays["label"] = np.asarray(f"{series} [{sweep_value:g}]")
        arrays["concept_schedule"] = np.asarray(
            _scalar_text(arrays.get("concept_schedule", ""))
            or source.concept_schedule
        )
        arrays["b_detect"] = np.asarray(b_detect, dtype=np.int32)
        arrays["delta_feddrift"] = np.asarray(delta, dtype=np.float64)
        arrays["sweep_parameter"] = np.asarray(parameter)
        arrays["sweep_value"] = np.asarray(sweep_value, dtype=np.float64)

        target = (
            output_root / dataset / "raw"
            / _raw_target_name(seed, parameter, sweep_value)
        )
        if target.exists():
            raise FileExistsError(f"出力先が重複しています: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
        copied.append({
            "source": str(path),
            "target": target.relative_to(output_root).as_posix(),
            "sha256": _hash_file(target),
        })
    return copied


def build_baseline(sources=DEFAULT_SOURCES, output_root=DEFAULT_OUTPUT,
                   batches=DEFAULT_BATCHES):
    """複数の既存結果からデータセット別の固定ベースラインを構築する。"""
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(
            f"既存ベースラインを上書きしません: {output_root}"
        )
    output_root.mkdir(parents=True)

    rows_by_dataset = {}
    fields_by_dataset = {}
    raw_files = []
    source_records = []
    for source in sources:
        fieldnames, rows = _read_source_rows(source, set(batches))
        for row in rows:
            dataset = row["dataset"]
            rows_by_dataset.setdefault(dataset, []).append(row)
            fields_by_dataset.setdefault(dataset, fieldnames)
        raw_files.extend(_copy_raw_files(source, output_root, set(batches)))
        source_records.append({
            "root": str(source.root),
            "metrics_csv": str(source.csv_path),
            "metrics_csv_sha256": _hash_file(source.csv_path),
            "datasets": list(source.datasets),
            "concept_schedule": source.concept_schedule,
        })

    dataset_records = {}
    for dataset, rows in sorted(rows_by_dataset.items()):
        rows.sort(key=lambda row: (
            int(row["seed"]), row["sweep_parameter"], float(row["sweep_value"])
        ))
        metrics_path = output_root / dataset / "metrics.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields_by_dataset[dataset])
            writer.writeheader()
            writer.writerows(rows)
        raw_count = sum(
            record["target"].startswith(f"{dataset}/raw/")
            for record in raw_files
        )
        if raw_count != len(rows):
            raise ValueError(
                f"{dataset}: CSV {len(rows)}行に対してNPZ {raw_count}件です"
            )
        dataset_records[dataset] = {
            "metrics_rows": len(rows),
            "raw_files": raw_count,
            "metrics_sha256": _hash_file(metrics_path),
        }

    manifest = {
        "baseline": "FedDrift",
        "status": "immutable_reference",
        "schema_version": 1,
        "selection": {
            "datasets": sorted(dataset_records),
            "seeds": [0, 1, 2, 3, 4],
            "total_data": 5000,
            "concept_schedule": "random",
            "b_detect": list(batches),
            "delta_feddrift": [0.05, 0.1, 0.15, 0.2],
        },
        "name_mapping": {
            "FedDrift_v2": "FedDrift",
            "sea": "sea4",
            "circle": "circle2",
            "sine": "sine2",
            "feddrift_batch": "b_detect",
            "distance_threshold": "delta_feddrift",
            "batch sweep": "B_detect sweep",
            "delta sweep": "delta_feddrift sweep",
        },
        "code_mapping": {
            "B_detect": "FEDDRIFT_DETECT_BATCH",
            "delta_feddrift": "DISTANCE_THRESHOLD",
        },
        "unavailable_new_metrics": [
            "alarm_precision", "alarm_recall", "alarm_f1", "alarm_total",
            "switch_fp_early", "switch_fp_late",
            "switch_fp_duplicate", "switch_fp_isolated",
            "adaptation_reuse_count", "adaptation_reuse_precision",
            "adaptation_create_count", "adaptation_create_precision",
            "server_mapping_change_count",
        ],
        "not_applicable_fedsda_metrics": [
            "adaptation_create_rejected_count",
            "provisional_*",
            "adaptation_episode_suppressed_count",
        ],
        "sources": source_records,
        "datasets": dataset_records,
        "raw_files": raw_files,
    }
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="既存結果からFedDrift固定ベースラインを非破壊で作成する"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_baseline(output_root=args.output)
    print(json.dumps(manifest["datasets"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
