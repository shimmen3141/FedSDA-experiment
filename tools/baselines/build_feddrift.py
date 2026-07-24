"""既存実験結果から、再利用可能なFedDrift固定ベースラインを作成する。"""

import argparse
import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from federated_drift_experiment.data import normalize_dataset_name
from federated_drift_experiment.parameter_schema import (
    PARAMETER_SCHEMA_VERSION,
    parameter,
)


DEFAULT_OUTPUT = Path("results/baselines/feddrift")
DEFAULT_BATCHES = (50, 100, 200, 500)

# FedDriftでは使用しない設定値は固定ベースラインのCSVから除外する。
IRRELEVANT_COLUMNS = {
    "aggregation_interval",
    "adwin_delta",
    "e_detector_baseline_strategy",
    "e_detector_baseline_beta",
    "clustering_policy",
    "detection_episodes",
    "new_model_creation_policy",
    "fifo_size",
    "new_model_validation_fraction",
}

FEDDRIFT_BATCH = parameter("feddrift_detection_batch_size").csv_name
FEDDRIFT_DISTANCE = parameter("feddrift_distance_threshold").csv_name


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


DEFAULT_SOURCE_CONFIG = Path("tools/baselines/feddrift_sources.json")


def load_sources(config_paths):
    """JSON設定から入力実験を読み込む。"""
    sources = []
    for config_path in config_paths:
        config_path = Path(config_path)
        records = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"入力設定は配列である必要があります: {config_path}")
        for record in records:
            root = Path(record["root"])
            csv_path = Path(record["csv_path"])
            sources.append(BaselineSource(
                root=root,
                csv_path=csv_path,
                datasets=tuple(record["datasets"]),
                concept_schedule=record.get("concept_schedule", "random"),
            ))
    return tuple(sources)


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


def _canonical_series(value, old_mode):
    """正規スキーマのFedDrift系列名を検証して返す。"""
    if old_mode != "FedDrift":
        raise ValueError(f"FedDrift以外の入力モードです: {old_mode!r}")
    normalized = str(value)
    if "sweep" in normalized and not (
        "B_detect sweep" in normalized or "δ_FedDrift sweep" in normalized
    ):
        raise ValueError(f"旧形式または未知の系列名です: {normalized!r}")
    return normalized


def _sweep_parameter(series):
    return (
        FEDDRIFT_BATCH
        if "B_detect sweep" in series
        else FEDDRIFT_DISTANCE
    )


def _canonical_fieldnames(source_fieldnames):
    fieldnames = []
    for name in source_fieldnames:
        if name in IRRELEVANT_COLUMNS:
            continue
        if name not in fieldnames:
            fieldnames.append(name)
    if "concept_schedule" not in fieldnames:
        fieldnames.insert(fieldnames.index("seed"), "concept_schedule")
    if "parameter_schema_version" not in fieldnames:
        fieldnames.insert(0, "parameter_schema_version")
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
        result[name] = value
    result["mode"] = old_mode
    result["parameter_schema_version"] = PARAMETER_SCHEMA_VERSION
    result["dataset"] = normalize_dataset_name(row["dataset"])
    result["series"] = series
    result["concept_schedule"] = row.get("concept_schedule") or concept_schedule
    sweep_parameter = row.get("sweep_parameter", "")
    if sweep_parameter not in {FEDDRIFT_BATCH, FEDDRIFT_DISTANCE}:
        sweep_value = float(row["sweep_value"])
        batch_value = float(row[FEDDRIFT_BATCH])
        distance_value = float(row[FEDDRIFT_DISTANCE])
        matches_batch = np.isclose(sweep_value, batch_value)
        matches_distance = np.isclose(sweep_value, distance_value)
        if matches_batch and not matches_distance:
            sweep_parameter = FEDDRIFT_BATCH
        elif matches_distance and not matches_batch:
            sweep_parameter = FEDDRIFT_DISTANCE
        else:
            sweep_parameter = _sweep_parameter(series)
    result["sweep_parameter"] = sweep_parameter
    return result


def _read_source_rows(source, batches):
    with source.csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"CSVヘッダーがありません: {source.csv_path}")
        fieldnames = _canonical_fieldnames(reader.fieldnames)
        rows = []
        for row in reader:
            if row.get("mode", "") != "FedDrift":
                continue
            if int(row.get("parameter_schema_version") or -1) != PARAMETER_SCHEMA_VERSION:
                raise ValueError(
                    f"未対応のパラメータスキーマです: {source.csv_path}"
                )
            if row.get("dataset") not in source.datasets:
                continue
            batch_value = row.get(FEDDRIFT_BATCH)
            if batch_value in (None, ""):
                raise ValueError(
                    f"正規列 {FEDDRIFT_BATCH!r} がありません: {source.csv_path}"
                )
            if int(float(batch_value)) not in batches:
                continue
            rows.append(_canonical_row(row, source.concept_schedule))
    return fieldnames, rows


def _raw_parameters(
    label,
    filename,
    source_mode,
    stored_parameter="",
    stored_sweep_value=None,
):
    series = _canonical_series(label, source_mode)
    sweep_parameter = (
        stored_parameter
        if stored_parameter in {FEDDRIFT_BATCH, FEDDRIFT_DISTANCE}
        else _sweep_parameter(series)
    )
    match = re.search(r"_sv(.+)\.npz$", filename)
    if not match:
        raise ValueError(f"掃引値をファイル名から取得できません: {filename}")
    sweep_text = match.group(1).replace("_", ".").replace("p", ".")
    filename_sweep_value = float(sweep_text)
    sweep_value = (
        float(stored_sweep_value)
        if stored_sweep_value is not None
        and np.isfinite(float(stored_sweep_value))
        else filename_sweep_value
    )
    if sweep_parameter == FEDDRIFT_BATCH:
        return series, sweep_parameter, int(sweep_value), 0.1, sweep_value
    return series, sweep_parameter, 50, sweep_value, sweep_value


def _number_slug(value):
    return f"{float(value):g}".replace(".", "p")


def _raw_target_name(seed, sweep_parameter, sweep_value):
    short = "b" if sweep_parameter == FEDDRIFT_BATCH else "d"
    return f"{sweep_parameter}_sweep_seed{seed}_{short}{_number_slug(sweep_value)}.npz"


def _copy_raw_files(source, output_root, batches):
    copied = []
    for path in sorted(source.raw_dir.glob("FedDrift*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if int(np.asarray(arrays.get("parameter_schema_version", -1)).item()) != PARAMETER_SCHEMA_VERSION:
            raise ValueError(f"未対応のパラメータスキーマです: {path}")
        source_dataset = _scalar_text(arrays.get("dataset", ""))
        if source_dataset not in source.datasets:
            continue
        dataset = normalize_dataset_name(source_dataset)
        label = _scalar_text(arrays.get("label", "")) or ""
        source_mode = _scalar_text(arrays.get("mode", "")) or ""
        stored_parameter = _scalar_text(
            arrays.get("sweep_parameter", "")
        ) or ""
        stored_sweep_value = np.asarray(
            arrays.get("sweep_value", np.nan)
        ).item()
        series, parameter, b_detect, delta, sweep_value = _raw_parameters(
            label,
            path.name,
            source_mode,
            stored_parameter=stored_parameter,
            stored_sweep_value=stored_sweep_value,
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
        arrays["parameter_schema_version"] = np.asarray(
            PARAMETER_SCHEMA_VERSION, dtype=np.int32
        )
        arrays[FEDDRIFT_BATCH] = np.asarray(b_detect, dtype=np.int32)
        arrays[FEDDRIFT_DISTANCE] = np.asarray(delta, dtype=np.float64)
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


def build_baseline(sources=None, output_root=DEFAULT_OUTPUT,
                   batches=DEFAULT_BATCHES):
    """複数の既存結果からデータセット別の固定ベースラインを構築する。"""
    if sources is None:
        sources = load_sources((DEFAULT_SOURCE_CONFIG,))
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

    all_rows = [
        row for rows in rows_by_dataset.values() for row in rows
    ]
    total_data_values = sorted({
        int(float(row["total_data"]))
        for row in all_rows if row.get("total_data")
    })
    manifest = {
        "baseline": "FedDrift",
        "status": "immutable_reference",
        "schema_version": 1,
        "parameter_schema_version": PARAMETER_SCHEMA_VERSION,
        "selection": {
            "datasets": sorted(dataset_records),
            "seeds": sorted({int(row["seed"]) for row in all_rows}),
            "total_data": total_data_values,
            "concept_schedule": sorted({
                row["concept_schedule"] for row in all_rows
            }),
            FEDDRIFT_BATCH: sorted({
                int(float(row[FEDDRIFT_BATCH])) for row in all_rows
            }),
            FEDDRIFT_DISTANCE: sorted({
                float(row[FEDDRIFT_DISTANCE]) for row in all_rows
            }),
        },
        "code_mapping": {
            parameter("feddrift_detection_batch_size").paper_symbol:
                "FEDDRIFT_DETECTION_BATCH_SIZE",
            parameter("feddrift_distance_threshold").paper_symbol:
                "FEDDRIFT_DISTANCE_THRESHOLD",
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


def _read_metrics(path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"CSVヘッダーがありません: {path}")
        return list(reader.fieldnames), list(reader)


def _result_key(row):
    return (
        str(row["seed"]),
        row.get("concept_schedule", "random"),
        row["sweep_parameter"],
        f"{float(row['sweep_value']):g}",
    )


def _merge_duplicate_row(existing, incoming, source):
    merged = dict(existing)
    for key in set(existing) | set(incoming):
        old_value = existing.get(key, "")
        new_value = incoming.get(key, "")
        if old_value and new_value and old_value != new_value:
            raise ValueError(
                f"同じ実験キーの結果が競合しています: {source} "
                f"({_result_key(existing)}, {key})"
            )
        if not old_value and new_value:
            merged[key] = new_value
    return merged


def _arrays_equal(left, right):
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    )


def _merge_raw_file(existing, incoming):
    with np.load(existing, allow_pickle=False) as archive:
        old_arrays = {key: archive[key] for key in archive.files}
    with np.load(incoming, allow_pickle=False) as archive:
        new_arrays = {key: archive[key] for key in archive.files}
    for key in old_arrays.keys() & new_arrays.keys():
        if not _arrays_equal(old_arrays[key], new_arrays[key]):
            raise ValueError(
                f"同じ実験キーのNPZが競合しています: {incoming} ({key})"
            )
    if new_arrays.keys() - old_arrays.keys():
        np.savez_compressed(existing, **(old_arrays | new_arrays))


def _merge_baseline_trees(staging_root, incoming_root):
    """検証用コピーへ新しいbaselineを統合し、追加件数を返す。"""
    added = 0
    datasets = {
        path.name for path in staging_root.iterdir() if path.is_dir()
    } | {
        path.name for path in incoming_root.iterdir() if path.is_dir()
    }
    for dataset in sorted(datasets):
        existing_metrics = staging_root / dataset / "metrics.csv"
        incoming_metrics = incoming_root / dataset / "metrics.csv"
        if not incoming_metrics.is_file():
            continue
        new_fields, new_rows = _read_metrics(incoming_metrics)
        if existing_metrics.is_file():
            old_fields, old_rows = _read_metrics(existing_metrics)
        else:
            old_fields, old_rows = [], []
            existing_metrics.parent.mkdir(parents=True, exist_ok=True)

        fields = old_fields + [field for field in new_fields if field not in old_fields]
        rows = {_result_key(row): row for row in old_rows}
        for new_row in new_rows:
            key = _result_key(new_row)
            incoming_raw = (
                incoming_root / dataset / "raw"
                / _raw_target_name(new_row["seed"], key[2], key[3])
            )
            existing_raw = (
                staging_root / dataset / "raw"
                / _raw_target_name(new_row["seed"], key[2], key[3])
            )
            if key in rows:
                rows[key] = _merge_duplicate_row(rows[key], new_row, incoming_metrics)
                _merge_raw_file(existing_raw, incoming_raw)
                continue
            existing_raw.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(incoming_raw, existing_raw)
            rows[key] = new_row
            added += 1

        ordered_rows = sorted(rows.values(), key=lambda row: (
            int(row["seed"]), row["sweep_parameter"], float(row["sweep_value"])
        ))
        with existing_metrics.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(ordered_rows)
    return added


def _refresh_manifest(root, existing_manifest, incoming_manifest, added):
    """統合後の件数・ハッシュ・選択範囲を実ファイルから再計算する。"""
    datasets = {}
    seeds = set()
    batches = set()
    deltas = set()
    schedules = set()
    total_data_values = set()
    raw_files = {
        record["target"]: record
        for record in existing_manifest.get("raw_files", [])
    }
    raw_files.update({
        record["target"]: record
        for record in incoming_manifest.get("raw_files", [])
    })
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        metrics = dataset_dir / "metrics.csv"
        if not metrics.is_file():
            continue
        _, rows = _read_metrics(metrics)
        raw_paths = sorted((dataset_dir / "raw").glob("*.npz"))
        if len(rows) != len(raw_paths):
            raise ValueError(
                f"{dataset_dir.name}: CSV {len(rows)}行に対して"
                f"NPZ {len(raw_paths)}件です"
            )
        for row in rows:
            seeds.add(int(row["seed"]))
            batches.add(int(float(row[FEDDRIFT_BATCH])))
            deltas.add(float(row[FEDDRIFT_DISTANCE]))
            schedules.add(row.get("concept_schedule", "random"))
            if row.get("total_data"):
                total_data_values.add(int(float(row["total_data"])))
        datasets[dataset_dir.name] = {
            "metrics_rows": len(rows),
            "raw_files": len(raw_paths),
            "metrics_sha256": _hash_file(metrics),
        }
        for raw_path in raw_paths:
            target = raw_path.relative_to(root).as_posix()
            record = dict(raw_files.get(target, {"source": "existing_baseline"}))
            record["target"] = target
            record["sha256"] = _hash_file(raw_path)
            raw_files[target] = record

    manifest = dict(existing_manifest)
    manifest["schema_version"] = 2
    manifest["status"] = "versioned_reference"
    manifest["selection"] = dict(existing_manifest.get("selection", {}))
    manifest["selection"].update({
        "datasets": sorted(datasets),
        "seeds": sorted(seeds),
        "total_data": sorted(total_data_values),
        "concept_schedule": sorted(schedules),
        FEDDRIFT_BATCH: sorted(batches),
        FEDDRIFT_DISTANCE: sorted(deltas),
    })
    manifest["sources"] = (
        existing_manifest.get("sources", []) + incoming_manifest.get("sources", [])
    )
    manifest["datasets"] = datasets
    manifest["raw_files"] = [raw_files[key] for key in sorted(raw_files)]
    manifest["last_extension"] = {
        "added_results": added,
        "extended_at": datetime.now().astimezone().isoformat(),
    }
    with (root / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return manifest


def extend_baseline(sources, output_root=DEFAULT_OUTPUT,
                    batches=DEFAULT_BATCHES, backup_root=None):
    """既存baselineへ追加結果を安全なディレクトリ切り替えで統合する。"""
    output_root = Path(output_root).resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"既存baselineがありません: {output_root}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging = output_root.with_name(f"{output_root.name}__staging_{timestamp}")
    incoming = output_root.with_name(f"{output_root.name}__incoming_{timestamp}")
    backup = (
        Path(backup_root).resolve() if backup_root
        else output_root.with_name(f"{output_root.name}__backup_{timestamp}")
    )
    for path in (staging, incoming, backup):
        if path.exists():
            raise FileExistsError(f"作業先が既に存在します: {path}")

    existing_manifest = json.loads(
        (output_root / "manifest.json").read_text(encoding="utf-8")
    )
    shutil.copytree(output_root, staging)
    try:
        incoming_manifest = build_baseline(
            sources=sources, output_root=incoming, batches=batches
        )
        added = _merge_baseline_trees(staging, incoming)
        manifest = _refresh_manifest(
            staging, existing_manifest, incoming_manifest, added
        )
        output_root.rename(backup)
        try:
            staging.rename(output_root)
        except Exception:
            backup.rename(output_root)
            raise
    finally:
        if incoming.exists():
            shutil.rmtree(incoming)
    return manifest, backup


def main():
    parser = argparse.ArgumentParser(
        description="既存結果からFedDrift固定ベースラインを非破壊で作成する"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-config",
        type=Path,
        nargs="+",
        default=[DEFAULT_SOURCE_CONFIG],
        help="入力実験を列挙したJSON（複数指定可）",
    )
    parser.add_argument(
        "--extend",
        action="store_true",
        help="既存outputへ追加結果を検証後に統合する",
    )
    parser.add_argument(
        "--backup",
        type=Path,
        default=None,
        help="--extend時の旧baseline退避先",
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCHES),
        help="取り込むB_detect（既定: 50 100 200 500）",
    )
    args = parser.parse_args()
    sources = load_sources(args.source_config)
    if args.extend:
        manifest, backup = extend_baseline(
            sources=sources,
            output_root=args.output,
            batches=args.batches,
            backup_root=args.backup,
        )
        print(f"旧baseline: {backup}")
    else:
        manifest = build_baseline(
            sources=sources, output_root=args.output, batches=args.batches
        )
    print(json.dumps(manifest["datasets"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
