"""過去の実験結果を現在のモード名へ非破壊で移行する。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from federated_drift_experiment.data import (
    normalize_dataset_in_text,
    normalize_dataset_name,
)
from federated_drift_experiment.mode_names import (
    LEGACY_MODE_NAMES,
    normalize_legacy_mode,
    normalize_legacy_series,
    normalize_series_notation,
)


UCB_MARKERS = ("_ucb", "ucb_")


def _is_ucb(value):
    text = str(value).lower()
    return any(marker in text for marker in UCB_MARKERS) or text.endswith("ucb")


def _normalized_text(value, old_mode):
    if value is None:
        return value
    renamed = normalize_legacy_series(
        str(value), old_mode, normalize_legacy_mode(old_mode)
    )
    return normalize_series_notation(renamed)


def _normalized_name(name):
    for old_mode, new_mode in sorted(
        LEGACY_MODE_NAMES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if old_mode in name:
            name = name.replace(old_mode, new_mode, 1)
            break
    return normalize_dataset_in_text(normalize_series_notation(name))


def _target_path(source_file, source_root, output_root):
    relative = source_file.relative_to(source_root)
    return output_root / relative.parent / _normalized_name(relative.name)


def _ensure_new_target(target, source):
    if target.exists():
        raise FileExistsError(
            f"移行先が重複しています: {target}（入力: {source}）"
        )
    target.parent.mkdir(parents=True, exist_ok=True)


def migrate_csv(source, target):
    with source.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"CSVヘッダーがありません: {source}")
        rows = list(reader)

    migrated = []
    removed = 0
    for row in rows:
        old_mode = row.get("mode", "")
        if _is_ucb(old_mode) or _is_ucb(row.get("series", "")):
            removed += 1
            continue
        if "mode" in row:
            row["mode"] = normalize_legacy_mode(old_mode)
        if "dataset" in row:
            row["dataset"] = normalize_dataset_name(row["dataset"])
        for key in ("series", "label"):
            if key in row:
                row[key] = _normalized_text(row[key], old_mode)
        migrated.append(row)

    _ensure_new_target(target, source)
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(migrated)
    return len(migrated), removed


def _scalar_text(value):
    array = np.asarray(value)
    if array.ndim != 0:
        return None
    scalar = array.item()
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    return str(scalar)


def migrate_npz(source, target):
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}

    old_mode = _scalar_text(arrays.get("mode", "")) or ""
    label = _scalar_text(arrays.get("label", "")) or ""
    if _is_ucb(old_mode) or _is_ucb(label) or _is_ucb(source.name):
        return False

    if "mode" in arrays:
        arrays["mode"] = np.asarray(normalize_legacy_mode(old_mode))
    if "dataset" in arrays:
        dataset = _scalar_text(arrays["dataset"])
        arrays["dataset"] = np.asarray(normalize_dataset_name(dataset))
    if "label" in arrays:
        arrays["label"] = np.asarray(_normalized_text(label, old_mode))

    _ensure_new_target(target, source)
    np.savez_compressed(target, **arrays)
    return True


def migrate_results(source_root, output_root):
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("出力先は入力ディレクトリの外側に指定してください")
    output_root.mkdir(parents=True, exist_ok=False)

    manifest = {
        "source": str(source_root),
        "output": str(output_root),
        "csv_rows_kept": 0,
        "ucb_rows_removed": 0,
        "npz_files_kept": 0,
        "ucb_npz_files_removed": 0,
        "derived_files_not_copied": [],
    }
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        target = _target_path(source, source_root, output_root)
        if source.suffix.lower() == ".csv":
            kept, removed = migrate_csv(source, target)
            manifest["csv_rows_kept"] += kept
            manifest["ucb_rows_removed"] += removed
        elif source.suffix.lower() == ".npz":
            if migrate_npz(source, target):
                manifest["npz_files_kept"] += 1
            else:
                manifest["ucb_npz_files_removed"] += 1
        else:
            manifest["derived_files_not_copied"].append(
                str(source.relative_to(source_root))
            )

    manifest_path = output_root / "migration_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="過去のCSV/NPZからUCB結果を除外し、モード名・凡例表記を非破壊で正規化する"
    )
    parser.add_argument("source", help="移行元のresultsディレクトリ")
    parser.add_argument("--output", required=True, help="新規作成する移行先ディレクトリ")
    args = parser.parse_args()
    manifest = migrate_results(args.source, args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
