"""results全体の手法名・データセット名を非破壊で正規化する。"""

import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from federated_drift_experiment.data import (
    normalize_dataset_in_text,
    normalize_dataset_name,
)
from federated_drift_experiment.mode_names import (
    LEGACY_MODE_NAMES,
    normalize_legacy_mode,
)


TEXT_SUFFIXES = {".md", ".log", ".txt"}
DERIVED_SUFFIXES = {".png"}
UCB_MARKERS = ("_ucb", "ucb_")


def _io_path(path):
    """Windowsの長い絶対パスをI/O APIへ安全に渡す。"""
    resolved = str(Path(path).resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return f"\\\\?\\{resolved}"
    return resolved


def _is_ucb(value):
    text = str(value).lower()
    return any(marker in text for marker in UCB_MARKERS) or text.endswith("ucb")


def normalize_method_in_text(value):
    """文章・ファイル名中の旧手法名だけを正規名へ変換する。"""
    normalized = str(value)
    for old_mode, new_mode in sorted(
        LEGACY_MODE_NAMES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if old_mode == "FedSDA":
            # 現行名（FedSDA_Cached_*等）の接頭辞は旧v1名とみなさない。
            pattern = rf"(?<![0-9A-Za-z_]){re.escape(old_mode)}(?![0-9A-Za-z_])"
            normalized = re.sub(pattern, new_mode, normalized)
        else:
            # バージョン付き旧名は、ファイル名の区切り「_」の前でも置換する。
            normalized = normalized.replace(old_mode, new_mode)
    return normalized


def normalize_names_in_text(value):
    """パラメータ表記には触れず、手法名・データセット名だけを変換する。"""
    return normalize_dataset_in_text(normalize_method_in_text(value))


def _scalar_text(value):
    array = np.asarray(value)
    if array.ndim != 0:
        return None
    scalar = array.item()
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    return str(scalar)


def _target_relative_path(source, source_root):
    relative = source.relative_to(source_root)
    parts = [normalize_names_in_text(part) for part in relative.parts]
    return Path(*parts)


def _ensure_target(target, source):
    if target.exists():
        raise FileExistsError(
            f"正規化後の出力先が衝突しました: {target} (入力: {source})"
        )
    target.parent.mkdir(parents=True, exist_ok=True)


def _migrate_csv(source, target):
    with source.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"CSVヘッダーがありません: {source}")
        fieldnames = list(reader.fieldnames)
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
                row[key] = normalize_names_in_text(row[key])
        migrated.append(row)

    _ensure_target(target, source)
    with open(_io_path(target), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(migrated)
    return len(rows), len(migrated), removed


def _migrate_npz(source, target):
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}

    old_mode = _scalar_text(arrays.get("mode", "")) or ""
    label = _scalar_text(arrays.get("label", "")) or ""
    if _is_ucb(old_mode) or _is_ucb(label) or _is_ucb(source.name):
        return False

    if "mode" in arrays:
        arrays["mode"] = np.asarray(normalize_legacy_mode(old_mode))
    if "dataset" in arrays:
        arrays["dataset"] = np.asarray(normalize_dataset_name(
            _scalar_text(arrays["dataset"])
        ))
    if "label" in arrays:
        arrays["label"] = np.asarray(normalize_names_in_text(label))

    _ensure_target(target, source)
    np.savez_compressed(_io_path(target), **arrays)
    return True


def _migrate_text(source, target):
    text = source.read_text(encoding="utf-8-sig")
    _ensure_target(target, source)
    with open(_io_path(target), "w", encoding="utf-8") as file:
        file.write(normalize_names_in_text(text))


def _normalize_json_names(value):
    """JSONの構造と数値を保ち、文字列および文字列キーの名称だけを変換する。"""
    if isinstance(value, str):
        return normalize_names_in_text(value)
    if isinstance(value, list):
        return [_normalize_json_names(item) for item in value]
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            new_key = normalize_names_in_text(key)
            if new_key in normalized:
                raise ValueError(f"JSONキーの正規化後に衝突しました: {new_key}")
            normalized[new_key] = _normalize_json_names(item)
        return normalized
    return value


def _migrate_json(source, target):
    """JSON内の手法名・データセット名だけを変換する。"""
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    _ensure_target(target, source)
    with open(_io_path(target), "w", encoding="utf-8") as file:
        json.dump(_normalize_json_names(data), file, ensure_ascii=False, indent=2)
        file.write("\n")


def migrate_results_tree(source_root, staging_root):
    """構造化結果をstagingへ移行し、元resultsは変更しない。"""
    source_root = Path(source_root).resolve()
    staging_root = Path(staging_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"入力resultsがありません: {source_root}")
    if staging_root.exists():
        raise FileExistsError(f"stagingは空の新規パスを指定してください: {staging_root}")
    if source_root == staging_root or source_root in staging_root.parents:
        raise ValueError("stagingは入力resultsの外側に作成してください")
    staging_root.mkdir(parents=True)

    manifest = {
        "schema_version": 1,
        "migration_scope": "method_and_dataset_names_only",
        "source": str(source_root),
        "staging": str(staging_root),
        "csv_files": 0,
        "csv_rows_source": 0,
        "csv_rows_kept": 0,
        "ucb_rows_removed": 0,
        "npz_files_source": 0,
        "npz_files_kept": 0,
        "ucb_npz_files_removed": 0,
        "text_files_migrated": 0,
        "json_files_migrated": 0,
        "derived_files_not_copied": [],
    }
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        relative = _target_relative_path(source, source_root)
        target = staging_root / relative
        suffix = source.suffix.lower()
        if suffix == ".csv":
            source_rows, kept, removed = _migrate_csv(source, target)
            manifest["csv_files"] += 1
            manifest["csv_rows_source"] += source_rows
            manifest["csv_rows_kept"] += kept
            manifest["ucb_rows_removed"] += removed
        elif suffix == ".npz":
            manifest["npz_files_source"] += 1
            if _migrate_npz(source, target):
                manifest["npz_files_kept"] += 1
            else:
                manifest["ucb_npz_files_removed"] += 1
        elif suffix in TEXT_SUFFIXES:
            _migrate_text(source, target)
            manifest["text_files_migrated"] += 1
        elif suffix == ".json":
            _migrate_json(source, target)
            manifest["json_files_migrated"] += 1
        elif suffix in DERIVED_SUFFIXES:
            manifest["derived_files_not_copied"].append(normalize_names_in_text(
                source.relative_to(source_root).as_posix()
            ))
        else:
            manifest["derived_files_not_copied"].append(normalize_names_in_text(
                source.relative_to(source_root).as_posix()
            ))

    validate_migration(source_root, staging_root, manifest)
    manifest_path = staging_root / "migration_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _contains_legacy_method(value):
    text = str(value)
    versioned_names = (
        old_mode for old_mode in LEGACY_MODE_NAMES if old_mode != "FedSDA"
    )
    if any(old_mode in text for old_mode in versioned_names):
        return True
    return re.search(r"(?<![0-9A-Za-z_])FedSDA(?![0-9A-Za-z_])", text) is not None


def _contains_legacy_dataset_token(value):
    text = str(value)
    return any(
        re.search(rf"(?<![0-9A-Za-z]){re.escape(old)}(?![0-9A-Za-z])", text)
        for old in ("sea", "circle", "sine")
    )


def validate_migration(source_root, staging_root, manifest):
    """件数・必須配列・旧名残存を検査する。"""
    if manifest["csv_rows_source"] != (
        manifest["csv_rows_kept"] + manifest["ucb_rows_removed"]
    ):
        raise AssertionError("CSV行数の保存則を満たしていません")
    if manifest["npz_files_source"] != (
        manifest["npz_files_kept"] + manifest["ucb_npz_files_removed"]
    ):
        raise AssertionError("NPZ件数の保存則を満たしていません")

    csv_files = list(staging_root.rglob("*.csv"))
    npz_files = list(staging_root.rglob("*.npz"))
    if len(csv_files) != manifest["csv_files"]:
        raise AssertionError("移行後CSVファイル数が一致しません")
    if len(npz_files) != manifest["npz_files_kept"]:
        raise AssertionError("移行後NPZファイル数が一致しません")

    # 名称以外のCSV値とNPZ配列がビット単位で保存されていることを確認する。
    name_fields = {"mode", "dataset", "series", "label"}
    for source in source_root.rglob("*.csv"):
        target = staging_root / _target_relative_path(source, source_root)
        with source.open("r", newline="", encoding="utf-8-sig") as file:
            source_rows = [
                row for row in csv.DictReader(file)
                if not _is_ucb(row.get("mode", ""))
                and not _is_ucb(row.get("series", ""))
            ]
        with open(_io_path(target), "r", newline="", encoding="utf-8-sig") as file:
            target_rows = list(csv.DictReader(file))
        if len(source_rows) != len(target_rows):
            raise AssertionError(f"CSV行数が一致しません: {source}")
        for old_row, new_row in zip(source_rows, target_rows):
            for key in old_row.keys() - name_fields:
                if old_row[key] != new_row[key]:
                    raise AssertionError(
                        f"CSVの名称以外の値が変化しました: {source} ({key})"
                    )

    for source in source_root.rglob("*.npz"):
        with np.load(source, allow_pickle=False) as old_archive:
            old_mode = _scalar_text(old_archive["mode"]) if "mode" in old_archive else ""
            old_label = _scalar_text(old_archive["label"]) if "label" in old_archive else ""
            if _is_ucb(old_mode or "") or _is_ucb(old_label or "") or _is_ucb(source.name):
                continue
            target = staging_root / _target_relative_path(source, source_root)
            with np.load(_io_path(target), allow_pickle=False) as new_archive:
                if old_archive.files != new_archive.files:
                    raise AssertionError(f"NPZキーが一致しません: {source}")
                for key in old_archive.files:
                    if key in name_fields:
                        continue
                    old_array = old_archive[key]
                    new_array = new_archive[key]
                    if (
                        old_array.dtype != new_array.dtype
                        or old_array.shape != new_array.shape
                        or old_array.tobytes() != new_array.tobytes()
                    ):
                        raise AssertionError(
                            f"NPZの名称以外の配列が変化しました: {source} ({key})"
                        )

    for path in csv_files:
        if _contains_legacy_method(path) or _contains_legacy_dataset_token(path):
            raise AssertionError(f"CSVファイル名に旧名が残っています: {path}")
        with open(_io_path(path), "r", newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                for key in ("mode", "dataset", "series", "label"):
                    value = row.get(key, "")
                    if _contains_legacy_method(value):
                        raise AssertionError(f"{path}: {key}に旧手法名が残っています")
                    if key == "dataset" and _contains_legacy_dataset_token(value):
                        raise AssertionError(f"{path}: datasetに旧名が残っています")

    for path in npz_files:
        if _contains_legacy_method(path) or _contains_legacy_dataset_token(path):
            raise AssertionError(f"NPZファイル名に旧名が残っています: {path}")
        with np.load(_io_path(path), allow_pickle=False) as archive:
            for key in ("mode", "dataset", "label"):
                if key not in archive.files:
                    continue
                value = _scalar_text(archive[key]) or ""
                if _contains_legacy_method(value):
                    raise AssertionError(f"{path}: {key}に旧手法名が残っています")
                if key == "dataset" and _contains_legacy_dataset_token(value):
                    raise AssertionError(f"{path}: datasetに旧名が残っています")

    for path in staging_root.rglob("*"):
        if (
            not path.is_file()
            or path.name == "migration_manifest.json"
            or path.suffix.lower() not in TEXT_SUFFIXES | {".json"}
        ):
            continue
        with open(_io_path(path), encoding="utf-8-sig") as file:
            text = file.read()
        if _contains_legacy_method(text):
            raise AssertionError(f"テキストに旧手法名が残っています: {path}")
        if _contains_legacy_dataset_token(text):
            raise AssertionError(f"テキストに旧データセット名が残っています: {path}")


def activate_migration(source_root, staging_root, backup_root):
    """検証済みstagingをresultsへ切り替え、元resultsをバックアップする。"""
    source_root = Path(source_root).resolve()
    staging_root = Path(staging_root).resolve()
    backup_root = Path(backup_root).resolve()
    if not source_root.is_dir() or not staging_root.is_dir():
        raise FileNotFoundError("切り替え対象のresultsまたはstagingがありません")
    if backup_root.exists():
        raise FileExistsError(f"バックアップ先が既に存在します: {backup_root}")
    if not (staging_root / "migration_manifest.json").is_file():
        raise ValueError("検証済みmigration_manifest.jsonがありません")

    source_root.rename(backup_root)
    try:
        staging_root.rename(source_root)
    except Exception:
        backup_root.rename(source_root)
        raise
    manifest_path = source_root / "migration_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["activated_at"] = datetime.now().astimezone().isoformat()
    manifest["backup"] = str(backup_root)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return backup_root


def main():
    parser = argparse.ArgumentParser(
        description="results全体の手法名・データセット名を安全に正規化する"
    )
    parser.add_argument("--source", type=Path, default=Path("results"))
    parser.add_argument(
        "--staging", type=Path, default=Path("results__name_migration_staging")
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup", type=Path, default=None)
    args = parser.parse_args()

    manifest_path = args.staging / "migration_manifest.json"
    if args.apply and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_migration(args.source, args.staging, manifest)
    else:
        manifest = migrate_results_tree(args.source, args.staging)
    print(json.dumps({
        key: value for key, value in manifest.items()
        if key != "derived_files_not_copied"
    }, ensure_ascii=False, indent=2))
    print(f"派生成果物の非コピー件数: {len(manifest['derived_files_not_copied'])}")
    if args.apply:
        backup = args.backup or Path(
            f"results_legacy_names_{datetime.now():%Y%m%d_%H%M%S}"
        )
        activated = activate_migration(args.source, args.staging, backup)
        print(f"移行完了。旧results: {activated}")
    else:
        print("stagingの検証まで完了しました。切り替えには--applyを指定してください。")


if __name__ == "__main__":
    main()
