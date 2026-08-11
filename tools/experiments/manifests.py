"""既存実験の正規CSV・NPZ・ログから実行manifestを補完するCLI。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np

from federated_drift_experiment.experiment_spec.manifests import (
    MANIFEST_KIND,
    MANIFEST_SCHEMA_VERSION,
    configuration_from_result_row,
    fingerprint,
    write_json_atomic,
)
from federated_drift_experiment.experiment_spec.parameters import (
    PARAMETER_SCHEMA_VERSION,
)


def _infer_total_data(csv_paths, result_dir=None):
    for path in csv_paths:
        match = re.search(r"(?:^|_)n(\d+)(?:_|\.)", path.name)
        if match:
            return int(match.group(1))
    if result_dir is not None:
        for path in sorted(Path(result_dir).rglob("*.npz")):
            try:
                with np.load(path, allow_pickle=False) as arrays:
                    if "total_data" in arrays:
                        return int(arrays["total_data"].item())
            except (OSError, ValueError, KeyError):
                continue
    raise ValueError(
        "CSV名・NPZからtotal_dataを推定できません。--total-dataを指定してください"
    )


def _infer_concept_schedule(result_dir):
    values = set()
    for path in sorted(Path(result_dir).rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as arrays:
                if "concept_schedule" in arrays:
                    value = str(arrays["concept_schedule"].item())
                    if value:
                        values.add(value)
        except (OSError, ValueError, KeyError):
            continue
        if len(values) > 1:
            break
    if not values:
        pattern = re.compile(r"\bschedule=([A-Za-z0-9_]+)")
        for path in sorted(Path(result_dir).rglob("*.log")):
            text = path.read_text(encoding="utf-8", errors="replace")
            match = pattern.search(text)
            if match:
                values.add(match.group(1))
    return next(iter(values)) if len(values) == 1 else None


def _canonical_csv_paths(result_dir):
    """回復分析CSVなどを除き、正規run行を持つCSVだけを返す。"""
    paths = []
    for path in sorted(Path(result_dir).rglob("*.csv")):
        try:
            with path.open(newline="", encoding="utf-8-sig") as file:
                fields = set(next(csv.reader(file), ()))
        except (OSError, UnicodeDecodeError):
            continue
        required = {"parameter_schema_version", "mode", "dataset", "seed"}
        if required <= fields:
            paths.append(path)
    return paths


def discover_execution_roots(results_root):
    """正規CSVの配置から、manifestを置くべき実験variant直下を列挙する。"""
    roots = set()
    for csv_path in _canonical_csv_paths(results_root):
        if "baselines" in csv_path.parts:
            continue
        if "comparison_pareto" in csv_path.parts:
            continue
        root = csv_path.parent.parent if csv_path.parent.name == "pareto" else csv_path.parent
        roots.add(root.resolve())
    return sorted(roots)


def backfill_manifest(
    result_dir, total_data=None, force=False, concept_schedule=None,
):
    """正規CSVから由来不明の事後manifestを生成する。"""
    result_dir = Path(result_dir).resolve()
    manifest_path = result_dir / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"manifest already exists: {manifest_path}")
    csv_paths = _canonical_csv_paths(result_dir)
    if not csv_paths:
        raise FileNotFoundError(f"CSVが見つかりません: {result_dir}")
    total_data = total_data or _infer_total_data(csv_paths, result_dir)
    inferred_schedule = concept_schedule or _infer_concept_schedule(result_dir)

    runs_by_fingerprint = {}
    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            required = {
                "parameter_schema_version", "mode", "dataset", "seed",
                "sweep_parameter", "sweep_value",
            }
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"正規設定列が不足しています: {csv_path}: {sorted(missing)}"
                )
            for row in reader:
                if not row.get("concept_schedule"):
                    if inferred_schedule is None:
                        raise ValueError(
                            f"concept_scheduleをNPZ・ログから推定できません: {csv_path}"
                        )
                    row["concept_schedule"] = inferred_schedule
                if int(row.get("parameter_schema_version") or -1) != PARAMETER_SCHEMA_VERSION:
                    raise ValueError(f"旧パラメータスキーマです: {csv_path}")
                configuration = configuration_from_result_row(row, total_data)
                configuration_fingerprint = fingerprint(configuration)
                runs_by_fingerprint[configuration_fingerprint] = {
                    "configuration_fingerprint": configuration_fingerprint,
                    "execution_fingerprint": None,
                    "configuration": configuration,
                }

    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "kind": MANIFEST_KIND,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "parameter_schema_version": PARAMETER_SCHEMA_VERSION,
        "status": "completed",
        "provenance_status": "unknown_backfill",
        "started_at": None,
        "completed_at": now,
        "source_csvs": [str(path) for path in csv_paths],
        "run_count": len(runs_by_fingerprint),
        "runs": [runs_by_fingerprint[key] for key in sorted(runs_by_fingerprint)],
    }
    manifest["plan_fingerprint"] = fingerprint(sorted(runs_by_fingerprint))
    write_json_atomic(manifest_path, manifest)
    return manifest_path, manifest


def backfill_tree(
    results_root, total_data=None, force=False, concept_schedule=None,
):
    """結果木に含まれる各実験variantへ独立したmanifestを補完する。"""
    outcomes = []
    for execution_root in discover_execution_roots(results_root):
        manifest_path = execution_root / "manifest.json"
        if manifest_path.exists() and not force:
            outcomes.append((manifest_path, "skipped", None))
            continue
        try:
            path, manifest = backfill_manifest(
                execution_root, total_data=total_data, force=force,
                concept_schedule=concept_schedule,
            )
        except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
            outcomes.append((manifest_path, "failed", str(error)))
        else:
            outcomes.append((path, "created", manifest["run_count"]))
    return outcomes


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    backfill = subparsers.add_parser("backfill", help="既存CSVからmanifestを事後生成")
    backfill.add_argument("result_dir")
    backfill.add_argument("--total-data", type=int, default=None)
    backfill.add_argument("--force", action="store_true")
    backfill.add_argument(
        "--concept-schedule", choices=("random", "feddrift_fixed"), default=None,
        help="旧CSV・NPZ・ログから推定不能な場合の明示値",
    )
    backfill.add_argument(
        "--recursive", action="store_true",
        help="配下の各pareto出力を独立した実験として一括補完",
    )

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "backfill":
        if args.recursive:
            outcomes = backfill_tree(
                args.result_dir, total_data=args.total_data, force=args.force,
                concept_schedule=args.concept_schedule,
            )
            for path, status, detail in outcomes:
                suffix = "" if detail is None else f" ({detail})"
                print(f"{status}: {path}{suffix}")
            failed = sum(status == "failed" for _, status, _ in outcomes)
            print(
                f"Backfill: created={sum(s == 'created' for _, s, _ in outcomes)}, "
                f"skipped={sum(s == 'skipped' for _, s, _ in outcomes)}, "
                f"failed={failed}"
            )
            if failed:
                raise SystemExit(1)
            return
        path, manifest = backfill_manifest(
            args.result_dir, total_data=args.total_data, force=args.force,
            concept_schedule=args.concept_schedule,
        )
        print(f"Manifest saved: {path} ({manifest['run_count']} runs)")
        return

if __name__ == "__main__":
    main()
