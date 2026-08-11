"""実験manifestの事後生成と重複監査を行うCLI。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

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


def _infer_total_data(csv_paths):
    for path in csv_paths:
        match = re.search(r"(?:^|_)n(\d+)(?:_|\.)", path.name)
        if match:
            return int(match.group(1))
    raise ValueError("CSV名からtotal_dataを推定できません。--total-dataを指定してください")


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
        root = csv_path.parent.parent if csv_path.parent.name == "pareto" else csv_path.parent
        roots.add(root.resolve())
    return sorted(roots)


def backfill_manifest(result_dir, total_data=None, force=False):
    """正規CSVから由来不明の事後manifestを生成する。"""
    result_dir = Path(result_dir).resolve()
    manifest_path = result_dir / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"manifest already exists: {manifest_path}")
    csv_paths = _canonical_csv_paths(result_dir)
    if not csv_paths:
        raise FileNotFoundError(f"CSVが見つかりません: {result_dir}")
    total_data = total_data or _infer_total_data(csv_paths)

    runs_by_fingerprint = {}
    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
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


def backfill_tree(results_root, total_data=None, force=False):
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
            )
        except (FileNotFoundError, ValueError) as error:
            outcomes.append((manifest_path, "failed", str(error)))
        else:
            outcomes.append((path, "created", manifest["run_count"]))
    return outcomes


def load_execution_manifests(results_root):
    manifests = []
    for path in Path(results_root).rglob("manifest.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if value.get("kind") == MANIFEST_KIND:
            manifests.append((path, value))
    return manifests


def build_audit_report(results_root):
    """実行一覧と完全重複・由来違いをMarkdownにする。"""
    manifests = load_execution_manifests(results_root)
    configuration_groups = {}
    execution_groups = {}
    for path, manifest in manifests:
        for run in manifest.get("runs", []):
            config_key = run.get("configuration_fingerprint")
            execution_key = run.get("execution_fingerprint")
            if config_key:
                configuration_groups.setdefault(config_key, set()).add(str(path))
            if execution_key:
                execution_groups.setdefault(execution_key, set()).add(str(path))

    exact_groups = [values for values in execution_groups.values() if len(values) > 1]
    related_groups = [
        values for values in configuration_groups.values() if len(values) > 1
    ]
    lines = [
        "# 実験manifest監査",
        "",
        f"- 対象: `{Path(results_root)}`",
        f"- manifest数: {len(manifests)}",
        f"- 完全重複run群: {len(exact_groups)}",
        f"- 設定一致run群（コード・golden違いを含む）: {len(related_groups)}",
        "",
        "## 実行一覧",
        "",
        "| manifest | status | runs | modes | datasets | schedule | provenance |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for path, manifest in manifests:
        provenance = manifest.get("provenance", {}).get("fingerprint")
        if provenance is None:
            provenance = manifest.get("provenance_status", "unknown")
        configurations = [
            run.get("configuration", {}) for run in manifest.get("runs", [])
        ]
        modes = ", ".join(sorted({str(item.get("mode")) for item in configurations}))
        datasets = ", ".join(sorted({str(item.get("dataset")) for item in configurations}))
        schedules = ", ".join(sorted({str(item.get("concept_schedule")) for item in configurations}))
        lines.append(
            f"| `{path}` | {manifest.get('status', 'unknown')} | "
            f"{manifest.get('run_count', 0)} | {modes} | {datasets} | "
            f"{schedules} | `{str(provenance)[:12]}` |"
        )

    def append_groups(title, groups):
        lines.extend(["", f"## {title}", ""])
        if not groups:
            lines.append("該当なし。")
            return
        for index, paths in enumerate(sorted(groups, key=lambda item: sorted(item)), 1):
            lines.append(f"### group {index}")
            lines.append("")
            lines.extend(f"- `{path}`" for path in sorted(paths))
            lines.append("")

    append_groups("完全重複", exact_groups)
    append_groups("設定一致・由来相違または不明", related_groups)
    return "\n".join(lines).rstrip() + "\n"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    backfill = subparsers.add_parser("backfill", help="既存CSVからmanifestを事後生成")
    backfill.add_argument("result_dir")
    backfill.add_argument("--total-data", type=int, default=None)
    backfill.add_argument("--force", action="store_true")
    backfill.add_argument(
        "--recursive", action="store_true",
        help="配下の各pareto出力を独立した実験として一括補完",
    )

    audit = subparsers.add_parser("audit", help="manifest一覧と重複をMarkdown化")
    audit.add_argument("--results-root", default="results")
    audit.add_argument("--output", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "backfill":
        if args.recursive:
            outcomes = backfill_tree(
                args.result_dir, total_data=args.total_data, force=args.force,
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
        )
        print(f"Manifest saved: {path} ({manifest['run_count']} runs)")
        return

    report = build_audit_report(args.results_root)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"Audit report saved: {output}")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
