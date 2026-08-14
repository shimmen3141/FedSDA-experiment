"""実験計画・実装由来・出力状態を記録するmanifest。"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

from .parameters import PARAMETER_SCHEMA_VERSION


MANIFEST_SCHEMA_VERSION = 1
MANIFEST_KIND = "experiment_execution"


def _json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def fingerprint(value):
    """JSONとして同値な値へ安定したSHA-256を与える。"""
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def file_sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256(repo_root):
    """結果へ影響する実験コードをGit状態に依存せずハッシュ化する。"""
    repo_root = Path(repo_root)
    paths = [repo_root / "run_pareto_sweep.py", repo_root / "experiment_runtime.py"]
    paths.extend(sorted(
        (repo_root / "federated_drift_experiment").rglob("*.py")
    ))
    digest = hashlib.sha256()
    for path in sorted({path.resolve() for path in paths if path.is_file()}):
        relative = path.relative_to(repo_root.resolve()).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_head(repo_root):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_provenance(repo_root=None, golden_path=None):
    """コード・golden・数値環境をまとめ、重複判定用の由来を返す。"""
    repo_root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    golden_path = Path(
        golden_path or repo_root / "tests" / "regression_golden.json"
    )
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _package_version("numpy"),
        "torch": _package_version("torch"),
    }
    provenance = {
        "git_commit": _git_head(repo_root),
        "implementation_sha256": _source_tree_sha256(repo_root),
        "regression_golden_sha256": (
            file_sha256(golden_path) if golden_path.is_file() else None
        ),
        "environment": environment,
    }
    provenance["fingerprint"] = fingerprint(provenance)
    return provenance


def experiment_configuration(experiment, total_data):
    """一runの意味的設定を順序に依存しない辞書へ変換する。"""
    parameters = {
        assignment.parameter_id: assignment.value
        for assignment in experiment.parameters
    }
    algorithm = asdict(experiment.algorithm)
    if "SharedBackbone" not in experiment.mode and "ResidualAdapter" not in experiment.mode:
        algorithm.pop("shared_backbone_training", None)
        algorithm.pop("shared_backbone_gradient_strategy", None)
        algorithm.pop("shared_backbone_routing_recalibration", None)
    if "SoftRouting" not in experiment.mode:
        algorithm.pop("soft_routing_context", None)
        algorithm.pop("soft_routing_meta_loss", None)
    else:
        if algorithm.get("soft_routing_context") not in {
            "predicted_class", "meta_predicted_class",
        }:
            algorithm.pop("soft_routing_meta_loss", None)
        if (
            algorithm.get("shared_backbone_training") != "joint"
            or algorithm.get("shared_backbone_gradient_strategy") == "mean"
        ):
            algorithm.pop("shared_backbone_gradient_strategy", None)
    if "ResidualAdapter" not in experiment.mode:
        algorithm.pop("shared_adapter_rank", None)
    return {
        "mode": experiment.mode,
        "dataset": experiment.dataset,
        "seed": experiment.seed,
        "concept_schedule": experiment.concept_schedule,
        "total_data": total_data,
        "sweep_parameter": experiment.sweep_parameter,
        "sweep_value": experiment.sweep_value,
        "parameters": dict(sorted(parameters.items())),
        "algorithm": algorithm,
    }


def _optional_number(value, number_type=float):
    if value in (None, "", "None", "nan"):
        return None
    if number_type is int:
        return int(float(value))
    return number_type(value)


def _optional_bool(value):
    if value in (None, "", "None"):
        return None
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes"}


def configuration_from_result_row(row, total_data):
    """正規CSVの非指標列から、実行時と同じ設定表現を復元する。"""
    mode = row["mode"]
    parameter_types = {
        "adwin_delta": float,
        "aggregation_interval": int,
        "fedsda_distance_threshold": float,
        "feddrift_detection_batch_size": int,
        "feddrift_distance_threshold": float,
    }
    parameters = {}
    for parameter_id, number_type in parameter_types.items():
        value = _optional_number(row.get(parameter_id), number_type)
        if value is not None:
            parameters[parameter_id] = value

    algorithm = {
        "clustering_policy": row.get("clustering_policy"),
        "clustering_decision": row.get("clustering_decision"),
        "detection_episodes": _optional_bool(row.get("detection_episodes")),
        "new_model_creation_policy": row.get("new_model_creation_policy"),
        "fifo_size": _optional_number(row.get("fifo_size"), int),
        "new_model_validation_fraction": _optional_number(
            row.get("new_model_validation_fraction"), float,
        ),
        "new_model_forward_validation_samples": _optional_number(
            row.get("new_model_forward_validation_samples"), int,
        ),
    }
    if "SoftRouting" in mode:
        algorithm["soft_routing_context"] = (
            row.get("soft_routing_context") or "global"
        )
        if algorithm["soft_routing_context"] in {
            "predicted_class", "meta_predicted_class",
        }:
            algorithm["soft_routing_meta_loss"] = (
                # 列追加前のMeta実験はbounded_scoreで実行されている。
                row.get("soft_routing_meta_loss") or "bounded_score"
            )
    if "SharedBackbone" in mode or "ResidualAdapter" in mode:
        algorithm.update({
            "shared_backbone_training": row.get("shared_backbone_training"),
            "shared_backbone_routing_recalibration": row.get(
                "shared_backbone_routing_recalibration"
            ),
        })
        gradient_strategy = row.get("shared_backbone_gradient_strategy")
        if (
            algorithm["shared_backbone_training"] == "joint"
            and gradient_strategy not in (None, "", "mean")
        ):
            algorithm["shared_backbone_gradient_strategy"] = gradient_strategy
    if "ResidualAdapter" in mode:
        algorithm["shared_adapter_rank"] = _optional_number(
            row.get("shared_adapter_rank"), int,
        )

    sweep_parameter = row.get("sweep_parameter") or None
    sweep_type = parameter_types.get(sweep_parameter, float)
    return {
        "mode": mode,
        "dataset": row["dataset"],
        "seed": int(row["seed"]),
        "concept_schedule": row["concept_schedule"],
        "total_data": int(total_data),
        "sweep_parameter": sweep_parameter,
        "sweep_value": _optional_number(row.get("sweep_value"), sweep_type),
        "parameters": dict(sorted(parameters.items())),
        "algorithm": algorithm,
    }


def build_run_records(plan, total_data, provenance):
    records = []
    for experiment in plan.iter_experiments():
        configuration = experiment_configuration(experiment, total_data)
        configuration_fingerprint = fingerprint(configuration)
        records.append({
            "configuration_fingerprint": configuration_fingerprint,
            "execution_fingerprint": fingerprint({
                "configuration": configuration_fingerprint,
                "provenance": provenance["fingerprint"],
            }),
            "configuration": configuration,
        })
    return records


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path, value):
    """途中終了で壊れたJSONを残さないよう同一ディレクトリ内で置換する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, path)


def find_overlaps(candidate_manifest, results_root, exclude_path=None):
    """完了済みmanifestとrun単位で完全一致・旧由来一致を照合する。"""
    results_root = Path(results_root)
    exclude = None if exclude_path is None else Path(exclude_path).resolve()
    candidate_by_configuration = {
        run["configuration_fingerprint"]: run for run in candidate_manifest["runs"]
    }
    exact = {}
    stale = {}
    if not results_root.exists():
        return {"exact": [], "different_provenance": []}

    for path in results_root.rglob("manifest.json"):
        if exclude is not None and path.resolve() == exclude:
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if (
            manifest.get("kind") != MANIFEST_KIND
            or manifest.get("status") != "completed"
        ):
            continue
        exact_count = 0
        stale_count = 0
        for previous in manifest.get("runs", []):
            current = candidate_by_configuration.get(
                previous.get("configuration_fingerprint")
            )
            if current is None:
                continue
            if previous.get("execution_fingerprint") == current["execution_fingerprint"]:
                exact_count += 1
            else:
                stale_count += 1
        if exact_count:
            exact[str(path)] = exact_count
        if stale_count:
            stale[str(path)] = stale_count

    def records(values):
        return [
            {"manifest": path, "overlapping_runs": count}
            for path, count in sorted(values.items())
        ]

    return {
        "exact": records(exact),
        "different_provenance": records(stale),
    }


def overlap_run_count(overlaps, kind):
    return sum(item["overlapping_runs"] for item in overlaps.get(kind, []))


def infer_execution_root(out_dir, raw_dir=None):
    """pareto/rawの共通親を実験variantの出力ルートとする。"""
    paths = [str(Path(out_dir).resolve())]
    if raw_dir:
        paths.append(str(Path(raw_dir).resolve()))
    common = Path(os.path.commonpath(paths))
    if len(paths) == 1:
        common = common.parent
    return common


def preview_overlaps(
    *, plan, total_data, results_root="results", repo_root=None,
):
    """実験を開始せず、計画内の各runと既存manifestの重複を調べる。"""
    repo_root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    provenance = build_provenance(repo_root)
    candidate = {
        "kind": MANIFEST_KIND,
        "runs": build_run_records(plan, total_data, provenance),
    }
    return find_overlaps(candidate, results_root)


class ExperimentManifestSession:
    """実験開始・成功・失敗の状態遷移を一つのmanifestへ保存する。"""

    def __init__(self, path, manifest):
        self.path = Path(path)
        self.manifest = manifest

    @classmethod
    def start(
        cls, *, plan, total_data, argv, out_dir, raw_dir, tag,
        results_root="results", repo_root=None,
    ):
        repo_root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
        execution_root = infer_execution_root(out_dir, raw_dir)
        path = execution_root / "manifest.json"
        provenance = build_provenance(repo_root)
        runs = build_run_records(plan, total_data, provenance)
        manifest = {
            "kind": MANIFEST_KIND,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "parameter_schema_version": PARAMETER_SCHEMA_VERSION,
            "status": "running",
            "tag": tag,
            "started_at": _utc_now(),
            "completed_at": None,
            "command_arguments": list(argv),
            "outputs": {
                "root": str(execution_root),
                "pareto_dir": str(Path(out_dir).resolve()),
                "raw_dir": str(Path(raw_dir).resolve()) if raw_dir else None,
            },
            "provenance": provenance,
            "plan_fingerprint": fingerprint(sorted(
                run["configuration_fingerprint"] for run in runs
            )),
            "run_count": len(runs),
            "runs": runs,
        }
        manifest["overlaps"] = find_overlaps(
            manifest, results_root, exclude_path=path,
        )
        write_json_atomic(path, manifest)
        return cls(path, manifest)

    def complete(self, csv_path=None, raw_dir=None):
        self.manifest["status"] = "completed"
        self.manifest["completed_at"] = _utc_now()
        if csv_path and Path(csv_path).is_file():
            self.manifest["outputs"]["metrics_csv"] = str(Path(csv_path).resolve())
            self.manifest["outputs"]["metrics_csv_sha256"] = file_sha256(csv_path)
        if raw_dir and Path(raw_dir).is_dir():
            self.manifest["outputs"]["raw_file_count"] = sum(
                1 for _ in Path(raw_dir).glob("*.npz")
            )
        write_json_atomic(self.path, self.manifest)

    def fail(self, error):
        self.manifest["status"] = "failed"
        self.manifest["completed_at"] = _utc_now()
        self.manifest["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        write_json_atomic(self.path, self.manifest)


def format_overlap_summary(overlaps):
    exact = overlap_run_count(overlaps, "exact")
    stale = overlap_run_count(overlaps, "different_provenance")
    lines = [(
        f"既存実験照合: 完全一致={exact} runs, "
        f"設定一致・由来相違={stale} runs"
    )]
    for label, kind in (
        ("完全一致", "exact"),
        ("設定一致・由来相違", "different_provenance"),
    ):
        for item in overlaps.get(kind, []):
            lines.append(
                f"  - {label}: {item['overlapping_runs']} runs: "
                f"{item['manifest']}"
            )
    return "\n".join(lines)
