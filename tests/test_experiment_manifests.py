import csv
from dataclasses import replace
import json
from pathlib import Path

from federated_drift_experiment.experiment_spec.configuration import (
    AlgorithmOptions,
    ExperimentConfiguration,
    ParameterAssignment,
)
from federated_drift_experiment.experiment_spec.manifests import (
    ExperimentManifestSession,
    build_provenance,
    configuration_from_result_row,
    experiment_configuration,
    find_overlaps,
    fingerprint,
    format_overlap_summary,
    preview_overlaps,
    write_json_atomic,
)
from tools.baselines.build_fedsda_study import build_study_manifest
from tools.experiments.manifests import (
    backfill_manifest,
    backfill_tree,
    discover_execution_roots,
)


def _algorithm():
    return AlgorithmOptions(
        clustering_policy="on_new_model",
        clustering_decision="distance",
        detection_episodes=False,
        new_model_creation_policy="forward_persistent",
        fifo_size=30,
        new_model_validation_fraction=0.2,
        new_model_forward_validation_samples=10,
        shared_backbone_training="joint",
        shared_backbone_routing_recalibration="fifo_replay",
        shared_adapter_rank=8,
    )


def _experiment(seed=0):
    return ExperimentConfiguration(
        mode="FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting",
        dataset="circle2", seed=seed, concept_schedule="random",
        series="test", sweep_parameter="aggregation_interval", sweep_value=50,
        parameters=(
            ParameterAssignment("aggregation_interval", 50),
            ParameterAssignment("fedsda_distance_threshold", 0.1),
        ),
        algorithm=_algorithm(),
    )


class _Plan:
    def __init__(self, experiments):
        self.experiments = experiments

    def iter_experiments(self):
        return iter(self.experiments)


def test_csv_backfill_configuration_matches_resolved_experiment():
    row = {
        "mode": _experiment().mode,
        "dataset": "circle2", "seed": "0", "concept_schedule": "random",
        "sweep_parameter": "aggregation_interval", "sweep_value": "50",
        "aggregation_interval": "50", "fedsda_distance_threshold": "0.1",
        "feddrift_detection_batch_size": "", "feddrift_distance_threshold": "",
        "adwin_delta": "", "clustering_policy": "on_new_model",
        "clustering_decision": "distance", "detection_episodes": "False",
        "new_model_creation_policy": "forward_persistent", "fifo_size": "30",
        "new_model_validation_fraction": "0.2",
        "new_model_forward_validation_samples": "10",
        "shared_backbone_training": "joint",
        "shared_backbone_routing_recalibration": "fifo_replay",
        "shared_adapter_rank": "8",
    }

    assert configuration_from_result_row(row, 5000) == experiment_configuration(
        _experiment(), 5000,
    )


def test_default_mean_gradient_strategy_keeps_legacy_configuration_identity():
    default_configuration = experiment_configuration(_experiment(), 5000)
    pcgrad_experiment = replace(
        _experiment(),
        algorithm=replace(
            _experiment().algorithm,
            shared_backbone_gradient_strategy="pcgrad",
        ),
    )

    assert "shared_backbone_gradient_strategy" not in (
        default_configuration["algorithm"]
    )
    assert experiment_configuration(pcgrad_experiment, 5000)["algorithm"][
        "shared_backbone_gradient_strategy"
    ] == "pcgrad"


def test_provenance_changes_when_regression_golden_changes(tmp_path):
    (tmp_path / "federated_drift_experiment").mkdir()
    (tmp_path / "federated_drift_experiment" / "a.py").write_text("x = 1\n")
    (tmp_path / "run_pareto_sweep.py").write_text("pass\n")
    (tmp_path / "experiment_runtime.py").write_text("pass\n")
    golden = tmp_path / "golden.json"
    golden.write_text('{"value": 1}\n')
    first = build_provenance(tmp_path, golden)
    golden.write_text('{"value": 2}\n')
    second = build_provenance(tmp_path, golden)

    assert first["implementation_sha256"] == second["implementation_sha256"]
    assert first["regression_golden_sha256"] != second["regression_golden_sha256"]
    assert first["fingerprint"] != second["fingerprint"]


def test_overlap_distinguishes_exact_and_different_provenance(tmp_path):
    config_key = fingerprint({"configuration": 1})
    candidate = {"runs": [{
        "configuration_fingerprint": config_key,
        "execution_fingerprint": "current",
    }]}
    for name, execution in (("exact", "current"), ("stale", "old")):
        write_json_atomic(tmp_path / name / "manifest.json", {
            "kind": "experiment_execution", "status": "completed",
            "runs": [{
                "configuration_fingerprint": config_key,
                "execution_fingerprint": execution,
            }],
        })

    overlaps = find_overlaps(candidate, tmp_path)

    assert overlaps["exact"][0]["overlapping_runs"] == 1
    assert overlaps["different_provenance"][0]["overlapping_runs"] == 1
    summary = format_overlap_summary(overlaps)
    assert "exact\\manifest.json" in summary or "exact/manifest.json" in summary
    assert "1 runs" in summary


def test_preview_overlaps_does_not_create_manifest(tmp_path):
    repo = tmp_path / "repo"
    package = repo / "federated_drift_experiment"
    package.mkdir(parents=True)
    (package / "a.py").write_text("x = 1\n")
    (repo / "run_pareto_sweep.py").write_text("pass\n")
    (repo / "experiment_runtime.py").write_text("pass\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "regression_golden.json").write_text("{}\n")

    overlaps = preview_overlaps(
        plan=_Plan([_experiment()]), total_data=5000,
        results_root=tmp_path / "results", repo_root=repo,
    )

    assert overlaps == {"exact": [], "different_provenance": []}
    assert not list(tmp_path.rglob("manifest.json"))


def test_manifest_session_records_lifecycle_and_outputs(tmp_path):
    repo = tmp_path / "repo"
    package = repo / "federated_drift_experiment"
    package.mkdir(parents=True)
    (package / "a.py").write_text("x = 1\n")
    (repo / "run_pareto_sweep.py").write_text("pass\n")
    (repo / "experiment_runtime.py").write_text("pass\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "regression_golden.json").write_text("{}\n")
    out = tmp_path / "run" / "pareto"
    raw = tmp_path / "run" / "raw"
    out.mkdir(parents=True)
    raw.mkdir()

    session = ExperimentManifestSession.start(
        plan=_Plan([_experiment()]), total_data=5000, argv=["--quick"],
        out_dir=out, raw_dir=raw, tag="test", results_root=tmp_path / "results",
        repo_root=repo,
    )
    assert json.loads(session.path.read_text(encoding="utf-8"))["status"] == "running"
    metrics = out / "metrics.csv"
    metrics.write_text("x\n1\n")
    (raw / "one.npz").write_bytes(b"raw")
    session.complete(metrics, raw)
    saved = json.loads(session.path.read_text(encoding="utf-8"))

    assert saved["status"] == "completed"
    assert saved["outputs"]["raw_file_count"] == 1
    assert saved["outputs"]["metrics_csv_sha256"]


def test_backfill_existing_csv(tmp_path):
    result = tmp_path / "results_20260811_test"
    result.mkdir()
    path = result / "pareto_circle2_seed0_n5000_test.csv"
    row = {
        "parameter_schema_version": "1", "mode": _experiment().mode,
        "dataset": "circle2", "seed": "0", "concept_schedule": "random",
        "sweep_parameter": "aggregation_interval", "sweep_value": "50",
        "aggregation_interval": "50", "fedsda_distance_threshold": "0.1",
        "clustering_policy": "on_new_model", "clustering_decision": "distance",
        "detection_episodes": "False", "new_model_creation_policy": "forward_persistent",
        "fifo_size": "30", "new_model_validation_fraction": "0.2",
        "new_model_forward_validation_samples": "10",
        "shared_backbone_training": "joint",
        "shared_backbone_routing_recalibration": "fifo_replay",
        "shared_adapter_rank": "8",
    }
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    manifest_path, manifest = backfill_manifest(result)
    assert manifest_path.is_file()
    assert manifest["provenance_status"] == "unknown_backfill"


def test_recursive_backfill_separates_variants_and_ignores_analysis_csv(tmp_path):
    fieldnames = [
        "parameter_schema_version", "mode", "dataset", "seed",
        "concept_schedule", "sweep_parameter", "sweep_value",
        "aggregation_interval", "fedsda_distance_threshold",
    ]
    for variant, seed in (("first", 0), ("second", 1)):
        pareto = tmp_path / variant / "pareto"
        pareto.mkdir(parents=True)
        path = pareto / f"pareto_circle2_seed{seed}_n5000.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({
                "parameter_schema_version": "1", "mode": _experiment().mode,
                "dataset": "circle2", "seed": str(seed),
                "concept_schedule": "random",
                "sweep_parameter": "aggregation_interval", "sweep_value": "50",
                "aggregation_interval": "50", "fedsda_distance_threshold": "0.1",
            })
        (tmp_path / variant / "recovery.csv").write_text(
            "metric,value\naccuracy,0.9\n", encoding="utf-8",
        )

    assert discover_execution_roots(tmp_path) == [
        (tmp_path / "first").resolve(), (tmp_path / "second").resolve(),
    ]
    outcomes = backfill_tree(tmp_path)

    assert [status for _, status, _ in outcomes] == ["created", "created"]
    assert (tmp_path / "first" / "manifest.json").is_file()
    assert (tmp_path / "second" / "manifest.json").is_file()


def test_backfill_infers_missing_schedule_and_total_data_from_raw(tmp_path):
    result = tmp_path / "legacy"
    pareto = result / "pareto"
    raw = result / "raw"
    pareto.mkdir(parents=True)
    raw.mkdir()
    row = {
        "parameter_schema_version": "1", "mode": _experiment().mode,
        "dataset": "circle2", "seed": "0",
        "sweep_parameter": "aggregation_interval", "sweep_value": "50",
        "aggregation_interval": "50", "fedsda_distance_threshold": "0.1",
    }
    path = pareto / "short.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    import numpy as np
    np.savez_compressed(
        raw / "run.npz", total_data=np.asarray(5000),
        concept_schedule=np.asarray("random"),
    )

    _, manifest = backfill_manifest(result)

    configuration = manifest["runs"][0]["configuration"]
    assert configuration["total_data"] == 5000
    assert configuration["concept_schedule"] == "random"


def test_study_manifest_is_generated_from_utf8_definition(tmp_path):
    study_root = tmp_path / "study"
    variant_root = study_root / "variants" / "reference"
    variant_root.mkdir(parents=True)
    write_json_atomic(variant_root / "manifest.json", {
        "variant_id": "reference", "mode": "FedSDA_Mode",
        "selection": {
            "datasets": ["circle2"], "missing_datasets": [],
            "routing": "hard",
        },
    })
    definition = tmp_path / "definition.json"
    write_json_atomic(definition, {
        "study_id": "study", "title": "日本語タイトル",
        "question": "何を比較するか", "comparison_axes": {"routing": ["hard"]},
        "common_configuration": {"datasets": ["circle2"]},
        "reference_variant": "reference",
        "variants": {"reference": {"path": "variants/reference"}},
    })

    manifest = build_study_manifest(definition, study_root)

    assert manifest["title"] == "日本語タイトル"
    assert "?" not in (study_root / "manifest.json").read_text(encoding="utf-8")
