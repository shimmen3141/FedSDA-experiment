"""対の集計はseedを単位とし、保存設定と入力の不整合を拒否する。"""

from copy import deepcopy

import numpy as np
import pytest

from tools import aggregate_soft_training_single_model_prediction as analysis


def test_paired_interval_uses_ten_seed_means_and_percentage_points():
    values = np.tile(np.arange(10) / 1000, (20, 1))
    result = analysis._paired_summary(values.mean(axis=0), accuracy=True)
    assert result["mean"] == pytest.approx(0.45)
    assert result["half_width_95"] == pytest.approx(
        2.2621571628540993 * np.std(np.arange(10) / 10, ddof=1) / np.sqrt(10)
    )
    assert (result["positive_seeds"], result["zero_seeds"], result["negative_seeds"]) == (9, 1, 0)


def test_settings_normalization_accepts_only_explicit_axis_differences():
    from test_fedsda_soft_training_configuration import build_settings

    from federated_learning_experiments.core.settings_serialization import (
        convert_settings_to_plain_mapping,
    )

    new = convert_settings_to_plain_mapping(build_settings(analysis._NEW))
    for name in (
        "fedsda",
        "fedsda-single-model-prediction",
        "fedsda-soft-training",
        "fedsda-training-mass-diagnostics",
    ):
        old = convert_settings_to_plain_mapping(build_settings(name))
        analysis._check_settings(new, old, name)
        old["run_metric_settings"]["maximum_detection_delay_sample_count"] += 1
        with pytest.raises(ValueError):
            analysis._check_settings(new, old, name)


@pytest.mark.parametrize(
    "column",
    [
        "sample_prediction.observed_class_id",
        "sample_prediction.observed_concept_id",
        "sample_prediction.sample_index",
        "input.feature_digest",
    ],
)
def test_same_seed_does_not_hide_input_mismatch(column):
    tables = {
        key: np.arange(4)
        for key in (
            "sample_prediction.observed_class_id",
            "sample_prediction.observed_concept_id",
            "sample_prediction.sample_index",
            "input.feature_digest",
        )
    }
    other = deepcopy(tables)
    other[column][0] = 100
    with pytest.raises(ValueError):
        analysis._check_inputs(tables, other)
    assert "未検証" in analysis._check_inputs(
        {k: v for k, v in tables.items() if k != "input.feature_digest"},
        {k: v for k, v in tables.items() if k != "input.feature_digest"},
    )


def test_missing_metrics_and_memory_are_not_zero_filled():
    metrics = analysis._metrics(
        {
            "run_metrics": {
                "prediction_accuracy": 0.5,
                "processed_sample_count": 4,
                "communication_volume": {"uploaded_byte_count": 100, "downloaded_byte_count": 200},
            }
        },
        {},
    )
    assert metrics["communication_total_bytes"] == 300
    assert metrics["communication_total_MiB"] == 300 / 1024**2
    assert metrics["peak_memory_bytes"] is None
    assert metrics["training_mass_purity"] is None


def test_final_held_model_count_uses_final_client_records():
    summary = {
        "run_metrics": {},
        "run_settings": {"execution_settings": {"experiment_run_conditions": {"client_count": 2}}},
    }
    metrics = analysis._metrics(summary, {"client_held_model.client_id": np.array([0, 0, 1])})
    assert metrics["final_held_model_count"] == 1.5


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "missing",
        "duplicate",
        "failed",
        "environment",
        "provenance",
        "summary",
        "npz",
        "wrong_configuration",
        "wrong_configuration_excluded_interval",
        "wrong_root",
    ],
)
def test_artifact_gate_checks_success_summary_provenance_and_npz(tmp_path, defect):
    import csv
    import json

    from federated_learning_experiments.configuration.experiment_run_conditions import (
        ExperimentRunConditions,
    )
    from federated_learning_experiments.core.settings_serialization import (
        convert_settings_to_plain_mapping,
    )
    from federated_learning_experiments.sweep.experiment_configurations import (
        build_experiment_run_settings,
    )

    root = tmp_path / analysis._NEW
    runs = root / "runs"
    runs.mkdir(parents=True)
    settings = convert_settings_to_plain_mapping(
        build_experiment_run_settings(
            configuration_name=analysis._NEW,
            experiment_run_conditions=ExperimentRunConditions(
                dataset_name="sine2",
                random_seed=0,
                client_count=10,
                per_client_sample_count=5000,
                server_aggregation_interval_per_client_samples=50,
            ),
        )
    )
    identifier = analysis._identifier(analysis._NEW, settings)
    summary = dict(
        format_version=2,
        run_settings=settings,
        run_metrics=dict(prediction_accuracy=1.0, processed_sample_count=50000),
        environment=dict(system="Linux", machine="x86_64", python="3.10", numpy="2", torch="2"),
        run_provenance=dict(
            code_commit="a" * 40,
            has_uncommitted_changes=False,
            code_revision_unavailable_reason=None,
            behavior_baseline_version=16,
        ),
        record_table_file_name=identifier + ".npz",
    )
    row = dict(
        run_identifier=identifier,
        configuration_name=analysis._NEW,
        **{
            k: v
            for k, v in settings["execution_settings"]["experiment_run_conditions"].items()
            if k != "settings_type"
        },
        **summary["run_metrics"],
        **{
            k: v
            for k, v in summary["run_provenance"].items()
            if k != "code_revision_unavailable_reason"
        },
    )
    if defect == "summary":
        row["prediction_accuracy"] = 0.0
    if defect in ("wrong_configuration", "wrong_configuration_excluded_interval"):
        row["configuration_name"] = "fedsda-single-model-prediction"
    if defect == "wrong_configuration_excluded_interval":
        settings["execution_settings"]["experiment_run_conditions"][
            "server_aggregation_interval_per_client_samples"
        ] = 100
        row["server_aggregation_interval_per_client_samples"] = 100
    with (root / "run_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    if defect == "environment":
        summary["environment"] = {}
    if defect == "provenance":
        summary["run_provenance"]["code_commit"] = None
    (tmp_path / "sweep_record_test.json").write_text(
        json.dumps(
            dict(
                run_outcomes=[
                    dict(
                        configuration_name=analysis._NEW,
                        run_identifier=identifier,
                        run_status="failed" if defect == "failed" else "completed",
                    )
                ]
            )
        ),
        encoding="utf-8",
    )
    if defect != "missing":
        (runs / (identifier + ".json")).write_text(json.dumps(summary), encoding="utf-8")
    if defect == "duplicate":
        extra = runs / "duplicate"
        extra.mkdir()
        (extra / (identifier + ".json")).write_text(json.dumps(summary), encoding="utf-8")
        np.savez(
            extra / (identifier + ".npz"),
            **{
                "sample_prediction.combined_prediction_is_correct": np.ones((10, 5000), dtype=bool),
                "sample_prediction.observed_class_id": np.zeros((10, 5000), dtype=np.int64),
                "sample_prediction.observed_concept_id": np.zeros((10, 5000), dtype=np.int64),
            },
        )
    if defect == "npz":
        (runs / (identifier + ".npz")).write_bytes(b"broken")
    else:
        np.savez(
            runs / (identifier + ".npz"),
            **{
                "sample_prediction.combined_prediction_is_correct": np.ones((10, 5000), dtype=bool),
                "sample_prediction.observed_class_id": np.zeros((10, 5000), dtype=np.int64),
                "sample_prediction.observed_concept_id": np.zeros((10, 5000), dtype=np.int64),
            },
        )
    checks = []
    if defect == "wrong_root":
        root = tmp_path
    result = analysis._read_group([root], "Ssingle", {("sine2", 50, 0, 10, 5000)}, checks)
    assert any(c["status"] == "error" for c in checks) == (defect is not None)
    if defect is None:
        assert len(result) == 1


@pytest.mark.parametrize("verified", [False, True])
def test_cli_stops_before_aggregation_when_input_identity_is_unverified(monkeypatch, verified):
    from io import StringIO
    from pathlib import Path

    from test_fedsda_soft_training_configuration import build_settings

    from federated_learning_experiments.core.settings_serialization import (
        convert_settings_to_plain_mapping,
    )

    monkeypatch.setattr(analysis, "_DATASETS", ("sine2",))

    def read_group(roots, role, expected, checks):
        tables = {
            "sample_prediction.observed_class_id": np.zeros((10, 5000), dtype=int),
            "sample_prediction.observed_concept_id": np.zeros((10, 5000), dtype=int),
            "sample_prediction.client_id": np.repeat(np.arange(10), 5000),
            "sample_prediction.sample_index": np.tile(np.arange(5000), 10),
        }
        if verified:
            tables["input.feature_digest"] = np.arange(50000)
        summary = dict(
            run_settings=convert_settings_to_plain_mapping(build_settings(analysis._NAMES[role])),
            environment={"system": "test"},
            run_metrics={"prediction_accuracy": 0.5},
        )
        return {key: (summary, tables, analysis._NAMES[role], roots[0]) for key in expected}

    monkeypatch.setattr(analysis, "_read_group", read_group)
    outputs = {}
    monkeypatch.setattr(
        analysis, "_write_csv", lambda path, rows: outputs.update({path.name: rows})
    )
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: StringIO())
    result = analysis.main(
        [
            "--new-root",
            "new",
            "--soft-roots",
            "soft",
            "--single-roots",
            "single",
            "--hard-roots",
            "hard",
        ]
    )
    assert result == (0 if verified else 2)
    assert "analysis_pairing_check.csv" in outputs
    assert ("analysis_soft_training_single_model_prediction.csv" in outputs) == verified


@pytest.mark.parametrize(
    "role, root", [("Smix", "fedsda"), ("Hmix", "parent"), ("Diagnostic", "fedsda")]
)
def test_wrong_group_root_is_rejected_before_loading(role, root):
    from pathlib import Path

    checks = []
    assert analysis._read_group([Path(root)], role, set(), checks) == {}
    assert checks == [dict(role=role, path=root, status="error", detail="構成ルートが違う")]


@pytest.mark.parametrize("interval", [50, 100, 200])
@pytest.mark.parametrize("wrong_field", ["configuration_name", "run_settings"])
def test_foreign_configuration_is_rejected_even_in_excluded_intervals(
    monkeypatch, interval, wrong_field
):
    import json
    from io import StringIO
    from pathlib import Path

    from test_fedsda_soft_training_configuration import build_settings

    from federated_learning_experiments.core.settings_serialization import (
        convert_settings_to_plain_mapping,
    )

    root = Path(analysis._NEW)
    path = root / "runs" / "test.json"
    configuration = "fedsda-single-model-prediction"
    settings = convert_settings_to_plain_mapping(
        build_settings(configuration if wrong_field == "run_settings" else analysis._NEW)
    )
    settings["execution_settings"]["experiment_run_conditions"][
        "server_aggregation_interval_per_client_samples"
    ] = interval
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(Path, "glob", lambda *args: iter(()))
    monkeypatch.setattr(Path, "rglob", lambda *args: iter((path,)))
    monkeypatch.setattr(
        Path, "read_text", lambda *args, **kwargs: json.dumps(dict(run_settings=settings))
    )
    monkeypatch.setattr(
        Path,
        "open",
        lambda *args, **kwargs: StringIO(
            "run_identifier,configuration_name\ntest,"
            + (configuration if wrong_field == "configuration_name" else analysis._NEW)
            + "\n"
        ),
    )
    checks = []
    assert analysis._read_group([root], "Ssingle", set(), checks) == {}
    assert any(
        c["status"] == "error" and "構成" in c["detail"] or "軸" in c["detail"] for c in checks
    )
