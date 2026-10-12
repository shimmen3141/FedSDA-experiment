"""新実装の重要動作goldenの構造・検出力・数値回帰を検証する。"""

from copy import deepcopy

import behavior_golden_cases as golden
import pytest


@pytest.mark.parametrize(
    "section", ("run_settings", "run_metrics", "record_tables", "global_parameters")
)
def test_changed_snapshot_identifies_case_and_item(section):
    expected = {section: {"example": 1}}
    actual = deepcopy(expected)
    actual[section]["example"] = 2
    with pytest.raises(AssertionError, match=rf"example-case.*{section}.*example"):
        golden.assert_case_matches("example-case", actual, expected)


@pytest.mark.parametrize(
    "values", ([1, 2], [[True, False]], ["a", "b"], [float("nan"), float("inf"), -float("inf")])
)
def test_array_digest_includes_values_shape_and_dtype(values):
    import numpy as np

    array = np.asarray(values)
    snapshot = golden.array_snapshot(array)
    assert snapshot == golden.array_snapshot(array.copy())
    assert set(snapshot) == {"shape", "dtype", "sha256"}
    assert len(snapshot["sha256"]) == 64
    if array.size > 1:
        assert snapshot != golden.array_snapshot(array.reshape(-1, 1))
    if array.dtype.kind == "f":
        assert snapshot != golden.array_snapshot(np.asarray([float("nan")] * 3))


def test_case_registry_preserves_all_registered_configurations():
    from federated_learning_experiments.sweep.experiment_configurations import (
        EXPERIMENT_CONFIGURATION_NAMES,
    )

    assert golden.CASE_NAMES[:13] == EXPERIMENT_CONFIGURATION_NAMES
    assert len(golden.CASE_NAMES_BY_BASELINE_VERSION[1]) == 17
    assert golden.CASE_NAMES_BY_BASELINE_VERSION[2] == (
        *golden.CASE_NAMES_BY_BASELINE_VERSION[1],
        "fedsda-sine2-reference-reuse",
    )
    assert golden.CASE_NAMES == golden.CASE_NAMES_BY_BASELINE_VERSION[golden.BASELINE_VERSION]
    for baseline_version, case_names in golden.CASE_NAMES_BY_BASELINE_VERSION.items():
        for case_name in case_names:
            golden.build_case_settings(case_name, baseline_version).__post_init__()
    with pytest.raises(ValueError):
        golden.build_case_settings("fedsda-sine2-reference-reuse", 1)


def test_helper_does_not_import_legacy_or_other_tests():
    import ast
    from pathlib import Path

    source = Path(golden.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        assert not any(
            name.startswith(("federated_drift_experiment", "test_", "tests.", "pytest"))
            for name in names
        )


def test_missing_column_and_changed_parameter_are_identified():
    expected = {
        "record_tables": {"sample_prediction.correct": {"sha256": "a"}},
        "global_parameters": {"0.head.weight": {"sha256": "b"}},
    }
    for section, key in (
        ("record_tables", "sample_prediction.correct"),
        ("global_parameters", "0.head.weight"),
    ):
        actual = deepcopy(expected)
        del actual[section][key]
        with pytest.raises(AssertionError, match=section):
            golden.assert_case_matches("missing-case", actual, expected)


@pytest.fixture
def structural_document():
    import numpy as np

    digest = golden.array_snapshot(np.asarray([1]))
    cases = {}
    for name in golden.CASE_NAMES:
        conditions = golden.build_case_settings(name).execution_settings.experiment_run_conditions
        count = conditions.per_client_sample_count
        interval = conditions.server_aggregation_interval_per_client_samples
        cases[name] = dict(
            run_settings=golden.comparable_run_settings(name),
            run_metrics={"test_value": 1},
            record_tables={"test.column": digest},
            global_parameters={"0.test.weight": digest},
            coverage=dict(
                generated_sample_count_per_client=count,
                processed_sample_count_per_client=count // interval * interval,
                unprocessed_tail_sample_count_per_client=count % interval,
                synchronization_round_count=count // interval,
                candidate_accepted_count=1,
                candidate_rejected_count=1,
                decision_reasons=[
                    "first_segment_margin_failed",
                    "second_segment_margin_failed",
                    "alternative_reference_within_historical_loss_tolerance",
                ],
                training_model_switch_count=1,
                merged_model_count=1,
                recalibration_replayed_sample_count=1,
                registered_model_count=2,
                detection_count=1,
                unregistered_adopted_model_count=0,
            ),
        )
    return dict(
        baseline_version=golden.BASELINE_VERSION,
        schema_version=1,
        source_commit="a" * 40,
        environment=dict(
            system="Test",
            machine="test64",
            python="1.2.3",
            numpy="1.2.3",
            torch="1.2.3",
            device="cpu",
            dtype="torch.float32",
            threads=1,
        ),
        mnist_sha256={
            "train-images-idx3-ubyte.gz": "a" * 64,
            "train-labels-idx1-ubyte.gz": "b" * 64,
        },
        cases=cases,
    )


def test_unknown_environment_has_no_match_but_reads_structure(tmp_path, structural_document):
    golden._write_exclusive(tmp_path / "valid.json", structural_document)
    assert golden.load_golden(tmp_path / "valid.json") == structural_document
    assert golden.find_golden_path({"system": "Other"}, tmp_path) is None


def test_duplicate_environment_is_rejected(tmp_path, structural_document):
    for name in ("first", "second"):
        golden._write_exclusive(tmp_path / f"{name}.json", structural_document)
    with pytest.raises(ValueError, match="duplicate"):
        golden.find_golden_path(structural_document["environment"], tmp_path)


@pytest.mark.parametrize(
    "damage",
    (
        "unknown_version",
        "boolean_version",
        "missing_case",
        "invalid_commit",
        "bad_digest",
        "negative_shape",
        "missing_column",
        "missing_path",
        "nonfinite_metric",
    ),
)
def test_corrupt_golden_is_rejected_before_environment_skip(tmp_path, structural_document, damage):
    import json

    document = deepcopy(structural_document)
    case = document["cases"]["fedsda-sine2-events"]
    if damage == "unknown_version":
        document["baseline_version"] = max(golden.SUPPORTED_BASELINE_VERSIONS) + 1
    elif damage == "boolean_version":
        document["schema_version"] = True
    elif damage == "missing_case":
        del document["cases"]["fedavg"]
    elif damage == "invalid_commit":
        document["source_commit"] = "missing"
    elif damage == "bad_digest":
        case["record_tables"]["test.column"]["sha256"] = "bad"
    elif damage == "negative_shape":
        case["record_tables"]["test.column"]["shape"] = [-1]
    elif damage == "missing_column":
        case["record_tables"] = {}
    elif damage == "missing_path":
        case["coverage"]["merged_model_count"] = 0
    else:
        case["run_metrics"]["bad"] = float("nan")
    path = tmp_path / "damaged.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="damaged.json"):
        golden.find_golden_path({"system": "Other"}, tmp_path)


def test_unicode_dtype_is_a_valid_digest(tmp_path, structural_document):
    import numpy as np

    structural_document["cases"]["fedavg"]["record_tables"]["test.column"] = golden.array_snapshot(
        np.asarray(["hello"])
    )
    path = tmp_path / "unicode.json"
    golden._write_exclusive(path, structural_document)
    assert golden.load_golden(path) == structural_document


@pytest.mark.parametrize("second_result", ("same", "different", "failure"))
def test_generation_requires_two_matching_process_results(
    monkeypatch, tmp_path, structural_document, second_result
):
    import subprocess

    source_commit = structural_document["source_commit"]
    calls = []
    monkeypatch.setattr(
        golden.subprocess,
        "check_output",
        lambda command, **kwargs: "" if command[-1] == "src" else source_commit,
    )

    def run(command, **kwargs):
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0)
        calls.append(command)
        document = deepcopy(structural_document)
        if len(calls) == 2:
            if second_result == "failure":
                raise subprocess.CalledProcessError(1, command)
            if second_result == "different":
                document["cases"]["fedavg"]["run_metrics"]["test_value"] = 2
        from pathlib import Path

        golden._write_exclusive(Path(command[command.index("--worker-output") + 1]), document)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(golden.subprocess, "run", run)
    output = tmp_path / "new.json"
    if second_result == "same":
        assert golden.generate_golden(output, source_commit) == output
        assert golden.load_golden(output) == structural_document
    else:
        with pytest.raises((AssertionError, subprocess.CalledProcessError)):
            golden.generate_golden(output, source_commit)
        assert not output.exists()
    assert len(calls) == 2
    assert calls[0][3] != calls[1][3]


def test_generation_preserves_existing_output_before_starting(monkeypatch, tmp_path):
    output = tmp_path / "existing.json"
    output.write_bytes(b"original")
    monkeypatch.setattr(
        golden.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected process")
    )
    with pytest.raises(FileExistsError):
        golden.generate_golden(output, "a" * 40)
    assert output.read_bytes() == b"original"


@pytest.mark.parametrize("baseline_version", golden.SUPPORTED_BASELINE_VERSIONS)
def test_committed_goldens_cover_both_reference_environments(baseline_version):
    paths = sorted(golden.golden_directory(baseline_version).glob("*.json"))
    assert len(paths) == 2, "Windows/WSL baselines are required"
    documents = [golden.load_golden(path, baseline_version) for path in paths]
    assert {d["environment"]["system"] for d in documents} == {"Windows", "Linux"}
    case_names = golden.CASE_NAMES_BY_BASELINE_VERSION[baseline_version]
    for name in case_names:
        assert (
            documents[0]["cases"][name]["run_settings"]
            == documents[1]["cases"][name]["run_settings"]
        )
    for name in case_names:
        for section in ("run_metrics", "record_tables"):
            assert set(documents[0]["cases"][name][section]) == set(
                documents[1]["cases"][name][section]
            ), (name, section)
    assert documents[0]["mnist_sha256"] == documents[1]["mnist_sha256"]


# ケースの実行の結果（ケース名と基準版ごとに1回だけ実行して、数値の照合と、版の差の説明のtestで共有する）。
_CAPTURED_CASES = {}


def capture_case_once(case_name, baseline_version):
    key = (case_name, baseline_version)
    if key not in _CAPTURED_CASES:
        _CAPTURED_CASES[key] = golden.capture_case_and_count_unregistered_adopted_models(
            case_name, baseline_version
        )
    return _CAPTURED_CASES[key]


@pytest.mark.numerical_golden
@pytest.mark.parametrize(
    ("baseline_version", "case_name"),
    [
        (baseline_version, case_name)
        for baseline_version, case_names in golden.CASE_NAMES_BY_BASELINE_VERSION.items()
        for case_name in case_names
    ],
)
def test_new_implementation_matches_important_behavior_golden(case_name, baseline_version):
    """全部の基準版を照合する。版1は、版1の動作に当たる設定（登録の扱いが、固定旧と同じ）で実行する。"""
    environment = golden.numerical_environment()
    path = golden.find_golden_path(environment, baseline_version=baseline_version)
    if path is None:
        pytest.skip(
            f"baseline v{baseline_version} has no numerical golden for {environment}; structural checks remain required"
        )
    document = golden.load_golden(path, baseline_version)
    assert document["mnist_sha256"] == golden.mnist_sha256(), "MNIST reference data changed"
    golden.assert_case_matches(
        case_name,
        capture_case_once(case_name, baseline_version)[0],
        document["cases"][case_name],
    )


@pytest.mark.parametrize("case_name", golden.CASE_NAMES_BY_BASELINE_VERSION[1])
def test_baseline_2_differs_from_baseline_1_only_where_adopted_models_stayed_unregistered(
    case_name,
):
    """版2と版1の差は、版1の扱いで、登録されないままの採用が起きるケースに限る（LEGACY-016の修正の範囲）。

    FedDrift・FedAvgのケースは、登録の扱いの設定を持たず、数は0（2つの版で、同じ設定）。
    """
    version_1_case, version_1_unregistered_model_count = capture_case_once(case_name, 1)
    version_2_case, version_2_unregistered_model_count = capture_case_once(case_name, 2)
    assert version_2_unregistered_model_count == 0
    compared_sections = ("run_metrics", "record_tables", "global_parameters")
    values_are_equal = all(
        version_1_case[section] == version_2_case[section] for section in compared_sections
    )
    if version_1_unregistered_model_count == 0:
        for section in compared_sections:
            golden.assert_case_matches(
                case_name, version_2_case[section], version_1_case[section], section
            )
    else:
        assert not values_are_equal, case_name
    # 修正の経路を通るケースが、基準に入っている。
    if case_name == "fedsda-sine2-events":
        assert version_1_unregistered_model_count == 2


@pytest.mark.parametrize("condition", ("wrong_head", "modified_source", "untracked_source"))
def test_generation_rejects_unrecorded_source_before_workers(monkeypatch, tmp_path, condition):
    def git_output(command, **kwargs):
        if command[-1] == "HEAD":
            return "b" * 40 if condition == "wrong_head" else "a" * 40
        return " M src/example.py" if condition == "modified_source" else "?? src/example.py"

    monkeypatch.setattr(golden.subprocess, "check_output", git_output)
    monkeypatch.setattr(
        golden.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected worker")
    )
    output = tmp_path / "rejected.json"
    with pytest.raises(ValueError, match="HEAD|src"):
        golden.generate_golden(output, "a" * 40)
    assert not output.exists()


def test_incomplete_candidate_is_not_counted_as_rejection():
    from types import SimpleNamespace

    import numpy as np

    result = SimpleNamespace(
        generated_sample_count_per_client=3,
        processed_sample_count_per_client=3,
        unprocessed_tail_sample_count_per_client=0,
        synchronization_interval_count=1,
    )
    run = SimpleNamespace(
        measured_run=SimpleNamespace(run_result=result),
        run_metrics=SimpleNamespace(
            training_model_switch_detection_metrics=SimpleNamespace(detection_count=0)
        ),
    )
    coverage = golden._coverage(
        "test-case",
        run,
        {
            "candidate_validation_decision.candidate_accepted": np.asarray([True, False, False]),
            "candidate_validation_decision.is_incomplete_at_run_end": np.asarray(
                [False, False, True]
            ),
        },
        baseline_version=1,
    )
    assert coverage["candidate_accepted_count"] == coverage["candidate_rejected_count"] == 1


@pytest.mark.parametrize(
    "section", ("run_settings", "run_metrics", "record_tables", "global_parameters")
)
def test_changed_real_golden_snapshot_is_detected(section):
    import re

    document = golden.load_golden(next(golden.GOLDEN_DIRECTORY.glob("windows-*.json")))
    expected = document["cases"]["fedavg"]
    changed = deepcopy(expected)
    if section == "run_settings":
        changed[section]["execution_settings"]["experiment_run_conditions"]["random_seed"] += 1
        field = "random_seed"
    elif section == "run_metrics":
        changed[section]["processed_sample_count"] += 1
        field = "processed_sample_count"
    else:
        field = next(iter(changed[section]))
        changed[section][field]["sha256"] = "f" * 64
    with pytest.raises(AssertionError, match=rf"fedavg.*{section}.*{re.escape(field)}"):
        golden.assert_case_matches("fedavg", changed, expected)
