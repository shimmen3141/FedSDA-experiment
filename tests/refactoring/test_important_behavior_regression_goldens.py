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
    from pathlib import Path

    from federated_learning_experiments.sweep.experiment_configurations import (
        EXPERIMENT_CONFIGURATION_NAMES,
    )

    # 構成の全部が、goldenの事例か、専用のtestで守る構成の、どちらか一方に入る。既存の構成の並びは変えない。
    assert golden.CASE_NAMES[:13] == golden.BASELINE_CONFIGURATION_CASE_NAMES
    assert EXPERIMENT_CONFIGURATION_NAMES[:13] == golden.BASELINE_CONFIGURATION_CASE_NAMES
    protected = set(golden.CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN)
    assert not protected & set(golden.BASELINE_CONFIGURATION_CASE_NAMES)
    assert set(EXPERIMENT_CONFIGURATION_NAMES) == (
        set(golden.BASELINE_CONFIGURATION_CASE_NAMES) | protected
    )
    repository_root = Path(__file__).resolve().parents[2]
    for test_file in golden.CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN.values():
        assert (repository_root / test_file).is_file(), test_file
    assert len(golden.CASE_NAMES_BY_BASELINE_VERSION[1]) == 17
    assert golden.CASE_NAMES_BY_BASELINE_VERSION[2] == (
        *golden.CASE_NAMES_BY_BASELINE_VERSION[1],
        "fedsda-sine2-reference-reuse",
    )
    assert golden.CASE_NAMES_BY_BASELINE_VERSION[3] == golden.CASE_NAMES_BY_BASELINE_VERSION[2]
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
        cases_run_settings = golden.comparable_run_settings(name)
        cases[name] = dict(
            run_settings=cases_run_settings,
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
            )
            | (
                dict(repeated_training_sample_assignment_count=0)
                if "run_client_settings" in cases_run_settings["run_participant_settings"]
                and "pending_model_registration_settings"
                in cases_run_settings["run_participant_settings"]["run_client_settings"]
                else {}
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
        _CAPTURED_CASES[key] = golden.capture_case_and_count_divergence_events(
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
@pytest.mark.slow
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


# 隣り合う版の組ごとの、（前の版の扱いで、2つの版の状態が分かれる事象の名前、実測で差を確かめるケースと、その事象の数）。
# 一般に言えるのは、「事象が0なら、2つの版の値は同じ」の向きだけ（事象があっても、値が違うとは限らない）。
BASELINE_VERSION_DIFFERENCES = {
    (1, 2): ("unregistered_adopted_model_count", {"fedsda-sine2-events": 2}),
    (2, 3): (
        "too_short_alarm_with_earlier_interval_count",
        {"fedsda-sine2-events": None, "fedsda-sine2-reference-reuse": None},
    ),
    # 版3の扱いで、待ちの間に現行でない送信保留が起きるケース（2026-10-14、Windowsで4件と1件）。
    (3, 4): (
        "pending_model_not_current_while_waiting_count",
        {"fedsda-sine2-events": None, "fedsda-sine2-reference-reuse": None},
    ),
    # 版4の扱いで、警報を待つ場面が起きるケース（2026-10-14、Windowsで10件・16件・16件）。ADWINの2構成は、
    # 版4・5とも即時の警報なので、事象は定義により0（数えない）。
    (4, 5): (
        "alarm_wait_situation_count",
        {
            "fedsda-sine2-events": None,
            "fedsda-sine2-reference-reuse": None,
            "fedsda-sine2-alternative-reference-reuse": None,
        },
    ),
}


# 版6では、予測と、その推論に由来する項目だけが変わる。学習の計算回数は許容しない。
_PREDICTION_SIDE_RECORD_TABLE_COLUMNS = {
    "client_weight_update_counts.fixed_share_weight_leader_switch_count",
    "model_registration.maximum_weight_prediction_sample_count",
    "server_round.local_processing_concept_specific_part_inference_example_count",
    "server_round.local_processing_concept_specific_part_inference_forward_multiply_accumulate_count",
    "server_round.local_processing_shared_part_inference_example_count",
    "server_round.local_processing_shared_part_inference_forward_multiply_accumulate_count",
}
_PREDICTION_SIDE_RUN_METRIC_PATHS = {
    "prediction_accuracy",
    "stable_period_prediction_accuracy",
    *(
        f"model_computation_counts.{name}"
        for name in (
            "shared_part_inference_example_count",
            "shared_part_inference_forward_multiply_accumulate_count",
            "concept_specific_part_inference_example_count",
            "concept_specific_part_inference_forward_multiply_accumulate_count",
        )
    ),
    *(
        f"computation_cost_summary.{name}"
        for name in (
            "client_concept_specific_part_forward_multiply_accumulate_count",
            "client_concept_specific_part_forward_multiply_accumulate_count_per_held_model_sample",
            "client_forward_and_backward_multiply_accumulate_count_per_held_model_sample",
            "client_forward_and_backward_multiply_accumulate_count_per_processed_sample",
            "client_forward_multiply_accumulate_count",
            "client_forward_multiply_accumulate_count_per_held_model_sample",
            "client_forward_multiply_accumulate_count_per_processed_sample",
            "client_shared_part_forward_multiply_accumulate_count",
            "client_shared_part_forward_multiply_accumulate_count_per_processed_sample",
        )
    ),
}


def _flatten_metric_paths(mapping, prefix=""):
    return {
        path: leaf
        for key, value in mapping.items()
        for path, leaf in (
            _flatten_metric_paths(value, f"{prefix}{key}.").items()
            if isinstance(value, dict)
            else [(f"{prefix}{key}", value)]
        )
    }


def test_baseline_6_retains_all_cases_and_changes_only_prediction_settings():
    assert 6 in golden.SUPPORTED_BASELINE_VERSIONS
    assert golden.CASE_NAMES_BY_BASELINE_VERSION[6] == golden.CASE_NAMES_BY_BASELINE_VERSION[5]
    for name in golden.CASE_NAMES_BY_BASELINE_VERSION[5]:
        previous = deepcopy(golden.comparable_run_settings(name, 5))
        current = golden.comparable_run_settings(name, 6)
        if name.startswith("fedsda"):
            prediction = previous["run_participant_settings"]["run_client_settings"][
                "prediction_combination_settings"
            ]
            prediction["prediction_model_parameter_source"] = (
                "current_local_parameters"
                if name == "fedsda-single-model-prediction"
                else "mixture_of_synchronized_and_local_parameters"
            )
            prediction["fixed_share_weight_redistribution_time_scale_samples"] = 120
        golden.assert_case_matches(name, current, previous, "run_settings")


@pytest.mark.slow
def test_baseline_6_differs_from_5_only_on_the_prediction_side():
    prediction_records_changed = []
    for name in golden.CASE_NAMES_BY_BASELINE_VERSION[5]:
        previous = capture_case_once(name, 5)[0]
        current = capture_case_once(name, 6)[0]
        for section in ("global_parameters", "coverage"):
            golden.assert_case_matches(name, current[section], previous[section], section)
        if not name.startswith("fedsda") or name == "fedsda-single-model-prediction":
            for section in ("run_metrics", "record_tables"):
                golden.assert_case_matches(name, current[section], previous[section], section)
            continue
        previous_tables = previous["record_tables"]
        current_tables = current["record_tables"]
        assert set(previous_tables) == set(current_tables), name
        changed_tables = {
            key for key in previous_tables if previous_tables[key] != current_tables[key]
        }
        assert all(
            key.startswith("sample_prediction.") or key in _PREDICTION_SIDE_RECORD_TABLE_COLUMNS
            for key in changed_tables
        ), (name, changed_tables)
        previous_metrics = _flatten_metric_paths(previous["run_metrics"])
        current_metrics = _flatten_metric_paths(current["run_metrics"])
        assert set(previous_metrics) == set(current_metrics), name
        changed_metrics = {
            key for key in previous_metrics if previous_metrics[key] != current_metrics[key]
        }
        assert all(
            key.startswith("prediction_diagnostic_metrics.")
            or key in _PREDICTION_SIDE_RUN_METRIC_PATHS
            for key in changed_metrics
        ), (name, changed_metrics)
        assert changed_tables or changed_metrics, name
        if any(key.startswith("sample_prediction.") for key in changed_tables):
            prediction_records_changed.append(name)
    assert prediction_records_changed


# 版7で、版6と同じ値であるケース: 候補検証の経路の4ケース（版6までの既定と同じ設定の構成で実行する）、
# FedDrift・FedAvg、候補検証を外した構成（設定は、版6と同じ）。ほかのFedSDAのケースは、新しい既定を引き継ぐ。
_BASELINE_7_CASES_KEEPING_BASELINE_6 = (
    *golden.CANDIDATE_VALIDATION_PATH_CASE_NAMES,
    "fedsda-without-candidate-validation",
    "feddrift",
    "fedavg",
    "feddrift-drift-tail",
    "fedavg-no-complete-interval",
)


def test_baseline_7_adds_the_new_default_cases_and_changes_only_candidate_settings():
    """版7のケースは、版6の全部と、新しい既定の2ケース。設定は、新しい既定を引き継ぐケースだけ、候補の2つの値が違う。"""
    assert 7 in golden.SUPPORTED_BASELINE_VERSIONS
    assert golden.CASE_NAMES_BY_BASELINE_VERSION[7] == (
        *golden.CASE_NAMES_BY_BASELINE_VERSION[6],
        *golden.ADOPTION_WITHOUT_VALIDATION_PATH_CASE_NAMES,
    )
    assert set(_BASELINE_7_CASES_KEEPING_BASELINE_6) <= set(
        golden.CASE_NAMES_BY_BASELINE_VERSION[6]
    )
    for name in golden.CASE_NAMES_BY_BASELINE_VERSION[6]:
        previous = deepcopy(golden.comparable_run_settings(name, 6))
        current = deepcopy(golden.comparable_run_settings(name, 7))
        if name.startswith("fedsda"):
            current_client = current["run_participant_settings"]["run_client_settings"]
            shared_feature_policy = current_client["candidate_epoch_training_settings"].pop(
                "candidate_shared_feature_update_policy"
            )
            if name not in _BASELINE_7_CASES_KEEPING_BASELINE_6:
                # 新しい既定を引き継ぐケース: 候補を検証せずに採用し、共有部を固定する。
                assert shared_feature_policy == "keep_shared_features_fixed", name
                previous["run_participant_settings"]["run_client_settings"][
                    "candidate_model_training_and_acceptance_settings"
                ][
                    "candidate_model_acceptance_policy"
                ] = "current_model_first_reuse_then_candidate_adoption_without_validation"
            else:
                assert shared_feature_policy == "update_shared_features_with_candidate", name
        golden.assert_case_matches(name, current, previous, "run_settings")
    for name in golden.ADOPTION_WITHOUT_VALIDATION_PATH_CASE_NAMES:
        client = golden.build_case_settings(name, 7).run_participant_settings.run_client_settings
        assert (
            client.candidate_model_training_and_acceptance_settings.candidate_model_acceptance_policy,
            client.candidate_epoch_training_settings.candidate_shared_feature_update_policy,
        ) == (
            "current_model_first_reuse_then_candidate_adoption_without_validation",
            "keep_shared_features_fixed",
        )


@pytest.mark.slow
@pytest.mark.parametrize("case_name", _BASELINE_7_CASES_KEEPING_BASELINE_6)
def test_baseline_7_keeps_baseline_6_outside_the_new_default(case_name):
    """候補検証の経路の4ケース、FedDrift・FedAvg、候補検証を外した構成は、版7でも、版6と同じ値。"""
    previous = capture_case_once(case_name, 6)[0]
    current = deepcopy(capture_case_once(case_name, 7)[0])
    if case_name.startswith("fedsda"):
        # 版7で足した経路の記録の項目（採用した候補の数）。版6には無い。
        current["coverage"].pop("adopted_candidate_count")
    for section in ("run_metrics", "record_tables", "global_parameters", "coverage"):
        golden.assert_case_matches(case_name, current[section], previous[section], section)


@pytest.mark.parametrize(
    ("previous_version", "next_version", "case_name"),
    [
        (previous_version, next_version, case_name)
        for previous_version, next_version in BASELINE_VERSION_DIFFERENCES
        for case_name in golden.CASE_NAMES_BY_BASELINE_VERSION[previous_version]
    ],
)
def test_next_baseline_differs_only_where_the_fixed_behavior_occurred(
    previous_version, next_version, case_name
):
    """次の版との差は、前の版の扱いで、直した挙動の事象が起きるケースに限る。

    版1→2: 登録されないままの採用（LEGACY-016）。版2→3: 前区間のある短い警報（LEGACY-002）。
    版3→4: 待ちの間に現行でない送信保留（LEGACY-016の扱いの見直し。事象が1以上のケースは、全部、名前を挙げる）。
    版4→5: 警報を待つ場面（IMPROVE-006。事象が1以上のケースは、全部、名前を挙げる）。
    事象が0のケースは、指標・表・パラメータが、前の版と同じ。事象があるケースの差は、名前を挙げたケースで確かめる。
    FedDrift・FedAvgのケースは、これらの扱いの設定を持たず、事象は0（2つの版で、同じ設定）。
    """
    divergence_event_name, differing_case_event_counts = BASELINE_VERSION_DIFFERENCES[
        (previous_version, next_version)
    ]
    previous_case, previous_event_counts = capture_case_once(case_name, previous_version)
    next_case, next_event_counts = capture_case_once(case_name, next_version)
    # 次の版の扱いでは、直した挙動は起きない。
    assert next_event_counts["unregistered_adopted_model_count"] == 0
    if next_version >= 3:
        assert next_event_counts["repeated_training_sample_assignment_count"] == 0
    compared_sections = ("run_metrics", "record_tables", "global_parameters")
    if previous_event_counts[divergence_event_name] == 0:
        assert case_name not in differing_case_event_counts
        for section in compared_sections:
            golden.assert_case_matches(
                case_name, next_case[section], previous_case[section], section
            )
    if (previous_version, next_version) in ((3, 4), (4, 5)):
        # 事象が1以上のケースは、名前を挙げたケースに限る（挙げていないケースの差を、見落とさない）。
        assert (previous_event_counts[divergence_event_name] > 0) == (
            case_name in differing_case_event_counts
        ), (case_name, previous_event_counts)
    if case_name in differing_case_event_counts:
        expected_event_count = differing_case_event_counts[case_name]
        assert previous_event_counts[divergence_event_name] > 0
        if expected_event_count is not None:
            assert previous_event_counts[divergence_event_name] == expected_event_count
        if (previous_version, next_version) == (2, 3):
            # 版2の扱いでは、同じ標本の2回目の取込みが、実際に起きている。
            assert previous_event_counts["repeated_training_sample_assignment_count"] > 0
        assert not all(
            previous_case[section] == next_case[section] for section in compared_sections
        ), case_name


def test_named_difference_cases_exist_in_both_baseline_versions():
    for (previous_version, next_version), (
        _,
        differing_cases,
    ) in BASELINE_VERSION_DIFFERENCES.items():
        for case_name in differing_cases:
            assert case_name in golden.CASE_NAMES_BY_BASELINE_VERSION[previous_version]
            assert case_name in golden.CASE_NAMES_BY_BASELINE_VERSION[next_version]


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
