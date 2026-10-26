"""新実装の重要動作goldenの構造・検出力・数値回帰を検証する。"""

from copy import deepcopy
from dataclasses import replace

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
                dict(
                    repeated_training_sample_assignment_count=0,
                    adopted_candidate_count=1,
                    selected_current_model_count=1,
                    selected_held_model_count=1,
                    selected_new_model_count=1,
                    selected_reinitialized_learner_count=1,
                    selected_warm_started_learner_count=1,
                    # 版12で足した、クロス評価の記録の行の数と、そのうち標本が0件の行の数。
                    cross_evaluation_record_count=1,
                    zero_sample_cross_evaluation_count=0,
                    # 版13で足した、損失の監視の基準を下げた回数（下の2つの必須の経路のケースは1）。
                    loss_monitoring_baseline_lowering_count=0,
                )
                if "run_client_settings" in cases_run_settings["run_participant_settings"]
                and "pending_model_registration_settings"
                in cases_run_settings["run_participant_settings"]["run_client_settings"]
                else {}
            ),
        )
        if name in (
            *golden.ADOPTION_WITHOUT_VALIDATION_PATH_CASE_NAMES,
            golden.WARM_STARTED_LEARNER_PATH_CASE_NAME,
        ):
            # 新しい既定のケースは、検証の判定が無く、切替えが採用より多い。
            cases[name]["coverage"].update(
                candidate_accepted_count=0,
                candidate_rejected_count=0,
                decision_reasons=[],
                training_model_switch_count=2,
            )
        if name in (
            "fedsda-sine2-adoption-without-validation",
            golden.WARM_STARTED_LEARNER_PATH_CASE_NAME,
        ):
            cases[name]["coverage"]["loss_monitoring_baseline_lowering_count"] = 1
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


def test_runs_are_reused_only_for_identical_full_run_settings(monkeypatch):
    """同じ完全なrun設定では1回だけ実行し、1つの項目でも違えば別に実行する。経路の記録は版ごとに組み立てる。"""
    monkeypatch.setattr(golden, "_REUSED_CASE_RUNS", {})
    executed = []
    real_execute = golden._execute_case_run

    def counting_execute(case_name, settings):
        executed.append(case_name)
        return real_execute(case_name, settings)

    monkeypatch.setattr(golden, "_execute_case_run", counting_execute)
    # FedDriftのケースは、版1と版2で、完全なrun設定が同じ。
    assert golden._plain(golden.build_case_settings("feddrift-drift-tail", 1)) == golden._plain(
        golden.build_case_settings("feddrift-drift-tail", 2)
    )
    first = golden.capture_case_reusing_runs("feddrift-drift-tail", 1)
    second = golden.capture_case_reusing_runs("feddrift-drift-tail", 2)
    assert executed == ["feddrift-drift-tail"]
    # 経路の記録は、版ごと（版2からの項目は、版2だけが持つ）。ほかの結果は同じ。
    assert "unregistered_adopted_model_count" not in first[0]["coverage"]
    assert "unregistered_adopted_model_count" in second[0]["coverage"]
    for section in ("run_metrics", "record_tables", "global_parameters"):
        assert first[0][section] == second[0][section], section
    # 毎回実行する関数と、同じ結果。
    assert golden.capture_case_and_count_divergence_events("feddrift-drift-tail", 2) == second
    # 1つの項目（seed）だけが違う設定は、別に実行する。
    real_build = golden.build_case_settings

    def build_with_other_seed(case_name, baseline_version=golden.BASELINE_VERSION):
        settings = real_build(case_name, baseline_version)
        if baseline_version != 3:
            return settings
        conditions = settings.execution_settings.experiment_run_conditions
        return replace(
            settings,
            execution_settings=replace(
                settings.execution_settings,
                experiment_run_conditions=replace(
                    conditions, random_seed=conditions.random_seed + 1
                ),
            ),
        )

    monkeypatch.setattr(golden, "build_case_settings", build_with_other_seed)
    # seedを変えた設定では、ケースの必須の経路を通るとは限らない（ここでは実行の回数だけを確かめる）。
    monkeypatch.setattr(golden, "assert_required_paths", lambda *arguments: None)
    monkeypatch.setattr(golden, "_REUSED_CASE_RUNS", {})
    executed.clear()
    golden.capture_case_reusing_runs("feddrift-drift-tail", 2)
    golden.capture_case_reusing_runs("feddrift-drift-tail", 3)
    assert executed == ["feddrift-drift-tail", "feddrift-drift-tail"]


# ケースの実行の結果（ケース名と基準版ごとに1回だけ実行して、数値の照合と、版の差の説明のtestで共有する）。
_CAPTURED_CASES = {}


def capture_case_once(case_name, baseline_version):
    key = (case_name, baseline_version)
    if key not in _CAPTURED_CASES:
        _CAPTURED_CASES[key] = golden.capture_case_reusing_runs(case_name, baseline_version)
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


# 版の差の説明のtestで、差が実際に起きることを確かめるケース（比較が空にならないことを保つ。2026-10-20に、
# 全ケースの結果で、経路を通ることを確かめて選んだ）。
_CASE_WITH_CHANGED_PREDICTION_RECORDS_IN_BASELINE_6 = "fedsda-sine2-events"
_CASE_WITH_REDUCED_DOWNLOADS_IN_BASELINE_8 = "fedsda-sine2-adoption-without-validation"
_CASE_WITH_CONSOLIDATION_IN_BASELINE_9 = "fedsda-sine2-adoption-without-validation"


@pytest.mark.slow
@pytest.mark.parametrize("case_name", golden.CASE_NAMES_BY_BASELINE_VERSION[5])
def test_baseline_6_differs_from_5_only_on_the_prediction_side(case_name):
    name = case_name
    previous = capture_case_once(name, 5)[0]
    current = capture_case_once(name, 6)[0]
    for section in ("global_parameters", "coverage"):
        golden.assert_case_matches(name, current[section], previous[section], section)
    if not name.startswith("fedsda") or name == "fedsda-single-model-prediction":
        for section in ("run_metrics", "record_tables"):
            golden.assert_case_matches(name, current[section], previous[section], section)
        return
    previous_tables = previous["record_tables"]
    current_tables = current["record_tables"]
    assert set(previous_tables) == set(current_tables), name
    changed_tables = {key for key in previous_tables if previous_tables[key] != current_tables[key]}
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
        key.startswith("prediction_diagnostic_metrics.") or key in _PREDICTION_SIDE_RUN_METRIC_PATHS
        for key in changed_metrics
    ), (name, changed_metrics)
    assert changed_tables or changed_metrics, name
    if name == _CASE_WITH_CHANGED_PREDICTION_RECORDS_IN_BASELINE_6:
        assert any(key.startswith("sample_prediction.") for key in changed_tables), name


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


def test_baseline_8_keeps_cases_and_changes_only_the_cross_evaluation_transfer_setting():
    """版8のケースは版7と同じ。設定は、FedSDAのケースのクロス評価のモデルの送り方だけが違う。"""
    assert 8 in golden.SUPPORTED_BASELINE_VERSIONS
    assert golden.CASE_NAMES_BY_BASELINE_VERSION[8] == golden.CASE_NAMES_BY_BASELINE_VERSION[7]
    for name in golden.CASE_NAMES_BY_BASELINE_VERSION[7]:
        previous = golden.comparable_run_settings(name, 7)
        current = deepcopy(golden.comparable_run_settings(name, 8))
        if "cross_evaluation_model_transfer_policy" in current["run_participant_settings"]:
            assert (
                current["run_participant_settings"].pop("cross_evaluation_model_transfer_policy")
                == "reuse_distributed_models_for_cross_evaluation"
            ), name
        golden.assert_case_matches(name, current, previous, "run_settings")


@pytest.mark.slow
@pytest.mark.parametrize("case_name", golden.CASE_NAMES_BY_BASELINE_VERSION[7])
def test_baseline_8_differs_from_7_only_in_communication_volume(case_name):
    """版7→8の差は通信量だけ。クロス評価が起きたFedSDAのケースは、下りのbyteが減る。"""
    name = case_name
    previous = capture_case_once(name, 7)[0]
    current = capture_case_once(name, 8)[0]
    for section in ("record_tables", "global_parameters", "coverage"):
        golden.assert_case_matches(name, current[section], previous[section], section)
    previous_metrics = _flatten_metric_paths(previous["run_metrics"])
    current_metrics = _flatten_metric_paths(current["run_metrics"])
    assert set(previous_metrics) == set(current_metrics), name
    changed_metrics = {
        key for key in previous_metrics if previous_metrics[key] != current_metrics[key]
    }
    assert all(key.startswith("communication_volume.") for key in changed_metrics), (
        name,
        changed_metrics,
    )
    cross_evaluated = any(
        key.startswith("client_cross_evaluation.")
        and previous["record_tables"][key]["shape"][0] > 0
        for key in previous["record_tables"]
    )
    if name == _CASE_WITH_REDUCED_DOWNLOADS_IN_BASELINE_8:
        assert cross_evaluated, name
    if not name.startswith("fedsda") or not cross_evaluated:
        assert not changed_metrics, (name, changed_metrics)
        return
    downloaded = "communication_volume.downloaded_byte_count"
    assert current_metrics[downloaded] < previous_metrics[downloaded], name


def test_baseline_9_keeps_cases_and_changes_only_the_consolidated_shared_feature_setting():
    """版9のケースは版8と同じ。設定は、FedSDAのケースの統合での共有部の扱いだけが違う。"""
    assert 9 in golden.SUPPORTED_BASELINE_VERSIONS
    assert golden.CASE_NAMES_BY_BASELINE_VERSION[9] == golden.CASE_NAMES_BY_BASELINE_VERSION[8]
    for name in golden.CASE_NAMES_BY_BASELINE_VERSION[8]:
        previous = golden.comparable_run_settings(name, 8)
        current = deepcopy(golden.comparable_run_settings(name, 9))
        if "consolidated_shared_feature_policy" in current["run_participant_settings"]:
            assert (
                current["run_participant_settings"].pop("consolidated_shared_feature_policy")
                == "keep_identical_shared_features"
            ), name
        golden.assert_case_matches(name, current, previous, "run_settings")


@pytest.mark.slow
@pytest.mark.parametrize("case_name", golden.CASE_NAMES_BY_BASELINE_VERSION[8])
def test_baseline_9_differs_from_8_only_where_consolidation_occurred(case_name):
    """版8→9: FedDrift・FedAvgと、統合が起きないFedSDAのケースは、全部が版8と同じ。統合が起きたケースは差があってよい。"""
    name = case_name
    previous = capture_case_once(name, 8)[0]
    current = capture_case_once(name, 9)[0]
    if name == _CASE_WITH_CONSOLIDATION_IN_BASELINE_9:
        assert previous["coverage"]["merged_model_count"] > 0, name
    if name.startswith("fedsda") and previous["coverage"]["merged_model_count"] > 0:
        return
    for section in ("run_metrics", "record_tables", "global_parameters", "coverage"):
        golden.assert_case_matches(name, current[section], previous[section], section)


# 版10で、警報の後の選び方が今の扱い（検出器の推定と許容量）のままのケース: FedDrift・FedAvgと、候補検証の
# 経路の4ケース（候補検証ありの構成）。
_BASELINE_10_CASES_KEEPING_BASELINE_9 = (
    *golden.CANDIDATE_VALIDATION_PATH_CASE_NAMES,
    "feddrift",
    "fedavg",
    "feddrift-drift-tail",
    "fedavg-no-complete-interval",
)
_ALARM_INTERVAL_ASSIGNMENT_COVERAGE_KEYS = (
    "selected_current_model_count",
    "selected_held_model_count",
    "selected_new_model_count",
)


def test_baseline_10_keeps_cases_and_changes_only_the_alarm_interval_assignment_setting():
    """版10のケースは版9と同じ。設定は、FedSDAのケースの警報の後の選び方だけが違う（候補検証ありの構成は今の扱い）。"""
    assert 10 in golden.SUPPORTED_BASELINE_VERSIONS
    assert golden.CASE_NAMES_BY_BASELINE_VERSION[10] == golden.CASE_NAMES_BY_BASELINE_VERSION[9]
    for name in golden.CASE_NAMES_BY_BASELINE_VERSION[9]:
        previous = golden.comparable_run_settings(name, 9)
        current = deepcopy(golden.comparable_run_settings(name, 10))
        client_settings = current["run_participant_settings"].get("run_client_settings", {})
        if "training_data_assignment_settings" in client_settings:
            policy = client_settings["training_data_assignment_settings"].pop(
                "alarm_interval_assignment_policy"
            )
            expected_policy = (
                "detector_change_span_and_mean_loss_increase_tolerance"
                if name in golden.CANDIDATE_VALIDATION_PATH_CASE_NAMES
                else "minimum_description_length"
            )
            assert policy == expected_policy, name
        golden.assert_case_matches(name, current, previous, "run_settings")


@pytest.mark.slow
@pytest.mark.parametrize("case_name", _BASELINE_10_CASES_KEEPING_BASELINE_9)
def test_baseline_10_keeps_baseline_9_where_the_alarm_interval_assignment_is_unchanged(case_name):
    """FedDrift・FedAvgと、候補検証ありの構成のケースは、版10でも、版9と同じ値。

    版10で足した選択の記録の表と経路の記録の項目は、取り除いてから比べ、空（どれも0）であることを確かめる。
    """
    previous = capture_case_once(case_name, 9)[0]
    current = deepcopy(capture_case_once(case_name, 10)[0])
    selection_tables = {
        name: snapshot
        for name, snapshot in current["record_tables"].items()
        if name.startswith(golden.ALARM_INTERVAL_ASSIGNMENT_SELECTION_TABLE_PREFIX)
    }
    if case_name.startswith("fedsda"):
        assert selection_tables, case_name
    assert all(snapshot["shape"][0] == 0 for snapshot in selection_tables.values()), case_name
    for name in selection_tables:
        del current["record_tables"][name]
    for key in _ALARM_INTERVAL_ASSIGNMENT_COVERAGE_KEYS:
        if key in current["coverage"]:
            assert current["coverage"].pop(key) == 0, (case_name, key)
    for section in ("run_metrics", "record_tables", "global_parameters", "coverage"):
        golden.assert_case_matches(case_name, current[section], previous[section], section)


_NEW_DEFAULT_SETTINGS_IN_BASELINE_11 = (
    (("run_client_settings", "training_data_assignment_settings"), "new_model_hypothesis_learners"),
    ((), "initial_model_loss_statistics_estimation"),
)
_BASELINE_11_COVERAGE_KEYS = (
    "selected_reinitialized_learner_count",
    "selected_warm_started_learner_count",
)


def test_baseline_11_adds_the_mnist4_case_and_changes_only_the_two_settings():
    """版11のケースは、版10のケースにMNIST4のケースを1つ足したもの。共通のケースの設定は、FedSDAのケースの
    2つの項目（学習者の組、初期モデルの統計の推定）だけが違う（候補検証ありの構成は今の扱い）。"""
    assert 11 in golden.SUPPORTED_BASELINE_VERSIONS
    assert golden.CASE_NAMES_BY_BASELINE_VERSION[11] == (
        *golden.CASE_NAMES_BY_BASELINE_VERSION[10],
        golden.WARM_STARTED_LEARNER_PATH_CASE_NAME,
    )
    for name in golden.CASE_NAMES_BY_BASELINE_VERSION[10]:
        previous = golden.comparable_run_settings(name, 10)
        current = deepcopy(golden.comparable_run_settings(name, 11))
        participant = current["run_participant_settings"]
        if "training_data_assignment_settings" in participant.get("run_client_settings", {}):
            values = []
            for path, field_name in _NEW_DEFAULT_SETTINGS_IN_BASELINE_11:
                holder = participant
                for key in path:
                    holder = holder[key]
                values.append(holder.pop(field_name))
            expected = (
                ["reinitialized_learner", "resubstitution"]
                if name in golden.CANDIDATE_VALIDATION_PATH_CASE_NAMES
                else ["reinitialized_and_warm_started_learners", "five_fold_cross_validation"]
            )
            assert values == expected, name
        golden.assert_case_matches(name, current, previous, "run_settings")
    mnist4 = golden.build_case_settings(golden.WARM_STARTED_LEARNER_PATH_CASE_NAME, 11)
    assert mnist4.run_participant_settings.initial_model_loss_statistics_estimation == (
        "five_fold_cross_validation"
    )


@pytest.mark.slow
@pytest.mark.parametrize("case_name", _BASELINE_10_CASES_KEEPING_BASELINE_9)
def test_baseline_11_keeps_baseline_10_where_the_two_settings_are_unchanged(case_name):
    """FedDrift・FedAvgと、候補検証ありの構成のケースは、版11でも、版10と同じ値。

    版11で足した選択の記録の列と経路の記録の項目は、取り除いてから比べる（列は空、項目はどれも0）。
    """
    previous = capture_case_once(case_name, 10)[0]
    current = deepcopy(capture_case_once(case_name, 11)[0])
    column = current["record_tables"].pop(golden.NEW_MODEL_LEARNER_NAME_TABLE_NAME, None)
    if case_name.startswith("fedsda"):
        assert column is not None and column["shape"][0] == 0, case_name
    for key in _BASELINE_11_COVERAGE_KEYS:
        if key in current["coverage"]:
            assert current["coverage"].pop(key) == 0, (case_name, key)
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


def test_baseline_12_keeps_cases_and_changes_only_the_client_selection_setting():
    assert 12 in golden.SUPPORTED_BASELINE_VERSIONS
    assert golden.CASE_NAMES_BY_BASELINE_VERSION[12] == golden.CASE_NAMES_BY_BASELINE_VERSION[11]
    for case_name in golden.CASE_NAMES_BY_BASELINE_VERSION[11]:
        previous = golden._plain(golden.build_case_settings(case_name, 11))
        current = golden._plain(golden.build_case_settings(case_name, 12))
        if (
            case_name.startswith("fedsda")
            and case_name not in golden.CANDIDATE_VALIDATION_PATH_CASE_NAMES
        ):
            assert (
                previous["run_participant_settings"].pop("cross_evaluation_client_selection_policy")
                == "clients_holding_target_model"
            )
            assert (
                current["run_participant_settings"].pop("cross_evaluation_client_selection_policy")
                == "clients_with_target_model_samples"
            )
        assert current == previous, case_name


@pytest.mark.parametrize("case_name", golden.CASE_NAMES_BY_BASELINE_VERSION[11])
@pytest.mark.slow
def test_baseline_12_runs_both_versions_without_goldens_and_keeps_unaffected_results(case_name):
    """goldenが未配置でも、版11と12をその場で実行し、経路と、影響を受けない構成の値を守る。"""
    previous = capture_case_once(case_name, 11)[0]
    current = deepcopy(capture_case_once(case_name, 12)[0])
    for key in ("cross_evaluation_record_count", "zero_sample_cross_evaluation_count"):
        current["coverage"].pop(key, None)
    if (
        not case_name.startswith("fedsda")
        or case_name in golden.CANDIDATE_VALIDATION_PATH_CASE_NAMES
    ):
        for section in ("run_metrics", "record_tables", "global_parameters", "coverage"):
            golden.assert_case_matches(case_name, current[section], previous[section], section)


@pytest.mark.slow
def test_baseline_11_really_contains_zero_sample_cross_evaluations():
    """0件を除く検証が空振りでないことを、版11の実行の記録から数えて確かめる。"""
    assert any(
        capture_case_once(case_name, 11)[1]["zero_sample_cross_evaluation_count"] > 0
        for case_name in golden.CASE_NAMES_BY_BASELINE_VERSION[11]
        if case_name.startswith("fedsda")
    )


def test_baseline_13_keeps_cases_and_changes_only_baseline_policy_and_client_limit():
    assert 13 in golden.SUPPORTED_BASELINE_VERSIONS
    assert golden.CASE_NAMES_BY_BASELINE_VERSION[13] == golden.CASE_NAMES_BY_BASELINE_VERSION[12]
    for version in range(1, 14):
        for case_name in golden.CASE_NAMES_BY_BASELINE_VERSION[version]:
            settings = golden.build_case_settings(case_name, version)
            if not case_name.startswith("fedsda"):
                continue
            participant = settings.run_participant_settings
            candidate_validation = case_name in golden.CANDIDATE_VALIDATION_PATH_CASE_NAMES
            assert participant.maximum_evaluating_client_count_per_model == (
                3
                if version <= 12 or candidate_validation
                else settings.execution_settings.experiment_run_conditions.client_count
            ), (version, case_name)
            policy = participant.run_client_settings.loss_change_detection_settings.loss_monitoring_baseline_update_policy
            assert policy == (
                "keep_baseline_until_alarm"
                if version <= 12 or candidate_validation or "adwin" in case_name
                else "lower_baseline_on_loss_decrease"
            ), (version, case_name)
    for case_name in golden.CASE_NAMES_BY_BASELINE_VERSION[12]:
        previous = golden._plain(golden.build_case_settings(case_name, 12))
        current = golden._plain(golden.build_case_settings(case_name, 13))
        if (
            case_name.startswith("fedsda")
            and case_name not in golden.CANDIDATE_VALIDATION_PATH_CASE_NAMES
        ):
            for mapping in (previous, current):
                participant = mapping["run_participant_settings"]
                participant.pop("maximum_evaluating_client_count_per_model")
                participant["run_client_settings"]["loss_change_detection_settings"].pop(
                    "loss_monitoring_baseline_update_policy"
                )
        assert current == previous, case_name


@pytest.mark.parametrize("case_name", golden.CASE_NAMES_BY_BASELINE_VERSION[12])
@pytest.mark.slow
def test_baseline_13_runs_both_versions_without_goldens_and_keeps_unaffected_results(case_name):
    previous = capture_case_once(case_name, 12)[0]
    current = deepcopy(capture_case_once(case_name, 13)[0])
    if (
        not case_name.startswith("fedsda")
        or case_name in golden.CANDIDATE_VALIDATION_PATH_CASE_NAMES
    ):
        current["coverage"].pop("loss_monitoring_baseline_lowering_count", None)
        if case_name.startswith("fedsda"):
            assert current["run_metrics"].pop("loss_monitoring_baseline_lowering_count") == 0
            counts = current["run_metrics"]["loss_monitoring_computation_counts"]
            assert counts.pop("decrease_detector_component_update_count") == 0
            assert counts.pop("decrease_detector_evaluated_candidate_bet_count") == 0
            for name in tuple(current["record_tables"]):
                if name.startswith("loss_monitoring_baseline_lowering."):
                    assert current["record_tables"].pop(name)["shape"] == [0]
        for section in ("run_metrics", "record_tables", "global_parameters", "coverage"):
            golden.assert_case_matches(case_name, current[section], previous[section], section)


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


def test_parallel_scope_keeps_tests_of_one_golden_case_in_one_process(request):
    """並列の実行で、ケースの名前を持つ全部のtestが、そのケースの組へ入る。同じケースは同じ組、重いケースは別々の組。"""
    from conftest import GOLDEN_CASE_NAMES_IN_OWN_GROUP, select_parallel_test_scope

    all_case_names = golden.ALL_CASE_NAMES
    scopes_by_case_name = {}
    for test_item in request.session.items:
        if test_item.path.name != "test_important_behavior_regression_goldens.py":
            continue
        test_scope = select_parallel_test_scope(
            test_node_id=test_item.nodeid, golden_case_names=all_case_names
        )
        call_specification = getattr(test_item, "callspec", None)
        if call_specification is None or "case_name" not in call_specification.params:
            assert "::" not in test_scope, test_item.nodeid
            continue
        case_name = call_specification.params["case_name"]
        assert "::golden-case-group-" in test_scope, test_item.nodeid
        scopes_by_case_name.setdefault(case_name, set()).add(test_scope)
    assert all(len(scopes) == 1 for scopes in scopes_by_case_name.values())
    # 組の割り当ての全体は、全部のケースの名前で確かめる（`-k`などで一部のtestだけを選んだときも同じ）。
    file_path = "tests/refactoring/test_important_behavior_regression_goldens.py"
    scope_by_case_name = {
        name: select_parallel_test_scope(
            test_node_id=f"{file_path}::test[9-{name}]", golden_case_names=all_case_names
        )
        for name in all_case_names
    }
    for name, scopes in scopes_by_case_name.items():
        assert scopes == {scope_by_case_name[name]}, name
    heavy_scopes = [scope_by_case_name[name] for name in GOLDEN_CASE_NAMES_IN_OWN_GROUP]
    assert len(set(heavy_scopes)) == len(GOLDEN_CASE_NAMES_IN_OWN_GROUP)
    other_scopes = {
        scope
        for name, scope in scope_by_case_name.items()
        if name not in GOLDEN_CASE_NAMES_IN_OWN_GROUP
    }
    assert len(other_scopes) == 2 and not other_scopes & set(heavy_scopes)
