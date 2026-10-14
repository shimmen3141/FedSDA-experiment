"""runの来歴（コードの版と、期待する動作の基準版）の値と変換。"""

import pytest
from behavior_golden_cases import BASELINE_VERSION, SUPPORTED_BASELINE_VERSIONS

from federated_learning_experiments.core.run_provenance import (
    BEHAVIOR_BASELINE_VERSION,
    RunProvenance,
    convert_run_provenance_to_mapping,
    read_run_provenance,
)

COMMITTED_RUN_PROVENANCE = RunProvenance(
    code_commit="0123456789abcdef0123456789abcdef01234567",
    has_uncommitted_changes=False,
    code_revision_unavailable_reason=None,
    behavior_baseline_version=BEHAVIOR_BASELINE_VERSION,
)
UNAVAILABLE_RUN_PROVENANCE = RunProvenance(
    code_commit=None,
    has_uncommitted_changes=None,
    code_revision_unavailable_reason="not a git work tree: somewhere",
    behavior_baseline_version=BEHAVIOR_BASELINE_VERSION,
)


def test_behavior_baseline_version_matches_current_golden_version():
    """goldenの版を上げたら、実行のコードの基準版も上げる（runに残る番号がずれないように）。"""
    assert BEHAVIOR_BASELINE_VERSION == BASELINE_VERSION == SUPPORTED_BASELINE_VERSIONS[-1]


@pytest.mark.parametrize("run_provenance", [COMMITTED_RUN_PROVENANCE, UNAVAILABLE_RUN_PROVENANCE])
def test_run_provenance_round_trips_through_mapping(run_provenance):
    run_provenance_mapping = convert_run_provenance_to_mapping(run_provenance)
    assert list(run_provenance_mapping) == [
        "code_commit",
        "has_uncommitted_changes",
        "code_revision_unavailable_reason",
        "behavior_baseline_version",
    ]
    assert read_run_provenance(run_provenance_mapping) == run_provenance


@pytest.mark.parametrize(
    "changed_fields,error_type",
    [
        (dict(behavior_baseline_version=True), TypeError),
        (dict(behavior_baseline_version=5.0), TypeError),
        (dict(behavior_baseline_version=0), ValueError),
        (dict(code_commit=""), ValueError),
        (dict(code_commit=b"abc"), TypeError),
        (dict(has_uncommitted_changes=None), TypeError),
        (dict(has_uncommitted_changes=0), TypeError),
        (dict(code_revision_unavailable_reason="reason"), ValueError),
        (dict(code_commit=None), ValueError),
        (dict(code_commit=None, has_uncommitted_changes=None), TypeError),
        (
            dict(
                code_commit=None,
                has_uncommitted_changes=False,
                code_revision_unavailable_reason="r",
            ),
            ValueError,
        ),
        (
            dict(
                code_commit=None, has_uncommitted_changes=None, code_revision_unavailable_reason=""
            ),
            ValueError,
        ),
    ],
)
def test_run_provenance_rejects_invalid_combinations(changed_fields, error_type):
    fields = convert_run_provenance_to_mapping(COMMITTED_RUN_PROVENANCE) | changed_fields
    with pytest.raises(error_type):
        RunProvenance(**fields)


@pytest.mark.parametrize(
    "run_provenance_mapping,error_type",
    [
        (None, TypeError),
        ([], TypeError),
        ({}, ValueError),
        (
            {
                key: value
                for key, value in convert_run_provenance_to_mapping(
                    COMMITTED_RUN_PROVENANCE
                ).items()
                if key != "behavior_baseline_version"
            },
            ValueError,
        ),
        (convert_run_provenance_to_mapping(COMMITTED_RUN_PROVENANCE) | {"extra": 1}, ValueError),
        (
            convert_run_provenance_to_mapping(COMMITTED_RUN_PROVENANCE)
            | {"behavior_baseline_version": "5"},
            TypeError,
        ),
    ],
)
def test_read_run_provenance_rejects_invalid_mappings(run_provenance_mapping, error_type):
    with pytest.raises(error_type):
        read_run_provenance(run_provenance_mapping)


def test_convert_run_provenance_rejects_other_types():
    with pytest.raises(TypeError):
        convert_run_provenance_to_mapping(
            convert_run_provenance_to_mapping(COMMITTED_RUN_PROVENANCE)
        )
