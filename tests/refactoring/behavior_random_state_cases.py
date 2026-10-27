"""候補検証を使わない版14のrunの終端で、借用乱数源の全状態を採取する。"""

import json
from pathlib import Path
from unittest.mock import patch

import behavior_golden_cases as golden
import torch

import federated_learning_experiments.runtime.single_run_execution as execution
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)

RANDOM_STATE_FIXTURE_PATH = (
    Path(__file__).parent / "golden" / "fedsda_without_candidate_validation_random_states.json"
)
RANDOM_STATE_CASE_NAMES = tuple(
    name
    for name in golden.CASE_NAMES_BY_BASELINE_VERSION[14]
    if name.startswith("fedsda") and name not in golden.CANDIDATE_VALIDATION_PATH_CASE_NAMES
)


def capture_run_end_random_states(case_name):
    """隔離torch状態の復元前、全clientと通信の終端処理の後に採取する。"""
    sources = []
    captured = []
    real_create = execution.create_run_random_sources
    real_intervals = execution.run_stream_protocol_intervals

    def create(**arguments):
        result = real_create(**arguments)
        sources.append(result)
        return result

    def intervals(**arguments):
        result = real_intervals(**arguments)
        assert len(sources) == 1
        python_state = sources[0].python_random_generator.getstate()
        numpy_state = sources[0].numpy_random_generator.get_state()
        captured.append(
            dict(
                python_random=[python_state[0], list(python_state[1]), python_state[2]],
                numpy_random_state=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                torch_cpu=torch.get_rng_state().tolist(),
            )
        )
        return result

    with (
        patch.object(execution, "create_run_random_sources", create),
        patch.object(execution, "run_stream_protocol_intervals", intervals),
    ):
        execute_fedsda_experiment_run(run_settings=golden.build_case_settings(case_name, 14))
    assert len(captured) == 1
    return captured[0]


def generate_random_state_fixture():
    """移設前のHEADとsrc無変更を検査し、既存fixtureを上書きせず生成する。"""
    import subprocess

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert commit == "6adb3ee77ae57af4bcef612916fc6389d075ab80"
    assert not subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src"], text=True
    ).strip()
    document = dict(
        source_commit=commit,
        baseline_version=14,
        environment=golden.numerical_environment(),
        mnist_sha256=golden.mnist_sha256(),
        capture_point="after_run_stream_protocol_intervals_before_torch_state_restoration",
        cases={name: capture_run_end_random_states(name) for name in RANDOM_STATE_CASE_NAMES},
    )
    with RANDOM_STATE_FIXTURE_PATH.open("x", encoding="utf-8") as output:
        json.dump(document, output, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        output.write("\n")


if __name__ == "__main__":
    generate_random_state_fixture()
