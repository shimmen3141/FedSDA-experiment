"""別processの実旧で再現したLinux指標・最終parameterを固定して照合する。"""

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from federated_learning_experiments.runtime.feddrift_run_metric_derivation import (
    derive_feddrift_run_metrics,
)
from federated_learning_experiments.runtime.feddrift_run_participant_factory import (
    FeddriftRunParticipantFactory,
)
from federated_learning_experiments.runtime.feddrift_stream_run_execution import (
    execute_feddrift_stream_protocol_run,
)
from tests.refactoring.test_feddrift_run_metric_derivation import (
    METRIC_SETTINGS,
    basic_metric_values,
)
from tests.refactoring.test_feddrift_stream_run_execution import run_inputs


@pytest.mark.parametrize("case_name", ("sine2", "sea4", "mnist2", "sine2_drift"))
def test_linux_full_run_core_metrics_and_parameters_match_fixed_legacy(monkeypatch, case_name):
    golden = json.loads(
        Path(__file__)
        .with_name("feddrift_regression_goldens")
        .joinpath("linux-x86_64-python3.14.4-numpy2.4.6-torch2.12.1+cpu.json")
        .read_text(encoding="utf-8")
    )
    if (
        sys.platform,
        platform.machine(),
        platform.python_version(),
        np.__version__,
        torch.__version__,
    ) != ("linux", "x86_64", golden["python"], golden["numpy"], golden["torch"]):
        pytest.skip("固定Linux CPU基準環境専用。実旧全体run対照は各環境で別に実行する。")
    settings, run_settings = (
        run_inputs(
            monkeypatch, "sine2", count=17, threshold=0.03, rounds=3, seed=71, probability=1.0
        )
        if case_name == "sine2_drift"
        else run_inputs(monkeypatch, case_name)
    )
    factory = FeddriftRunParticipantFactory(run_participant_settings=settings)
    result = execute_feddrift_stream_protocol_run(
        execution_settings=run_settings, participant_factory=factory
    )
    participants = factory.prepared_run_participants
    metrics = derive_feddrift_run_metrics(
        run_result=result, participants=participants, run_metric_settings=METRIC_SETTINGS
    )
    expected = golden["cases"][case_name]
    assert basic_metric_values(metrics) == expected["metrics"]
    if case_name == "sine2_drift":
        assert metrics.training_model_switch_detection_metrics.detection_count > 0
        assert metrics.training_model_switch_detection_metrics.detection_precision > 0.0
        assert metrics.training_model_switch_detection_metrics.detection_recall > 0.0
    repository = participants.run_server.global_model_repository
    values = [
        (
            mid,
            [
                tensor.tolist()
                for tensor in repository.get_global_model_parameters(model_id=mid).values()
            ],
        )
        for mid in repository.global_model_ids
    ]
    actual_hash = hashlib.sha256(
        json.dumps(values, separators=(",", ":"), allow_nan=True).encode()
    ).hexdigest()
    assert actual_hash == expected["global_parameter_sha256"]
