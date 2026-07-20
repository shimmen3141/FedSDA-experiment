import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from federated_drift_experiment import config
from federated_drift_experiment.clients import (
    ClassConditionalEDetectorFedSDAClient,
    EDetectorFedSDAClient,
    FedSDAClient,
)
from federated_drift_experiment.e_detector import BoundedMeanEDetector
from federated_drift_experiment.experiment import MODE_SPECS
from federated_drift_experiment.models import SimpleMLP
from federated_drift_experiment.servers import FedSDACachedServer, FedSDANoCachedServer


def test_bounded_mean_e_detector_detects_upward_shift_and_returns_split():
    detector = BoundedMeanEDetector(baseline=0.2, alpha=0.001)

    for _ in range(200):
        detector.update(0.1)
        assert not detector.drift_detected

    for _ in range(20):
        detector.update(0.8)
        if detector.drift_detected:
            break

    assert detector.drift_detected
    assert detector.e_value >= 1.0 / detector.alpha
    assert 1 <= detector.width <= 20


def test_bounded_mean_e_detector_stays_quiet_below_baseline():
    detector = BoundedMeanEDetector(baseline=0.2, alpha=0.001, max_candidates=500)
    for _ in range(400):
        detector.update(0.1)
    assert not detector.drift_detected


def test_e_detector_modes_reuse_server_flows_without_changing_existing_modes():
    assert MODE_SPECS["FedSDA_NoCached_ADWIN"].client_cls is FedSDAClient
    assert MODE_SPECS["FedSDA_Cached_ADWIN"].client_cls is FedSDAClient
    assert MODE_SPECS["FedSDA_NoCached_ESR"].client_cls is EDetectorFedSDAClient
    assert MODE_SPECS["FedSDA_Cached_ESR"].client_cls is EDetectorFedSDAClient
    assert MODE_SPECS["FedSDA_NoCached_ESR"].server_cls is FedSDANoCachedServer
    assert MODE_SPECS["FedSDA_Cached_ESR"].server_cls is FedSDACachedServer


def test_e_detector_client_disables_uncontrolled_forced_check():
    client = EDetectorFedSDAClient(
        client_id=0,
        initial_models={0: SimpleMLP()},
        initial_stats={0: {"n": 100, "mean": 0.2, "M2": 1.0}},
        verbose=False,
    )
    assert not client._forced_drift_check(100)


def test_forced_check_can_be_disabled_without_changing_default(monkeypatch):
    client = FedSDAClient(
        client_id=0,
        initial_models={0: SimpleMLP()},
        initial_stats={0: {"n": 100, "mean": 0.2, "M2": 1.0}},
        verbose=False,
    )
    monkeypatch.setattr(config, "FEDSDA_ENABLE_FORCED_DRIFT_CHECK", False)
    assert not client._forced_drift_check(100)


def test_class_conditional_e_detector_finds_class_local_increase():
    client = ClassConditionalEDetectorFedSDAClient(
        client_id=0,
        initial_models={0: SimpleMLP()},
        initial_stats={0: {"n": 100, "mean": 0.6, "M2": 1.0}},
        verbose=False,
    )

    detected = None
    for sample_idx in range(800):
        class_id = sample_idx % 2
        if sample_idx < 400:
            error = 0.1 if class_id == 0 else 0.5
        else:
            error = 1.0 if class_id == 0 else 0.0
        y = torch.tensor([[float(class_id)]])
        if client._update_drift_detectors(error, y, sample_idx):
            detected = sample_idx
            break

    assert detected is not None
    assert detected >= 400
    assert client._class_drift_start is not None
    assert client._class_drift_start >= 400


def test_class_conditional_e_detector_modes_reuse_protocol_servers():
    assert MODE_SPECS["FedSDA_NoCached_ClassESR"].client_cls is ClassConditionalEDetectorFedSDAClient
    assert MODE_SPECS["FedSDA_Cached_ClassESR"].client_cls is ClassConditionalEDetectorFedSDAClient
    assert MODE_SPECS["FedSDA_NoCached_ClassESR"].server_cls is FedSDANoCachedServer
    assert MODE_SPECS["FedSDA_Cached_ClassESR"].server_cls is FedSDACachedServer
