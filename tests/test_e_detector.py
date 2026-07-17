import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from federated_drift_experiment.clients import EDetectorFedSDAClient, FedSDAClient
from federated_drift_experiment.drift_detectors import BoundedMeanEDetector
from federated_drift_experiment.experiment import MODE_SPECS
from federated_drift_experiment.models import SimpleMLP
from federated_drift_experiment.servers import FedSDAV2Server, FedSDAV3Server


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
    assert MODE_SPECS["FedSDA_v2"].client_cls is FedSDAClient
    assert MODE_SPECS["FedSDA_v3"].client_cls is FedSDAClient
    assert MODE_SPECS["FedSDA_v2.3"].client_cls is EDetectorFedSDAClient
    assert MODE_SPECS["FedSDA_v3.3"].client_cls is EDetectorFedSDAClient
    assert MODE_SPECS["FedSDA_v2.3"].server_cls is FedSDAV2Server
    assert MODE_SPECS["FedSDA_v3.3"].server_cls is FedSDAV3Server


def test_e_detector_client_disables_uncontrolled_forced_check():
    client = EDetectorFedSDAClient(
        client_id=0,
        initial_models={0: SimpleMLP()},
        initial_stats={0: {"n": 100, "mean": 0.2, "M2": 1.0}},
        verbose=False,
    )
    assert not client._forced_drift_check(100)
