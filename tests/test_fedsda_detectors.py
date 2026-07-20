import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from federated_drift_experiment.clients import ClassConditionalFedSDAClient, FedSDAClient
from federated_drift_experiment.experiment import MODE_SPECS
from federated_drift_experiment.models import SimpleMLP
from federated_drift_experiment.servers import FedSDACachedServer, FedSDANoCachedServer


def _make_client():
    return ClassConditionalFedSDAClient(
        client_id=0,
        initial_models={0: SimpleMLP()},
        initial_stats={0: {"n": 10, "mean": 0.1, "M2": 0.0}},
        verbose=False,
    )


def test_class_adwin_detects_change_hidden_in_overall_loss():
    client = _make_client()
    detected = None

    # 全体では常に誤差0/1が半数ずつだが、各正解クラス内では誤差が反転する。
    for sample_idx in range(800):
        class_id = sample_idx % 2
        before_drift = sample_idx < 400
        error = float(class_id) if before_drift else float(1 - class_id)
        y = torch.tensor([[float(class_id)]])
        if client._update_drift_detectors(error, y, sample_idx):
            detected = sample_idx
            break

    assert detected is not None
    assert not client.adwin.drift_detected
    assert any(detector.drift_detected for detector in client.class_adwins.values())
    assert client._class_drift_start == 400


def test_class_adwin_reuses_no_cached_server():
    assert MODE_SPECS["FedSDA_NoCached_ADWIN"].client_cls is FedSDAClient
    assert MODE_SPECS["FedSDA_NoCached_ClassADWIN"].client_cls is ClassConditionalFedSDAClient
    assert MODE_SPECS["FedSDA_NoCached_ClassADWIN"].server_cls is FedSDANoCachedServer


def test_cached_class_adwin_reuses_class_conditional_client():
    assert MODE_SPECS["FedSDA_Cached_ADWIN"].client_cls is FedSDAClient
    assert MODE_SPECS["FedSDA_Cached_ClassADWIN"].client_cls is ClassConditionalFedSDAClient
    assert MODE_SPECS["FedSDA_Cached_ClassADWIN"].server_cls is FedSDACachedServer
