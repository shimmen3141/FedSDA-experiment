import pytest
import torch

from federated_drift_experiment import config
from federated_drift_experiment.clients import (
    ClassConditionalHDDMAFedSDAClient,
    HDDMFedSDAClient,
)
from federated_drift_experiment.experiment import MODE_SPECS
from federated_drift_experiment.hddm import HDDMA, HDDMW
from federated_drift_experiment.models import SimpleMLP
from federated_drift_experiment.servers import FedSDACachedServer, FedSDANoCachedServer


@pytest.mark.parametrize("detector", [HDDMA(), HDDMW()])
def test_hddm_detects_bounded_loss_increase(detector):
    for _ in range(300):
        detector.update(0.1)
        assert not detector.drift_detected

    for _ in range(200):
        detector.update(0.9)
        if detector.drift_detected:
            break

    assert detector.drift_detected
    assert 1 <= detector.width <= 200


@pytest.mark.parametrize("detector", [HDDMA(), HDDMW()])
def test_hddm_stays_quiet_on_constant_loss(detector):
    for _ in range(1000):
        detector.update(0.2)
    assert not detector.drift_detected


def test_hddm_rejects_unbounded_input():
    with pytest.raises(ValueError):
        HDDMA().update(1.1)
    with pytest.raises(ValueError):
        HDDMW().update(-0.1)


def test_hddm_modes_reuse_both_server_protocols():
    expected = {
        "FedSDA_NoCached_HDDMA": (FedSDANoCachedServer, "A"),
        "FedSDA_NoCached_HDDMW": (FedSDANoCachedServer, "W"),
        "FedSDA_Cached_HDDMA": (FedSDACachedServer, "A"),
        "FedSDA_Cached_HDDMW": (FedSDACachedServer, "W"),
    }
    for mode, (server_cls, variant) in expected.items():
        spec = MODE_SPECS[mode]
        assert spec.client_cls is HDDMFedSDAClient
        assert spec.server_cls is server_cls
        assert spec.client_kwargs == {"hddm_variant": variant}

    assert MODE_SPECS["FedSDA_NoCached_ClassHDDMA"].client_cls is (
        ClassConditionalHDDMAFedSDAClient
    )
    assert MODE_SPECS["FedSDA_Cached_ClassHDDMA"].server_cls is FedSDACachedServer


def test_hddm_client_disables_adwin_forced_check():
    client = HDDMFedSDAClient(
        client_id=0,
        initial_models={0: SimpleMLP()},
        initial_stats={0: {"n": 100, "mean": 0.2, "M2": 1.0}},
        hddm_variant="W",
        verbose=False,
    )
    assert not hasattr(client, "adwin")
    assert hasattr(client, "hddm")
    assert not client._forced_drift_check(100)


def test_class_hddma_detects_class_local_change_hidden_in_overall_mean():
    client = ClassConditionalHDDMAFedSDAClient(
        client_id=0,
        initial_models={0: SimpleMLP()},
        initial_stats={0: {"n": 100, "mean": 0.5, "M2": 1.0}},
        verbose=False,
    )

    detected = None
    for sample_idx in range(1200):
        class_id = sample_idx % 2
        before_drift = sample_idx < 600
        error = float(class_id) if before_drift else float(1 - class_id)
        y = torch.tensor([[float(class_id)]])
        if client._update_drift_detectors(error, y, sample_idx):
            detected = sample_idx
            break

    assert detected is not None
    assert not client.hddm.drift_detected
    assert any(detector.drift_detected for detector in client.class_hddms.values())
    assert client._class_drift_start == 600


def test_class_hddma_applies_configured_confidence_to_each_detector():
    client = ClassConditionalHDDMAFedSDAClient(
        client_id=0,
        initial_models={0: SimpleMLP()},
        initial_stats={0: {"n": 100, "mean": 0.5, "M2": 1.0}},
        verbose=False,
    )

    assert client.component_drift_confidence == config.HDDM_DRIFT_CONFIDENCE
    assert client.component_warning_confidence == config.HDDM_WARNING_CONFIDENCE
    assert client.hddm.drift_confidence == config.HDDM_DRIFT_CONFIDENCE
    assert client.hddm.warning_confidence == config.HDDM_WARNING_CONFIDENCE
