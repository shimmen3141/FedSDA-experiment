import pytest

from federated_drift_experiment.clients import HDDMFedSDAClient
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
