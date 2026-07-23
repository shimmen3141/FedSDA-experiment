import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from federated_drift_experiment import config, run_random_drift_experiment


def test_experiment_records_compute_and_model_telemetry(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATASET", "blobs")
    monkeypatch.setattr(config, "TOTAL_DATA_POINTS", 100)
    monkeypatch.setattr(config, "N_CLIENTS", 2)
    monkeypatch.setattr(config, "PRETRAIN_SAMPLES", 20)
    monkeypatch.setattr(config, "AGGREGATION_INTERVAL", 20)
    monkeypatch.setattr(config, "CLIENT_BATCH_SIZE", 10)
    raw_path = tmp_path / "telemetry.npz"

    results = run_random_drift_experiment(
        mode="Oblivious",
        random_seed=0,
        verbose=False,
        show_plot=False,
        raw_path=str(raw_path),
    )

    assert results["compute_prediction_examples_total"] == 200
    assert results["compute_optimizer_steps_total"] > 0
    assert results["compute_model_examples_total"] >= 200
    assert results["mean_model_count"] == 1.0
    assert results["max_model_count"] == 1.0
    assert results["client_compute_seconds_sum"] >= 0.0

    with np.load(raw_path) as raw:
        assert raw["round_global_model_count"].shape == (5,)
        assert raw["round_client_held_model_count"].shape == (5, 2)
        assert raw["round_client_prediction_examples"].shape == (5, 2)
        assert raw["round_client_optimizer_steps"].sum() == results[
            "compute_optimizer_steps_total"
        ]
        assert np.all(raw["round_global_model_count"] == 1.0)
        assert raw["parameter_schema_version"].item() == 1
        assert raw["aggregation_interval"].item() == 20
        assert raw["feddrift_detection_batch_size"].item() == -1
        assert np.isnan(raw["fedsda_distance_threshold"].item())
        assert np.isnan(raw["feddrift_distance_threshold"].item())
