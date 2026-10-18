"""研究用介入の専用検証。pytestへパスを明示して実行する。"""

import importlib.util
import json
import random
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights import (
    FixedShareWeightController,
)
from federated_learning_experiments.methods.fedsda.prediction_combination.prediction_combination_settings import (
    PredictionCombinationSettings,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_run_record_tables import (
    build_fedsda_run_record_tables,
)
from federated_learning_experiments.sweep.experiment_configurations import (
    build_experiment_run_settings,
)


def _settings(scale=120):
    return PredictionCombinationSettings(
        prediction_combination_strategy="fixed_share_weighted_prediction",
        prediction_mixture_activation_policy="always",
        prediction_weight_recalibration_after_aggregation_policy="recompute_buffer_losses_and_replay_weight_updates",
        prediction_state_reset_on_training_assignment_change_policy="restart_adahedge_preserve_fixed_share_prediction_state",
        prediction_model_parameter_source="mixture_of_synchronized_and_local_parameters",
        fixed_share_weight_redistribution_time_scale_samples=scale,
    )


@pytest.fixture
def diagnostic():
    path = Path(__file__).with_name("run.py")
    spec = importlib.util.spec_from_file_location("_initial_mass_checks", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode,new_mass", [("ratio-preserved", 1 / 3), ("small-new-mass", 1 / 360)])
def test_added_id_preserves_old_ratios_and_uses_declared_mass(diagnostic, mode, new_mass):
    controller = FixedShareWeightController(prediction_combination_settings=_settings(120))
    controller._weights_by_model_id = {0: 0.8, 1: 0.2}
    controller._cumulative_observed_loss_variance = 7.0
    events = []
    original = FixedShareWeightController._synchronize_model_pool
    with diagnostic._prediction_mass_policy(mode, events):
        weights = controller.get_weights_before_label_observation(model_ids=[2, 1, 0])
        assert weights == pytest.approx(
            {0: (1 - new_mass) * 0.8, 1: (1 - new_mass) * 0.2, 2: new_mass}
        )
        assert controller.model_pool_reset_count == 1
        assert controller.cumulative_observed_loss_variance == 0
        assert len(events) == 1
        assert events[0]["old_weights"] == {0: 0.8, 1: 0.2}
        assert events[0]["new_weights"] == weights
        assert events[0]["new_model_ids"] == [2]
        assert events[0]["time_scale"] == 120
    assert FixedShareWeightController._synchronize_model_pool is original


@pytest.mark.parametrize(
    "old,ids",
    [
        ({}, [0, 1]),
        ({0: 0.8, 1: 0.2}, [0, 1]),
        ({0: 0.8, 1: 0.2}, [0]),
        ({0: 1.0}, [1, 2]),
        ({0: 0.0, 1: 1.0}, [0, 2]),
    ],
)
def test_fallback_matches_original_state(diagnostic, old, ids):
    a = FixedShareWeightController(prediction_combination_settings=_settings())
    b = FixedShareWeightController(prediction_combination_settings=_settings())
    for c in (a, b):
        c._weights_by_model_id = dict(old)
        c._cumulative_observed_loss_variance = 3.0
    expected = b.get_weights_before_label_observation(model_ids=ids)
    events = []
    with diagnostic._prediction_mass_policy("small-new-mass", events):
        assert a.get_weights_before_label_observation(model_ids=ids) == expected
    assert a.__dict__ == b.__dict__
    assert events == []


@pytest.mark.parametrize("scale", [2, 30, 120])
def test_multiple_additions_and_removal(diagnostic, scale):
    c = FixedShareWeightController(prediction_combination_settings=_settings(scale))
    c._weights_by_model_id = {0: 0.6, 1: 0.3, 2: 0.1}
    with diagnostic._prediction_mass_policy("small-new-mass", []):
        w = c.get_weights_before_label_observation(model_ids=[0, 3, 4])
    assert w == pytest.approx({0: 1 - 2 / (3 * scale), 3: 1 / (3 * scale), 4: 1 / (3 * scale)})
    assert sum(w.values()) == pytest.approx(1)


def test_restores_on_exception_without_rng_changes(diagnostic):
    original = FixedShareWeightController._synchronize_model_pool
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    with pytest.raises(RuntimeError, match="expected"):
        with diagnostic._prediction_mass_policy("small-new-mass", []):
            c = FixedShareWeightController(prediction_combination_settings=_settings())
            c.get_weights_before_label_observation(model_ids=[0])
            c.get_weights_before_label_observation(model_ids=[0, 1])
            raise RuntimeError("expected")
    assert FixedShareWeightController._synchronize_model_pool is original
    assert random.getstate() == state[0]
    now = np.random.get_state()
    np.testing.assert_array_equal(now[1], state[1][1])
    assert now[0] == state[1][0] and now[2:] == state[1][2:]
    assert torch.equal(torch.get_rng_state(), state[2])


@pytest.mark.parametrize(
    "mode,cfg", [("validation", "fedsda"), ("immediate", "fedsda-without-candidate-validation")]
)
def test_control_matches_every_array_metrics_and_rng(diagnostic, mode, cfg):
    settings = build_experiment_run_settings(
        configuration_name=cfg,
        experiment_run_conditions=ExperimentRunConditions(
            dataset_name="sea2",
            random_seed=2,
            client_count=2,
            per_client_sample_count=100,
            server_aggregation_interval_per_client_samples=50,
        ),
    )
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    actual, events = diagnostic._run_with_policy(settings, mode)
    end = diagnostic._rng_fingerprint()
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    expected = execute_fedsda_experiment_run(run_settings=settings)
    assert diagnostic._rng_fingerprint() == end
    assert events == []
    assert convert_settings_to_plain_mapping(
        actual.run_metrics, non_finite_float_as_none=True
    ) == convert_settings_to_plain_mapping(expected.run_metrics, non_finite_float_as_none=True)
    a = build_fedsda_run_record_tables(experiment_run=actual)
    b = build_fedsda_run_record_tables(experiment_run=expected)
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_plan_is_read_only_and_rejects_existing_reservation(diagnostic, tmp_path, capsys):
    output = tmp_path / "diagnostic"
    argv = [
        "--cases",
        "sea2:50",
        "--seeds",
        "0",
        "--sample-count",
        "100",
        "--workers",
        "1",
        "--output-directory",
        str(output),
        "--purpose",
        "control",
        "--print-plan",
    ]
    assert diagnostic._main(argv) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["jobs"]) == 4
    assert not output.exists()
    output.mkdir()
    (output / plan["jobs"][0]["run_name"]).mkdir()
    with pytest.raises(FileExistsError):
        diagnostic._main(argv)


@pytest.mark.parametrize(
    "case,count", [("unknown:50", "100"), ("sea2:0", "100"), ("sea2:50", "101")]
)
def test_invalid_plan_does_not_create_outputs(diagnostic, tmp_path, case, count):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        diagnostic._main(
            [
                "--cases",
                case,
                "--sample-count",
                count,
                "--output-directory",
                str(output),
                "--purpose",
                "test",
                "--print-plan",
            ]
        )
    assert not output.exists()


def test_worker_reserves_output_before_training_and_writes_provenance(
    diagnostic, tmp_path, capsys, monkeypatch
):
    output = tmp_path / "worker"
    diagnostic._main(
        [
            "--cases",
            "sea2:50",
            "--seeds",
            "0",
            "--sample-count",
            "100",
            "--modes",
            "small-new-mass",
            "--output-directory",
            str(output),
            "--purpose",
            "worker integration",
            "--print-plan",
        ]
    )
    job = json.loads(capsys.readouterr().out)["jobs"][0]
    output.mkdir()
    diagnostic._execute_job(job)
    directory = output / job["run_name"]
    payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    assert payload["format"] == "prediction-initial-mass-diagnostic-v1"
    assert payload["final_global_parameters"]
    assert payload["processed_sample_count"] == 1000
    assert payload["mode"] == "small-new-mass"
    assert payload["purpose"] == "worker integration"
    assert payload["source_tree_sha256"] == job["code_identity"]["source_tree_sha256"]
    assert payload["rng_before_sha256"] == payload["rng_after_sha256"]
    assert payload["environment"]["torch_threads"] == 1
    import hashlib

    npz = directory / payload["record_table_file_name"]
    assert hashlib.sha256(npz.read_bytes()).hexdigest() == payload["record_table_sha256"]
    before = npz.read_bytes()
    monkeypatch.setattr(
        diagnostic, "_run_with_policy", Mock(side_effect=AssertionError("must not train"))
    )
    with pytest.raises(FileExistsError):
        diagnostic._execute_job(job)
    assert npz.read_bytes() == before


def test_failed_worker_retains_reservation_and_no_completion(
    diagnostic, tmp_path, capsys, monkeypatch
):
    output = tmp_path / "failed"
    diagnostic._main(
        [
            "--cases",
            "sea2:50",
            "--seeds",
            "0",
            "--sample-count",
            "100",
            "--modes",
            "validation",
            "--output-directory",
            str(output),
            "--purpose",
            "failure protection",
            "--print-plan",
        ]
    )
    job = json.loads(capsys.readouterr().out)["jobs"][0]
    output.mkdir()
    monkeypatch.setattr(
        diagnostic, "_run_with_policy", Mock(side_effect=RuntimeError("failed run"))
    )
    with pytest.raises(RuntimeError, match="failed run"):
        diagnostic._execute_job(job)
    directory = output / job["run_name"]
    assert directory.is_dir()
    assert not (directory / "result.json").exists()
