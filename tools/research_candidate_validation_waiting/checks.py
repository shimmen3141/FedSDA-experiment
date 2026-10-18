"""待機後強制採用の境界・通常実行との一致・研究成果物の保護を確認する。"""

import importlib.util
import json
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import federated_learning_experiments.runtime.post_alarm_candidate_validation_progress as progress
from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.core.settings_serialization import (
    convert_settings_to_plain_mapping,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_collection import (
    PostAlarmCandidateLossCollection,
)
from federated_learning_experiments.methods.fedsda.candidate_model_selection.post_alarm_candidate_loss_evaluation import (
    PostAlarmCandidateLossEvaluation,
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


# 専用検証だけで既存test helperを参照する。共通設定には経路を追加しない。
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "refactoring"))


@pytest.fixture
def diagnostic():
    path = Path(__file__).resolve().parents[2] / "run_candidate_validation_waiting_diagnostic.py"
    spec = importlib.util.spec_from_file_location("_waiting_diagnostic_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("accepted,reusable", [(False, None), (False, 9), (True, None)])
def test_forced_evaluation_retains_original_and_restores(
    diagnostic, monkeypatch, accepted, reusable
):
    original = PostAlarmCandidateLossEvaluation(
        comparison_reference_model_id=4,
        reusable_reference_model_id=reusable,
        candidate_accepted=accepted,
        decision_reason="original_reason",
        validation_sample_count=10,
        candidate_full_interval_mean_loss=0.6,
        reference_full_interval_mean_loss=0.3,
        candidate_second_segment_mean_loss=0.7,
        reference_second_segment_mean_loss=0.2,
        reference_historical_mean_loss=0.1,
    )
    operation = Mock(return_value=original)
    monkeypatch.setattr(progress, "evaluate_candidate_using_post_alarm_losses", operation)
    events = []
    rng = diagnostic._rng_fingerprint()
    with pytest.raises(RuntimeError, match="test failure"):
        with diagnostic._force_adoption_after_wait(events):
            result = progress.evaluate_candidate_using_post_alarm_losses(marker=42)
            assert result == replace(
                original,
                candidate_accepted=True,
                reusable_reference_model_id=None,
                decision_reason="diagnostic_forced_adoption_after_validation_wait",
            )
            assert events == [
                {"original_evaluation": asdict(original), "applied_evaluation": asdict(result)}
            ]
            assert diagnostic._rng_fingerprint() == rng
            raise RuntimeError("test failure")
    assert progress.evaluate_candidate_using_post_alarm_losses is operation
    operation.assert_called_once_with(marker=42)


def test_rng_fingerprint_includes_gaussian_cache(diagnostic):
    before = np.random.get_state()
    try:
        a = diagnostic._rng_fingerprint()
        state = list(before)
        state[3] = 1 - state[3]
        np.random.set_state(tuple(state))
        assert diagnostic._rng_fingerprint() != a
        state = list(before)
        state[4] += 0.5
        np.random.set_state(tuple(state))
        assert diagnostic._rng_fingerprint() != a
    finally:
        np.random.set_state(before)


def test_nine_samples_wait_and_tenth_forces_real_adoption(diagnostic, monkeypatch):
    from test_post_alarm_candidate_validation_progress import (
        build_validation_progress_oracle,
        make_acceptance_settings,
        set_scripted_validation_losses,
    )

    with torch.random.fork_rng(devices=[]):
        args, _, _, legacy, _ = build_validation_progress_oracle(
            class_count=2, legacy_resolution_case="reuse", monkeypatch=monkeypatch
        )
        settings = make_acceptance_settings(sample_count=10)
        session = args["validation_session"]
        collection = PostAlarmCandidateLossCollection(
            candidate_model_training_and_acceptance_settings=settings,
            proposal_sample_index=53,
            reference_model_ids=(4, 9),
        )
        args.update(
            candidate_model_training_and_acceptance_settings=settings,
            validation_session=replace(session, post_alarm_candidate_loss_collection=collection),
        )
        set_scripted_validation_losses(
            progress_arguments=args, legacy_client=legacy, monkeypatch=monkeypatch
        )
        events = []
        with diagnostic._force_adoption_after_wait(events):
            for n in range(1, 10):
                result = progress.advance_post_alarm_candidate_validation(
                    **dict(args, sample_index=53 + n)
                )
                assert result.completed_validation is None
                assert events == []
            result = progress.advance_post_alarm_candidate_validation(**dict(args, sample_index=63))
        assert len(events) == 1
        assert events[0]["original_evaluation"]["candidate_accepted"] is False
        assert (
            result.completed_validation.validation_resolution.resolution_outcome
            == "candidate_adopted_as_new_model"
        )
        assert result.completed_validation.decision_record.resolution_sample_index == 63


def test_incomplete_session_uses_original_finalization(diagnostic, monkeypatch):
    from test_incomplete_post_alarm_candidate_validation_finalization import (
        build_incomplete_validation_finalization_oracle,
    )

    from federated_learning_experiments.runtime.incomplete_post_alarm_candidate_validation_finalization import (
        finalize_incomplete_post_alarm_candidate_validation,
    )

    with torch.random.fork_rng(devices=[]):
        args, *_ = build_incomplete_validation_finalization_oracle(
            class_count=2, monkeypatch=monkeypatch, observed_validation_sample_count=2
        )
        events = []
        with diagnostic._force_adoption_after_wait(events):
            result = finalize_incomplete_post_alarm_candidate_validation(**args)
        assert events == []
        assert result.decision_record.validation_sample_count == 2


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
    assert len(plan["jobs"]) == 3
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
            "delayed-forced",
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
    assert payload["format"] == "candidate-validation-waiting-diagnostic-v1"
    assert payload["processed_sample_count"] == 1000
    assert payload["mode"] == "delayed-forced"
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
