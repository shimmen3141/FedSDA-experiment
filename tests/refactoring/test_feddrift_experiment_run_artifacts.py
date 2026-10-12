"""単一runの実旧対応、表の全列、保存失敗と新形式roundtripを検証する。"""

from copy import deepcopy
from dataclasses import replace
from math import isnan

import pytest

from federated_learning_experiments.runtime import feddrift_experiment_run as execution
from federated_learning_experiments.runtime.feddrift_run_settings import FeddriftRunSettings
from tests.refactoring.test_feddrift_run_client import assert_rng_equal, rng_snapshot
from tests.refactoring.test_feddrift_run_metric_derivation import (
    METRIC_SETTINGS,
    basic_metric_values,
)
from tests.refactoring.test_feddrift_server_synchronization import assert_server_equal
from tests.refactoring.test_feddrift_stream_finalization import (
    assert_terminal_values_equal,
    terminal_snapshot,
)
from tests.refactoring.test_feddrift_stream_run_execution import run_inputs, run_legacy


@pytest.mark.parametrize("count", (2, 11))
def test_save_load_roundtrip_and_nonfinite_summary(monkeypatch, tmp_path, count):
    import json

    import numpy as np

    from federated_learning_experiments.core.settings_serialization import (
        convert_settings_to_plain_mapping,
    )
    from federated_learning_experiments.runtime.feddrift_run_artifact_files import (
        load_feddrift_run_artifacts,
        save_feddrift_experiment_run,
    )
    from federated_learning_experiments.runtime.feddrift_run_record_tables import (
        build_feddrift_run_record_tables,
    )

    result = execution.execute_feddrift_experiment_run(
        run_settings=make_run_settings(monkeypatch, count=count)
    )
    before, rng = (
        deepcopy(terminal_snapshot(result.measured_run.participants.run_server)),
        rng_snapshot(),
    )
    paths = save_feddrift_experiment_run(
        experiment_run=result, output_directory=tmp_path, file_stem="test-run"
    )
    summary, tables = load_feddrift_run_artifacts(summary_file_path=paths.summary_file_path)
    assert {p.name for p in tmp_path.iterdir()} == {"test-run.json", "test-run.npz"}
    assert (
        summary["method_name"] == "feddrift"
        and type(summary["format_version"]) is int
        and summary["format_version"] == 1
    )
    assert summary["record_table_file_name"] == "test-run.npz"
    assert summary["run_settings"] == json.loads(
        json.dumps(convert_settings_to_plain_mapping(result.run_settings))
    )
    assert summary["run_metrics"] == json.loads(
        json.dumps(
            convert_settings_to_plain_mapping(result.run_metrics, non_finite_float_as_none=True)
        )
    )
    assert set(summary["environment"]) == {"system", "machine", "python", "numpy", "torch"}
    expected = build_feddrift_run_record_tables(experiment_run=result)
    assert tables.keys() == expected.keys()
    for key, value in expected.items():
        np.testing.assert_array_equal(tables[key], value)
        assert tables[key].dtype == value.dtype
    assert_terminal_values_equal(
        terminal_snapshot(result.measured_run.participants.run_server), before
    )
    assert_rng_equal(rng)
    if count == 2:
        assert summary["run_metrics"]["stable_period_prediction_accuracy"] is None


@pytest.mark.parametrize(
    "invalid", ("", " ", ".", "..", "a/b", "a\\b", "a:b", " spaced ", None, True)
)
def test_save_rejects_invalid_name_without_writing(artifact_run, tmp_path, invalid):
    from federated_learning_experiments.runtime.feddrift_run_artifact_files import (
        save_feddrift_experiment_run,
    )

    with pytest.raises((TypeError, ValueError)):
        save_feddrift_experiment_run(
            experiment_run=artifact_run, output_directory=tmp_path, file_stem=invalid
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("suffix", ("json", "npz"))
def test_existing_output_is_preserved(artifact_run, tmp_path, suffix):
    from federated_learning_experiments.runtime.feddrift_run_artifact_files import (
        save_feddrift_experiment_run,
    )

    existing = tmp_path / f"result.{suffix}"
    existing.write_bytes(b"existing data")
    with pytest.raises(FileExistsError):
        save_feddrift_experiment_run(
            experiment_run=artifact_run, output_directory=tmp_path, file_stem="result"
        )
    assert existing.read_bytes() == b"existing data"
    assert list(tmp_path.iterdir()) == [existing]


@pytest.mark.parametrize("invalid", ("wrong_result", "wrong_directory", "missing_directory"))
def test_save_rejects_invalid_result_or_directory(artifact_run, tmp_path, invalid):
    from federated_learning_experiments.runtime.feddrift_run_artifact_files import (
        save_feddrift_experiment_run,
    )

    with pytest.raises((TypeError, FileNotFoundError)):
        save_feddrift_experiment_run(
            experiment_run=None if invalid == "wrong_result" else artifact_run,
            output_directory="directory"
            if invalid == "wrong_directory"
            else tmp_path / "missing"
            if invalid == "missing_directory"
            else tmp_path,
            file_stem="result",
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "stage", ("table_build", "npz_write", "json_write", "npz_replace", "json_replace")
)
def test_save_failure_cleans_its_files(monkeypatch, artifact_run, tmp_path, stage):
    from pathlib import Path

    from federated_learning_experiments.runtime import feddrift_run_artifact_files as artifacts

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    if stage == "table_build":
        monkeypatch.setattr(artifacts, "build_feddrift_run_record_tables", fail, raising=False)
    elif stage == "npz_write":
        monkeypatch.setattr(artifacts, "savez_compressed", fail, raising=False)
    elif stage == "json_write":
        monkeypatch.setattr(Path, "write_text", fail)
    else:
        original = getattr(artifacts, "replace", None)

        def fail_selected(source, destination):
            if destination.suffix == (".npz" if stage == "npz_replace" else ".json"):
                fail()
            return original(source, destination)

        monkeypatch.setattr(artifacts, "replace", fail_selected, raising=False)
    with pytest.raises(OSError, match="injected failure"):
        artifacts.save_feddrift_experiment_run(
            experiment_run=artifact_run, output_directory=tmp_path, file_stem="result"
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "change",
    (
        {"format_version": True},
        {"format_version": 2},
        {"method_name": "fedsda"},
        {"record_table_file_name": "../escape.npz"},
        {"record_table_file_name": "..\\escape.npz"},
        {"record_table_file_name": "C:escape.npz"},
        {"record_table_file_name": "x.json"},
        {"record_table_file_name": ""},
    ),
)
def test_loader_rejects_wrong_version_method_or_path(tmp_path, change):
    import json

    from federated_learning_experiments.runtime.feddrift_run_artifact_files import (
        load_feddrift_run_artifacts,
    )

    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            dict(
                format_version=1, method_name="feddrift", record_table_file_name="result.npz", **{}
            )
            | change
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_feddrift_run_artifacts(summary_file_path=path)


@pytest.mark.parametrize("invalid", ("nonobject", "missing", "corrupt", "pickle", "path_type"))
def test_loader_rejects_unreadable_table_or_summary(tmp_path, invalid):
    import json

    import numpy as np

    from federated_learning_experiments.runtime.feddrift_run_artifact_files import (
        load_feddrift_run_artifacts,
    )

    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            []
            if invalid == "nonobject"
            else dict(format_version=1, method_name="feddrift", record_table_file_name="result.npz")
        ),
        encoding="utf-8",
    )
    if invalid == "corrupt":
        (tmp_path / "result.npz").write_bytes(b"broken")
    elif invalid == "pickle":
        np.savez(tmp_path / "result.npz", object=np.array([object()], dtype=object))
    with pytest.raises((TypeError, ValueError, FileNotFoundError, OSError)):
        load_feddrift_run_artifacts(summary_file_path=str(path) if invalid == "path_type" else path)


@pytest.mark.parametrize("count,threshold", ((2, 1.0), (11, 0.0), (17, 0.03)))
def test_record_tables_match_actual_legacy_and_owner_fields(monkeypatch, count, threshold):
    from dataclasses import fields

    import numpy as np

    from federated_drift_experiment.data import extract_true_drift_events
    from federated_learning_experiments.evaluation.adaptation_record_store import AdaptationRecord
    from federated_learning_experiments.evaluation.model_clustering_record_store import (
        ModelClusteringObservation,
        ModelPairClusteringObservation,
    )
    from federated_learning_experiments.learning.models.model_computation_measurement import (
        ModelComputationCounts,
    )
    from federated_learning_experiments.runtime.feddrift_detection_batch_processing import (
        FeddriftDetectionBatchTransition,
    )
    from federated_learning_experiments.runtime.feddrift_run_record_tables import (
        build_feddrift_run_record_tables,
    )

    settings = make_run_settings(monkeypatch, count=count, threshold=threshold)
    old = run_legacy(monkeypatch, settings.execution_settings, threshold)
    result = execution.execute_feddrift_experiment_run(run_settings=settings)
    server = result.measured_run.participants.run_server
    before, rng = deepcopy(terminal_snapshot(server)), rng_snapshot()
    tables = build_feddrift_run_record_tables(experiment_run=result)
    assert_terminal_values_equal(terminal_snapshot(server), before)
    assert_rng_equal(rng)
    old_server, old_clients = old["participants"]
    old_changes = extract_true_drift_events(old["concepts"])
    expected_changes = [
        (client_id, index) for client_id, indices in old_changes.items() for index in indices
    ]
    assert (
        list(
            zip(
                tables["concept_change.client_id"].tolist(),
                tables["concept_change.first_new_concept_sample_index"].tolist(),
                strict=True,
            )
        )
        == expected_changes
    )
    if count >= 11:
        assert expected_changes
    for column, old_name in (
        ("prediction_is_correct", "history_accuracy"),
        ("prediction_model_id", "history_model_id"),
        ("observed_concept_id", "history_concept"),
    ):
        np.testing.assert_array_equal(
            tables[f"sample_prediction.{column}"], [getattr(c, old_name) for c in old_clients]
        )
    processed = result.measured_run.run_result.processed_sample_count_per_client
    client_count = len(old_clients)
    assert tables["sample_prediction.sample_index"].shape == (client_count, processed)
    assert tables["server_timestep.client_held_model_count_after_synchronization"].shape == (
        count // 3,
        client_count,
    )
    np.testing.assert_array_equal(
        tables["training_model_switch.switch_sample_index"],
        [v for c in old_clients for v in c.local_switch_positions],
    )
    np.testing.assert_array_equal(
        tables["model_registration.model_id"],
        [r.model_id for r in old_server.model_lineage.registrations],
    )
    np.testing.assert_array_equal(
        tables["model_registration.registered_server_timestep_index"],
        [r.round_index for r in old_server.model_lineage.registrations],
    )
    for table, kind, records in (
        (
            "model_clustering",
            ModelClusteringObservation,
            server.model_clustering_record_store.snapshot_model_clustering_observations(),
        ),
        (
            "model_pair_clustering",
            ModelPairClusteringObservation,
            server.model_clustering_record_store.snapshot_pair_clustering_observations(),
        ),
        (
            "detection_batch",
            FeddriftDetectionBatchTransition,
            tuple(
                r
                for c in server.run_clients
                for r in c.stream_processor.completed_detection_batch_transitions
            ),
        ),
        (
            "adaptation",
            AdaptationRecord,
            tuple(
                r
                for c in server.run_clients
                for r in c.owners.adaptation_record_store.get_state_snapshot().adaptation_records
            ),
        ),
    ):
        expected_columns = {
            f.name if f.name != "round_index" else "server_timestep_index" for f in fields(kind)
        }
        actual_columns = {key.split(".")[1] for key in tables if key.startswith(table + ".")}
        if table in ("adaptation", "detection_batch"):
            expected_columns.add("client_id")
        assert actual_columns == expected_columns
        for field in fields(kind):
            column = field.name if field.name != "round_index" else "server_timestep_index"
            values = [getattr(record, field.name) for record in records]
            array = tables[f"{table}.{column}"]
            expected = [
                np.nan
                if value is None and array.dtype.kind == "f"
                else -1
                if value is None
                else value
                for value in values
            ]
            np.testing.assert_array_equal(array, expected)
    for phase, counts in (
        ("preparation", result.measured_run.preparation_model_computation_counts),
        ("finalization", result.measured_run.finalization_model_computation_counts),
    ):
        for field in fields(ModelComputationCounts):
            assert tables[f"{phase}_model_computation.{field.name}"].tolist() == [
                getattr(counts, field.name)
            ]
    for phase, attribute in (
        ("sample_processing", "sample_processing_model_computation_counts"),
        ("synchronization", "synchronization_model_computation_counts"),
    ):
        for field in fields(ModelComputationCounts):
            assert tables[f"server_timestep.{phase}_{field.name}"].tolist() == [
                getattr(getattr(r, attribute), field.name)
                for r in result.measured_run.timestep_model_computation_counts
            ]
    for key, array in tables.items():
        assert array.dtype.kind in "ibfU", (key, array.dtype)
        assert array.ndim == (
            2
            if key.startswith("sample_prediction.")
            or key == "server_timestep.client_held_model_count_after_synchronization"
            else 1
        )
        if array.dtype.kind == "i":
            assert array.dtype == np.dtype("int64")
        if array.dtype.kind == "f":
            assert array.dtype == np.dtype("float64")

    clients = server.run_clients
    for column in ("sample_index", "predicted_class_id", "observed_class_id"):
        np.testing.assert_array_equal(
            tables[f"sample_prediction.{column}"],
            [[getattr(r, column) for r in c.stream_processor.sample_predictions] for c in clients],
        )
    np.testing.assert_array_equal(
        tables["sample_prediction.held_model_count_at_prediction"],
        [
            c.stream_processor.held_model_count_record_store.snapshot_held_model_counts()
            for c in clients
        ],
    )
    amounts = result.run_metrics.model_diagnostic_metrics.client_held_model_training_amounts
    for column in (
        "client_id",
        "model_id",
        "stored_training_sample_count",
        "trained_sample_count",
        "parameter_update_step_count",
    ):
        assert tables[f"client_held_model.{column}"].tolist() == [
            getattr(r, column) for r in amounts
        ]
    for column, attribute in (
        ("global_model_count_after_synchronization", "global_model_count"),
        ("client_held_model_count_after_synchronization", "client_held_model_counts"),
        ("timestep_index", "timestep_index"),
    ):
        expected = np.asarray(
            [
                getattr(r, attribute)
                for r in result.run_metrics.model_diagnostic_metrics.synchronized_timestep_model_counts
            ],
            dtype=np.int64,
        )
        if column == "client_held_model_count_after_synchronization":
            expected = expected.reshape(count // 3, client_count)
        np.testing.assert_array_equal(tables[f"server_timestep.{column}"], expected)
    registrations = old_server.model_lineage.registrations
    assert tables["model_registration.registering_client_id"].tolist() == [
        r.client_id for r in registrations
    ]
    assert tables["model_registration.is_final_global_model"].tolist() == [
        r.model_id in server.global_model_repository.global_model_ids for r in registrations
    ]
    crosses = server.cross_evaluation_record_store.snapshot_client_cross_evaluation_records()
    for column, attribute in (
        ("server_timestep_index", "round_index"),
        ("client_id", "client_id"),
        ("candidate_model_id", "candidate_model_id"),
        ("target_model_id", "target_model_id"),
        ("evaluated_sample_count", "evaluated_sample_count"),
        ("bounded_loss_sum", "bounded_loss_sum"),
        ("squared_bounded_loss_sum", "squared_bounded_loss_sum"),
    ):
        np.testing.assert_array_equal(
            tables[f"client_cross_evaluation.{column}"], [getattr(r, attribute) for r in crosses]
        )
    for column in (
        "candidate_only_correct_sample_count",
        "target_only_correct_sample_count",
        "both_correct_sample_count",
        "both_wrong_sample_count",
    ):
        assert tables[f"client_cross_evaluation.{column}"].tolist() == [-1] * len(crosses)
        assert tables[f"client_cross_evaluation_class.{column}"].shape == (0,)
    # 全ての表のスキーマを固定する。追加も欠落もこの契約の変更として検出する。
    assert {key.split(".")[0] for key in tables} == {
        "sample_prediction",
        "concept_change",
        "training_model_switch",
        "detection_batch",
        "adaptation",
        "model_registration",
        "model_clustering",
        "model_pair_clustering",
        "client_cross_evaluation",
        "client_cross_evaluation_class",
        "client_held_model",
        "server_timestep",
        "preparation_model_computation",
        "finalization_model_computation",
    }


@pytest.fixture(scope="module")
def artifact_run():
    with pytest.MonkeyPatch.context() as patch:
        return execution.execute_feddrift_experiment_run(run_settings=make_run_settings(patch))


def test_temporary_prediction_id_is_preserved(artifact_run):

    from federated_learning_experiments.runtime.feddrift_run_record_tables import (
        build_feddrift_run_record_tables,
    )

    result = deepcopy(artifact_run)
    client = result.measured_run.participants.run_clients[0]
    client.stream_processor._predictions[0] = replace(
        client.stream_processor._predictions[0], prediction_model_id=-105
    )
    tables = build_feddrift_run_record_tables(experiment_run=result)
    assert tables["sample_prediction.prediction_model_id"][0, 0] == -105


def test_unprocessed_tail_concept_changes_are_saved(monkeypatch):
    from federated_drift_experiment.data import extract_true_drift_events
    from federated_learning_experiments.runtime.feddrift_run_record_tables import (
        build_feddrift_run_record_tables,
    )

    settings = make_run_settings(monkeypatch, count=2)
    settings = replace(
        settings,
        execution_settings=replace(
            settings.execution_settings,
            concept_schedule_settings=replace(
                settings.execution_settings.concept_schedule_settings,
                minimum_sample_index_gap_before_change_trial=0,
                per_eligible_sample_concept_change_probability=1.0,
            ),
        ),
    )
    result = execution.execute_feddrift_experiment_run(run_settings=settings)
    tables = build_feddrift_run_record_tables(experiment_run=result)
    assert result.measured_run.run_result.processed_sample_count_per_client == 0
    assert tables["sample_prediction.prediction_is_correct"].shape == (2, 0)
    expected = extract_true_drift_events(
        [
            trace.concept_ids_by_sample_index
            for trace in result.measured_run.run_result.evaluation_concept_traces
        ]
    )
    assert expected == {0: [1], 1: [1]}
    assert tables["concept_change.client_id"].tolist() == [0, 1]
    assert tables["concept_change.first_new_concept_sample_index"].tolist() == [1, 1]
    assert result.run_metrics.training_model_switch_detection_metrics.concept_change_count == 2


@pytest.mark.parametrize("invalid", ("settings_type", "stale_metrics"))
def test_record_tables_reject_inconsistent_summary(artifact_run, invalid):
    from federated_learning_experiments.runtime.feddrift_run_record_tables import (
        build_feddrift_run_record_tables,
    )

    result = (
        replace(artifact_run, run_settings=None)
        if invalid == "settings_type"
        else replace(
            artifact_run, run_metrics=replace(artifact_run.run_metrics, prediction_accuracy=-1.0)
        )
    )
    before, rng = (
        deepcopy(terminal_snapshot(artifact_run.measured_run.participants.run_server)),
        rng_snapshot(),
    )
    with pytest.raises((TypeError, ValueError)):
        build_feddrift_run_record_tables(experiment_run=result)
    assert_terminal_values_equal(
        terminal_snapshot(artifact_run.measured_run.participants.run_server), before
    )
    assert_rng_equal(rng)


def make_run_settings(monkeypatch, dataset="sine2", count=11, threshold=0.03):
    participants, stream = run_inputs(monkeypatch, dataset, "sgd", count, threshold)
    return FeddriftRunSettings(
        execution_settings=stream,
        run_participant_settings=participants,
        run_metric_settings=METRIC_SETTINGS,
    )


@pytest.mark.parametrize(
    "dataset,count", (("sine2", 11), ("sea4", 12), ("mnist2", 11), ("sine2", 2))
)
def test_complete_run_matches_actual_legacy(monkeypatch, dataset, count):
    settings = make_run_settings(monkeypatch, dataset, count)
    old = run_legacy(monkeypatch, settings.execution_settings, 0.03)
    before = rng_snapshot()
    result = execution.execute_feddrift_experiment_run(run_settings=settings)
    assert result.run_settings is settings
    assert (
        result.run_metrics.model_computation_counts == result.measured_run.model_computation_counts
    )
    for key, value in basic_metric_values(result.run_metrics).items():
        if isinstance(value, float) and isnan(value):
            assert isnan(old["metrics"][key])
        else:
            assert value == old["metrics"][key], key
    assert_server_equal(old["participants"][0], result.measured_run.participants.run_server)
    assert_rng_equal(before)


@pytest.mark.parametrize(
    "invalid",
    (
        "outer",
        "execution",
        "participants",
        "metrics",
        "metric_value",
        "threshold",
        "batch_interval",
        "seed",
    ),
)
def test_invalid_settings_rejected_before_execution(monkeypatch, invalid):
    settings = deepcopy(make_run_settings(monkeypatch))
    if invalid == "outer":
        settings = None
    elif invalid in ("execution", "participants", "metrics"):
        field = {
            "execution": "execution_settings",
            "participants": "run_participant_settings",
            "metrics": "run_metric_settings",
        }[invalid]
        object.__setattr__(settings, field, None)
    elif invalid == "metric_value":
        object.__setattr__(
            settings.run_metric_settings, "maximum_detection_delay_sample_count", True
        )
    elif invalid == "threshold":
        object.__setattr__(
            settings.run_participant_settings.run_client_settings, "loss_increase_threshold", -1.0
        )
    else:
        field = (
            "random_seed" if invalid == "seed" else "server_aggregation_interval_per_client_samples"
        )
        object.__setattr__(
            settings.execution_settings.experiment_run_conditions,
            field,
            True if invalid == "seed" else 4,
        )
    calls = []
    monkeypatch.setattr(
        execution,
        "execute_feddrift_stream_protocol_run_with_computation_measurement",
        lambda **kwargs: calls.append(kwargs),
        raising=False,
    )
    before = rng_snapshot()
    with pytest.raises((TypeError, ValueError)):
        execution.execute_feddrift_experiment_run(run_settings=settings)
    assert calls == []
    assert_rng_equal(before)


def test_constructor_validates_batch_correspondence(monkeypatch):
    settings = make_run_settings(monkeypatch)
    with pytest.raises(ValueError):
        replace(
            settings,
            execution_settings=replace(
                settings.execution_settings,
                experiment_run_conditions=replace(
                    settings.execution_settings.experiment_run_conditions,
                    server_aggregation_interval_per_client_samples=4,
                ),
            ),
        )
