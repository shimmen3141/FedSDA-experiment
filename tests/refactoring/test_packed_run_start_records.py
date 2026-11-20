"""runの開始の記録の読み方（個別のファイル→zip）と、掃引の終わりのまとめ。"""

import json
import zipfile

import pytest
from run_provenance_examples import TEST_RUN_PROVENANCE

from federated_learning_experiments.sweep import experiment_sweep_execution as execution
from federated_learning_experiments.sweep import experiment_sweep_resumption as resumption
from federated_learning_experiments.sweep.experiment_sweep_plan import (
    ExperimentSweepPlan,
    build_planned_experiment_runs,
)

ARCHIVE = "run_start_records.zip"


def _runs(seeds=(0,)):
    return build_planned_experiment_runs(
        sweep_plan=ExperimentSweepPlan(
            configuration_names=("fedavg",),
            dataset_names=("circle2",),
            random_seeds=seeds,
            client_count=2,
            per_client_sample_count=15,
            server_aggregation_intervals_per_client_samples=(5,),
        )
    )


def _record_path(root, run):
    return root / ".resume" / run.configuration_name / f"{run.run_identifier}.json"


def _move_records_into_zip(root, runs, *, mutate=None):
    """個別の開始の記録を、本番と同じLZMAのzipへ手で移す（2.1のtestはpackの前に書く）。"""
    with zipfile.ZipFile(root / ".resume" / ARCHIVE, "w", compression=zipfile.ZIP_LZMA) as archive:
        for run in runs:
            path = _record_path(root, run)
            data = path.read_bytes()
            if mutate is not None:
                data = mutate(data)
            archive.writestr(f"{run.configuration_name}/{run.run_identifier}.json", data)
            path.unlink()


def _orphan(root, run):
    path = root / run.configuration_name / "runs" / f"{run.run_identifier}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"interrupted artifact")
    return path


def _tree(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != ".sweep.lock"
    }


# ---- 2.1 読み方 ----


def test_an_interrupted_run_is_recovered_with_the_record_in_the_zip(tmp_path):
    run = _runs()[0]
    resumption.write_run_start_record(tmp_path, run, TEST_RUN_PROVENANCE)
    _move_records_into_zip(tmp_path, [run])
    orphan = _orphan(tmp_path, run)
    assert resumption.can_recover_interrupted_run(tmp_path, run, TEST_RUN_PROVENANCE)
    outcomes = execution.execute_experiment_sweep(
        planned_runs=(run,),
        output_directory=tmp_path,
        worker_count=1,
        run_provenance=TEST_RUN_PROVENANCE,
    )
    assert outcomes[0].run_status == "completed"
    assert orphan.read_bytes() != b"interrupted artifact"
    backups = list((tmp_path / ".resume" / "recovered").rglob("*.npz"))
    assert [backup.read_bytes() for backup in backups] == [b"interrupted artifact"]


def test_a_same_record_in_the_zip_is_accepted_without_writing_a_file(tmp_path):
    run = _runs()[0]
    resumption.write_run_start_record(tmp_path, run, TEST_RUN_PROVENANCE)
    _move_records_into_zip(tmp_path, [run])
    returned = resumption.write_run_start_record(tmp_path, run, TEST_RUN_PROVENANCE)
    assert returned == tmp_path / ".resume" / ARCHIVE
    assert not _record_path(tmp_path, run).exists()


def test_a_different_record_in_the_zip_is_refused(tmp_path):
    run = _runs()[0]
    resumption.write_run_start_record(tmp_path, run, TEST_RUN_PROVENANCE)

    def change(data):
        document = json.loads(data)
        document["environment"]["torch"] = "other"
        return json.dumps(document).encode("utf-8")

    _move_records_into_zip(tmp_path, [run], mutate=change)
    with pytest.raises(ValueError, match="differs"):
        resumption.write_run_start_record(tmp_path, run, TEST_RUN_PROVENANCE)
    assert not _record_path(tmp_path, run).exists()


def _broken_zip(root, _run):
    (root / ".resume").mkdir(parents=True, exist_ok=True)
    (root / ".resume" / ARCHIVE).write_bytes(b"not a zip file")


def _truncated_zip(root, run):
    resumption.write_run_start_record(root, run, TEST_RUN_PROVENANCE)
    _move_records_into_zip(root, [run])
    path = root / ".resume" / ARCHIVE
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])


@pytest.mark.parametrize("damage", ["broken", "truncated"])
def test_an_unreadable_zip_blocks_recovery_and_new_starts_without_changing_files(tmp_path, damage):
    runs = _runs(seeds=(0, 1))
    (_broken_zip if damage == "broken" else _truncated_zip)(tmp_path, runs[0])
    _orphan(tmp_path, runs[0])
    assert not resumption.can_recover_interrupted_run(tmp_path, runs[0], TEST_RUN_PROVENANCE)
    with pytest.raises(ValueError):
        resumption.write_run_start_record(tmp_path, runs[1], TEST_RUN_PROVENANCE)
    before = _tree(tmp_path)
    outcomes = execution.execute_experiment_sweep(
        planned_runs=runs,
        output_directory=tmp_path,
        worker_count=1,
        run_provenance=TEST_RUN_PROVENANCE,
    )
    assert [outcome.run_status for outcome in outcomes] == ["failed", "failed"]
    assert _tree(tmp_path) == before


def test_a_record_in_the_zip_is_unreadable_without_lzma(tmp_path, monkeypatch):
    """lzmaの無い環境のzipfile（LZMAのmemberを開くとRuntimeErrorを出す）では、zipにある記録を読めない。"""
    run = _runs()[0]
    resumption.write_run_start_record(tmp_path, run, TEST_RUN_PROVENANCE)
    _move_records_into_zip(tmp_path, [run])
    _orphan(tmp_path, run)
    monkeypatch.setattr(zipfile, "lzma", None)
    assert not resumption.can_recover_interrupted_run(tmp_path, run, TEST_RUN_PROVENANCE)
    (tmp_path / run.configuration_name / "runs" / f"{run.run_identifier}.npz").unlink()
    with pytest.raises(ValueError, match="unreadable"):
        resumption.write_run_start_record(tmp_path, run, TEST_RUN_PROVENANCE)
