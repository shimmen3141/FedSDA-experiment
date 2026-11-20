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


# ---- 2.2 まとめる関数 ----


def _write_records(root, runs):
    for run in runs:
        resumption.write_run_start_record(root, run, TEST_RUN_PROVENANCE)


def _zip_members(root):
    with zipfile.ZipFile(root / ".resume" / ARCHIVE) as archive:
        assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_LZMA}
        return {name: archive.read(name) for name in archive.namelist()}


def test_records_are_packed_and_the_plan_and_recovered_files_are_kept(tmp_path):
    runs = _runs(seeds=(0, 1))
    _write_records(tmp_path, runs)
    # 計画に無いrunの記録もまとめる。
    extra = tmp_path / ".resume" / "fedsda" / "other_run.json"
    extra.parent.mkdir(parents=True)
    extra.write_text('{"x": 1}', encoding="utf-8")
    (tmp_path / ".resume" / "sweep.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".resume" / "recovered" / "abc").mkdir(parents=True)
    (tmp_path / ".resume" / "recovered" / "abc" / "a.npz").write_bytes(b"backup")
    (tmp_path / ".resume" / "fedavg" / "r.json.123.partial").write_text("p", encoding="utf-8")
    originals = {
        f"fedavg/{run.run_identifier}.json": _record_path(tmp_path, run).read_bytes()
        for run in runs
    }
    originals["fedsda/other_run.json"] = extra.read_bytes()
    assert resumption.pack_completed_sweep_run_start_records(tmp_path) is None
    assert _zip_members(tmp_path) == originals
    assert not any(_record_path(tmp_path, run).exists() for run in runs)
    assert not (tmp_path / ".resume" / "fedsda").exists()
    # `*.json`でない途中のファイルは残る（そのディレクトリも残る）。
    assert (tmp_path / ".resume" / "fedavg" / "r.json.123.partial").exists()
    assert (tmp_path / ".resume" / "sweep.json").read_text(encoding="utf-8") == "{}"
    assert (tmp_path / ".resume" / "recovered" / "abc" / "a.npz").read_bytes() == b"backup"
    assert not (tmp_path / ".resume" / f"{ARCHIVE}.partial").exists()


def test_nothing_to_pack_leaves_the_zip_untouched(tmp_path):
    runs = _runs()
    _write_records(tmp_path, runs)
    resumption.pack_completed_sweep_run_start_records(tmp_path)
    before = _tree(tmp_path)
    stat = (tmp_path / ".resume" / ARCHIVE).stat().st_mtime_ns
    assert resumption.pack_completed_sweep_run_start_records(tmp_path) is None
    assert _tree(tmp_path) == before
    assert (tmp_path / ".resume" / ARCHIVE).stat().st_mtime_ns == stat
    assert resumption.pack_completed_sweep_run_start_records(tmp_path / "missing") is None


def test_new_records_are_added_to_an_existing_zip(tmp_path):
    runs = _runs(seeds=(0, 1))
    _write_records(tmp_path, runs[:1])
    resumption.pack_completed_sweep_run_start_records(tmp_path)
    first = _zip_members(tmp_path)
    _write_records(tmp_path, runs[1:])
    second_bytes = _record_path(tmp_path, runs[1]).read_bytes()
    assert resumption.pack_completed_sweep_run_start_records(tmp_path) is None
    assert _zip_members(tmp_path) == {
        **first,
        f"fedavg/{runs[1].run_identifier}.json": second_bytes,
    }
    assert not _record_path(tmp_path, runs[1]).exists()


def test_only_duplicates_are_removed_without_rewriting_the_zip(tmp_path):
    runs = _runs()
    _write_records(tmp_path, runs)
    data = _record_path(tmp_path, runs[0]).read_bytes()
    resumption.pack_completed_sweep_run_start_records(tmp_path)
    zip_bytes = (tmp_path / ".resume" / ARCHIVE).read_bytes()
    _record_path(tmp_path, runs[0]).parent.mkdir(parents=True, exist_ok=True)
    _record_path(tmp_path, runs[0]).write_bytes(data)
    assert resumption.pack_completed_sweep_run_start_records(tmp_path) is None
    assert (tmp_path / ".resume" / ARCHIVE).read_bytes() == zip_bytes
    assert not _record_path(tmp_path, runs[0]).exists()


def test_a_different_duplicate_leaves_everything_and_warns(tmp_path):
    runs = _runs()
    _write_records(tmp_path, runs)
    resumption.pack_completed_sweep_run_start_records(tmp_path)
    _record_path(tmp_path, runs[0]).parent.mkdir(parents=True, exist_ok=True)
    _record_path(tmp_path, runs[0]).write_bytes(b'{"other": true}')
    before = _tree(tmp_path)
    warning = resumption.pack_completed_sweep_run_start_records(tmp_path)
    assert warning is not None and "まとめなかった" in warning
    assert _tree(tmp_path) == before


def _fail_verification(*arguments, **keyword_arguments):
    raise ValueError("simulated mismatch")


def _fail_writing(path, *arguments, **keyword_arguments):
    if str(path).endswith(".partial") and arguments and arguments[0] == "w":
        raise OSError("simulated disk full")
    return zipfile.ZipFile(path, *arguments, **keyword_arguments)


@pytest.mark.parametrize("failure", ["verification", "writing", "broken_existing_zip"])
def test_a_failure_before_placing_the_zip_keeps_the_originals(tmp_path, monkeypatch, failure):
    runs = _runs(seeds=(0, 1))
    _write_records(tmp_path, runs)
    if failure == "verification":
        monkeypatch.setattr(resumption, "_verify_run_start_record_archive", _fail_verification)
    elif failure == "writing":
        monkeypatch.setattr(resumption, "ZipFile", _fail_writing)
    else:
        (tmp_path / ".resume" / ARCHIVE).write_bytes(b"not a zip file")
    (tmp_path / ".resume" / f"{ARCHIVE}.partial").write_bytes(b"left over")
    before = _tree(tmp_path)
    del before[f".resume/{ARCHIVE}.partial"]
    warning = resumption.pack_completed_sweep_run_start_records(tmp_path)
    assert warning is not None and "まとめなかった" in warning
    assert _tree(tmp_path) == before


def test_a_failure_while_removing_the_originals_is_reported_after_placing_the_zip(
    tmp_path, monkeypatch
):
    runs = _runs()
    _write_records(tmp_path, runs)
    data = _record_path(tmp_path, runs[0]).read_bytes()
    real_unlink = type(tmp_path).unlink

    def fail_unlink(path, *arguments, **keyword_arguments):
        if path.suffix == ".json":
            raise OSError("simulated failure while removing")
        return real_unlink(path, *arguments, **keyword_arguments)

    monkeypatch.setattr(type(tmp_path), "unlink", fail_unlink)
    warning = resumption.pack_completed_sweep_run_start_records(tmp_path)
    monkeypatch.undo()
    assert warning is not None and "まとめたが" in warning
    assert _zip_members(tmp_path) == {f"fedavg/{runs[0].run_identifier}.json": data}
    # 次のまとめで、zipと同じ内容の重複として消える。
    assert resumption.pack_completed_sweep_run_start_records(tmp_path) is None
    assert not _record_path(tmp_path, runs[0]).exists()


def test_a_linked_resume_directory_is_not_packed(tmp_path):
    runs = _runs()
    target = tmp_path / "elsewhere"
    _write_records(target, runs)
    output = tmp_path / "output"
    output.mkdir()
    try:
        (output / ".resume").symlink_to(target / ".resume", target_is_directory=True)
    except OSError:
        pytest.skip("この環境ではsymlinkを作れない")
    before = _tree(target)
    warning = resumption.pack_completed_sweep_run_start_records(output)
    assert warning is not None and "まとめなかった" in warning
    assert _tree(target) == before


# ---- 2.3 掃引の終わり ----


def _sweep(root, runs, **keyword_arguments):
    return execution.execute_experiment_sweep(
        planned_runs=runs,
        output_directory=root,
        worker_count=1,
        run_provenance=TEST_RUN_PROVENANCE,
        **keyword_arguments,
    )


def test_a_completed_sweep_packs_its_start_records_and_continuing_changes_nothing(tmp_path):
    """`execute_experiment_sweep`を直接呼ぶ（比較scriptと同じ呼び方。`sweep.json`が無い）掃引でも、まとまる。"""
    runs = _runs(seeds=(0, 1))
    outcomes = _sweep(tmp_path, runs)
    assert [outcome.run_status for outcome in outcomes] == ["completed", "completed"]
    assert set(_zip_members(tmp_path)) == {f"fedavg/{run.run_identifier}.json" for run in runs}
    assert not (tmp_path / ".resume" / "fedavg").exists()
    # 記録の表はLZMA。
    for run in runs:
        npz = tmp_path / "fedavg" / "runs" / f"{run.run_identifier}.npz"
        with zipfile.ZipFile(npz) as archive:
            assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_LZMA}
    before = _tree(tmp_path)
    outcomes = _sweep(tmp_path, runs)
    assert [outcome.run_status for outcome in outcomes] == ["skipped_existing"] * 2
    assert _tree(tmp_path) == before


def test_a_sweep_with_a_failed_run_does_not_pack(tmp_path, monkeypatch):
    runs = _runs(seeds=(0, 1))

    def fail(**_):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(execution, "execute_fedavg_experiment_run", fail)
    outcomes = _sweep(tmp_path, runs)
    assert [outcome.run_status for outcome in outcomes] == ["failed", "failed"]
    assert not (tmp_path / ".resume" / ARCHIVE).exists()
    assert all(_record_path(tmp_path, run).is_file() for run in runs)


def test_a_packing_warning_is_reported_and_the_sweep_does_not_fail(tmp_path, monkeypatch):
    runs = _runs()
    monkeypatch.setattr(
        execution,
        "pack_completed_sweep_run_start_records",
        lambda _: "開始の記録をまとめなかった: x",
    )
    messages = []
    outcomes = _sweep(tmp_path, runs, report_progress=messages.append)
    assert [outcome.run_status for outcome in outcomes] == ["completed"]
    assert any("まとめなかった" in message for message in messages)
    # 報告先が無ければ、警告は捨てる（例外にしない）。
    outcomes = _sweep(tmp_path / "other", runs)
    assert [outcome.run_status for outcome in outcomes] == ["completed"]


def test_a_sweep_stops_before_any_run_without_lzma(tmp_path, monkeypatch):
    from federated_learning_experiments.runtime import lzma_record_table_files

    def refuse(name):
        raise ImportError(name)

    monkeypatch.setattr(lzma_record_table_files, "import_module", refuse)
    with pytest.raises(RuntimeError, match="lzma"):
        _sweep(tmp_path / "output", _runs())
    assert not (tmp_path / "output").exists()
