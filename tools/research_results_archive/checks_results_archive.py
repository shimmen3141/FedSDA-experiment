"""`archive.py`の検査（研究ツールなので明示して実行する）。

    python -m pytest tools/research_results_archive/checks_results_archive.py -q -p no:cacheprovider
"""

import csv
import io
import json
import zipfile

import numpy as np
import pytest

from tools.research_results_archive import archive


def _make_sweep(tmp_path, *, run_count=3, summary_rows=None, with_resume=True):
    sweep = tmp_path / "study" / "20261120_000000_label"
    configuration = sweep / "fedsda"
    runs = configuration / "runs"
    runs.mkdir(parents=True)
    rng = np.random.RandomState(0)
    for index in range(run_count):
        stem = f"sea4_interval50_seed{index}_abc"
        (runs / f"{stem}.json").write_text(json.dumps({"seed": index, "x": [1, 2, 3]}), encoding="utf-8")
        values = rng.rand(4, 7)
        values[0, 0] = np.nan
        np.savez_compressed(
            runs / f"{stem}.npz",
            correct=rng.rand(4, 7) > 0.5,
            ids=rng.randint(0, 100, size=(4, 7)).astype(np.int64),
            values=values,
            empty=np.zeros((0,), dtype=np.float32),
        )
    with (configuration / "run_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_identifier"])
        for index in range(run_count if summary_rows is None else summary_rows):
            writer.writerow([f"sea4_interval50_seed{index}_abc"])
    (configuration / "README.md").write_text("readme", encoding="utf-8")
    if with_resume:
        resume = sweep / ".resume" / "fedsda"
        resume.mkdir(parents=True)
        (resume / "sea4_interval50_seed0_abc.json").write_text('{"started": true}', encoding="utf-8")
        (sweep / ".resume" / "sweep.json").write_text("{}", encoding="utf-8")
    return sweep, configuration


def _snapshot(directory):
    """ファイルのパス→（NPZなら配列、それ以外ならバイト列）。"""
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            key = path.relative_to(directory).as_posix()
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as loaded:
                    result[key] = {name: loaded[name] for name in loaded.files}
            else:
                result[key] = path.read_bytes()
    return result


def _assert_same_snapshot(first, second):
    assert list(first) == list(second)
    for key, value in first.items():
        other = second[key]
        if isinstance(value, dict):
            assert list(value) == list(other)
            for name, array in value.items():
                assert array.dtype == other[name].dtype and array.shape == other[name].shape
                assert np.array_equal(array, other[name], equal_nan=array.dtype.kind == "f")
        else:
            assert value == other


def test_pack_then_unpack_restores_every_file(tmp_path):
    sweep, configuration = _make_sweep(tmp_path)
    before = _snapshot(sweep)
    assert archive.main(["pack", str(tmp_path)]) == 0
    assert not (configuration / "runs").exists()
    assert not (sweep / ".resume" / "fedsda").exists()
    # 掃引の記録など、まとめない物は残る。
    assert (configuration / "run_summary.csv").is_file() and (configuration / "README.md").is_file()
    assert (sweep / ".resume" / "sweep.json").is_file()
    with zipfile.ZipFile(configuration / archive.ARCHIVE_NAME) as packed:
        names = set(packed.namelist())
        manifest = json.loads(packed.read(archive.MANIFEST_NAME))
        # zipの中のNPZは、numpyでそのまま読める。
        with np.load(
            io.BytesIO(packed.read("runs/sea4_interval50_seed0_abc.npz"))
        ) as loaded:
            assert loaded["ids"].dtype == np.int64
    assert names == {
        archive.MANIFEST_NAME,
        ".resume/sea4_interval50_seed0_abc.json",
        *(f"runs/sea4_interval50_seed{i}_abc.{ext}" for i in range(3) for ext in ("json", "npz")),
    }
    assert len(manifest["files"]) == 7
    assert archive.main(["unpack", str(tmp_path)]) == 0
    assert not (configuration / archive.ARCHIVE_NAME).exists()
    _assert_same_snapshot(before, _snapshot(sweep))


@pytest.mark.parametrize("summary_rows", [2, None])
def test_incomplete_or_unsummarized_configurations_are_skipped(tmp_path, summary_rows, capsys):
    sweep, configuration = _make_sweep(tmp_path, summary_rows=summary_rows)
    if summary_rows is None:
        (configuration / "run_summary.csv").unlink()
    before = _snapshot(sweep)
    archive.main(["pack", str(tmp_path)])
    assert "skip" in capsys.readouterr().out
    assert not (configuration / archive.ARCHIVE_NAME).exists()
    _assert_same_snapshot(before, _snapshot(sweep))


def test_dry_run_changes_nothing(tmp_path):
    sweep, _ = _make_sweep(tmp_path)
    before = _snapshot(sweep)
    archive.main(["pack", str(tmp_path), "--dry-run"])
    _assert_same_snapshot(before, _snapshot(sweep))


def test_a_mismatch_in_the_check_keeps_the_originals_and_removes_the_zip(tmp_path, monkeypatch):
    sweep, configuration = _make_sweep(tmp_path)
    before = _snapshot(sweep)

    # zipへ書くNPZの配列を1つだけ変え、照合が止めることを確かめる。
    original = archive._lzma_npz_bytes
    monkeypatch.setattr(
        archive, "_lzma_npz_bytes", lambda arrays: original({**arrays, "ids": arrays["ids"] + 1})
    )
    with pytest.raises(ValueError, match="NPZ"):
        archive.main(["pack", str(tmp_path)])
    assert not (configuration / archive.ARCHIVE_NAME).exists()
    assert not list(configuration.glob("*.tmp"))
    _assert_same_snapshot(before, _snapshot(sweep))


def test_unpack_refuses_to_overwrite_existing_files(tmp_path):
    sweep, configuration = _make_sweep(tmp_path)
    archive.main(["pack", str(tmp_path)])
    (configuration / "runs").mkdir()
    (configuration / "runs" / "sea4_interval50_seed0_abc.json").write_text("other", encoding="utf-8")
    with pytest.raises(FileExistsError):
        archive.main(["unpack", str(tmp_path)])
    assert (configuration / archive.ARCHIVE_NAME).exists()


def test_already_packed_configurations_are_skipped(tmp_path, capsys):
    _make_sweep(tmp_path)
    archive.main(["pack", str(tmp_path)])
    capsys.readouterr()
    archive.main(["pack", str(tmp_path)])
    assert "まとめ済み" in capsys.readouterr().out


def test_parallel_workers_give_the_same_round_trip(tmp_path):
    first, _ = _make_sweep(tmp_path / "a")
    second, _ = _make_sweep(tmp_path / "b")
    before = (_snapshot(first), _snapshot(second))
    assert archive.main(["pack", str(tmp_path), "--workers", "2"]) == 0
    assert all((sweep / "fedsda" / archive.ARCHIVE_NAME).exists() for sweep in (first, second))
    assert archive.main(["unpack", str(tmp_path), "--workers", "2"]) == 0
    _assert_same_snapshot(before[0], _snapshot(first))
    _assert_same_snapshot(before[1], _snapshot(second))


def test_a_sweep_with_failed_runs_is_skipped(tmp_path, capsys):
    sweep, configuration = _make_sweep(tmp_path)
    (sweep / "sweep_record_20261120_000000.json").write_text(
        json.dumps({"run_status_counts": {"completed": 3, "skipped_existing": 0, "failed": 1}}),
        encoding="utf-8",
    )
    before = _snapshot(sweep)
    archive.main(["pack", str(tmp_path)])
    assert "失敗したrun" in capsys.readouterr().out
    _assert_same_snapshot(before, _snapshot(sweep))


def test_a_failure_while_writing_the_zip_leaves_no_temporary_file(tmp_path, monkeypatch):
    sweep, configuration = _make_sweep(tmp_path)
    before = _snapshot(sweep)

    def fail(*arguments, **keyword_arguments):
        raise OSError("simulated disk full")

    monkeypatch.setattr(archive, "_lzma_npz_bytes", fail)
    with pytest.raises(OSError):
        archive.main(["pack", str(tmp_path)])
    assert not list(configuration.glob("*.tmp"))
    assert not (configuration / archive.ARCHIVE_NAME).exists()
    _assert_same_snapshot(before, _snapshot(sweep))
