"""runのまとめはバイト列を保ち、戻しの検査失敗では何も変更しない。"""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from federated_learning_experiments.sweep import packed_run_artifact_archive as packing


@pytest.fixture
def configuration(tmp_path):
    runs = tmp_path / "configuration" / "runs"
    runs.mkdir(parents=True)
    buffer = io.BytesIO()
    np.savez(buffer, values=np.array([1, 2, 3]))
    for identifier in ("a[1]", "b"):
        (runs / f"{identifier}.json").write_bytes(b'{"value": 1}')
        (runs / f"{identifier}.npz").write_bytes(buffer.getvalue())
    return runs.parent


def _snapshot(directory):
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


def _rewrite_archive(configuration, change):
    path = configuration / packing.RUN_ARTIFACT_ARCHIVE_NAME
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members.pop("archive_manifest.json"))
    change(members, manifest)
    members["archive_manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_LZMA) as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def test_pack_preserves_bytes_manifest_compression_and_restore(configuration):
    original = _snapshot(configuration / "runs")
    result = packing.pack_configuration_run_artifacts(configuration)
    assert result == packing.ConfigurationRunArtifactPackingResult(True, None, None)
    assert not (configuration / "runs").exists()
    with zipfile.ZipFile(configuration / "runs.zip") as archive:
        manifest = json.loads(archive.read("archive_manifest.json"))
        assert manifest == {
            "format_version": 1,
            "files": [
                {
                    "path": f"runs/{name}",
                    "byte_count": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
                for name, data in sorted(original.items())
            ],
        }
        for name, data in original.items():
            assert archive.read(f"runs/{name}") == data
            assert archive.getinfo(f"runs/{name}").compress_type == (
                zipfile.ZIP_STORED if name.endswith(".npz") else zipfile.ZIP_LZMA
            )
    assert packing.list_packed_run_identifiers(configuration) == frozenset(("a[1]", "b"))
    assert packing.read_packed_run_summary_and_record_table(configuration, "absent") is None
    assert packing.read_packed_run_summary_and_record_table(configuration, "b") == (
        original["b.json"],
        original["b.npz"],
    )
    packing.restore_packed_run_artifacts(configuration)
    assert _snapshot(configuration / "runs") == original
    assert not (configuration / "runs.zip").exists()


@pytest.mark.parametrize("invalid", ["orphan.json", "other.txt", "directory", "existing_zip"])
def test_pack_refuses_unpaired_or_unexpected_files(configuration, invalid):
    if invalid == "directory":
        (configuration / "runs" / invalid).mkdir()
    elif invalid == "existing_zip":
        (configuration / "runs.zip").write_bytes(b"keep")
    else:
        (configuration / "runs" / invalid).write_bytes(b"keep")
    original = _snapshot(configuration / "runs")
    result = packing.pack_configuration_run_artifacts(configuration)
    assert not result.archive_placed and result.not_packed_reason
    assert _snapshot(configuration / "runs") == original


@pytest.mark.parametrize("damage", ["extra", "path", "hash", "unpaired", "research", "corrupt"])
def test_invalid_archive_is_rejected_before_any_restoration(configuration, damage):
    packing.pack_configuration_run_artifacts(configuration)

    def change(members, manifest):
        if damage == "extra":
            members["unexpected"] = b"extra"
        elif damage == "path":
            entry = manifest["files"][0]
            members["runs/../escape.json"] = members.pop(entry["path"])
            entry["path"] = "runs/../escape.json"
        elif damage == "hash":
            manifest["files"][-1]["sha256"] = "0" * 64
        elif damage == "unpaired":
            entry = manifest["files"].pop()
            del members[entry["path"]]
        elif damage == "research":
            manifest["files"][-1]["arrays"] = []

    _rewrite_archive(configuration, change)
    if damage == "corrupt":
        path = configuration / "runs.zip"
        with zipfile.ZipFile(path) as archive:
            info = archive.infolist()[1]
        data = bytearray(path.read_bytes())
        offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
        data[offset + info.compress_size // 2] ^= 0xFF
        path.write_bytes(data)
    with pytest.raises(ValueError):
        packing.restore_packed_run_artifacts(configuration)
    assert not (configuration / "runs").exists()
    assert (configuration / "runs.zip").exists()
    if damage != "corrupt":
        operations = [lambda p: packing.read_packed_run_summary_and_record_table(p, "b")]
        if damage != "hash":
            operations.append(packing.list_packed_run_identifiers)
        for operation in operations:
            with pytest.raises(ValueError):
                operation(configuration)


def test_restore_accepts_identical_files_and_cleans_only_matching_partials(configuration):
    original = _snapshot(configuration / "runs")
    packing.pack_configuration_run_artifacts(configuration)
    runs = configuration / "runs"
    runs.mkdir()
    (runs / "b.json").write_bytes(original["b.json"])
    (runs / "a[1].json.old.partial").write_bytes(b"interrupted")
    (runs / "a1.json.old.partial").write_bytes(b"unrelated")
    packing.restore_packed_run_artifacts(configuration)
    assert not (runs / "a[1].json.old.partial").exists()
    assert (runs / "a1.json.old.partial").read_bytes() == b"unrelated"
    for name, data in original.items():
        assert (runs / name).read_bytes() == data


def test_restore_conflict_changes_nothing(configuration):
    packing.pack_configuration_run_artifacts(configuration)
    runs = configuration / "runs"
    runs.mkdir()
    (runs / "b.npz").write_bytes(b"different")
    before = _snapshot(runs)
    with pytest.raises(ValueError):
        packing.restore_packed_run_artifacts(configuration)
    assert _snapshot(runs) == before


@pytest.mark.parametrize(
    "operation",
    [
        packing.list_packed_run_identifiers,
        packing.restore_packed_run_artifacts,
        lambda p: packing.read_packed_run_summary_and_record_table(p, "b"),
    ],
)
def test_missing_lzma_is_a_distinct_value_error(configuration, monkeypatch, operation):
    packing.pack_configuration_run_artifacts(configuration)
    monkeypatch.setattr(zipfile, "lzma", None)
    with pytest.raises(ValueError, match="LZMA"):
        operation(configuration)
    assert not (configuration / "runs").exists()


@pytest.mark.parametrize("failure", ["write", "verify", "source_changed", "replace"])
def test_pack_failure_keeps_sources_and_removes_partial(configuration, monkeypatch, failure):
    original = _snapshot(configuration / "runs")
    (configuration / "runs.zip.partial").write_bytes(b"stale")
    if failure == "write":
        monkeypatch.setattr(
            packing.ZipFile, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("full"))
        )
    elif failure == "verify":
        monkeypatch.setattr(
            packing, "_verify_archive", lambda *a: (_ for _ in ()).throw(ValueError("mismatch"))
        )
    elif failure == "source_changed":
        verify = packing._verify_archive

        def change_source(*args):
            verify(*args)
            (configuration / "runs" / "b.json").write_bytes(b"changed")

        monkeypatch.setattr(packing, "_verify_archive", change_source)
    else:
        monkeypatch.setattr(
            packing, "replace_file", lambda *a: (_ for _ in ()).throw(OSError("replace"))
        )
    result = packing.pack_configuration_run_artifacts(configuration)
    assert not result.archive_placed and result.not_packed_reason
    assert not (configuration / "runs.zip").exists()
    assert not (configuration / "runs.zip.partial").exists()
    assert set(_snapshot(configuration / "runs")) == set(original)
    if failure != "source_changed":
        assert _snapshot(configuration / "runs") == original


def test_pack_deletion_failure_reports_placed_archive(configuration, monkeypatch):
    unlink = Path.unlink

    def fail_source(path, *args, **kwargs):
        if path.parent.name == "runs":
            raise OSError("cannot delete")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_source)
    result = packing.pack_completed_sweep_run_artifacts(configuration.parent, (configuration.name,))
    assert result.packed_configuration_names == (configuration.name,)
    assert result.warning_descriptions
    assert (configuration / "runs.zip").exists()


def test_restore_write_failure_keeps_verified_file_and_archive(configuration, monkeypatch):
    packing.pack_configuration_run_artifacts(configuration)
    replace_file = packing.replace_file
    calls = []

    def fail_second(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError("full")
        replace_file(source, destination)

    monkeypatch.setattr(packing, "replace_file", fail_second)
    with pytest.raises(OSError):
        packing.restore_packed_run_artifacts(configuration)
    assert len(list((configuration / "runs").iterdir())) == 1
    assert (configuration / "runs.zip").exists()


@pytest.mark.parametrize("persistent", [False, True])
def test_restore_retries_archive_deletion(configuration, monkeypatch, persistent):
    packing.pack_configuration_run_artifacts(configuration)
    unlink = Path.unlink
    calls = []

    def locked(path, *args, **kwargs):
        if path.name == "runs.zip":
            calls.append(path)
            if persistent or len(calls) == 1:
                raise PermissionError("locked")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked)
    monkeypatch.setattr(packing, "sleep", lambda _: None)
    if persistent:
        with pytest.raises(ValueError):
            packing.restore_packed_run_artifacts(configuration)
        assert len(calls) == 8
    else:
        packing.restore_packed_run_artifacts(configuration)
        assert len(calls) == 2


def test_research_tool_can_unpack_and_skips_packed_configuration(configuration):
    from tools.research_results_archive.archive import _process, unpack_configuration

    original = _snapshot(configuration / "runs")
    packing.pack_configuration_run_artifacts(configuration)
    assert "まとめ済み" in str(_process(("pack", configuration, False, False)))
    unpack_configuration(configuration)
    assert _snapshot(configuration / "runs") == original


def test_research_archive_requires_unpack(configuration):
    from tools.research_results_archive.archive import pack_configuration

    pack_configuration(configuration, dry_run=False)
    with pytest.raises(ValueError, match="unpack"):
        packing.restore_packed_run_artifacts(configuration)
    assert not (configuration / "runs").exists()


def test_empty_runs_is_noop(configuration):
    for path in (configuration / "runs").iterdir():
        path.unlink()
    assert packing.pack_configuration_run_artifacts(configuration) == (
        packing.ConfigurationRunArtifactPackingResult(False, None, None)
    )


@pytest.mark.parametrize("linked", ["configuration", "runs", "runs.zip", "b.json"])
def test_links_are_rejected_before_changes(configuration, monkeypatch, linked):
    # Windowsでリンク作成の権限が無くても、リンク検出後の拒否の位置を確認する。
    if linked == "b.json":
        link_path = configuration / "runs" / linked
    else:
        link_path = configuration if linked == "configuration" else configuration / linked
    original = _snapshot(configuration / "runs")
    is_link = packing._is_link
    monkeypatch.setattr(packing, "_is_link", lambda path: path == link_path or is_link(path))
    result = packing.pack_configuration_run_artifacts(configuration)
    assert not result.archive_placed and result.not_packed_reason
    assert _snapshot(configuration / "runs") == original


@pytest.mark.parametrize("linked", ["configuration", "runs", "runs.zip", "b.json"])
def test_restore_rejects_links_without_writing(configuration, monkeypatch, linked):
    packing.pack_configuration_run_artifacts(configuration)
    link_path = (
        configuration
        if linked == "configuration"
        else (configuration / "runs" / linked if linked == "b.json" else configuration / linked)
    )
    is_link = packing._is_link
    monkeypatch.setattr(packing, "_is_link", lambda path: path == link_path or is_link(path))
    with pytest.raises(ValueError):
        packing.restore_packed_run_artifacts(configuration)
    assert not (configuration / "runs").exists()
    assert (configuration / "runs.zip").exists()


def test_pack_retries_temporary_archive_placement(configuration, monkeypatch):
    replace_file = packing.replace_file
    calls = []

    def locked(source, destination):
        calls.append(destination)
        if len(calls) == 1:
            raise PermissionError("locked")
        replace_file(source, destination)

    monkeypatch.setattr(packing, "replace_file", locked)
    monkeypatch.setattr(packing, "sleep", lambda _: None)
    assert packing.pack_configuration_run_artifacts(configuration).archive_placed
    assert len(calls) == 2
