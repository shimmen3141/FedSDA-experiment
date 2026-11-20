"""記録の表をLZMAのNPZとして書く関数と、LZMAの有無の確認。"""

import io
import sys
import zipfile

import numpy as np
import pytest

from federated_learning_experiments.runtime import lzma_record_table_files
from federated_learning_experiments.runtime.lzma_record_table_files import (
    ensure_lzma_compression_available,
    write_lzma_compressed_record_tables,
)


def _record_tables():
    values = np.arange(12, dtype=np.float64).reshape(3, 4) / 7
    values[0, 1] = np.nan
    return {
        "sample_prediction.prediction_is_correct": np.array([[True, False], [False, True]]),
        "sample_prediction.observed_class_id": np.arange(6, dtype=np.int64).reshape(2, 3),
        "model_clustering.distance": values,
        "empty.column": np.zeros((0,), dtype=np.float32),
        "scalar.value": np.float64(1.5),
    }


def _read(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def test_every_entry_is_lzma_and_the_arrays_equal_savez_compressed():
    tables = _record_tables()
    lzma_stream = io.BytesIO()
    write_lzma_compressed_record_tables(lzma_stream, tables)
    deflate_stream = io.BytesIO()
    np.savez_compressed(deflate_stream, **tables)
    with zipfile.ZipFile(io.BytesIO(lzma_stream.getvalue())) as archive:
        assert [info.filename for info in archive.infolist()] == [f"{name}.npy" for name in tables]
        assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_LZMA}
    written, reference = _read(lzma_stream.getvalue()), _read(deflate_stream.getvalue())
    assert list(written) == list(reference) == list(tables)
    for name, array in reference.items():
        assert written[name].dtype == array.dtype and written[name].shape == array.shape
        assert np.array_equal(written[name], array, equal_nan=array.dtype.kind == "f")


def test_object_arrays_are_refused():
    with pytest.raises(ValueError):
        write_lzma_compressed_record_tables(
            io.BytesIO(), {"bad.column": np.array([{"a": 1}], dtype=object)}
        )


def test_lzma_availability_is_checked(monkeypatch):
    ensure_lzma_compression_available()

    def refuse(name):
        raise ImportError(name)

    monkeypatch.setattr(lzma_record_table_files, "import_module", refuse)
    with pytest.raises(RuntimeError, match="lzma"):
        ensure_lzma_compression_available()


def test_the_module_does_not_import_lzma_at_load_time():
    """lzmaの無い環境でも、deflateのNPZを読むだけの経路はこのmoduleの読込みで失敗しない。"""
    source = sys.modules[lzma_record_table_files.__name__].__file__
    text = open(source, encoding="utf-8").read()
    assert "import lzma" not in text and "from lzma" not in text
