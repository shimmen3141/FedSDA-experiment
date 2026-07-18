"""MNIST IDXデータの取得・読込みとFedDrift論文の概念変換。"""
import gzip
import os
from pathlib import Path
import struct
import urllib.request

import numpy as np


_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist"
_FILES = {
    "images": "train-images-idx3-ubyte.gz",
    "labels": "train-labels-idx1-ubyte.gz",
}
_CACHE = {}


def default_data_dir():
    configured = os.environ.get("FDE_MNIST_DATA_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "data" / "mnist"


def _ensure_file(path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        urllib.request.urlretrieve(f"{_BASE_URL}/{path.name}", temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_images(path):
    with gzip.open(path, "rb") as stream:
        magic, count, rows, cols = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image magic number: {magic}")
        raw = np.frombuffer(stream.read(), dtype=np.uint8)
    expected = count * rows * cols
    if raw.size != expected:
        raise ValueError(f"Invalid MNIST image payload: expected {expected}, got {raw.size}")
    return raw.reshape(count, rows * cols).astype(np.float32) / 255.0


def _read_labels(path):
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label magic number: {magic}")
        labels = np.frombuffer(stream.read(), dtype=np.uint8)
    if labels.size != count:
        raise ValueError(f"Invalid MNIST label payload: expected {count}, got {labels.size}")
    return labels.astype(np.int64)


def load_mnist(data_dir=None):
    data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    cache_key = str(data_dir.resolve())
    if cache_key not in _CACHE:
        paths = {name: data_dir / filename for name, filename in _FILES.items()}
        for path in paths.values():
            _ensure_file(path)
        images = _read_images(paths["images"])
        labels = _read_labels(paths["labels"])
        if len(images) != len(labels):
            raise ValueError("MNIST image and label counts differ")
        _CACHE[cache_key] = (images, labels)
    return _CACHE[cache_key]


def apply_mnist_concept(labels, concept_id):
    """A=元ラベル、B=1/2、C=3/4、D=5/6の交換を適用する。"""
    if concept_id not in range(4):
        raise ValueError(f"Unknown MNIST concept: {concept_id}")
    transformed = np.asarray(labels, dtype=np.int64).copy()
    if concept_id == 0:
        return transformed
    first = 2 * concept_id - 1
    second = first + 1
    first_mask = transformed == first
    second_mask = transformed == second
    transformed[first_mask] = second
    transformed[second_mask] = first
    return transformed


def sample_mnist(concept_id, n_samples, data_dir=None):
    images, labels = load_mnist(data_dir)
    indices = np.random.randint(0, len(images), size=n_samples)
    return images[indices], apply_mnist_concept(labels[indices], concept_id)
