"""書き出した系列（研究ツール`stream_export`の形式）を読み、処理の前に検査する。riverとtorchをimportしない。"""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from federated_learning_experiments.data.dataset_definitions import get_dataset_definition


def compute_stream_array_digest(arrays: dict[str, np.ndarray]) -> str:
    """書き出した配列の内容のhash（`stream_export.array_digest`と同じ: 名前の順に、名前・形・型・値を連結）。"""
    digest = sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, kw_only=True)
class SingleStreamInput:
    """検査した系列。事前学習の標本は持たない（この手法は使わない）。"""

    dataset_name: str
    random_seed: int
    class_count: int
    features: np.ndarray
    labels: np.ndarray
    concept_ids: np.ndarray
    array_sha256: str


def load_single_stream_input(*, stream_path: Path) -> SingleStreamInput:
    """NPZとJSONを読み、hash・長さ・クラス数・ラベルの範囲を確かめる。"""
    stream_path = Path(stream_path)
    metadata = json.loads(stream_path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(stream_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    for name in ("features", "labels", "concept_ids"):
        if name not in arrays:
            raise ValueError(f"系列に{name}がありません: {stream_path}")
    for name in ("dataset_name", "random_seed", "array_sha256"):
        if name not in metadata:
            raise ValueError(f"系列のJSONに{name}がありません: {stream_path}")
    digest = compute_stream_array_digest(arrays)
    if digest != metadata["array_sha256"]:
        raise ValueError(f"系列の内容が書き出しの記録と違います: {stream_path}")
    features, labels, concept_ids = arrays["features"], arrays["labels"], arrays["concept_ids"]
    if features.ndim != 2 or labels.ndim != 1 or concept_ids.ndim != 1:
        raise ValueError("featuresは2次元、labelsとconcept_idsは1次元が必要です。")
    if not len(features) == len(labels) == len(concept_ids):
        raise ValueError("features・labels・concept_idsの長さが違います。")
    try:
        class_count = get_dataset_definition(dataset_name=metadata["dataset_name"]).class_count
    except (KeyError, ValueError) as error:
        raise ValueError(f"データセットの定義がありません: {metadata['dataset_name']}") from error
    if labels.size and (labels.min() < 0 or labels.max() >= class_count):
        raise ValueError("ラベルは0以上class_count未満が必要です。")
    return SingleStreamInput(
        dataset_name=str(metadata["dataset_name"]),
        random_seed=int(metadata["random_seed"]),
        class_count=int(class_count),
        features=features,
        labels=labels.astype(np.int64),
        concept_ids=concept_ids.astype(np.int64),
        array_sha256=digest,
    )
