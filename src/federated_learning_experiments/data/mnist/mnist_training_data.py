"""置いてあるgzipのIDXファイルから、MNISTの学習用データ（画素と数字のラベル）を読む。取得はしない。"""

from dataclasses import dataclass
from gzip import open as open_gzip_file
from os import environ
from pathlib import Path
from struct import unpack

from numpy import frombuffer, int64, ndarray, uint8

# 置き場所を指定する環境変数（旧実装と同じ名前）。
_DATA_DIRECTORY_ENVIRONMENT_VARIABLE = "FDE_MNIST_DATA_DIR"
_IMAGE_FILE_NAME = "train-images-idx3-ubyte.gz"
_LABEL_FILE_NAME = "train-labels-idx1-ubyte.gz"
# IDX形式の先頭の、識別の数値。
_IMAGE_MAGIC_NUMBER = 2051
_LABEL_MAGIC_NUMBER = 2049
# 1件の画像の画素数（28×28）。
_PIXEL_COUNT_PER_IMAGE = 784


@dataclass(frozen=True, kw_only=True, eq=False)
class MnistTrainingData:
    """MNISTの学習用データ。配列は書込み不可で、processの中で使い回す。"""

    # uint8の画素（件数×784）。float32への変換は、生成器が行う。
    pixel_values: ndarray
    # 画像の数字（int64、件数）。概念で交換する前のラベル。
    digit_labels: ndarray

    def __post_init__(self) -> None:
        if type(self.pixel_values) is not ndarray:
            raise TypeError("pixel_values must be numpy.ndarray")
        if (
            self.pixel_values.dtype != uint8
            or self.pixel_values.ndim != 2
            or self.pixel_values.shape[0] < 1
            or self.pixel_values.shape[1] != _PIXEL_COUNT_PER_IMAGE
        ):
            raise ValueError(
                f"pixel_values must be uint8 with shape (count >= 1, {_PIXEL_COUNT_PER_IMAGE})"
            )
        if self.pixel_values.flags.writeable:
            raise ValueError("pixel_values must be read-only")
        if type(self.digit_labels) is not ndarray:
            raise TypeError("digit_labels must be numpy.ndarray")
        if self.digit_labels.dtype != int64 or self.digit_labels.shape != (
            self.pixel_values.shape[0],
        ):
            raise ValueError("digit_labels must be int64 with one label per image")
        if self.digit_labels.flags.writeable:
            raise ValueError("digit_labels must be read-only")


# 解決したディレクトリごとの、読んだ結果（読込みに成功したものだけ）。
_LOADED_TRAINING_DATA_BY_DIRECTORY: dict[Path, MnistTrainingData] = {}


def resolve_mnist_data_directory() -> Path:
    """MNISTのファイルの置き場所を返す。環境変数があればその値、なければ、リポジトリ直下の`data/mnist`。"""
    configured_directory = environ.get(_DATA_DIRECTORY_ENVIRONMENT_VARIABLE)
    if configured_directory:
        return Path(configured_directory)
    return Path(__file__).resolve().parents[4] / "data" / "mnist"


def _read_pixel_values(image_file_path: Path) -> ndarray:
    with open_gzip_file(image_file_path, "rb") as image_stream:
        magic_number, image_count, row_count, column_count = unpack(">IIII", image_stream.read(16))
        if magic_number != _IMAGE_MAGIC_NUMBER:
            raise ValueError(f"invalid MNIST image magic number: {magic_number}")
        if row_count * column_count != _PIXEL_COUNT_PER_IMAGE:
            raise ValueError(
                f"MNIST images must have {_PIXEL_COUNT_PER_IMAGE} pixels: {row_count}x{column_count}"
            )
        pixel_bytes = image_stream.read()
    if len(pixel_bytes) != image_count * _PIXEL_COUNT_PER_IMAGE:
        raise ValueError(
            "invalid MNIST image payload: "
            f"expected {image_count * _PIXEL_COUNT_PER_IMAGE} bytes, got {len(pixel_bytes)}"
        )
    # bytesから作った配列は、書込み不可。
    return frombuffer(pixel_bytes, dtype=uint8).reshape(image_count, _PIXEL_COUNT_PER_IMAGE)


def _read_digit_labels(label_file_path: Path) -> ndarray:
    with open_gzip_file(label_file_path, "rb") as label_stream:
        magic_number, label_count = unpack(">II", label_stream.read(8))
        if magic_number != _LABEL_MAGIC_NUMBER:
            raise ValueError(f"invalid MNIST label magic number: {magic_number}")
        label_bytes = label_stream.read()
    if len(label_bytes) != label_count:
        raise ValueError(
            f"invalid MNIST label payload: expected {label_count} bytes, got {len(label_bytes)}"
        )
    digit_labels = frombuffer(label_bytes, dtype=uint8).astype(int64)
    digit_labels.setflags(write=False)
    return digit_labels


def load_mnist_training_data(*, data_directory: Path) -> MnistTrainingData:
    """ディレクトリの2つのファイルから、学習用データを読む。同じディレクトリは、読み直さない。

    ファイルがなければ、取得を試みずに拒否する。
    """
    if not isinstance(data_directory, Path):
        raise TypeError("data_directory must be pathlib.Path")
    resolved_directory = data_directory.resolve()
    if resolved_directory in _LOADED_TRAINING_DATA_BY_DIRECTORY:
        return _LOADED_TRAINING_DATA_BY_DIRECTORY[resolved_directory]
    missing_file_names = [
        file_name
        for file_name in (_IMAGE_FILE_NAME, _LABEL_FILE_NAME)
        if not (resolved_directory / file_name).is_file()
    ]
    if missing_file_names:
        raise FileNotFoundError(
            f"MNIST files are missing in {resolved_directory}: {', '.join(missing_file_names)}. "
            f"Place them there, or set {_DATA_DIRECTORY_ENVIRONMENT_VARIABLE} to the directory "
            "that holds them (this implementation does not download them)."
        )
    pixel_values = _read_pixel_values(resolved_directory / _IMAGE_FILE_NAME)
    digit_labels = _read_digit_labels(resolved_directory / _LABEL_FILE_NAME)
    if len(pixel_values) != len(digit_labels):
        raise ValueError(
            f"MNIST image and label counts differ: {len(pixel_values)} and {len(digit_labels)}"
        )
    mnist_training_data = MnistTrainingData(pixel_values=pixel_values, digit_labels=digit_labels)
    _LOADED_TRAINING_DATA_BY_DIRECTORY[resolved_directory] = mnist_training_data
    return mnist_training_data
