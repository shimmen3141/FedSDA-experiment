"""MNIST（mnist2・mnist4）の学習用データの読込みと生成器を、実旧と照合する。

実旧との照合は、`FDE_MNIST_DATA_DIR`（なければ、リポジトリ直下の`data/mnist`）に置いたファイルを使う。
"""

import gzip
import random
import struct
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from federated_learning_experiments.data.dataset_definitions import get_dataset_definition
from federated_learning_experiments.data.mnist.mnist_sample_generation import MnistSampleGenerator
from federated_learning_experiments.data.mnist.mnist_training_data import (
    MnistTrainingData,
    load_mnist_training_data,
    resolve_mnist_data_directory,
)
from federated_learning_experiments.data.observed_sample_generation import (
    OBSERVED_SAMPLE_GENERATOR_TYPES,
    create_observed_sample_generator,
    is_observed_sample_generator_of_dataset,
)
from federated_learning_experiments.data.observed_streams import ObservedSample

IMAGE_FILE_NAME = "train-images-idx3-ubyte.gz"
LABEL_FILE_NAME = "train-labels-idx1-ubyte.gz"
MNIST_DATASET_NAMES = ("mnist2", "mnist4")
MNIST_DATASET_NAME_AND_CONCEPT_ID_PAIRS = [
    (dataset_name, concept_id)
    for dataset_name in MNIST_DATASET_NAMES
    for concept_id in range(get_dataset_definition(dataset_name=dataset_name).concept_count)
]
# 概念ごとの、入れ替える数字の対（概念0は、入れ替えない）。
SWAPPED_DIGITS_BY_CONCEPT_ID = {0: None, 1: (1, 2), 2: (3, 4), 3: (5, 6)}


def write_mnist_files(
    directory,
    *,
    pixel_rows,
    digit_labels,
    image_magic_number=2051,
    label_magic_number=2049,
    declared_image_count=None,
    declared_label_count=None,
    row_count=28,
    column_count=28,
):
    """gzipのIDX形式で、画像とラベルのファイルを書く（不正な形式も書ける）。"""
    directory.mkdir(parents=True, exist_ok=True)
    pixel_bytes = np.asarray(pixel_rows, dtype=np.uint8).tobytes()
    with gzip.open(directory / IMAGE_FILE_NAME, "wb") as image_stream:
        image_stream.write(
            struct.pack(
                ">IIII",
                image_magic_number,
                len(pixel_rows) if declared_image_count is None else declared_image_count,
                row_count,
                column_count,
            )
        )
        image_stream.write(pixel_bytes)
    with gzip.open(directory / LABEL_FILE_NAME, "wb") as label_stream:
        label_stream.write(
            struct.pack(
                ">II",
                label_magic_number,
                len(digit_labels) if declared_label_count is None else declared_label_count,
            )
        )
        label_stream.write(np.asarray(digit_labels, dtype=np.uint8).tobytes())


def make_pixel_rows(image_count):
    """画像ごと・位置ごとに違う、0〜255の画素（784個）。"""
    return [
        [(image_index * 37 + pixel_index) % 256 for pixel_index in range(784)]
        for image_index in range(image_count)
    ]


def make_small_training_data(digit_labels=tuple(range(10))):
    """10件（数字0〜9が1件ずつ）の、小さい学習用データ。"""
    pixel_values = np.array(make_pixel_rows(len(digit_labels)), dtype=np.uint8)
    pixel_values.setflags(write=False)
    label_values = np.array(digit_labels, dtype=np.int64)
    label_values.setflags(write=False)
    return MnistTrainingData(pixel_values=pixel_values, digit_labels=label_values)


class FixedPositionRandomState(np.random.RandomState):
    """決めた位置を返す。"""

    def __init__(self, *, expected_sample_count, position):
        super().__init__(0)
        self.expected_sample_count = expected_sample_count
        self.position = position
        self.draw_count = 0

    def randint(self, low, high, *, size):
        assert (low, high, size) == (0, self.expected_sample_count, 1)
        self.draw_count += 1
        return np.array([self.position])


def assert_numpy_random_states_equal(actual_numpy_random_state, expected_numpy_random_state):
    assert actual_numpy_random_state[0] == expected_numpy_random_state[0]
    np.testing.assert_array_equal(actual_numpy_random_state[1], expected_numpy_random_state[1])
    assert actual_numpy_random_state[2:] == expected_numpy_random_state[2:]


# ---- 読込み ----


def test_real_training_data_matches_legacy_load():
    """置いてあるファイルから読んだ画素とラベルが、実旧の読込みの結果と、全件で一致する。"""
    from federated_drift_experiment.data import mnist as legacy_mnist_module

    legacy_images, legacy_labels = legacy_mnist_module.load_mnist()
    mnist_training_data = load_mnist_training_data(data_directory=resolve_mnist_data_directory())
    assert type(mnist_training_data) is MnistTrainingData
    assert mnist_training_data.pixel_values.dtype == np.uint8
    assert mnist_training_data.pixel_values.shape == (60000, 784)
    assert mnist_training_data.digit_labels.dtype == np.int64
    assert mnist_training_data.digit_labels.shape == (60000,)
    # 旧と同じ変換（float32へ直して、255.0で割る）をすると、旧の画像と一致する。
    assert np.array_equal(
        mnist_training_data.pixel_values.astype(np.float32) / 255.0, legacy_images
    )
    assert np.array_equal(mnist_training_data.digit_labels, legacy_labels)
    assert set(mnist_training_data.digit_labels.tolist()) == set(range(10))


def test_data_directory_follows_environment_variable_then_repository_default(monkeypatch, tmp_path):
    """置き場所は、環境変数があればその値、なければ、リポジトリ直下の`data/mnist`（旧と同じ規則）。"""
    from federated_drift_experiment.data import mnist as legacy_mnist_module

    monkeypatch.setenv("FDE_MNIST_DATA_DIR", str(tmp_path / "configured"))
    assert resolve_mnist_data_directory() == tmp_path / "configured"
    assert resolve_mnist_data_directory() == legacy_mnist_module.default_data_dir()
    for unset_value in (None, ""):
        if unset_value is None:
            monkeypatch.delenv("FDE_MNIST_DATA_DIR")
        else:
            monkeypatch.setenv("FDE_MNIST_DATA_DIR", unset_value)
        resolved_directory = resolve_mnist_data_directory()
        assert type(resolved_directory) is type(Path())
        assert resolved_directory == Path(__file__).resolve().parents[2] / "data" / "mnist"
        assert resolved_directory == legacy_mnist_module.default_data_dir()


def test_small_files_are_read_as_written(tmp_path):
    """書いた画素とラベルを、そのまま読む。配列は、書き換えられない。"""
    pixel_rows = make_pixel_rows(5)
    write_mnist_files(tmp_path, pixel_rows=pixel_rows, digit_labels=[3, 1, 4, 1, 5])
    mnist_training_data = load_mnist_training_data(data_directory=tmp_path)
    assert mnist_training_data.pixel_values.tolist() == pixel_rows
    assert mnist_training_data.digit_labels.tolist() == [3, 1, 4, 1, 5]
    assert mnist_training_data.pixel_values.dtype == np.uint8
    assert mnist_training_data.digit_labels.dtype == np.int64
    with pytest.raises(ValueError):
        mnist_training_data.pixel_values[0, 0] = 1
    with pytest.raises(ValueError):
        mnist_training_data.digit_labels[0] = 1
    with pytest.raises(FrozenInstanceError):
        mnist_training_data.digit_labels = np.zeros(5, dtype=np.int64)  # type: ignore[misc]


def test_loaded_training_data_is_reused_per_resolved_directory(tmp_path, monkeypatch):
    """同じディレクトリは、読み直さずに同じ結果を返す（相対の指定でも同じ）。別のディレクトリは、別に読む。"""
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    write_mnist_files(first_directory, pixel_rows=make_pixel_rows(2), digit_labels=[1, 2])
    write_mnist_files(second_directory, pixel_rows=make_pixel_rows(3), digit_labels=[4, 5, 6])
    first_training_data = load_mnist_training_data(data_directory=first_directory)
    # ファイルを消しても、同じprocessの中では、読んだ結果を返す。
    (first_directory / IMAGE_FILE_NAME).unlink()
    assert load_mnist_training_data(data_directory=first_directory) is first_training_data
    monkeypatch.chdir(tmp_path)
    assert load_mnist_training_data(data_directory=Path("first")) is first_training_data
    second_training_data = load_mnist_training_data(data_directory=second_directory)
    assert second_training_data is not first_training_data
    assert second_training_data.digit_labels.tolist() == [4, 5, 6]


@pytest.mark.parametrize("missing_file_name", [IMAGE_FILE_NAME, LABEL_FILE_NAME, "both"])
def test_missing_files_are_rejected_without_fetching(tmp_path, missing_file_name):
    """ファイルがなければ、取得を試みずに、ファイル名と環境変数の名前が分かる例外で拒否する。"""
    write_mnist_files(tmp_path, pixel_rows=make_pixel_rows(1), digit_labels=[7])
    missing_file_names = (
        (IMAGE_FILE_NAME, LABEL_FILE_NAME) if missing_file_name == "both" else (missing_file_name,)
    )
    for file_name in missing_file_names:
        (tmp_path / file_name).unlink()
    remaining_file_names = sorted(path.name for path in tmp_path.iterdir())
    with pytest.raises(FileNotFoundError) as exception_info:
        load_mnist_training_data(data_directory=tmp_path)
    for file_name in missing_file_names:
        assert file_name in str(exception_info.value)
    assert "FDE_MNIST_DATA_DIR" in str(exception_info.value)
    # 何も作らない（取得しない。途中のファイルも残さない）。
    assert sorted(path.name for path in tmp_path.iterdir()) == remaining_file_names
    # 失敗は、保持しない: ファイルを置けば、読める。
    write_mnist_files(tmp_path, pixel_rows=make_pixel_rows(1), digit_labels=[7])
    assert load_mnist_training_data(data_directory=tmp_path).digit_labels.tolist() == [7]


@pytest.mark.parametrize(
    "invalid_file_arguments",
    [
        dict(image_magic_number=2049),
        dict(label_magic_number=2051),
        dict(declared_image_count=3),
        dict(declared_label_count=3),
        dict(row_count=14, column_count=28, declared_image_count=4),
        dict(row_count=28, column_count=14, declared_image_count=4),
    ],
)
def test_invalid_file_formats_are_rejected(tmp_path, invalid_file_arguments):
    """識別の数値、中身の長さ、1件の画素数（784）が不正なファイルを拒否する。"""
    write_mnist_files(
        tmp_path, pixel_rows=make_pixel_rows(2), digit_labels=[1, 2], **invalid_file_arguments
    )
    with pytest.raises(ValueError):
        load_mnist_training_data(data_directory=tmp_path)


@pytest.mark.parametrize("damaged_file_name", [IMAGE_FILE_NAME, LABEL_FILE_NAME])
@pytest.mark.parametrize("damage", ["truncated_gzip", "not_gzip", "header_too_short", "empty"])
def test_damaged_files_are_rejected_as_invalid_format(tmp_path, damaged_file_name, damage):
    """途中で切れたgzip、gzipでないファイル、先頭の数値に足りないファイルを、形式の不正として拒否する。"""
    write_mnist_files(tmp_path, pixel_rows=make_pixel_rows(2), digit_labels=[1, 2])
    damaged_file_path = tmp_path / damaged_file_name
    if damage == "truncated_gzip":
        damaged_file_path.write_bytes(damaged_file_path.read_bytes()[:-12])
    elif damage == "not_gzip":
        damaged_file_path.write_bytes(b"not a gzip file" * 4)
    elif damage == "header_too_short":
        with gzip.open(damaged_file_path, "wb") as damaged_stream:
            damaged_stream.write(bytes(3))
    else:
        damaged_file_path.write_bytes(b"")
    with pytest.raises(ValueError, match=damaged_file_name):
        load_mnist_training_data(data_directory=tmp_path)
    # 失敗は、保持しない。
    write_mnist_files(tmp_path, pixel_rows=make_pixel_rows(2), digit_labels=[1, 2])
    assert load_mnist_training_data(data_directory=tmp_path).digit_labels.tolist() == [1, 2]


def test_differing_image_and_label_counts_are_rejected(tmp_path):
    write_mnist_files(tmp_path, pixel_rows=make_pixel_rows(2), digit_labels=[1, 2, 3])
    with pytest.raises(ValueError):
        load_mnist_training_data(data_directory=tmp_path)


def test_loading_requires_keyword_path_argument(tmp_path):
    with pytest.raises(TypeError):
        load_mnist_training_data(tmp_path)  # type: ignore[misc]
    with pytest.raises(TypeError, match="data_directory"):
        load_mnist_training_data(data_directory=str(tmp_path))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "make_invalid_value", "is_writable"),
    [
        ("pixel_values", lambda data: data.pixel_values.tolist(), False),
        ("pixel_values", lambda data: data.pixel_values.astype(np.float32), False),
        ("pixel_values", lambda data: data.pixel_values[:, :783], False),
        ("pixel_values", lambda data: data.pixel_values[:0], False),
        ("pixel_values", lambda data: data.pixel_values.copy(), True),
        ("digit_labels", lambda data: data.digit_labels.tolist(), False),
        ("digit_labels", lambda data: data.digit_labels.astype(np.uint8), False),
        ("digit_labels", lambda data: data.digit_labels.reshape(10, 1), False),
        ("digit_labels", lambda data: data.digit_labels[:9], False),
        ("digit_labels", lambda data: data.digit_labels.copy(), True),
    ],
)
def test_training_data_rejects_invalid_arrays(field_name, make_invalid_value, is_writable):
    """画素は、書込み不可のuint8の（件数≥1, 784）、ラベルは、書込み不可のint64の（件数,）だけを受け取る。"""
    valid_training_data = make_small_training_data()
    field_values = dict(
        pixel_values=valid_training_data.pixel_values,
        digit_labels=valid_training_data.digit_labels,
    )
    invalid_value = make_invalid_value(valid_training_data)
    if isinstance(invalid_value, np.ndarray):
        # 書込みの可否だけが不正な場合と、型・形だけが不正な場合を、分けて確かめる。
        invalid_value.setflags(write=is_writable)
    field_values[field_name] = invalid_value
    with pytest.raises((TypeError, ValueError), match=field_name):
        MnistTrainingData(**field_values)


# ---- 生成 ----


@pytest.mark.parametrize("concept_id", [0, 1, 2, 3])
def test_labels_are_swapped_by_concept_and_features_follow_the_chosen_image(concept_id):
    """選んだ位置の画像の特徴（float32の値）と、概念で入れ替えたラベルを返す。乱数は、1回だけ引く。"""
    mnist_training_data = make_small_training_data()
    swapped_digits = SWAPPED_DIGITS_BY_CONCEPT_ID[concept_id]
    for position in range(10):
        numpy_random_generator = FixedPositionRandomState(
            expected_sample_count=10, position=position
        )
        sample_generator = MnistSampleGenerator(
            numpy_random_generator=numpy_random_generator,
            concept_count=4,
            mnist_training_data=mnist_training_data,
        )
        observed_sample = sample_generator.generate_sample(concept_id=concept_id)
        assert type(observed_sample) is ObservedSample
        assert numpy_random_generator.draw_count == 1
        expected_class_label = position
        if swapped_digits is not None and position in swapped_digits:
            expected_class_label = swapped_digits[1 - swapped_digits.index(position)]
        assert observed_sample.class_label == expected_class_label
        assert type(observed_sample.class_label) is int
        assert observed_sample.feature_values == tuple(
            float(np.float32(pixel_value) / np.float32(255.0))
            for pixel_value in mnist_training_data.pixel_values[position].tolist()
        )
        assert len(observed_sample.feature_values) == 784
        assert all(type(feature_value) is float for feature_value in observed_sample.feature_values)


def test_feature_floats_are_shared_between_samples():
    """同じ画素の値は、標本の間で、同じfloatのobjectを使い回す（標本ごとのメモリを小さくする）。"""
    mnist_training_data = make_small_training_data()
    observed_samples = [
        MnistSampleGenerator(
            numpy_random_generator=FixedPositionRandomState(
                expected_sample_count=10, position=position
            ),
            concept_count=2,
            mnist_training_data=mnist_training_data,
        ).generate_sample(concept_id=0)
        for position in (0, 1)
    ]
    float_object_by_pixel_value = {}
    for position, observed_sample in zip((0, 1), observed_samples, strict=True):
        for pixel_value, feature_value in zip(
            mnist_training_data.pixel_values[position].tolist(),
            observed_sample.feature_values,
            strict=True,
        ):
            assert float_object_by_pixel_value.setdefault(pixel_value, feature_value) is (
                feature_value
            )
    assert len(float_object_by_pixel_value) == 256


@pytest.mark.parametrize("random_seed", [0, 17])
@pytest.mark.parametrize(("dataset_name", "concept_id"), MNIST_DATASET_NAME_AND_CONCEPT_ID_PAIRS)
def test_samples_match_legacy_generation_and_random_state(dataset_name, concept_id, random_seed):
    """同じ乱数から、実旧の`generate_data`と、特徴（float32）・ラベル・生成後の乱数が一致する。"""
    from federated_drift_experiment.data import streams as legacy_streams_module

    numpy_random_generator = np.random.RandomState(random_seed)
    sample_generator = create_observed_sample_generator(
        dataset_name=dataset_name, numpy_random_generator=numpy_random_generator
    )
    python_random_state = random.getstate()
    global_numpy_random_state = np.random.get_state()
    try:
        # 実旧は、NumPyの全体の乱数を使う。
        np.random.seed(random_seed)
        observed_class_labels = set()
        for _sample_index in range(400):
            legacy_feature_tensor, legacy_label_tensor = legacy_streams_module.generate_data(
                concept_id, dataset=dataset_name
            )
            observed_sample = sample_generator.generate_sample(concept_id=concept_id)
            assert observed_sample.feature_values == tuple(legacy_feature_tensor.tolist())
            assert observed_sample.class_label == int(legacy_label_tensor.item())
            observed_class_labels.add(observed_sample.class_label)
        assert_numpy_random_states_equal(numpy_random_generator.get_state(), np.random.get_state())
    finally:
        np.random.set_state(global_numpy_random_state)
    assert random.getstate() == python_random_state
    assert observed_class_labels == set(range(10))


@pytest.mark.parametrize("dataset_name", MNIST_DATASET_NAMES)
@pytest.mark.parametrize("invalid_concept_id", [-1, "count", True, False, 0.0, "0", None, []])
def test_samples_reject_invalid_concepts_without_consuming_randomness(
    dataset_name, invalid_concept_id
):
    """datasetの概念数以上・負・整数以外の概念IDを、乱数を進めずに拒否する（範囲外は、実旧も拒否する）。"""
    from federated_drift_experiment.data import streams as legacy_streams_module

    concept_count = get_dataset_definition(dataset_name=dataset_name).concept_count
    if invalid_concept_id == "count":
        invalid_concept_id = concept_count
    numpy_random_generator = np.random.RandomState(0)
    initial_numpy_random_state = numpy_random_generator.get_state()
    sample_generator = create_observed_sample_generator(
        dataset_name=dataset_name, numpy_random_generator=numpy_random_generator
    )
    with pytest.raises((TypeError, ValueError), match="concept_id"):
        sample_generator.generate_sample(concept_id=invalid_concept_id)
    assert_numpy_random_states_equal(numpy_random_generator.get_state(), initial_numpy_random_state)
    if type(invalid_concept_id) is int:
        with pytest.raises(ValueError, match="concept_id"):
            legacy_streams_module.generate_data(invalid_concept_id, dataset=dataset_name)


@pytest.mark.parametrize(
    ("argument_name", "invalid_value"),
    [
        ("concept_count", 0),
        ("concept_count", 5),
        ("concept_count", True),
        ("concept_count", 2.0),
        ("concept_count", None),
        ("mnist_training_data", None),
        ("mnist_training_data", (np.zeros((1, 784), dtype=np.uint8), np.zeros(1, dtype=np.int64))),
    ],
)
def test_generator_rejects_invalid_construction_arguments(argument_name, invalid_value):
    arguments = dict(
        numpy_random_generator=np.random.RandomState(0),
        concept_count=2,
        mnist_training_data=make_small_training_data(),
    )
    arguments[argument_name] = invalid_value
    with pytest.raises((TypeError, ValueError), match=argument_name):
        MnistSampleGenerator(**arguments)
    with pytest.raises(TypeError):
        MnistSampleGenerator(np.random.RandomState(0), 2, make_small_training_data())  # type: ignore[misc]


@pytest.mark.parametrize("dataset_name", MNIST_DATASET_NAMES)
def test_created_generator_uses_loaded_training_data_without_consuming_randomness(dataset_name):
    """dataset名から作った生成器は、置き場所の規則で読んだ学習用データと、定義の概念数を持つ。"""
    numpy_random_generator = np.random.RandomState(3)
    initial_numpy_random_state = numpy_random_generator.get_state()
    sample_generator = create_observed_sample_generator(
        dataset_name=dataset_name, numpy_random_generator=numpy_random_generator
    )
    assert type(sample_generator) is MnistSampleGenerator
    assert MnistSampleGenerator in OBSERVED_SAMPLE_GENERATOR_TYPES
    assert sample_generator.numpy_random_generator is numpy_random_generator
    assert (
        sample_generator.concept_count
        == get_dataset_definition(dataset_name=dataset_name).concept_count
    )
    assert sample_generator.mnist_training_data is load_mnist_training_data(
        data_directory=resolve_mnist_data_directory()
    )
    assert_numpy_random_states_equal(numpy_random_generator.get_state(), initial_numpy_random_state)


@pytest.mark.parametrize("generator_dataset_name", MNIST_DATASET_NAMES)
@pytest.mark.parametrize("dataset_name", ["sine2", "sea2", "sea4", "circle2", "mnist2", "mnist4"])
def test_mnist_generator_belongs_only_to_its_own_dataset(generator_dataset_name, dataset_name):
    """mnist2とmnist4の生成器は、同じ型でも、概念数で区別される。合成データの生成器でもない。"""
    sample_generator = create_observed_sample_generator(
        dataset_name=generator_dataset_name, numpy_random_generator=np.random.RandomState(0)
    )
    assert is_observed_sample_generator_of_dataset(
        sample_generator=sample_generator, dataset_name=dataset_name
    ) == (generator_dataset_name == dataset_name)
    other_generator = create_observed_sample_generator(
        dataset_name=dataset_name, numpy_random_generator=np.random.RandomState(0)
    )
    assert is_observed_sample_generator_of_dataset(
        sample_generator=other_generator, dataset_name=generator_dataset_name
    ) == (generator_dataset_name == dataset_name)
