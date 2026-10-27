"""追加画像データの概念・選択位置・IDXの読込みを確認する。"""

from pathlib import Path

import numpy as np
import pytest

from federated_learning_experiments.data.observed_sample_generation import (
    create_observed_sample_generator,
    is_observed_sample_generator_of_dataset,
)
from tests.refactoring.test_mnist_sample_generation import (
    IMAGE_FILE_NAME,
    LABEL_FILE_NAME,
    assert_numpy_random_states_equal,
    make_pixel_rows,
    write_mnist_files,
)


@pytest.fixture
def image_directories(tmp_path, monkeypatch):
    """画像の位置が同じなら特徴が一致し、元のラベルだけが違う2つの入力。"""
    mnist_directory = tmp_path / "mnist"
    fashion_directory = tmp_path / "fashion-mnist"
    for directory, labels in (
        (mnist_directory, list(reversed(range(10)))),
        (fashion_directory, list(range(10))),
    ):
        write_mnist_files(directory, pixel_rows=make_pixel_rows(10), digit_labels=labels)
    monkeypatch.setenv("FDE_MNIST_DATA_DIR", str(mnist_directory))
    monkeypatch.setenv("FDE_FASHION_MNIST_DATA_DIR", str(fashion_directory))
    return mnist_directory, fashion_directory


@pytest.mark.parametrize("dataset_name", ["fmnist2", "fmnist4"])
def test_fashion_labels_and_sampling_follow_mnist(image_directories, dataset_name):
    """全ラベルの入れ替えと、seedごとの画像選択・乱数の消費を一緒に照合する。"""
    concept_count = int(dataset_name[-1])
    for concept_id in range(concept_count):
        fashion_random = np.random.RandomState(17)
        mnist_random = np.random.RandomState(17)
        fashion = create_observed_sample_generator(
            dataset_name=dataset_name, numpy_random_generator=fashion_random
        )
        mnist = create_observed_sample_generator(
            dataset_name=f"mnist{concept_count}", numpy_random_generator=mnist_random
        )
        labels_seen = set()
        swapped_labels = {1: (1, 2), 2: (3, 4), 3: (5, 6)}.get(concept_id)
        for _ in range(100):
            fashion_sample = fashion.generate_sample(concept_id=concept_id)
            mnist_sample = mnist.generate_sample(concept_id=0)
            original_label = 9 - mnist_sample.class_label
            labels_seen.add(original_label)
            expected_label = original_label
            if swapped_labels and original_label in swapped_labels:
                expected_label = swapped_labels[1 - swapped_labels.index(original_label)]
            assert fashion_sample.class_label == expected_label
            assert fashion_sample.feature_values == mnist_sample.feature_values
        assert labels_seen == set(range(10))
        assert_numpy_random_states_equal(fashion_random.get_state(), mnist_random.get_state())


@pytest.mark.parametrize("dataset_name", ["rotmnist2", "rotmnist4"])
def test_rotation_matches_official_numpy_operations(image_directories, dataset_name):
    """公式のreshape→rot90→flattenをoracleにし、ラベル・乱数も照合する。"""
    concept_count = int(dataset_name[-1])
    for concept_id in range(concept_count):
        rotated_random = np.random.RandomState(3)
        mnist_random = np.random.RandomState(3)
        rotated = create_observed_sample_generator(
            dataset_name=dataset_name, numpy_random_generator=rotated_random
        )
        mnist = create_observed_sample_generator(
            dataset_name=f"mnist{concept_count}", numpy_random_generator=mnist_random
        )
        rotation_k = concept_id * 2 if concept_count == 2 else concept_id
        for _ in range(20):
            actual = rotated.generate_sample(concept_id=concept_id)
            original = mnist.generate_sample(concept_id=0)
            expected = np.rot90(np.asarray(original.feature_values).reshape(28, 28), k=rotation_k)
            assert actual.feature_values == tuple(expected.flatten())
            assert actual.class_label == original.class_label
            if concept_id == 0:
                assert actual == original
        assert_numpy_random_states_equal(rotated_random.get_state(), mnist_random.get_state())
        state = rotated_random.get_state()
        for invalid_concept in (-1, concept_count, True):
            with pytest.raises((TypeError, ValueError), match="concept_id"):
                rotated.generate_sample(concept_id=invalid_concept)
            assert_numpy_random_states_equal(rotated_random.get_state(), state)


def test_generators_belong_only_to_their_dataset(image_directories):
    """同じ概念数でも、画像の元・ラベル交換・回転を取り違えない。"""
    dataset_names = ("mnist2", "mnist4", "fmnist2", "fmnist4", "rotmnist2", "rotmnist4")
    for generator_dataset_name in dataset_names:
        generator = create_observed_sample_generator(
            dataset_name=generator_dataset_name, numpy_random_generator=np.random.RandomState(0)
        )
        for dataset_name in dataset_names:
            assert is_observed_sample_generator_of_dataset(
                sample_generator=generator, dataset_name=dataset_name
            ) == (generator_dataset_name == dataset_name)


def test_fashion_directory_resolution(monkeypatch, tmp_path):
    from federated_learning_experiments.data.mnist.mnist_training_data import (
        resolve_fashion_mnist_data_directory,
    )

    monkeypatch.delenv("FDE_FASHION_MNIST_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert resolve_fashion_mnist_data_directory() == (
        Path(__file__).resolve().parents[2] / "data" / "fashion-mnist"
    )
    monkeypatch.setenv("FDE_FASHION_MNIST_DATA_DIR", str(tmp_path))
    assert resolve_fashion_mnist_data_directory() == tmp_path


@pytest.mark.parametrize("missing_file_name", [IMAGE_FILE_NAME, LABEL_FILE_NAME])
def test_fashion_missing_file_is_rejected_without_download(
    tmp_path, monkeypatch, missing_file_name
):
    write_mnist_files(tmp_path, pixel_rows=make_pixel_rows(1), digit_labels=[1])
    (tmp_path / missing_file_name).unlink()
    monkeypatch.setenv("FDE_FASHION_MNIST_DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="FDE_FASHION_MNIST_DATA_DIR"):
        create_observed_sample_generator(
            dataset_name="fmnist2", numpy_random_generator=np.random.RandomState(0)
        )
    assert not (tmp_path / missing_file_name).exists()


def test_fashion_reuses_idx_validation(tmp_path, monkeypatch):
    write_mnist_files(
        tmp_path, pixel_rows=make_pixel_rows(1), digit_labels=[1], image_magic_number=0
    )
    monkeypatch.setenv("FDE_FASHION_MNIST_DATA_DIR", str(tmp_path))
    with pytest.raises(ValueError, match=IMAGE_FILE_NAME):
        create_observed_sample_generator(
            dataset_name="fmnist2", numpy_random_generator=np.random.RandomState(0)
        )
