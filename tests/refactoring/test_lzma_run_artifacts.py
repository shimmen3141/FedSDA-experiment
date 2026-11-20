"""4手法（FedSDA・FedDrift・FedAvg・DWM）のrunの記録の表が、LZMAのNPZとして書かれ、配列が変わらないこと。

- 書いたNPZの全エントリがLZMAで、書き出し関数へ渡した表と、読んだ配列の名前の順・型・形・値が同じ。
- 記録の表にobjectの配列が無い。
- 同じ表をdeflate（`numpy.savez_compressed`）で書き直したNPZも、読込み関数が同じに読む（既存の結果）。
- 書き出しの失敗で、途中のファイルも要約も残らない。
"""

import zipfile

import numpy as np
import pytest

from federated_learning_experiments.configuration.experiment_run_conditions import (
    ExperimentRunConditions,
)
from federated_learning_experiments.runtime import (
    dynamic_weighted_majority_run_artifact_files,
    fedavg_run_artifact_files,
    feddrift_run_artifact_files,
    fedsda_run_artifact_files,
)
from federated_learning_experiments.runtime.dynamic_weighted_majority_experiment_run import (
    execute_dynamic_weighted_majority_experiment_run,
)
from federated_learning_experiments.runtime.fedavg_experiment_run import (
    execute_fedavg_experiment_run,
)
from federated_learning_experiments.runtime.feddrift_experiment_run import (
    execute_feddrift_experiment_run,
)
from federated_learning_experiments.runtime.fedsda_experiment_run import (
    execute_fedsda_experiment_run,
)
from federated_learning_experiments.sweep.experiment_configurations import (
    build_experiment_run_settings,
)
from tests.refactoring.run_provenance_examples import TEST_RUN_PROVENANCE

METHODS = {
    "fedsda": (
        execute_fedsda_experiment_run,
        fedsda_run_artifact_files,
        "save_fedsda_experiment_run",
        "load_fedsda_run_artifacts",
    ),
    "feddrift": (
        execute_feddrift_experiment_run,
        feddrift_run_artifact_files,
        "save_feddrift_experiment_run",
        "load_feddrift_run_artifacts",
    ),
    "fedavg": (
        execute_fedavg_experiment_run,
        fedavg_run_artifact_files,
        "save_fedavg_experiment_run",
        "load_fedavg_run_artifacts",
    ),
    "dwm-neural-network": (
        execute_dynamic_weighted_majority_experiment_run,
        dynamic_weighted_majority_run_artifact_files,
        "save_dynamic_weighted_majority_experiment_run",
        "load_dynamic_weighted_majority_run_artifacts",
    ),
}


@pytest.fixture(scope="module")
def experiment_runs():
    runs = {}
    for configuration_name, (execute, *_rest) in METHODS.items():
        run_settings = build_experiment_run_settings(
            configuration_name=configuration_name,
            experiment_run_conditions=ExperimentRunConditions(
                dataset_name="sea4",
                random_seed=0,
                client_count=2,
                per_client_sample_count=60,
                server_aggregation_interval_per_client_samples=30,
            ),
        )
        runs[configuration_name] = execute(run_settings=run_settings)
    return runs


def _save(configuration_name, experiment_run, directory):
    module, save_name = METHODS[configuration_name][1], METHODS[configuration_name][2]
    return getattr(module, save_name)(
        experiment_run=experiment_run,
        output_directory=directory,
        file_stem="run-a",
        run_provenance=TEST_RUN_PROVENANCE,
    )


def _same_arrays(first, second):
    assert list(first) == list(second)
    for name, array in first.items():
        other = np.asarray(second[name])
        assert array.dtype == other.dtype and array.shape == other.shape, name
        assert np.array_equal(array, other, equal_nan=array.dtype.kind in "fc"), name


@pytest.mark.parametrize("configuration_name", list(METHODS))
def test_record_tables_are_written_with_lzma_and_unchanged(
    configuration_name, experiment_runs, tmp_path, monkeypatch
):
    module = METHODS[configuration_name][1]
    captured = {}
    real_writer = module.write_lzma_compressed_record_tables

    def capture(stream, record_tables):
        captured.update(record_tables)
        real_writer(stream, record_tables)

    monkeypatch.setattr(module, "write_lzma_compressed_record_tables", capture)
    paths = _save(configuration_name, experiment_runs[configuration_name], tmp_path)
    with zipfile.ZipFile(paths.record_table_file_path) as archive:
        assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_LZMA}
    with np.load(paths.record_table_file_path, allow_pickle=False) as loaded:
        written = {name: loaded[name] for name in loaded.files}
    _same_arrays(written, captured)
    assert all(np.asarray(values).dtype.kind != "O" for values in captured.values())


@pytest.mark.parametrize("configuration_name", list(METHODS))
def test_deflate_and_lzma_record_tables_are_read_alike(
    configuration_name, experiment_runs, tmp_path
):
    module, load_name = METHODS[configuration_name][1], METHODS[configuration_name][3]
    paths = _save(configuration_name, experiment_runs[configuration_name], tmp_path)
    load = getattr(module, load_name)
    lzma_summary, lzma_tables = load(summary_file_path=paths.summary_file_path)
    # 既存の結果と同じdeflateのNPZへ書き直す（配列は同じ）。
    with np.load(paths.record_table_file_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    np.savez_compressed(paths.record_table_file_path, **arrays)
    with zipfile.ZipFile(paths.record_table_file_path) as archive:
        assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_DEFLATED}
    deflate_summary, deflate_tables = load(summary_file_path=paths.summary_file_path)
    assert deflate_summary == lzma_summary
    _same_arrays({k: np.asarray(v) for k, v in lzma_tables.items()}, deflate_tables)


@pytest.mark.parametrize("configuration_name", list(METHODS))
def test_a_failure_while_writing_the_tables_leaves_no_files(
    configuration_name, experiment_runs, tmp_path, monkeypatch
):
    module = METHODS[configuration_name][1]

    def fail(*arguments, **keyword_arguments):
        raise OSError("simulated failure while writing the tables")

    monkeypatch.setattr(module, "write_lzma_compressed_record_tables", fail)
    with pytest.raises(OSError, match="simulated failure"):
        _save(configuration_name, experiment_runs[configuration_name], tmp_path)
    assert list(tmp_path.iterdir()) == []
