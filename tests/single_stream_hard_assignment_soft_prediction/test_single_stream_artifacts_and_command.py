"""成果物の保存→読込みの往復、上書きの拒否、実行コマンド（飛ばし・競合・一覧のCSV）を確かめる（WSL）。"""

import csv
import json

import numpy as np
import pytest

pytest.importorskip("river")

from single_stream_hard_assignment_soft_prediction.single_stream_artifacts import (  # noqa: E402
    load_single_stream_run,
    save_single_stream_run,
)
from single_stream_hard_assignment_soft_prediction.single_stream_command import main  # noqa: E402
from single_stream_hard_assignment_soft_prediction.single_stream_input import (  # noqa: E402
    compute_stream_array_digest,
    load_single_stream_input,
)
from single_stream_hard_assignment_soft_prediction.single_stream_run import (
    run_single_stream,  # noqa: E402
)
from single_stream_hard_assignment_soft_prediction.single_stream_settings import (  # noqa: E402
    build_default_single_stream_settings,
)
from tests.single_stream_hard_assignment_soft_prediction.stream_examples import (
    make_stream,  # noqa: E402
)


def write_stream(directory, seed):
    stream = make_stream([(0, 400), (1, 400)])
    arrays = dict(
        features=np.array([[x[0], x[1]] for x, _, _ in stream], dtype=np.float32),
        labels=np.array([y for _, y, _ in stream], dtype=np.int64),
        concept_ids=np.array([c for _, _, c in stream], dtype=np.int64),
        pretraining_features=np.zeros((3, 2), dtype=np.float32),
        pretraining_labels=np.array([0, 1, 0]),
    )
    path = directory / f"sine2_seed{seed}_n800.npz"
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(
        json.dumps(
            dict(
                dataset_name="sine2",
                random_seed=seed,
                array_sha256=compute_stream_array_digest(arrays),
            )
        ),
        encoding="utf-8",
    )
    return path


def test_saved_run_round_trips_and_is_not_overwritten(tmp_path):
    stream = load_single_stream_input(stream_path=write_stream(tmp_path, 0))
    settings = build_default_single_stream_settings()
    result = run_single_stream(stream=stream, settings=settings)
    json_path, _ = save_single_stream_run(
        result=result,
        settings=settings,
        stream=stream,
        output_directory=tmp_path / "out",
        file_stem="run",
    )
    document, tables = load_single_stream_run(json_path=json_path)
    assert set(tables) == set(result.record_tables)
    for name, values in result.record_tables.items():
        np.testing.assert_array_equal(tables[name], values)
    assert document["summary"]["prequential_accuracy"] == pytest.approx(
        result.summary["prequential_accuracy"]
    )
    assert document["settings"]["warm_up_sample_count"] == 200
    assert document["input"]["array_sha256"] == stream.array_sha256
    assert document["environment"]["river"] == "0.27.0"
    with pytest.raises(FileExistsError):
        save_single_stream_run(
            result=result,
            settings=settings,
            stream=stream,
            output_directory=tmp_path / "out",
            file_stem="run",
        )


def test_command_runs_streams_skips_completed_runs_and_refuses_partial_outputs(tmp_path, capsys):
    streams = tmp_path / "streams"
    streams.mkdir()
    for seed in (0, 1):
        write_stream(streams, seed)
    output = tmp_path / "output"
    assert main(["--streams", str(streams), "--output", str(output), "--workers", "2"]) == 0
    with (output / "run_summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["run_identifier"] for r in rows] == ["sine2_seed0_n800", "sine2_seed1_n800"]
    assert all(0.0 <= float(r["prequential_accuracy"]) <= 1.0 for r in rows)
    capsys.readouterr()
    assert main(["--streams", str(streams), "--output", str(output), "--seeds", "0"]) == 0
    assert "skip sine2_seed0_n800" in capsys.readouterr().out
    (output / "runs" / "sine2_seed1_n800.npz").unlink()
    with pytest.raises(RuntimeError):
        main(["--streams", str(streams), "--output", str(output), "--seeds", "1"])
