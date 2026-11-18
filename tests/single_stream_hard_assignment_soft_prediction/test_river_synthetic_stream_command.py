"""書き出しのコマンドと、比較手法の研究ツールの事前学習の標本の検査（WSL）。"""

import json

import numpy as np
import pytest

pytest.importorskip("river")

import river  # noqa: E402

from single_stream_hard_assignment_soft_prediction.river_synthetic_recurring_streams import (  # noqa: E402
    RIVER_SYNTHETIC_STREAM_DEFINITIONS,
    generate_river_synthetic_stream,
)
from single_stream_hard_assignment_soft_prediction.river_synthetic_stream_command import (  # noqa: E402
    main,
)
from single_stream_hard_assignment_soft_prediction.single_stream_input import (  # noqa: E402
    compute_stream_array_digest,
    load_single_stream_input,
)
from single_stream_hard_assignment_soft_prediction.single_stream_run import (  # noqa: E402
    run_single_stream,
)
from single_stream_hard_assignment_soft_prediction.single_stream_settings import (  # noqa: E402
    build_default_single_stream_settings,
)

METADATA_KEYS = {
    "export_format_version",
    "stream_source",
    "dataset_name",
    "random_seed",
    "sample_count",
    "class_count",
    "concept_count",
    "feature_count",
    "feature_names",
    "array_sha256",
    "concept_change_count",
    "generator_settings",
    "concept_schedule_settings",
    "river_version",
}


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    output = tmp_path_factory.mktemp("streams")
    arguments = ["--datasets", "stagger3", "agrawal4", "randomrbf4", "--seeds", "0", "1"]
    assert main([*arguments, "--sample-count", "1200", "--output", str(output)]) == 0
    return output


def test_written_streams_are_loaded_and_match_the_generator(written):
    names = sorted(p.name for p in written.iterdir())
    assert len(names) == 12 and not any(".tmp." in n for n in names)
    for name, definition in RIVER_SYNTHETIC_STREAM_DEFINITIONS.items():
        for seed in (0, 1):
            path = written / f"{name}_seed{seed}_n1200.npz"
            stream = load_single_stream_input(stream_path=path)
            assert stream.class_count == 2 and stream.dataset_name == name
            arrays = generate_river_synthetic_stream(
                dataset_name=name, random_seed=seed, sample_count=1200
            )
            np.testing.assert_array_equal(stream.features, arrays["features"])
            np.testing.assert_array_equal(stream.labels, arrays["labels"])
            np.testing.assert_array_equal(stream.concept_ids, arrays["concept_ids"])
            with np.load(path) as loaded:
                assert set(loaded.files) == {"features", "labels", "concept_ids"}
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            assert set(metadata) == METADATA_KEYS
            assert metadata["array_sha256"] == compute_stream_array_digest(arrays)
            assert metadata["generator_settings"] == dict(definition.fixed_generator_arguments)
            assert metadata["feature_names"] == list(definition.feature_names)
            assert metadata["feature_count"] == len(definition.feature_names)
            assert (metadata["class_count"], metadata["concept_count"]) == (
                2,
                definition.concept_count,
            )
            assert metadata["concept_change_count"] == int(
                np.count_nonzero(np.diff(arrays["concept_ids"]))
            )
            assert metadata["concept_schedule_settings"] == {
                "minimum_sample_index_gap_before_change_trial": 300,
                "per_eligible_sample_concept_change_probability": 0.0015,
            }
            assert metadata["river_version"] == river.__version__
            assert metadata["stream_source"] == "river_synthetic_generator"
            assert (metadata["random_seed"], metadata["sample_count"]) == (seed, 1200)


def test_existing_pairs_are_skipped(written):
    before = {p.name: p.stat().st_mtime_ns for p in written.iterdir()}
    assert (
        main(
            [
                "--datasets",
                "stagger3",
                "--seeds",
                "0",
                "--sample-count",
                "1200",
                "--output",
                str(written),
            ]
        )
        == 0
    )
    assert {p.name: p.stat().st_mtime_ns for p in written.iterdir()} == before


def test_half_written_stream_stops_before_writing_anything(tmp_path):
    (tmp_path / "agrawal4_seed1_n300.npz").write_bytes(b"")
    with pytest.raises(SystemExit, match="agrawal4_seed1_n300"):
        main(
            [
                "--datasets",
                "stagger3",
                "agrawal4",
                "--seeds",
                "0",
                "1",
                "--sample-count",
                "300",
                "--output",
                str(tmp_path),
            ]
        )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["agrawal4_seed1_n300.npz"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--datasets", "sine2", "--seeds", "0"],
        ["--datasets", "stagger3", "unknown", "--seeds", "0"],
        ["--datasets", "--seeds", "0"],
        ["--datasets", "stagger3", "--seeds"],
        ["--datasets", "stagger3", "--seeds", "-1"],
    ],
)
def test_invalid_arguments_are_rejected_before_writing(tmp_path, arguments):
    output = tmp_path / "out"
    with pytest.raises(SystemExit):
        main([*arguments, "--sample-count", "100", "--output", str(output)])
    assert not output.exists()


def test_tree_version_processes_a_written_stream(written):
    stream = load_single_stream_input(stream_path=written / "randomrbf4_seed0_n1200.npz")
    result = run_single_stream(stream=stream, settings=build_default_single_stream_settings())
    assert result.summary["sample_count"] == 1200
    assert 0.0 <= result.summary["prequential_accuracy"] <= 1.0


def test_river_baselines_refuse_streams_without_pretraining_unless_requested(written, tmp_path):
    from tools.research_single_client_stream_baselines import river_baselines

    streams = tmp_path / "streams"
    streams.mkdir()
    for suffix in (".npz", ".json"):
        (streams / f"stagger3_seed0_n1200{suffix}").write_bytes(
            (written / f"stagger3_seed0_n1200{suffix}").read_bytes()
        )
    output = tmp_path / "river"
    with pytest.raises(ValueError, match="stagger3_seed0_n1200"):
        river_baselines.main(
            [
                "--streams",
                str(streams),
                "--output",
                str(output),
                "--methods",
                "dwm",
                "--workers",
                "1",
            ]
        )
    assert not output.exists()
    status = river_baselines.main(
        [
            "--streams",
            str(streams),
            "--output",
            str(output),
            "--methods",
            "dwm",
            "--workers",
            "1",
            "--without-pretraining",
        ]
    )
    assert status == 0
    (result_path,) = (output / "runs").glob("*.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["pretraining_samples_learned"] is False and result["dataset_name"] == "stagger3"
