import csv
from pathlib import Path

import numpy as np

from tools.baselines.build_feddrift import (
    BaselineSource,
    FEDDRIFT_BATCH,
    FEDDRIFT_DISTANCE,
    build_baseline,
    extend_baseline,
)


def _write_source(
    root, dataset, include_batch25=False, seed=0, accuracy=0.9, batches=None
):
    pareto = root / "pareto" / "pareto.csv"
    raw = root / "raw"
    pareto.parent.mkdir(parents=True)
    raw.mkdir()
    fieldnames = [
        "mode", "dataset", "seed", "series", "sweep_value",
        "feddrift_batch", "agg_interval", "distance_threshold",
        "adwin_delta", "accuracy",
    ]
    batches = batches or ([25, 50] if include_batch25 else [50])
    rows = []
    for batch in batches:
        rows.append({
            "mode": "FedDrift_v2",
            "dataset": dataset,
            "seed": seed,
            "series": "FedDrift_v2 batch sweep (delta=0.1)",
            "sweep_value": batch,
            "feddrift_batch": batch,
            "agg_interval": 50,
            "distance_threshold": 0.1,
            "adwin_delta": 0.05,
            "accuracy": accuracy,
        })
        np.savez_compressed(
            raw / f"FedDrift_v2_batch_sweep_0_1_{dataset}_seed{seed}_sv{batch}.npz",
            mode=np.asarray("FedDrift_v2"),
            dataset=np.asarray(dataset),
            label=np.asarray(f"FedDrift_v2 batch sweep (delta=0.1) [{batch}]"),
            seed=np.asarray(seed),
            history_accuracy=np.asarray([[1, 0]], dtype=np.int8),
        )
    with pareto.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return BaselineSource(
        root=root, csv_path=pareto, datasets=(dataset,)
    )


def test_build_baseline_normalizes_names_and_groups_by_dataset(tmp_path):
    source = _write_source(tmp_path / "source", "circle", include_batch25=True)
    output = tmp_path / "baseline"

    manifest = build_baseline(
        sources=(source,), output_root=output, batches=(50,)
    )

    with (output / "circle2" / "metrics.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))
        fieldnames = file.seek(0) or next(csv.reader(file))

    assert len(rows) == 1
    assert rows[0]["parameter_schema_version"] == "1"
    assert rows[0]["dataset"] == "circle2"
    assert rows[0]["mode"] == "FedDrift"
    assert rows[0]["series"] == (
        "FedDrift B_detect sweep (δ_FedDrift=0.1)"
    )
    assert rows[0]["sweep_parameter"] == FEDDRIFT_BATCH
    assert rows[0][FEDDRIFT_BATCH] == "50"
    assert rows[0][FEDDRIFT_DISTANCE] == "0.1"
    assert rows[0]["concept_schedule"] == "random"
    assert "feddrift_batch" not in fieldnames
    assert "distance_threshold" not in fieldnames
    assert "agg_interval" not in fieldnames
    assert "adwin_delta" not in fieldnames

    raw_path = (
        output / "circle2" / "raw"
        / f"{FEDDRIFT_BATCH}_sweep_seed0_b50.npz"
    )
    with np.load(raw_path, allow_pickle=False) as archive:
        assert archive["mode"].item() == "FedDrift"
        assert archive[FEDDRIFT_BATCH].item() == 50
        assert archive[FEDDRIFT_DISTANCE].item() == 0.1
        assert archive["concept_schedule"].item() == "random"

    assert manifest["datasets"]["circle2"]["metrics_rows"] == 1
    assert manifest["datasets"]["circle2"]["raw_files"] == 1


def test_build_baseline_refuses_to_overwrite_existing_output(tmp_path):
    source = _write_source(tmp_path / "source", "circle")
    output = tmp_path / "baseline"
    output.mkdir()

    try:
        build_baseline(sources=(source,), output_root=output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("既存の固定ベースラインを上書きしてはいけない")


def test_extend_baseline_adds_new_seed_atomically_and_keeps_backup(tmp_path):
    initial = _write_source(tmp_path / "initial", "circle", seed=0)
    additional = _write_source(tmp_path / "additional", "circle", seed=1)
    output = tmp_path / "baseline"
    backup = tmp_path / "baseline_before_extension"
    build_baseline(sources=(initial,), output_root=output)

    manifest, actual_backup = extend_baseline(
        sources=(additional,),
        output_root=output,
        backup_root=backup,
    )

    assert actual_backup == backup.resolve()
    assert (
        backup / "circle2" / "raw"
        / f"{FEDDRIFT_BATCH}_sweep_seed0_b50.npz"
    ).is_file()
    with (output / "circle2" / "metrics.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))
    assert [int(row["seed"]) for row in rows] == [0, 1]
    assert manifest["selection"]["seeds"] == [0, 1]
    assert manifest["last_extension"]["added_results"] == 1
    assert manifest["datasets"]["circle2"]["metrics_rows"] == 2
    assert (
        output / "circle2" / "raw"
        / f"{FEDDRIFT_BATCH}_sweep_seed1_b50.npz"
    ).is_file()


def test_extend_baseline_rejects_conflicting_duplicate(tmp_path):
    initial = _write_source(tmp_path / "initial", "circle", accuracy=0.9)
    conflicting = _write_source(tmp_path / "conflict", "circle", accuracy=0.8)
    output = tmp_path / "baseline"
    build_baseline(sources=(initial,), output_root=output)

    try:
        extend_baseline(sources=(conflicting,), output_root=output)
    except ValueError as exc:
        assert "競合" in str(exc)
    else:
        raise AssertionError("同じ実験キーの異なる結果を統合してはいけない")

    with (output / "circle2" / "metrics.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["accuracy"] == "0.9"


def test_extend_baseline_adds_new_sweep_value(tmp_path):
    initial = _write_source(tmp_path / "initial", "circle", batches=[50])
    additional = _write_source(tmp_path / "additional", "circle", batches=[100])
    output = tmp_path / "baseline"
    build_baseline(sources=(initial,), output_root=output, batches=(50,))

    manifest, _ = extend_baseline(
        sources=(additional,), output_root=output, batches=(100,)
    )

    assert manifest["selection"][FEDDRIFT_BATCH] == [50, 100]
    assert manifest["last_extension"]["added_results"] == 1
    assert (
        output / "circle2" / "raw"
        / f"{FEDDRIFT_BATCH}_sweep_seed0_b100.npz"
    ).is_file()
