import csv
from pathlib import Path

import numpy as np

from build_feddrift_baseline import BaselineSource, build_baseline


def _write_source(root, dataset, include_batch25=False):
    pareto = root / "pareto" / "pareto.csv"
    raw = root / "raw"
    pareto.parent.mkdir(parents=True)
    raw.mkdir()
    fieldnames = [
        "mode", "dataset", "seed", "series", "sweep_value",
        "feddrift_batch", "agg_interval", "distance_threshold",
        "adwin_delta", "accuracy",
    ]
    batches = [25, 50] if include_batch25 else [50]
    rows = []
    for batch in batches:
        rows.append({
            "mode": "FedDrift_v2",
            "dataset": dataset,
            "seed": 0,
            "series": "FedDrift_v2 batch sweep (delta=0.1)",
            "sweep_value": batch,
            "feddrift_batch": batch,
            "agg_interval": 50,
            "distance_threshold": 0.1,
            "adwin_delta": 0.05,
            "accuracy": 0.9,
        })
        np.savez_compressed(
            raw / f"FedDrift_v2_batch_sweep_0_1_{dataset}_seed0_sv{batch}.npz",
            mode=np.asarray("FedDrift_v2"),
            dataset=np.asarray(dataset),
            label=np.asarray(f"FedDrift_v2 batch sweep (delta=0.1) [{batch}]"),
            seed=np.asarray(0),
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
    assert rows[0]["dataset"] == "circle2"
    assert rows[0]["mode"] == "FedDrift"
    assert rows[0]["series"] == (
        "FedDrift B_detect sweep (delta_feddrift=0.1)"
    )
    assert rows[0]["sweep_parameter"] == "b_detect"
    assert rows[0]["b_detect"] == "50"
    assert rows[0]["delta_feddrift"] == "0.1"
    assert rows[0]["concept_schedule"] == "random"
    assert "feddrift_batch" not in fieldnames
    assert "distance_threshold" not in fieldnames
    assert "agg_interval" not in fieldnames
    assert "adwin_delta" not in fieldnames

    raw_path = output / "circle2" / "raw" / "b_detect_sweep_seed0_b50.npz"
    with np.load(raw_path, allow_pickle=False) as archive:
        assert archive["mode"].item() == "FedDrift"
        assert archive["b_detect"].item() == 50
        assert archive["delta_feddrift"].item() == 0.1
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
