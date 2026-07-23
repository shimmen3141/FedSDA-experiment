import csv

import numpy as np

from migrate_result_modes import migrate_results


def test_migration_normalizes_modes_and_removes_ucb(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    csv_path = source / "pareto.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["mode", "series", "accuracy"])
        writer.writeheader()
        writer.writerow({
            "mode": "FedSDA_v2.2",
            "series": "FedSDA_v2.2 AGG_INTERVAL sweep (δ_adwin=0.05)",
            "accuracy": 0.9,
        })
        writer.writerow({
            "mode": "FedSDA_v2.2_ucb", "series": "FedSDA_v2.2_ucb sweep", "accuracy": 0.8,
        })
    np.savez_compressed(
        source / "FedSDA_v3.3_circle.npz",
        mode=np.asarray("FedSDA_v3.3"),
        label=np.asarray("FedSDA_v3.3 AGG_INTERVAL sweep (δ_adwin=0.05)"),
        history_accuracy=np.asarray([[1, 0]], dtype=np.int8),
    )
    np.savez_compressed(
        source / "FedSDA_v3.3_ucb_circle.npz",
        mode=np.asarray("FedSDA_v3.3_ucb"),
        label=np.asarray("FedSDA_v3.3_ucb"),
    )

    output = tmp_path / "migrated"
    manifest = migrate_results(source, output)

    with (output / "pareto.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows == [{
        "mode": "FedSDA_NoCached_ESR",
        "series": "FedSDA_NoCached_ESR A sweep (δ_ADWIN=0.05)",
        "accuracy": "0.9",
    }]
    migrated_npz = output / "FedSDA_Cached_ClassESR_circle2.npz"
    with np.load(migrated_npz, allow_pickle=False) as archive:
        assert archive["mode"].item() == "FedSDA_Cached_ClassESR"
        assert archive["label"].item() == "FedSDA_Cached_ClassESR A sweep (δ_ADWIN=0.05)"
        assert archive["history_accuracy"].tolist() == [[1, 0]]
    assert manifest["ucb_rows_removed"] == 1
    assert manifest["ucb_npz_files_removed"] == 1


def test_migration_does_not_copy_derived_artifacts(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "plot.png").write_bytes(b"png")
    (source / "summary.md").write_text("FedSDA_v2", encoding="utf-8")

    output = tmp_path / "migrated"
    manifest = migrate_results(source, output)

    assert not (output / "plot.png").exists()
    assert not (output / "summary.md").exists()
    assert manifest["derived_files_not_copied"] == ["plot.png", "summary.md"]
