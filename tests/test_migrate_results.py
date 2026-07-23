import csv

import numpy as np

from tools.migrations.migrate_results import (
    activate_migration,
    migrate_results_tree,
)


def _make_source(root):
    pareto = root / "run" / "pareto_FedSDA_v3.3_circle.csv"
    pareto.parent.mkdir(parents=True)
    with pareto.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "mode", "dataset", "series", "distance_threshold", "accuracy"
            ],
        )
        writer.writeheader()
        writer.writerow({
            "mode": "FedSDA_v3.3",
            "dataset": "circle",
            "series": "FedSDA_v3.3 batch sweep",
            "distance_threshold": "0.1",
            "accuracy": "0.9",
        })
        writer.writerow({
            "mode": "FedSDA_v3.3_ucb",
            "dataset": "circle",
            "series": "FedSDA_v3.3_ucb",
            "distance_threshold": "0.1",
            "accuracy": "0.8",
        })
    np.savez_compressed(
        root / "run" / "FedDrift_v2_sine_seed0.npz",
        mode=np.asarray("FedDrift_v2"),
        dataset=np.asarray("sine"),
        label=np.asarray("FedDrift_v2 batch sweep"),
        history_accuracy=np.asarray([[1, 0]], dtype=np.int8),
    )
    (root / "run" / "summary.md").write_text(
        "FedSDA_v3.3 on circle", encoding="utf-8"
    )
    (root / "run" / "manifest.json").write_text(
        '{"source": "FedDrift_v2_sine_seed0.npz", "count": 1}',
        encoding="utf-8",
    )
    (root / "run" / "plot_FedSDA_v3.3_circle.png").write_bytes(b"png")


def test_tree_migration_changes_only_names_and_omits_derived_images(tmp_path):
    source = tmp_path / "results"
    staging = tmp_path / "staging"
    _make_source(source)

    manifest = migrate_results_tree(source, staging)

    csv_path = staging / "run" / "pareto_FedSDA_Cached_ClassESR_circle2.csv"
    with csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows == [{
        "mode": "FedSDA_Cached_ClassESR",
        "dataset": "circle2",
        "series": "FedSDA_Cached_ClassESR batch sweep",
        "distance_threshold": "0.1",
        "accuracy": "0.9",
    }]
    npz_path = staging / "run" / "FedDrift_sine2_seed0.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        assert archive["mode"].item() == "FedDrift"
        assert archive["dataset"].item() == "sine2"
        assert archive["label"].item() == "FedDrift batch sweep"
        assert archive["history_accuracy"].tolist() == [[1, 0]]
    assert (staging / "run" / "summary.md").read_text(
        encoding="utf-8"
    ) == "FedSDA_Cached_ClassESR on circle2"
    assert (staging / "run" / "manifest.json").read_text(
        encoding="utf-8"
    ) == '{\n  "source": "FedDrift_sine2_seed0.npz",\n  "count": 1\n}\n'
    assert not list(staging.rglob("*.png"))
    assert manifest["ucb_rows_removed"] == 1
    assert manifest["derived_files_not_copied"] == [
        "run/plot_FedSDA_Cached_ClassESR_circle2.png"
    ]


def test_activation_keeps_original_as_backup(tmp_path):
    source = tmp_path / "results"
    staging = tmp_path / "staging"
    backup = tmp_path / "results_legacy"
    _make_source(source)
    migrate_results_tree(source, staging)

    activate_migration(source, staging, backup)

    assert (backup / "run" / "FedDrift_v2_sine_seed0.npz").is_file()
    assert (source / "run" / "FedDrift_sine2_seed0.npz").is_file()
    assert not staging.exists()
