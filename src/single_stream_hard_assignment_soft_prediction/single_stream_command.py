"""書き出した系列のファイルを並列に処理し、成果物と一覧のCSVを書く実行コマンド。

使い方（WSLの~/river-env、worktreeのルートで）:
    PYTHONPATH=src python -m single_stream_hard_assignment_soft_prediction.single_stream_command \\
        --streams <系列のディレクトリ> --output <出力先> [--datasets sine2 sea4] [--seeds 0 1 2] [--workers 8]
出力先には`runs/<系列の名前>.json`・`.npz`と`run_summary.csv`を書く。成果物が両方あるrunは飛ばし、片方だけなら止める。
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _thread_count_variable_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_count_variable_name, "1")

_SUMMARY_COLUMNS = (
    "dataset_name",
    "random_seed",
    "sample_count",
    "prequential_accuracy",
    "kappa",
    "kappa_m",
    "kappa_temporal",
    "accuracy_first_30_after_change",
    "accuracy_first_100_after_change",
    "accuracy_first_500_after_change",
    "true_change_count",
    "alarm_count",
    "created_model_count",
    "final_held_model_count",
    "final_tree_node_count",
    "elapsed_seconds",
)


def _run_one(task: tuple[str, str]) -> str:
    from single_stream_hard_assignment_soft_prediction.single_stream_artifacts import (
        save_single_stream_run,
    )
    from single_stream_hard_assignment_soft_prediction.single_stream_input import (
        load_single_stream_input,
    )
    from single_stream_hard_assignment_soft_prediction.single_stream_run import run_single_stream
    from single_stream_hard_assignment_soft_prediction.single_stream_settings import (
        build_default_single_stream_settings,
    )

    stream_path, output_directory = Path(task[0]), Path(task[1])
    runs_directory = output_directory / "runs"
    json_path = runs_directory / f"{stream_path.stem}.json"
    npz_path = runs_directory / f"{stream_path.stem}.npz"
    if json_path.exists() and npz_path.exists():
        return f"skip {stream_path.stem}"
    if json_path.exists() or npz_path.exists():
        raise RuntimeError(f"成果物が片方だけあります: {stream_path.stem}")
    stream = load_single_stream_input(stream_path=stream_path)
    settings = build_default_single_stream_settings()
    result = run_single_stream(stream=stream, settings=settings)
    save_single_stream_run(
        result=result,
        settings=settings,
        stream=stream,
        output_directory=runs_directory,
        file_stem=stream_path.stem,
    )
    return f"done {stream_path.stem} ({result.summary['elapsed_seconds']:.0f}s)"


def select_stream_paths(
    *, streams: Path, datasets: list[str] | None, seeds: list[int] | None
) -> list[Path]:
    """系列のディレクトリから、データ・seedで絞ったNPZを、名前の順に返す（JSONの対が無いものは除く）。"""
    paths = []
    for path in sorted(Path(streams).glob("*.npz")):
        metadata_path = path.with_suffix(".json")
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if datasets and metadata.get("dataset_name") not in datasets:
            continue
        if seeds is not None and metadata.get("random_seed") not in seeds:
            continue
        paths.append(path)
    return paths


def write_run_summary_table(*, output_directory: Path) -> Path:
    """`runs/`の全成果物から、一覧のCSVを書く（上書きする。成果物から作り直せるため）。"""
    from single_stream_hard_assignment_soft_prediction.single_stream_artifacts import (
        load_single_stream_run,
    )

    rows = []
    for json_path in sorted((Path(output_directory) / "runs").glob("*.json")):
        document, _ = load_single_stream_run(json_path=json_path)
        summary = document["summary"]
        row = {column: summary.get(column) for column in _SUMMARY_COLUMNS}
        row["run_identifier"] = json_path.stem
        rows.append(row)
    table_path = Path(output_directory) / "run_summary.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_identifier", *_SUMMARY_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)
    return table_path


def main(command_arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args(command_arguments)
    paths = select_stream_paths(
        streams=arguments.streams, datasets=arguments.datasets, seeds=arguments.seeds
    )
    if not paths:
        raise SystemExit("処理する系列がありません。")
    tasks = [(str(path), str(arguments.output)) for path in paths]
    if arguments.workers <= 1:
        messages = [_run_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
            messages = list(executor.map(_run_one, tasks))
    for message in messages:
        print(message, flush=True)
    print(f"wrote {write_run_summary_table(output_directory=arguments.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
