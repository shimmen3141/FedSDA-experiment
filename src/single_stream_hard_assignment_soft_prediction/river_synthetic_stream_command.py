"""Riverの合成データの再来の系列を、既存の書き出し（研究ツール`stream_export`）と同じ形式で書き出すコマンド。

系列ごとに`<データ>_seed<seed>_n<標本数>.npz`（`features`・`labels`・`concept_ids`。事前学習の標本は書かない）と、
同じ名前の`.json`（クラス数・概念の数・特徴の名前・内容のhash・生成器の設定・Riverの版など）を書く。
書く前に、対象の全系列について、NPZとJSONの片方だけがある系列が無いことを確かめ、あれば何も書かずに止める。
両方ある系列は飛ばす。NPZ→JSONの順に、それぞれ一時ファイル（`.tmp.npz`・`.tmp.json`）から置き換える。

使い方（WSLの~/river-env、worktreeのルートで）:
    PYTHONPATH=src python -m single_stream_hard_assignment_soft_prediction.river_synthetic_stream_command \\
        --datasets stagger3 agrawal4 randomrbf4 --seeds $(seq 0 29) --sample-count 5000 --output <出力先> --workers 8
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _thread_count_variable_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_count_variable_name, "1")

_EXPORT_FORMAT_VERSION = 1
_STREAM_SOURCE = "river_synthetic_generator"


def _stream_path(*, output: Path, dataset_name: str, random_seed: int, sample_count: int) -> Path:
    return output / f"{dataset_name}_seed{random_seed}_n{sample_count}.npz"


def _write_one(task: tuple[str, int, int, str]) -> str:
    import numpy as np
    import river

    from single_stream_hard_assignment_soft_prediction.river_synthetic_recurring_streams import (
        RIVER_SYNTHETIC_STREAM_DEFINITIONS,
        generate_river_synthetic_stream,
    )
    from single_stream_hard_assignment_soft_prediction.single_stream_input import (
        compute_stream_array_digest,
    )

    dataset_name, random_seed, sample_count, output = task
    path = _stream_path(
        output=Path(output),
        dataset_name=dataset_name,
        random_seed=random_seed,
        sample_count=sample_count,
    )
    if path.exists() and path.with_suffix(".json").exists():
        return f"skip {path.name}"
    definition = RIVER_SYNTHETIC_STREAM_DEFINITIONS[dataset_name]
    arrays = generate_river_synthetic_stream(
        dataset_name=dataset_name, random_seed=random_seed, sample_count=sample_count
    )
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    metadata = dict(
        export_format_version=_EXPORT_FORMAT_VERSION,
        stream_source=_STREAM_SOURCE,
        dataset_name=dataset_name,
        random_seed=random_seed,
        sample_count=sample_count,
        class_count=definition.class_count,
        concept_count=definition.concept_count,
        feature_count=len(definition.feature_names),
        feature_names=list(definition.feature_names),
        array_sha256=compute_stream_array_digest(arrays),
        concept_change_count=int(np.count_nonzero(np.diff(arrays["concept_ids"]))),
        generator_settings=dict(definition.fixed_generator_arguments),
        concept_schedule_settings=dict(
            minimum_sample_index_gap_before_change_trial=300,
            per_eligible_sample_concept_change_probability=0.0015,
        ),
        river_version=river.__version__,
    )
    temporary_json = path.with_suffix(".tmp.json")
    temporary_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_json.replace(path.with_suffix(".json"))
    return f"done {path.name}"


def main(command_arguments: list[str] | None = None) -> int:
    from single_stream_hard_assignment_soft_prediction.river_synthetic_recurring_streams import (
        RIVER_SYNTHETIC_STREAM_DEFINITIONS,
    )

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args(command_arguments)
    unknown = [
        name for name in arguments.datasets if name not in RIVER_SYNTHETIC_STREAM_DEFINITIONS
    ]
    if unknown:
        parser.error(
            f"知らないデータです: {' '.join(unknown)}（{'・'.join(RIVER_SYNTHETIC_STREAM_DEFINITIONS)}）"
        )
    if any(seed < 0 for seed in arguments.seeds):
        parser.error("seedは0以上にしてください。")
    if arguments.sample_count < 0:
        parser.error("sample-countは0以上にしてください。")
    if arguments.workers < 1:
        parser.error("workersは1以上にしてください。")
    tasks = [
        (dataset_name, seed, arguments.sample_count, str(arguments.output))
        for dataset_name in dict.fromkeys(arguments.datasets)
        for seed in dict.fromkeys(arguments.seeds)
    ]
    half_written = []
    for dataset_name, seed, sample_count, _ in tasks:
        path = _stream_path(
            output=arguments.output,
            dataset_name=dataset_name,
            random_seed=seed,
            sample_count=sample_count,
        )
        if path.exists() != path.with_suffix(".json").exists():
            half_written.append(path.stem)
    if half_written:
        raise SystemExit(
            "NPZとJSONの片方だけがある系列があります（消してからやり直してください）: "
            + " ".join(half_written)
        )
    arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.workers == 1:
        messages = [_write_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
            messages = list(executor.map(_write_one, tasks))
    for message in messages:
        print(message, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
