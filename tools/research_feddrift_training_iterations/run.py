"""FedDriftの、標本1件あたりの更新の回数（`updates_per_sample`、既定1）だけを替える診断の掃引入口。

目的（2026-11-21）: FedHASPの学習の量を増やすと再来の合成データで精度が上がった
（`docs/research/training-iterations-recurring-synthetic-20261121.md`）。FedDriftも同じ計算量にして比べる。

製品の掃引の計画が使う`experiment_configurations.build_default_feddrift_run_settings`を、掃引のコマンドを呼ぶ間だけ
包み、組み立てた設定の`run_participant_settings.run_client_settings.updates_per_sample`だけを置き換える
（設定は掃引の計画のときに組み立てられ、runの計画と一緒にworkerへ渡る）。runの記録の設定に、使った値が残る。
`product-default`は差し替えずに、同じ手元の環境で既定を流すための変種。
"""

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

for _thread_count_variable_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ.setdefault(_thread_count_variable_name, "1")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from federated_learning_experiments.sweep import (  # noqa: E402
    experiment_configurations,
)
from tools.research_loss_monitor_rebaseline import (  # noqa: E402
    _provenance,
    _read_json,
    _sweep_arguments,
    _write_json,
)

UPDATES_BY_VARIANT = {"product-default": 1, "updates-2": 2, "updates-4": 4}
_VARIANT_RECORD_NAME = "feddrift_training_iterations_variant.json"
_original_build = experiment_configurations.build_default_feddrift_run_settings


def _build_with_updates(updates_per_sample):
    def build(**arguments):
        settings = _original_build(**arguments)
        participant = settings.run_participant_settings
        client = participant.run_client_settings
        if client.updates_per_sample != 1:
            raise ValueError("FedDriftの既定の回数が、この診断の前提（1）と違います")
        return replace(
            settings,
            run_participant_settings=replace(
                participant,
                run_client_settings=replace(client, updates_per_sample=updates_per_sample),
            ),
        )

    return build


def _run_variant(arguments, variant):
    from federated_learning_experiments.sweep.experiment_sweep_command import (
        main as run_sweep,
    )

    output_directory = arguments.output.resolve() / variant
    record_path = output_directory / _VARIANT_RECORD_NAME
    if record_path.exists():
        record = _read_json(record_path)
        if record.get("variant") != variant or record.get("updates_per_sample") != UPDATES_BY_VARIANT[variant]:
            raise ValueError(f"出力先に別の変種の記録があります: {record_path}")
    elif output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError(f"変種の記録のない出力先は使えません: {output_directory}")
    sweep_arguments = _sweep_arguments(arguments, variant, output_directory)
    index = sweep_arguments.index("--configurations") + 1
    if sweep_arguments[index] != "fedsda":
        raise ValueError("掃引の引数の構成が想定と違います")
    sweep_arguments[index] = "feddrift"
    if arguments.print_plan:
        return run_sweep([*sweep_arguments, "--print-plan"])
    if not record_path.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
        _write_json(
            record_path,
            dict(variant=variant, updates_per_sample=UPDATES_BY_VARIANT[variant], **_provenance()),
        )
    if variant != "product-default":
        experiment_configurations.build_default_feddrift_run_settings = _build_with_updates(
            UPDATES_BY_VARIANT[variant]
        )
    try:
        return run_sweep(sweep_arguments)
    finally:
        experiment_configurations.build_default_feddrift_run_settings = _original_build


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--intervals", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--client-count", type=int, default=10)
    parser.add_argument("--variants", nargs="+", choices=tuple(UPDATES_BY_VARIANT), required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--purpose", default="FedDriftの標本1件あたりの更新の回数を増やす診断")
    parser.add_argument("--print-plan", action="store_true")
    arguments = parser.parse_args(command_arguments)
    try:
        exit_codes = []
        for variant in arguments.variants:
            print(f"== {variant}（回数={UPDATES_BY_VARIANT[variant]}）", flush=True)
            exit_codes.append(_run_variant(arguments, variant))
        return max(exit_codes)
    except (ValueError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
