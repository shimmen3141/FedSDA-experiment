"""FedHASP基準版16の、学習要求1件あたりの共同更新の回数だけを替える診断の掃引入口。

目的（2026-11-21）: 再来の合成データ（agrawal4・randomrbf4）で、割り当てを正しくしても残るFedDriftとの差が、
学び方（保有モデルの学習の量）から来るか、を確かめる。製品の既定は、観測標本1件の学習要求ごとに、そのモデルの
保管標本から最大32件を抜き出して1回更新する（`_JOINT_UPDATE_ITERATIONS_PER_TRAINING_REQUEST`=1）。
これを2・4にして、保管標本を繰り返し学ぶ量を増やす。その他の設定（事前学習・候補の学習を含む）は変えない。

変種ごとに、製品の最終構成の設定の組立てが読む定数`_JOINT_UPDATE_ITERATIONS_PER_TRAINING_REQUEST`を、掃引のコマンドの
呼出しの間だけ置き換える（設定は掃引の計画のときに組み立てられ、runの計画と一緒にworkerへ渡る）。runの記録の設定（`joint_update_iterations_per_training_request`）に、使った値が残る。
既定（product-default）は差し替えずに実行する。
"""

import argparse
import os
import sys
from pathlib import Path

for _thread_count_variable_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ.setdefault(_thread_count_variable_name, "1")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from federated_learning_experiments.runtime import (  # noqa: E402
    fedsda_final_configuration_run_settings,
)
from tools.research_loss_monitor_rebaseline import (  # noqa: E402
    _provenance,
    _read_json,
    _sweep_arguments,
    _write_json,
)

PRODUCT_DEFAULT_ITERATIONS = 1
# 変種の名前と、使う回数。
ITERATIONS_BY_VARIANT = {
    "product-default": PRODUCT_DEFAULT_ITERATIONS,
    "iterations-2": 2,
    "iterations-4": 4,
}
_VARIANT_RECORD_NAME = "training_iterations_variant.json"


def _run_variant(arguments, variant):
    """1つの変種で、製品の掃引のコマンドを実行する。完了・再開・飛ばしの判定は製品のコマンドに任せる。"""
    from federated_learning_experiments.sweep.experiment_sweep_command import (
        main as run_sweep,
    )

    output_directory = arguments.output.resolve() / variant
    record_path = output_directory / _VARIANT_RECORD_NAME
    if record_path.exists():
        record = _read_json(record_path)
        if record.get("variant") != variant or record.get("iterations") != ITERATIONS_BY_VARIANT[variant]:
            raise ValueError(f"出力先に別の変種の記録があります: {record_path}")
    elif output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError(
            f"変種の記録のない出力先は使えません（別の出力先を指定してください）: {output_directory}"
        )
    sweep_arguments = _sweep_arguments(arguments, variant, output_directory)
    if arguments.print_plan:
        return run_sweep([*sweep_arguments, "--print-plan"])
    if not record_path.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
        _write_json(
            record_path,
            dict(variant=variant, iterations=ITERATIONS_BY_VARIANT[variant], **_provenance()),
        )
    # 製品の設定は掃引の計画のとき（このprocess）に組み立てられ、runの計画と一緒にworkerへ渡る。
    if fedsda_final_configuration_run_settings._JOINT_UPDATE_ITERATIONS_PER_TRAINING_REQUEST != PRODUCT_DEFAULT_ITERATIONS:
        raise ValueError("製品の既定の回数が、この診断の前提（1）と違います")
    fedsda_final_configuration_run_settings._JOINT_UPDATE_ITERATIONS_PER_TRAINING_REQUEST = ITERATIONS_BY_VARIANT[variant]
    try:
        exit_code = run_sweep(sweep_arguments)
    finally:
        fedsda_final_configuration_run_settings._JOINT_UPDATE_ITERATIONS_PER_TRAINING_REQUEST = (
            PRODUCT_DEFAULT_ITERATIONS
        )
    return exit_code


def _run_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--intervals", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--client-count", type=int, default=10)
    parser.add_argument("--variants", nargs="+", choices=tuple(ITERATIONS_BY_VARIANT), required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--purpose", default="FedHASP基準版16の学習要求1件あたりの共同更新の回数を増やす診断")
    parser.add_argument("--print-plan", action="store_true")
    return parser


def main(command_arguments=None):
    arguments = _run_parser().parse_args(
        sys.argv[1:] if command_arguments is None else list(command_arguments)
    )
    try:
        if min(arguments.workers, arguments.sample_count, arguments.client_count) < 1:
            raise ValueError("workers・sample-count・client-countは正の整数にしてください")
        if len(set(arguments.variants)) != len(arguments.variants):
            raise ValueError("変種に重複があります")
        exit_codes = []
        for variant in arguments.variants:
            print(f"== {variant}（回数={ITERATIONS_BY_VARIANT[variant]}）", flush=True)
            exit_codes.append(_run_variant(arguments, variant))
        return max(exit_codes)
    except (ValueError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
