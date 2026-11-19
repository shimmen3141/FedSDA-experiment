"""細部を全部外した簡素版FedHASPを、製品の掃引のコマンドで流す研究ツールの入口。

変種（`settings.VARIANT_ABLATIONS`）:
- `simplified`: クラス別の監視→全体の損失だけ、1件刻みの逆向きの探索→5件刻みの前向き、クラス別の機能的な
  クラスタリング→損失の距離、候補の共有部の固定→更新、warm startの学習者→外す、集約後の再較正→外す。
- `simplified-without-adapter`: さらに概念固有のadapterを外す。
製品の掃引の計画を作る関数（`experiment_sweep_plan.build_experiment_run_settings`）だけを、構成`fedsda`のときに
簡素版のrun設定を返すものへ差し替えて、`--configurations fedsda`で掃引する。run設定がFedHASPと違うので、runの
識別子（設定のhash）もFedHASPと違う。各runのJSONに完全なrun設定が残る。出力は`<--output>/<変種>/fedsda/`。

MNIST・回転MNISTはFDE_MNIST_DATA_DIR、Fashion-MNISTはFDE_FASHION_MNIST_DATA_DIRから読む。

使い方（プロジェクトの環境、リポジトリのルートで）:
    python -m tools.research_simplified_fedhasp.run --output <出力先> --datasets ... --intervals 50 500 \\
        --seeds 0 1 ... --variants simplified simplified-without-adapter --workers 64
"""

import argparse
import os
import sys
from pathlib import Path

for _thread_count_variable_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_count_variable_name, "1")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from federated_learning_experiments.sweep import experiment_sweep_plan  # noqa: E402
from tools.research_loss_monitor_rebaseline import (  # noqa: E402
    _provenance,
    _read_json,
    _sweep_arguments,
    _write_json,
)
from tools.research_simplified_fedhasp.settings import (  # noqa: E402
    BASE_CONFIGURATION_NAME,
    VARIANT_ABLATIONS,
    build_simplified_run_settings,
)

_VARIANT_RECORD_NAME = "simplified_fedhasp_variant.json"
_original_build = experiment_sweep_plan.build_experiment_run_settings


def make_patched_build(variant):
    """構成`fedsda`のときだけ簡素版のrun設定を返す、計画用の関数。"""

    def build(*, configuration_name, experiment_run_conditions):
        if configuration_name == BASE_CONFIGURATION_NAME:
            return build_simplified_run_settings(
                variant=variant, experiment_run_conditions=experiment_run_conditions
            )
        return _original_build(
            configuration_name=configuration_name, experiment_run_conditions=experiment_run_conditions
        )

    return build


def _run_variant(arguments, variant):
    from federated_learning_experiments.sweep.experiment_sweep_command import main as run_sweep

    output_directory = arguments.output.resolve() / variant
    record_path = output_directory / _VARIANT_RECORD_NAME
    if record_path.exists():
        if _read_json(record_path).get("variant") != variant:
            raise ValueError(f"出力先に別の変種の記録があります: {record_path}")
    elif output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError(f"変種の記録のない出力先は使えません: {output_directory}")
    sweep_arguments = _sweep_arguments(arguments, variant, output_directory)
    configuration_index = sweep_arguments.index("--configurations") + 1
    if sweep_arguments[configuration_index] != BASE_CONFIGURATION_NAME:
        raise ValueError("掃引の引数の構成が想定と違います")
    if not arguments.print_plan and not record_path.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
        _write_json(
            record_path,
            dict(variant=variant, removed_by_configurations=list(VARIANT_ABLATIONS[variant]), **_provenance()),
        )
    experiment_sweep_plan.build_experiment_run_settings = make_patched_build(variant)
    try:
        return run_sweep([*sweep_arguments, "--print-plan"] if arguments.print_plan else sweep_arguments)
    finally:
        experiment_sweep_plan.build_experiment_run_settings = _original_build


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--intervals", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--client-count", type=int, default=10)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANT_ABLATIONS), required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--purpose", default="FedHASPから細部を全部外した簡素版の精度")
    parser.add_argument("--print-plan", action="store_true")
    arguments = parser.parse_args(command_arguments)
    if min(arguments.workers, arguments.sample_count, arguments.client_count) < 1:
        parser.error("workers・sample-count・client-countは正の整数にしてください")
    if len(set(arguments.variants)) != len(arguments.variants):
        parser.error("変種に重複があります")
    exit_codes = []
    for variant in arguments.variants:
        print(f"== {variant}", flush=True)
        exit_codes.append(_run_variant(arguments, variant))
    return max(exit_codes)


if __name__ == "__main__":
    sys.exit(main())
