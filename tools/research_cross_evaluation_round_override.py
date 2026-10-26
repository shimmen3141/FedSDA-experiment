"""診断用: 指定したroundの交差評価だけ、評価clientの上限を外して、製品の掃引のコマンドを実行する。

上限を超えたときの無作為選択（`random.Random.sample`）は、今と同じに呼んで乱数を消費し、結果だけを捨てて、
標本を持つ全clientに評価させる。乱数の列が変わらないので、指定したroundの直前までは、既定のrunと同じ軌跡になる
（同じ環境なら）。指定したroundの統合の判定が、上限のせいで変わったかを、1つのrunの中で比べるために使う。

使い方:
    python -m tools.research_cross_evaluation_round_override --override-rounds 7 -- <run_experiment_sweep.pyの引数>
    python -m tools.research_cross_evaluation_round_override --override-rounds all -- <引数>
    python -m tools.research_cross_evaluation_round_override --override-rounds none -- <引数>（差し替えなし。照合用）

差し替えはこのprocessの中だけに効くので、`--workers 1`を求める。差し替えはrun設定に入らず、runの識別子は既定と同じになる。
出力先は、既定の結果と分けて`--output-directory`で指定し、差し替えの内容は出力先の`round_override.json`へ残す。
"""

import json
import os
import sys
from pathlib import Path

for _thread_count_variable_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_count_variable_name, "1")

_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from federated_learning_experiments.runtime import server_round_synchronization  # noqa: E402
from federated_learning_experiments.sweep.experiment_sweep_command import main as run_sweep  # noqa: E402

_original_cross_evaluate_global_models = server_round_synchronization.cross_evaluate_global_models


def _parse_override_rounds(text: str) -> frozenset[int] | None:
    """`all`ならNone（全round）、`none`なら空集合、それ以外はカンマ区切りのround番号。"""
    if text == "all":
        return None
    if text == "none":
        return frozenset()
    return frozenset(int(part) for part in text.split(","))


def _install_round_override(override_rounds: frozenset[int] | None) -> None:
    def cross_evaluate_with_round_override(**arguments):
        round_index = arguments["round_index"]
        if override_rounds is not None and round_index not in override_rounds:
            return _original_cross_evaluate_global_models(**arguments)
        random_generator = arguments["python_random_generator"]
        original_sample = random_generator.sample

        def sample_and_keep_whole_population(population, k, **keyword_arguments):
            # 乱数を今と同じに消費し、選ばれた部分ではなく、全部を返す（順は渡された列のまま）。
            original_sample(population, k, **keyword_arguments)
            return list(population)

        random_generator.sample = sample_and_keep_whole_population
        try:
            return _original_cross_evaluate_global_models(**arguments)
        finally:
            del random_generator.sample

    server_round_synchronization.cross_evaluate_global_models = cross_evaluate_with_round_override


def main(command_arguments: list[str]) -> int:
    if len(command_arguments) < 3 or command_arguments[0] != "--override-rounds" or command_arguments[2] != "--":
        raise SystemExit("使い方: --override-rounds <all|none|7,8> -- <run_experiment_sweep.pyの引数>")
    override_text = command_arguments[1]
    sweep_arguments = command_arguments[3:]
    if "--workers" not in sweep_arguments or sweep_arguments[sweep_arguments.index("--workers") + 1] != "1":
        raise SystemExit("差し替えはこのprocessの中だけに効くので、--workers 1を指定してください。")
    if "--output-directory" not in sweep_arguments:
        raise SystemExit("既定の結果と分けるため、--output-directoryを指定してください。")
    override_rounds = _parse_override_rounds(override_text)
    _install_round_override(override_rounds)
    output_directory = Path(sweep_arguments[sweep_arguments.index("--output-directory") + 1])
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "round_override.json").write_text(
        json.dumps(
            {
                "override_rounds": override_text,
                "description": "指定roundの交差評価だけ、評価clientの上限を外した（乱数の消費は今と同じ）",
                "sweep_arguments": sweep_arguments,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_sweep(sweep_arguments)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
