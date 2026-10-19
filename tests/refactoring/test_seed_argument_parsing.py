"""seedの表記差をCLIの入口で整数列へ正規化する。"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

from federated_learning_experiments.core.run_provenance import BEHAVIOR_BASELINE_VERSION
from federated_learning_experiments.sweep import experiment_sweep_command as command
from federated_learning_experiments.sweep.experiment_sweep_plan import build_planned_experiment_runs
from federated_learning_experiments.sweep.experiment_sweep_resumption import (
    describe_planned_run_set,
)
from federated_learning_experiments.sweep.seed_argument_parsing import SeedListArgumentAction


def parser(default=None):
    result = argparse.ArgumentParser()
    result.add_argument("--seeds", nargs="+", action=SeedListArgumentAction, default=default)
    return result


@pytest.mark.parametrize(
    "tokens,expected",
    [
        (["0", "2", "1"], [0, 2, 1]),
        (["0-9"], list(range(10))),
        (["0-2", "5", "8-9"], [0, 1, 2, 5, 8, 9]),
        (["9-9", "0", "3-4"], [9, 0, 3, 4]),
        (["+1", "02", "3-4"], [1, 2, 3, 4]),
        (["4294967294-4294967295"], [2**32 - 2, 2**32 - 1]),
    ],
)
def test_expands_tokens_in_order(tokens, expected):
    assert parser().parse_args(["--seeds", *tokens]).seeds == expected


@pytest.mark.parametrize(
    "tokens",
    [
        ["-1"],
        ["4294967296"],
        ["3-1"],
        ["0-4294967296"],
        ["4294967296-4294967296"],
        ["0", "0"],
        ["0-2", "2-4"],
        ["+1", "1"],
        ["1.5"],
        ["0,1"],
        ["0:9"],
        ["1--3"],
        ["a"],
        ["0-100000"],
        ["0-4294967295"],
        ["0-99999", "100000"],
    ],
)
def test_rejects_invalid_or_overlapping_tokens_before_expansion(tokens, monkeypatch):
    if tokens in (["0-100000"], ["0-4294967295"]):
        from federated_learning_experiments.sweep import seed_argument_parsing as seed_module

        def forbid_materialization(*args):
            raise AssertionError("過大範囲はrangeの反復前に拒否する")

        monkeypatch.setattr(seed_module, "range", forbid_materialization, raising=False)
    with pytest.raises(SystemExit) as error:
        parser().parse_args(["--seeds", *tokens])
    assert error.value.code == 2


def test_expansion_limit_boundary():
    seeds = parser().parse_args(["--seeds", "0-99999"]).seeds
    assert len(seeds) == 100000
    assert seeds[0] == 0 and seeds[-1] == 99999


def test_preserves_default_and_last_flag_wins():
    default = [0, 1, 2]
    argument_parser = parser(default=default)
    assert argument_parser.parse_args([]).seeds == default
    assert argument_parser.parse_args(["--seeds", "0-2", "--seeds", "5-6"]).seeds == [5, 6]
    assert default == [0, 1, 2]


def test_seed_syntax_does_not_change_plan_identifiers_or_legacy_resume(tmp_path, capsys):
    output = tmp_path / "unused"
    base = ["--configurations", "fedavg", "--datasets", "circle2", "--aggregation-intervals", "10"]
    arguments = [*base, "--seeds", "0-2", "5", "--output-directory", str(output), "--print-plan"]
    expanded = [
        *base,
        "--seeds",
        "0",
        "1",
        "2",
        "5",
        "--output-directory",
        str(output),
        "--print-plan",
    ]
    plans = [
        command._build_sweep_plan(command._build_argument_parser().parse_args(a))
        for a in (arguments, expanded)
    ]
    assert plans[0] == plans[1]
    runs = [build_planned_experiment_runs(sweep_plan=p) for p in plans]
    assert [r.run_identifier for r in runs[0]] == [r.run_identifier for r in runs[1]]
    assert command.main(arguments) == 0
    first = capsys.readouterr().out
    assert command.main(expanded) == 0
    assert capsys.readouterr().out == first
    assert not output.exists()
    readme = tmp_path / "README.md"
    readme.write_text(
        f"- 期待する動作の基準版: {BEHAVIOR_BASELINE_VERSION}\n\n## コマンドの引数\n\n```\n"
        + " ".join([*base, "--seeds", "0-2", "5"])
        + "\n```\n",
        encoding="utf-8",
    )
    assert command._read_legacy_plan(readme) == describe_planned_run_set(runs[1])


@pytest.mark.parametrize("seeds", [["3-1"], ["0-2", "2"], ["0-4294967295"]])
def test_invalid_main_arguments_create_no_output(tmp_path, seeds):
    output = tmp_path / "unused"
    assert (
        command.main(
            [
                "--datasets",
                "sine2",
                "--aggregation-intervals",
                "10",
                "--seeds",
                *seeds,
                "--output-directory",
                str(output),
                "--purpose",
                "test",
            ]
        )
        == 2
    )
    assert not output.exists()


@pytest.mark.parametrize(
    "relative_path,entry,base",
    [
        (
            "run_behavior_fix_comparison.py",
            "main",
            ["--datasets", "sine2", "--aggregation-intervals", "50"],
        ),
        (
            "tools/research_prediction_initial_mass/run.py",
            "_main",
            ["--cases", "sine2:50", "--output-directory", "unused", "--purpose", "test"],
        ),
        (
            "tools/research_component_ablation/analyze.py",
            "main",
            ["--baseline-directory", "unused-baseline", "--ablation-directory", "unused-ablation"],
        ),
    ],
)
def test_other_cli_parsers_deliver_the_same_integer_list(monkeypatch, relative_path, entry, base):
    # argparseの後で停止し、実行/解析/ファイル作成は起こさない。
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("seed_cli_under_test", root / relative_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    original_parse = argparse.ArgumentParser.parse_args

    class Parsed(BaseException):
        pass

    captured = []

    def parse(self, args=None, namespace=None):
        parsed = original_parse(self, args, namespace)
        captured.append(parsed.seeds)
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse)
    with pytest.raises(Parsed):
        getattr(module, entry)([*base, "--seeds", "0-2", "5", "8-9"])
    assert captured == [[0, 1, 2, 5, 8, 9]]
