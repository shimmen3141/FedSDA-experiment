"""全pytestを並列で実行するとき（pytest-xdist）の、testの配り方。

同じファイルのtestは、同じprocessで実行する（`--dist loadfile`と同じ。経路の網羅のtestや、moduleで1回だけ
実行するfixtureは、これを前提にしている）。例外として、下に挙げるファイルは、構成の名前で、少数の組へ分ける
（同じ構成の名前を持つtestは、同じprocessで実行する）。最も長いファイルが、全体の所要時間の下限になるため。
組を増やすと、同じprocessの中で使い回していた結果（同じ条件の提案構成の照合）を、組ごとに計算し直すので、
組の数は、少なくしてある。
重要動作goldenのtestのファイルは、ケースの名前で組へ分ける（同じケースの名前を持つtestは、同じprocessで
実行し、ケースの実行の結果を使い回す）。

pytest-xdistが無い環境（goldenの再現環境）では、配り方のhookは呼ばれない。

計測で重かったtestには、`slow`の印を付けてある（作業中は`-m "not slow"`で外せる。全pytestは、印に関係なく、
全部を実行する）。
"""

import re
import sys
from pathlib import Path

import pytest

# 構成の名前で、組へ分けるファイル。このファイルのtestが、別のtestの結果を読むときは、同じ構成の名前を
# parametrizeの値として持つこと（testのidの先頭が、構成の名前になる）。
TEST_FILES_SPLIT_BY_CONFIGURATION_NAME = (
    "tests/refactoring/test_fedsda_configuration_variants.py",
)
# 上のファイルを分ける、組の数（構成の名前の、一覧の中の位置を、この数で割った余りが、組の番号）。
CONFIGURATION_NAME_GROUP_COUNT = 3

# ケースの名前で、組へ分けるファイル。ケースの名前を持つtestは、parametrizeの値`case_name`を持つこと
# （testのidが、`<ケース>`、`<版>-<ケース>`、`<前の版>-<次の版>-<ケース>`の形になる）。
TEST_FILE_SPLIT_BY_GOLDEN_CASE_NAME = (
    "tests/refactoring/test_important_behavior_regression_goldens.py"
)
# 1ケースで1組にする、重いケース（大きい条件のFedSDAのケース）。ほかのケースは、一覧の中の位置を、
# 下の数で割った余りで組へ分ける。
GOLDEN_CASE_NAMES_IN_OWN_GROUP = (
    "fedsda-sine2-events",
    "fedsda-mnist2-decisions",
    "fedsda-sine2-reference-reuse",
    "fedsda-sine2-alternative-reference-reuse",
    "fedsda-sine2-adoption-without-validation",
    "fedsda-mnist2-adoption-without-validation",
)
OTHER_GOLDEN_CASE_GROUP_COUNT = 2

# 固定旧のtestのうち、重いもの（ファイルを変更しないので、ここでslowの印を付ける）。
SLOW_FIXED_LEGACY_TEST_NODE_IDS = (
    "tests/test_regression.py::test_regression",
    "tests/test_proposed_regression.py::test_proposed_regression",
)


def pytest_collection_modifyitems(items):
    for test_item in items:
        if test_item.nodeid in SLOW_FIXED_LEGACY_TEST_NODE_IDS:
            test_item.add_marker(pytest.mark.slow)


def select_parallel_test_scope(*, test_node_id, configuration_names=(), golden_case_names=()):
    """testのidから、同じprocessで実行する組の名前を返す。"""
    test_file_path, _, test_name = test_node_id.partition("::")
    if "[" not in test_name:
        return test_file_path
    parameter_id = test_name.partition("[")[2].removesuffix("]")
    if test_file_path == TEST_FILE_SPLIT_BY_GOLDEN_CASE_NAME:
        return _select_golden_case_scope(test_file_path, parameter_id, golden_case_names)
    if test_file_path not in TEST_FILES_SPLIT_BY_CONFIGURATION_NAME:
        return test_file_path
    matching_configuration_names = [
        configuration_name
        for configuration_name in configuration_names
        if parameter_id == configuration_name or parameter_id.startswith(configuration_name + "-")
    ]
    if not matching_configuration_names:
        return test_file_path
    # ある構成の名前が、別の構成の名前の先頭と一致する場合は、長い方。
    configuration_name = max(matching_configuration_names, key=len)
    group_number = configuration_names.index(configuration_name) % CONFIGURATION_NAME_GROUP_COUNT
    return f"{test_file_path}::configuration-group-{group_number}"


def _select_golden_case_scope(test_file_path, parameter_id, golden_case_names):
    # 先頭の版の番号（`<数字>-`）を外した残りが、ケースの名前と一致するときだけ、ケースの組へ入れる。
    case_name = re.sub(r"^(?:\d+-)+", "", parameter_id)
    if case_name not in golden_case_names:
        return test_file_path
    if case_name in GOLDEN_CASE_NAMES_IN_OWN_GROUP:
        group_name = case_name
    else:
        other_case_names = [
            name for name in golden_case_names if name not in GOLDEN_CASE_NAMES_IN_OWN_GROUP
        ]
        group_name = f"others-{other_case_names.index(case_name) % OTHER_GOLDEN_CASE_GROUP_COUNT}"
    return f"{test_file_path}::golden-case-group-{group_name}"


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_make_scheduler(config, log):
    if config.getoption("dist") != "loadfile":
        return None
    from xdist.scheduler import LoadScopeScheduling

    from federated_learning_experiments.sweep.experiment_configurations import (
        EXPERIMENT_CONFIGURATION_NAMES,
    )

    refactoring_test_directory = str(Path(__file__).resolve().parent / "refactoring")
    if refactoring_test_directory not in sys.path:
        sys.path.insert(0, refactoring_test_directory)
    from behavior_golden_cases import ALL_CASE_NAMES

    class LoadFileOrConfigurationScheduling(LoadScopeScheduling):
        def _split_scope(self, nodeid):
            return select_parallel_test_scope(
                test_node_id=nodeid,
                configuration_names=EXPERIMENT_CONFIGURATION_NAMES,
                golden_case_names=ALL_CASE_NAMES,
            )

    return LoadFileOrConfigurationScheduling(config, log)
