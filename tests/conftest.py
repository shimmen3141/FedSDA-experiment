"""全pytestを並列で実行するとき（pytest-xdist）の、testの配り方。

同じファイルのtestは、同じprocessで実行する（`--dist loadfile`と同じ。経路の網羅のtestや、moduleで1回だけ
実行するfixtureは、これを前提にしている）。例外として、下に挙げるファイルは、構成の名前で、少数の組へ分ける
（同じ構成の名前を持つtestは、同じprocessで実行する）。最も長いファイルが、全体の所要時間の下限になるため。
組を増やすと、同じprocessの中で使い回していた結果（同じ条件の提案構成の照合）を、組ごとに計算し直すので、
組の数は、少なくしてある。

pytest-xdistが無い環境（goldenの再現環境）では、配り方のhookは呼ばれない。

計測で重かったtestには、`slow`の印を付けてある（作業中は`-m "not slow"`で外せる。全pytestは、印に関係なく、
全部を実行する）。
"""

import pytest

# 構成の名前で、組へ分けるファイル。このファイルのtestが、別のtestの結果を読むときは、同じ構成の名前を
# parametrizeの値として持つこと（testのidの先頭が、構成の名前になる）。
TEST_FILES_SPLIT_BY_CONFIGURATION_NAME = (
    "tests/refactoring/test_fedsda_configuration_variants.py",
)
# 上のファイルを分ける、組の数（構成の名前の、一覧の中の位置を、この数で割った余りが、組の番号）。
CONFIGURATION_NAME_GROUP_COUNT = 3

# 固定旧のtestのうち、重いもの（ファイルを変更しないので、ここでslowの印を付ける）。
SLOW_FIXED_LEGACY_TEST_NODE_IDS = (
    "tests/test_regression.py::test_regression",
    "tests/test_proposed_regression.py::test_proposed_regression",
)


def pytest_collection_modifyitems(items):
    for test_item in items:
        if test_item.nodeid in SLOW_FIXED_LEGACY_TEST_NODE_IDS:
            test_item.add_marker(pytest.mark.slow)


def select_parallel_test_scope(*, test_node_id, configuration_names):
    """testのidから、同じprocessで実行する組の名前を返す。"""
    test_file_path, _, test_name = test_node_id.partition("::")
    if test_file_path not in TEST_FILES_SPLIT_BY_CONFIGURATION_NAME or "[" not in test_name:
        return test_file_path
    parameter_id = test_name.partition("[")[2].removesuffix("]")
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


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_make_scheduler(config, log):
    if config.getoption("dist") != "loadfile":
        return None
    from xdist.scheduler import LoadScopeScheduling

    from federated_learning_experiments.sweep.experiment_configurations import (
        EXPERIMENT_CONFIGURATION_NAMES,
    )

    class LoadFileOrConfigurationScheduling(LoadScopeScheduling):
        def _split_scope(self, nodeid):
            return select_parallel_test_scope(
                test_node_id=nodeid, configuration_names=EXPERIMENT_CONFIGURATION_NAMES
            )

    return LoadFileOrConfigurationScheduling(config, log)
