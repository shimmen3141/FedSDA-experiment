"""製品との境界: 新しいパッケージは許可した製品のmoduleだけをimportし、torchを使わず、製品からはimportされない。"""

import ast
import subprocess
import sys
from pathlib import Path

PACKAGE_NAME = "single_stream_hard_assignment_soft_prediction"
_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
PACKAGE_DIRECTORY = _SOURCE_ROOT / PACKAGE_NAME
PRODUCT_DIRECTORY = _SOURCE_ROOT / "federated_learning_experiments"
ALLOWED_PRODUCT_MODULES = {
    "federated_learning_experiments.methods.fedsda.loss_change_detection.overall_and_true_class_loss_monitoring",
    "federated_learning_experiments.methods.fedsda.loss_change_detection.loss_change_detection_settings",
    "federated_learning_experiments.methods.fedsda.prediction_combination.fixed_share_weights",
    "federated_learning_experiments.methods.fedsda.prediction_combination.prediction_combination_settings",
    "federated_learning_experiments.methods.fedsda.training_data_assignment.minimum_description_length_assignment_selection",
    "federated_learning_experiments.methods.fedsda.loss_statistics.loss_baseline_selection",
    "federated_learning_experiments.learning.loss_statistics.model_and_class_loss_statistics",
    "federated_learning_experiments.data.dataset_definitions",
}
ALLOWED_THIRD_PARTY_ROOTS = {"numpy", "river"}
# riverを使わないmodule（Windowsの固定venvでimportできる）。
RIVER_FREE_PACKAGE_MODULES = ("single_stream_settings", "single_stream_input", "single_stream_metrics")


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise AssertionError(f"相対importは使わない: {path}")
            names.add(node.module)
    return names


def test_package_imports_only_allowed_product_modules_stdlib_numpy_and_river():
    for path in sorted(PACKAGE_DIRECTORY.glob("*.py")):
        for name in imported_modules(path):
            root = name.split(".")[0]
            if root == "federated_learning_experiments":
                assert name in ALLOWED_PRODUCT_MODULES, (path.name, name)
            elif root == PACKAGE_NAME:
                continue
            else:
                assert root in ALLOWED_THIRD_PARTY_ROOTS or root in sys.stdlib_module_names, (path.name, name)
            assert root != "torch", (path.name, name)


def test_product_does_not_import_the_package():
    for path in PRODUCT_DIRECTORY.rglob("*.py"):
        assert PACKAGE_NAME not in path.read_text(encoding="utf-8"), path


def test_allowed_product_modules_and_river_free_package_modules_do_not_load_torch():
    modules = sorted(ALLOWED_PRODUCT_MODULES) + [f"{PACKAGE_NAME}.{name}" for name in RIVER_FREE_PACKAGE_MODULES]
    code = (
        "import importlib, sys\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if m.startswith('torch'))\n"
        "print('ok')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(_SOURCE_ROOT)},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_every_allowed_product_module_is_actually_imported():
    """許可集合が広すぎないこと: 許可した製品のmoduleは、どれもパッケージのどこかでimportされている。"""
    used = set()
    for path in PACKAGE_DIRECTORY.glob("*.py"):
        used |= imported_modules(path)
    assert ALLOWED_PRODUCT_MODULES <= used, sorted(ALLOWED_PRODUCT_MODULES - used)

