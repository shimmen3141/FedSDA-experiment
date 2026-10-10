"""最終構成の回帰の、Linux用のgolden。

既存の回帰test（tests/test_proposed_regression.py）とWindows用のgoldenは、変更しない。計算・比較・環境・条件の定義は、
既存の回帰testの関数を、そのまま使う。Linuxでは、旧実装の結果自体がWindows用のgoldenと違うので、Linux用の
goldenを、別のfileに持つ。

照合: `python -m pytest tests/refactoring/test_linux_proposed_regression.py`（Linux以外では、旧実装の照合をskipする）。
作成: Linuxで`python tests/refactoring/test_linux_proposed_regression.py --update`。更新前には、必ず差分の原因を確認する。
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "tests"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import test_proposed_regression as legacy_regression  # noqa: E402

LINUX_GOLDEN_PATH = Path(__file__).with_name("proposed_regression_golden_linux.json")
GOLDEN_KEYS = {
    "_env",
    "definition",
    "cases",
    "source_commit",
    "legacy_golden_sha256",
    "mnist_sha256",
}


def load_linux_golden():
    return json.loads(LINUX_GOLDEN_PATH.read_text(encoding="utf-8"))


def build_linux_golden_payload(*, source_commit=None):
    """Linuxで、旧実装の3ケースを実行して、Windows用のgoldenと同じ形のpayloadを作る。Linux以外では拒否する。

    `source_commit`を渡さなければ、`git rev-parse HEAD`で求める（WSLからWindows側のworktreeを使うときは、
    WSLのgitがworktreeを読めないので、Windows側で求めたcommitを渡す）。
    """
    if platform.system() != "Linux":
        raise RuntimeError(f"the Linux golden must be built on Linux, not on {platform.system()!r}")
    if source_commit is None:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    if len(source_commit) != 40 or set(source_commit) - set("0123456789abcdef"):
        raise ValueError(f"source_commit must be a full commit hash: {source_commit!r}")
    cases = legacy_regression.compute_all()
    return dict(
        _env=legacy_regression.environment(),
        definition=legacy_regression.definition(),
        cases=cases,
        source_commit=source_commit,
        legacy_golden_sha256=hashlib.sha256(
            legacy_regression.GOLDEN_PATH.with_name("regression_golden.json").read_bytes()
        ).hexdigest(),
        mnist_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(legacy_regression.default_data_dir().glob("train-*.gz"))
        },
    )


def test_linux_golden_has_the_same_form_and_definition_as_windows_golden():
    """Linux用のgoldenは、Windows用と同じ形・同じ条件の定義・同じMNISTのファイルで作られている（どの環境でも確かめる）。"""
    linux_golden = load_linux_golden()
    windows_golden = json.loads(legacy_regression.GOLDEN_PATH.read_text(encoding="utf-8"))
    assert set(linux_golden) == GOLDEN_KEYS == set(windows_golden)
    assert linux_golden["_env"]["system"] == "Linux"
    assert set(linux_golden["_env"]) == set(windows_golden["_env"])
    assert linux_golden["_env"] | dict(system="", machine="", python="") == windows_golden[
        "_env"
    ] | dict(system="", machine="", python="")
    # 条件の定義は、既存の回帰testの現在の定義（JSONのキーの正規化の後）と、Windows用のgoldenの定義。
    assert linux_golden["definition"] == json.loads(json.dumps(legacy_regression.definition()))
    assert linux_golden["definition"] == windows_golden["definition"]
    assert linux_golden["mnist_sha256"] == windows_golden["mnist_sha256"]
    assert len(linux_golden["mnist_sha256"]) == 2
    assert linux_golden["legacy_golden_sha256"] == windows_golden["legacy_golden_sha256"]
    assert len(linux_golden["source_commit"]) == 40
    assert (
        set(linux_golden["cases"]) == set(legacy_regression.CASES) == set(windows_golden["cases"])
    )
    for dataset_name, linux_case in linux_golden["cases"].items():
        windows_case = windows_golden["cases"][dataset_name]
        assert set(linux_case) == {"metrics", "traces", "coverage"}
        assert tuple(linux_case["metrics"]) == legacy_regression.METRICS
        assert tuple(linux_case["traces"]) == legacy_regression.TRACES
        assert set(linux_case["coverage"]) == set(windows_case["coverage"])
        for trace_name, linux_trace in linux_case["traces"].items():
            assert set(linux_trace) == {"shape", "sha256"}, trace_name
            assert len(linux_trace["sha256"]) == 64, trace_name
    # 別のfileで、Windows用のgoldenの写しではない（Linuxでは、sine2とsea2の結果が違う）。
    assert LINUX_GOLDEN_PATH != legacy_regression.GOLDEN_PATH
    assert linux_golden["cases"] != windows_golden["cases"]


def test_building_the_linux_golden_is_refused_outside_linux(monkeypatch):
    """Linux以外では、作成を、計算の前に拒否する。"""
    computed = []
    monkeypatch.setattr(legacy_regression, "compute_all", lambda: computed.append(1))
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="Linux"):
        build_linux_golden_payload()
    assert computed == []
    # commitの全体のhashでない値も、計算の前に拒否する。
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    with pytest.raises(ValueError, match="source_commit"):
        build_linux_golden_payload(source_commit="31e927b")
    assert computed == []


def check_linux_golden():
    linux_golden = load_linux_golden()
    assert linux_golden["definition"] == json.loads(json.dumps(legacy_regression.definition()))
    if legacy_regression.environment() != linux_golden["_env"]:
        # 既存の回帰testと同じ扱い: 警告して、比較を実行する（skipも、goldenの更新も、しない）。
        warnings.warn(
            "Linux用のgoldenの生成環境と実行環境が異なる: "
            f"{linux_golden['_env']} / {legacy_regression.environment()}",
            stacklevel=2,
        )
    legacy_regression.compare(legacy_regression.compute_all(), linux_golden["cases"])


def test_legacy_final_configuration_matches_linux_golden():
    """Linuxで、旧実装の最終構成の3ケースが、Linux用のgoldenと一致する（基準は、既存の回帰testの`compare`）。"""
    if platform.system() != "Linux":
        pytest.skip("Linux用のgoldenとの照合は、Linuxでだけ行う")
    check_linux_golden()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--source-commit", default=None)
    arguments = parser.parse_args()
    if arguments.update:
        payload = build_linux_golden_payload(source_commit=arguments.source_commit)
        LINUX_GOLDEN_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {key: value["coverage"] for key, value in payload["cases"].items()}, indent=2
            )
        )
    else:
        check_linux_golden()
        print("最終構成の回帰（Linux用のgolden）: PASS")
