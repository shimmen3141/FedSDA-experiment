"""最終構成の回帰の、実行環境ごとのgolden。

旧実装の結果は、実行環境（OS、Python・NumPy・torchの版）で変わることがある。goldenを、環境ごとに1ファイルで
`proposed_regression_goldens/`に置き、実行環境の記録（既存の回帰testの`environment()`）と、goldenの`_env`が
完全に一致するものを、自動で選ぶ。設定は要らない。

- 照合: `python -m pytest tests/refactoring/test_environment_proposed_regression.py`
  （実行環境に合うgoldenがなければ、goldenとの照合だけをskipする）。
- goldenを足す: その環境で`python tests/refactoring/test_environment_proposed_regression.py --update`。
  できたファイルを、commitする。
- goldenを消す: ファイルを消す。

固定のWindows用のgolden（tests/proposed_regression_golden.json）と既存の回帰test（tests/test_proposed_regression.py）は、
変更しない。固定のgoldenも、同じ規則で選ぶ対象に含める。計算・比較・環境・条件の定義は、既存の回帰testの関数を使う。
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "tests"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import test_proposed_regression as legacy_regression  # noqa: E402

# 環境ごとのgoldenの置き場所（1環境1ファイル）。
GOLDEN_DIRECTORY = Path(__file__).with_name("proposed_regression_goldens")
# 固定のgolden（Windowsの基準環境。既存の回帰testの`--update`でだけ更新する）。
FIXED_GOLDEN_PATH = legacy_regression.GOLDEN_PATH
GOLDEN_KEYS = {
    "_env",
    "definition",
    "cases",
    "source_commit",
    "legacy_golden_sha256",
    "mnist_sha256",
}
UPDATE_COMMAND = "python tests/refactoring/test_environment_proposed_regression.py --update"


def make_golden_file_name(environment):
    """環境の記録から、goldenのファイル名を作る（例: linux-x86_64-python3.14.4-numpy2.4.6-torch2.12.1+cpu.json）。"""
    return (
        f"{environment['system']}-{environment['machine']}-python{environment['python']}"
        f"-numpy{environment['numpy']}-torch{environment['torch']}.json"
    ).lower()


def list_golden_paths(golden_directory=GOLDEN_DIRECTORY):
    """選ぶ対象のgolden: 固定のgoldenと、ディレクトリの`*.json`（名前の順）。"""
    return (FIXED_GOLDEN_PATH, *sorted(golden_directory.glob("*.json")))


def load_golden(golden_path):
    """goldenを読む。壊れたJSONや、`_env`のないfileは、fileの名前つきで拒否する。"""
    try:
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as decode_error:
        raise ValueError(f"golden file is not valid JSON: {golden_path}: {decode_error}") from (
            decode_error
        )
    if not isinstance(golden, dict) or not isinstance(golden.get("_env"), dict):
        raise ValueError(f"golden file has no `_env` record: {golden_path}")
    return golden


def find_golden_path(environment, golden_directory=GOLDEN_DIRECTORY):
    """`_env`が、環境の記録と完全に一致するgoldenのpathを返す。なければ`None`。"""
    matching_paths = [
        golden_path
        for golden_path in list_golden_paths(golden_directory)
        if load_golden(golden_path)["_env"] == environment
    ]
    if len(matching_paths) > 1:
        raise RuntimeError(
            "more than one golden has the same environment: "
            + ", ".join(str(golden_path) for golden_path in matching_paths)
        )
    return matching_paths[0] if matching_paths else None


def describe_missing_golden(environment, golden_directory=GOLDEN_DIRECTORY):
    """goldenのない環境でのskipの理由（環境の記録、足し方、いまあるgoldenの環境）。

    版が変わってgoldenが合わなくなったときに、どのgoldenと、どこが違うかを、理由から読めるようにする。
    """
    available_environments = "; ".join(
        f"{golden_path.name}: {load_golden(golden_path)['_env']}"
        for golden_path in list_golden_paths(golden_directory)
    )
    return (
        f"この実行環境のgoldenがない: {environment}。"
        f"足すには、この環境で `{UPDATE_COMMAND}` を実行して、できたファイルをcommitする。"
        f"いまあるgolden: {available_environments}"
    )


def build_golden_payload(*, source_commit=None):
    """旧実装の3ケースを実行して、固定のgoldenと同じ形のpayloadを作る。

    `source_commit`を渡さなければ、`git rev-parse HEAD`で求める（WSLからWindows側のworktreeを使うときは、
    WSLのgitがworktreeを読めないので、Windows側で求めたcommitを渡す）。
    """
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
            FIXED_GOLDEN_PATH.with_name("regression_golden.json").read_bytes()
        ).hexdigest(),
        mnist_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(legacy_regression.default_data_dir().glob("train-*.gz"))
        },
    )


def write_environment_golden(
    *, golden_directory=GOLDEN_DIRECTORY, overwrite=False, source_commit=None
):
    """実行環境のgoldenを作って、ディレクトリへ書く。戻り値: 書いたpath。

    固定のgoldenの環境では、常に拒否する。すでにgoldenがある環境では、`overwrite`がなければ拒否する。
    拒否は、計算の前に行う。
    """
    environment = legacy_regression.environment()
    existing_golden_path = find_golden_path(environment, golden_directory)
    if existing_golden_path == FIXED_GOLDEN_PATH:
        raise RuntimeError(
            "this environment is the one of the fixed golden; "
            "update it with `python tests/test_proposed_regression.py --update`"
        )
    if existing_golden_path is not None and not overwrite:
        raise RuntimeError(
            f"a golden for this environment already exists: {existing_golden_path}. "
            "Pass --overwrite only after reviewing why the results changed"
        )
    golden_path = golden_directory / make_golden_file_name(environment)
    if golden_path.exists() and golden_path != existing_golden_path:
        # 名前は、環境の記録の一部から作る。名前が同じで、記録が違うgoldenを、黙って上書きしない。
        raise RuntimeError(
            f"another golden already uses the file name of this environment: {golden_path}"
        )
    payload = build_golden_payload(source_commit=source_commit)
    golden_directory.mkdir(parents=True, exist_ok=True)
    golden_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return golden_path


# ---- 全goldenの検査（どの環境でも） ----


def check_golden_collection(golden_directory):
    """ディレクトリのgoldenは、名前が`_env`から決まり、`_env`が、ほかのgolden（固定のものを含む）と重ならない。

    環境ごとのgoldenが1つもない状態（ディレクトリが空、または、ない）も、正しい状態である。
    """
    environment_golden_paths = sorted(golden_directory.glob("*.json"))
    if golden_directory.is_dir():
        assert sorted(path.name for path in golden_directory.iterdir()) == [
            path.name for path in environment_golden_paths
        ]
    for golden_path in environment_golden_paths:
        assert golden_path.name == make_golden_file_name(load_golden(golden_path)["_env"])
    all_golden_paths = list_golden_paths(golden_directory)
    all_environments = [load_golden(golden_path)["_env"] for golden_path in all_golden_paths]
    assert len({json.dumps(environment, sort_keys=True) for environment in all_environments}) == (
        len(all_environments)
    )
    # どのgoldenも、自分の環境の記録で、自分が選ばれる。
    for golden_path in all_golden_paths:
        assert find_golden_path(load_golden(golden_path)["_env"], golden_directory) == golden_path


def test_golden_directory_holds_only_goldens_named_after_their_environment():
    check_golden_collection(GOLDEN_DIRECTORY)


def test_golden_collection_may_be_empty_and_detects_misplaced_files(tmp_path):
    """環境ごとのgoldenを全部消しても（ディレクトリごと消しても）、検査は成功する。誤った置き方は、検出する。"""
    check_golden_collection(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    check_golden_collection(tmp_path / "empty")
    environment = make_environment()
    # 正しい名前のgolden。
    correct_directory = tmp_path / "correct"
    write_golden_with_environment(correct_directory, environment)
    check_golden_collection(correct_directory)
    # 名前が、環境の記録と合わない。
    misnamed_directory = tmp_path / "misnamed"
    write_golden_with_environment(misnamed_directory, environment, file_name="my-golden.json")
    with pytest.raises(AssertionError):
        check_golden_collection(misnamed_directory)
    # goldenでないfileが、混ざっている。
    stray_directory = tmp_path / "stray"
    write_golden_with_environment(stray_directory, environment)
    (stray_directory / "notes.txt").write_text("memo", encoding="utf-8")
    with pytest.raises(AssertionError):
        check_golden_collection(stray_directory)
    # 固定のgoldenと、同じ環境の記録（写しを置いた）。
    copied_directory = tmp_path / "copied"
    write_golden_with_environment(copied_directory, load_golden(FIXED_GOLDEN_PATH)["_env"])
    with pytest.raises((AssertionError, RuntimeError)):
        check_golden_collection(copied_directory)


@pytest.mark.parametrize(
    ("file_text", "expected_message"),
    [("{not json", "not valid JSON"), ("{}", "_env"), ("[]", "_env"), ('{"_env": 1}', "_env")],
)
def test_damaged_golden_files_are_reported_with_their_file_name(
    tmp_path, file_text, expected_message
):
    """壊れたJSONや、`_env`のないfileは、どのfileかが分かる例外になる。"""
    damaged_path = tmp_path / "damaged-golden.json"
    damaged_path.write_text(file_text, encoding="utf-8")
    for read_goldens in (
        lambda: load_golden(damaged_path),
        lambda: find_golden_path(make_environment(), tmp_path),
    ):
        with pytest.raises(ValueError, match=expected_message) as exception_info:
            read_goldens()
        assert "damaged-golden.json" in str(exception_info.value)


@pytest.mark.parametrize(
    "golden_path", sorted(GOLDEN_DIRECTORY.glob("*.json")), ids=lambda golden_path: golden_path.name
)
def test_environment_golden_has_the_same_form_and_definition_as_fixed_golden(golden_path):
    """環境ごとのgoldenは、固定のgoldenと同じ形・同じ条件の定義・同じMNISTのファイルで作られている。"""
    golden = load_golden(golden_path)
    fixed_golden = load_golden(FIXED_GOLDEN_PATH)
    assert set(golden) == GOLDEN_KEYS == set(fixed_golden)
    assert set(golden["_env"]) == set(fixed_golden["_env"]) == set(legacy_regression.environment())
    # 条件の定義は、既存の回帰testの現在の定義（JSONのキーの正規化の後）と、固定のgoldenの定義。
    assert golden["definition"] == json.loads(json.dumps(legacy_regression.definition()))
    assert golden["definition"] == fixed_golden["definition"]
    assert golden["mnist_sha256"] == fixed_golden["mnist_sha256"]
    assert len(golden["mnist_sha256"]) == 2
    assert golden["legacy_golden_sha256"] == fixed_golden["legacy_golden_sha256"]
    assert len(golden["source_commit"]) == 40
    assert set(golden["cases"]) == set(legacy_regression.CASES) == set(fixed_golden["cases"])
    for dataset_name, golden_case in golden["cases"].items():
        assert set(golden_case) == {"metrics", "traces", "coverage"}
        assert tuple(golden_case["metrics"]) == legacy_regression.METRICS
        assert tuple(golden_case["traces"]) == legacy_regression.TRACES
        assert set(golden_case["coverage"]) == set(fixed_golden["cases"][dataset_name]["coverage"])
        for trace_name, golden_trace in golden_case["traces"].items():
            assert set(golden_trace) == {"shape", "sha256"}, trace_name
            assert len(golden_trace["sha256"]) == 64, trace_name


# ---- 選択と作成の規則（一時ディレクトリで） ----


def make_environment(**changes):
    return (
        legacy_regression.environment()
        | dict(system="TestOS", machine="test64", python="9.9.9")
        | changes
    )


def write_golden_with_environment(golden_directory, environment, *, file_name=None):
    golden_directory.mkdir(parents=True, exist_ok=True)
    golden_path = golden_directory / (file_name or make_golden_file_name(environment))
    golden_path.write_text(json.dumps(dict(_env=environment)), encoding="utf-8")
    return golden_path


def test_golden_file_name_is_built_from_the_environment():
    assert (
        make_golden_file_name(
            dict(
                system="Linux",
                machine="x86_64",
                python="3.10.16",
                numpy="2.2.6",
                torch="2.12.1+cpu",
                device="cpu",
                dtype="torch.float32",
                threads=1,
            )
        )
        == "linux-x86_64-python3.10.16-numpy2.2.6-torch2.12.1+cpu.json"
    )


def test_golden_is_selected_only_by_exactly_matching_environment(tmp_path):
    """環境の記録が完全に一致するgoldenだけを選ぶ。版が1つでも違えば、選ばない。"""
    environment = make_environment()
    other_environment = make_environment(numpy="0.0.1")
    assert find_golden_path(environment, tmp_path) is None
    golden_path = write_golden_with_environment(tmp_path, environment)
    other_golden_path = write_golden_with_environment(tmp_path, other_environment)
    assert list_golden_paths(tmp_path) == (
        FIXED_GOLDEN_PATH,
        *sorted([golden_path, other_golden_path]),
    )
    assert find_golden_path(environment, tmp_path) == golden_path
    assert find_golden_path(other_environment, tmp_path) == other_golden_path
    for changed_field_name in environment:
        changed_environment = environment | {changed_field_name: "changed"}
        assert find_golden_path(changed_environment, tmp_path) is None, changed_field_name
    # 固定のgoldenも、同じ規則で選ばれる（ディレクトリが空でも）。
    fixed_environment = load_golden(FIXED_GOLDEN_PATH)["_env"]
    assert find_golden_path(fixed_environment, tmp_path / "empty") == FIXED_GOLDEN_PATH
    # 消せば、選ばれなくなる。
    golden_path.unlink()
    assert find_golden_path(environment, tmp_path) is None


def test_goldens_with_the_same_environment_are_rejected(tmp_path):
    environment = make_environment()
    write_golden_with_environment(tmp_path, environment)
    write_golden_with_environment(tmp_path, environment, file_name="copy.json")
    with pytest.raises(RuntimeError, match="same environment"):
        find_golden_path(environment, tmp_path)


def test_missing_golden_reason_tells_the_environment_and_how_to_add_one(tmp_path):
    """skipの理由から、実行環境の記録、足し方、いまあるgoldenの環境（版の違いを見比べられる）が読める。"""
    environment = make_environment()
    other_environment = make_environment(numpy="0.0.1")
    other_golden_path = write_golden_with_environment(tmp_path, other_environment)
    reason = describe_missing_golden(environment, tmp_path)
    assert str(environment) in reason
    assert UPDATE_COMMAND in reason
    assert other_golden_path.name in reason and str(other_environment) in reason
    assert FIXED_GOLDEN_PATH.name in reason
    assert str(load_golden(FIXED_GOLDEN_PATH)["_env"]) in reason


def test_writing_a_golden_follows_the_environment_and_refuses_unreviewed_overwrites(
    tmp_path, monkeypatch
):
    """作成は、実行環境の名前のファイルへ書く。すでにあれば、上書きの指定がなければ、計算の前に拒否する。"""
    environment = make_environment()
    computed = []

    def compute_all():
        computed.append(1)
        return dict(computed=len(computed))

    monkeypatch.setattr(legacy_regression, "environment", lambda: environment)
    monkeypatch.setattr(legacy_regression, "compute_all", compute_all)
    source_commit = "0123456789abcdef0123456789abcdef01234567"
    golden_path = write_environment_golden(golden_directory=tmp_path, source_commit=source_commit)
    assert golden_path == tmp_path / make_golden_file_name(environment)
    golden = load_golden(golden_path)
    assert set(golden) == GOLDEN_KEYS
    assert golden["_env"] == environment
    assert golden["cases"] == dict(computed=1)
    assert golden["source_commit"] == source_commit
    assert golden["definition"] == json.loads(json.dumps(legacy_regression.definition()))
    assert find_golden_path(environment, tmp_path) == golden_path
    # すでにある: 上書きの指定がなければ、計算の前に拒否する。
    with pytest.raises(RuntimeError, match="already exists"):
        write_environment_golden(golden_directory=tmp_path, source_commit=source_commit)
    assert computed == [1]
    assert load_golden(golden_path)["cases"] == dict(computed=1)
    assert (
        write_environment_golden(
            golden_directory=tmp_path, overwrite=True, source_commit=source_commit
        )
        == golden_path
    )
    assert load_golden(golden_path)["cases"] == dict(computed=2)
    # commitの全体のhashでない値は、計算の前に拒否する。
    with pytest.raises(ValueError, match="source_commit"):
        write_environment_golden(golden_directory=tmp_path / "other", source_commit="0123456")
    assert computed == [1, 1]
    assert not (tmp_path / "other").exists()
    # 名前が同じで、環境の記録が違うgolden（名前に入らない項目だけが違う）を、黙って上書きしない。
    monkeypatch.setattr(
        legacy_regression, "environment", lambda: environment | dict(dtype="torch.float64")
    )
    for overwrite in (False, True):
        with pytest.raises(RuntimeError, match="file name"):
            write_environment_golden(
                golden_directory=tmp_path, overwrite=overwrite, source_commit=source_commit
            )
    assert computed == [1, 1]
    assert load_golden(golden_path)["_env"] == environment


def test_writing_a_golden_is_always_refused_in_the_fixed_golden_environment(tmp_path, monkeypatch):
    """固定のgoldenの環境では、上書きの指定があっても、計算の前に拒否する（何も書かない）。"""
    computed = []
    monkeypatch.setattr(
        legacy_regression, "environment", lambda: load_golden(FIXED_GOLDEN_PATH)["_env"]
    )
    monkeypatch.setattr(legacy_regression, "compute_all", lambda: computed.append(1))
    fixed_golden_bytes = FIXED_GOLDEN_PATH.read_bytes()
    for overwrite in (False, True):
        with pytest.raises(RuntimeError, match="fixed golden"):
            write_environment_golden(
                golden_directory=tmp_path,
                overwrite=overwrite,
                source_commit="0123456789abcdef0123456789abcdef01234567",
            )
    assert computed == []
    assert not list(tmp_path.iterdir())
    assert FIXED_GOLDEN_PATH.read_bytes() == fixed_golden_bytes


# ---- 旧実装の照合（実行環境に合うgoldenがあるとき） ----


def check_environment_golden():
    environment = legacy_regression.environment()
    golden_path = find_golden_path(environment)
    if golden_path is None:
        raise LookupError(describe_missing_golden(environment))
    golden = load_golden(golden_path)
    assert golden["definition"] == json.loads(json.dumps(legacy_regression.definition()))
    legacy_regression.compare(legacy_regression.compute_all(), golden["cases"])
    return golden_path


def test_legacy_final_configuration_matches_environment_golden():
    """旧実装の最終構成の3ケースが、実行環境のgoldenと一致する（基準は、既存の回帰testの`compare`）。"""
    environment = legacy_regression.environment()
    golden_path = find_golden_path(environment)
    if golden_path is None:
        pytest.skip(describe_missing_golden(environment))
    if golden_path == FIXED_GOLDEN_PATH:
        pytest.skip(
            "固定のgoldenの環境では、既存の回帰test（tests/test_proposed_regression.py）が照合する"
        )
    check_environment_golden()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--source-commit", default=None)
    arguments = parser.parse_args()
    if arguments.update:
        written_golden_path = write_environment_golden(
            overwrite=arguments.overwrite, source_commit=arguments.source_commit
        )
        written_cases = load_golden(written_golden_path)["cases"]
        print(
            json.dumps({key: value["coverage"] for key, value in written_cases.items()}, indent=2)
        )
        print(f"書いたgolden: {written_golden_path}")
    else:
        print(f"最終構成の回帰（実行環境のgolden {check_environment_golden().name}）: PASS")
