"""specの命名表と承認・同一性を機械的に照合する。worktreeルートで実行する。

    python .kiro/settings/scripts/spec_checks.py names <spec名> <新規・変更した.py>... [--naming <命名表>]
    python .kiro/settings/scripts/spec_checks.py identity <spec名> [--junit <JUnit XML>] [--rev <commit>]
    python .kiro/settings/scripts/spec_checks.py progress <spec名>

names: 指定ファイルの束縛名（def・class・引数・代入・importの別名）を命名表と照合する。
    命名表の登録名は、表の先頭列の語と、表以外の行の語（表の説明列で触れただけの語は数えない）。
    未登録: 命名表になく、他のsrc/tests/refactoringにも現れない名前（exit 1）。
    役割の再利用: 命名表になく、他では関数・classの名前としてだけ現れる語を変数・引数に使っているもの（exit 1）。
    他のファイルでも変数・引数として使われている語は、同じ役割の再利用として報告しない。
identity: 承認hash、固定旧基準からの差分、source hash、作業ツリー、JUnitを照合する（不一致はexit 1）。
progress: tasks.mdの完了数とspec.jsonの進捗・phaseを照合する（不一致はexit 1）。あわせて、specのREADMEと
    再開案内・roadmapのうち、このspecに触れた行に残る「待ち」「未実施」などの語を表示する（人が確かめる。exitには数えない）。
    feature最終レビューへ出す前と、最終GOを記録した後に実行する。

独立レビューの代わりにはならない。名前の役割が実態と合うか、検査の順序が正しいかは人とレビュー担当が確かめる。
"""

import argparse
import ast
import hashlib
import io
import json
import pathlib
import re
import subprocess
import sys
import xml.etree.ElementTree as ElementTree

FIXED_LEGACY_COMMIT = "748c3aa"
FIXED_LEGACY_PATHS = (
    "federated_drift_experiment",
    "tests/regression_golden.json",
    "tests/proposed_regression_golden.json",
    "tests/test_regression.py",
    "tests/test_proposed_regression.py",
    "tools",
)
GOLDEN_PATHS = ("tests/regression_golden.json", "tests/proposed_regression_golden.json")
EXISTING_CODE_ROOTS = ("src", "tests/refactoring")
APPROVAL_STAGES = ("requirements", "design", "naming", "tasks")
PROGRESS_DOCUMENT_PATHS = (".kiro/steering/resume.md", ".kiro/steering/roadmap.md")
UNFINISHED_STATE_PATTERN = re.compile("待ち|未実施|ブロック中")


def collect_bound_names(source_path):
    """(関数・class名, 変数・引数・import名) を返す。"""
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    definition_names, variable_names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definition_names.add(node.name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = node.args
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                *filter(None, (arguments.vararg, arguments.kwarg)),
            ):
                variable_names.add(argument.arg)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            variable_names.add(node.id)
        if isinstance(node, ast.ExceptHandler) and node.name:
            variable_names.add(node.name)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # 定義元の名前のままのimportは新しい名前ではない。別名を付けたときだけ束縛名として数える。
            for alias in node.names:
                if alias.asname:
                    variable_names.add(alias.asname)
    return definition_names, variable_names


def collect_registered_names(naming_path):
    """命名表で名前として登録された語を集める。

    表の行は先頭列の語だけを登録名とする。説明列で既存のメソッド名などに触れただけの語は、
    その名前を登録したことにならない。表以外の行（箇条書き、再利用する既存名の列挙など）は行内の語を数える。
    """
    registered_names = set()
    for line in naming_path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("|") and line.count("|") >= 2:
            line = line.split("|")[1]
        # backtickの有無は問わない（過去の命名表はbacktickなしの表・箇条書き・列挙でも書いている）。
        registered_names.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", line))
    return registered_names


def check_names(arguments):
    spec_directory = pathlib.Path(".kiro/specs") / arguments.spec
    naming_path = (
        pathlib.Path(arguments.naming) if arguments.naming else spec_directory / "naming.md"
    )
    registered_words = collect_registered_names(naming_path)
    target_paths = [pathlib.Path(name).resolve() for name in arguments.files]
    other_definition_names, other_variable_names = set(), set()
    for root in EXISTING_CODE_ROOTS:
        for source_path in pathlib.Path(root).rglob("*.py"):
            if source_path.resolve() in target_paths:
                continue
            definition_names, variable_names = collect_bound_names(source_path)
            other_definition_names |= definition_names
            other_variable_names |= variable_names
    failed = False
    for target_path in target_paths:
        definition_names, variable_names = collect_bound_names(target_path)
        candidates = {
            name
            for name in definition_names | variable_names
            if name not in registered_words and name != "_"
        }
        unregistered = sorted(candidates - other_definition_names - other_variable_names)
        role_reused = sorted(
            (candidates & variable_names & other_definition_names) - other_variable_names
        )
        print(f"{target_path.name}: 束縛名 {len(definition_names | variable_names)}")
        print(f"  未登録（命名表にも既存コードにもない）: {unregistered or 'なし'}")
        print(
            f"  役割の再利用（他では関数・class名の語を変数・引数に使用、命名表に記載なし）: {role_reused or 'なし'}"
        )
        failed = failed or bool(unregistered or role_reused)
    return 1 if failed else 0


def lf_sha256(content):
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def run_git(*git_arguments):
    return subprocess.run(["git", *git_arguments], capture_output=True, check=True).stdout


def source_sha256(revision):
    """tracked Pythonと2goldenの、パス昇順・LF内容のhash（共通引継ぎ手順と同じ計算）。"""
    blob_entries = []
    for tree_entry in run_git("ls-tree", "-r", "-z", revision).split(b"\0"):
        if not tree_entry:
            continue
        metadata, file_path = tree_entry.split(b"\t", 1)
        file_path = file_path.decode("utf-8")
        if not (file_path.endswith(".py") or file_path in GOLDEN_PATHS):
            continue
        _, object_type, git_object_id = metadata.split()
        if object_type != b"blob":
            raise ValueError(f"source hash requires a Git blob: {file_path}")
        blob_entries.append((file_path, git_object_id))
    blob_entries.sort()
    # ファイルごとのGit起動を避ける。archiveのexport-ignore等に左右されないblobを読む。
    blob_stream = io.BytesIO(
        subprocess.run(
            ["git", "cat-file", "--batch"],
            input=b"".join(git_object_id + b"\n" for _, git_object_id in blob_entries),
            capture_output=True,
            check=True,
        ).stdout
    )
    digest = hashlib.sha256()
    for file_path, git_object_id in blob_entries:
        header = blob_stream.readline().split()
        if len(header) != 3 or header[:2] != [git_object_id, b"blob"]:
            raise ValueError(f"invalid Git blob header: {file_path}")
        blob_size = int(header[2])
        content = blob_stream.read(blob_size)
        if len(content) != blob_size or blob_stream.read(1) != b"\n":
            raise ValueError(f"incomplete Git blob: {file_path}")
        digest.update(file_path.encode("utf-8") + b"\0" + content.replace(b"\r\n", b"\n") + b"\0")
    return len(blob_entries), digest.hexdigest()


def check_identity(arguments):
    spec_directory = pathlib.Path(".kiro/specs") / arguments.spec
    spec = json.loads((spec_directory / "spec.json").read_text(encoding="utf-8"))
    failed = False

    def report(label, passed, detail=""):
        nonlocal failed
        failed = failed or not passed
        print(f"[{'OK' if passed else 'NG'}] {label}{' — ' + detail if detail else ''}")

    for stage in APPROVAL_STAGES:
        approval = spec.get("approvals", {}).get(stage, {})
        approved_hash = approval.get("approved_sha256_lf")
        content = (spec_directory / f"{stage}.md").read_bytes()
        if stage == "tasks":
            # 承認時は全taskが未完了。完了のcheckboxを戻した内容で比べる。
            content = content.replace(b"- [x] ", b"- [ ] ")
        if not approval.get("approved") or approved_hash is None:
            report(f"{stage}の承認hash", False, "spec.jsonに承認の記録がない")
            continue
        report(
            f"{stage} revision{approval.get('approved_revision')}の承認hash",
            lf_sha256(content) == approved_hash,
            approved_hash[:12],
        )
    for label, revisions in (
        ("commit済み", (FIXED_LEGACY_COMMIT, "HEAD")),
        ("作業ツリー", (FIXED_LEGACY_COMMIT,)),
    ):
        difference = run_git("diff", "--stat", *revisions, "--", *FIXED_LEGACY_PATHS).decode()
        report(f"固定旧基準{FIXED_LEGACY_COMMIT}からの差分が空（{label}）", not difference.strip())
    report("作業ツリーに未コミット差分がない", not run_git("status", "--short").strip())
    validation = spec.get("integration_validation", {})
    revision = arguments.rev or validation.get("tested_commit") or "HEAD"
    path_count, source_hash = source_sha256(revision)
    recorded_hash = validation.get("source_sha256_lf") or spec.get("verification_source_sha256")
    if recorded_hash is None:
        print(
            f"[--] source hash（{revision}、{path_count}パス）: {source_hash}（spec.jsonに記録なし）"
        )
    else:
        report(
            f"source hash（{revision}、{path_count}パス）",
            source_hash == recorded_hash,
            source_hash[:12],
        )
    if arguments.junit:
        test_cases = list(ElementTree.parse(arguments.junit).getroot().iter("testcase"))
        counts = {
            kind: sum(1 for test_case in test_cases if test_case.find(kind) is not None)
            for kind in ("failure", "error", "skipped")
        }
        passed_count = len(test_cases) - sum(counts.values())
        print(f"     JUnit: testcase {len(test_cases)}、passed {passed_count}、{counts}")
        report("JUnitにfailure・errorがない", counts["failure"] == counts["error"] == 0)
        for module_name in ("tests.test_regression", "tests.test_proposed_regression"):
            golden_cases = [case for case in test_cases if case.get("classname") == module_name]
            report(
                f"{module_name}が成功",
                bool(golden_cases) and all(len(case) == 0 for case in golden_cases),
                f"{len(golden_cases)}件",
            )
        for key, actual in (("full_passed", passed_count), ("full_skipped", counts["skipped"])):
            if key in validation:
                report(f"spec.jsonの{key}と一致", validation[key] == actual, str(actual))
    return 1 if failed else 0


def check_progress(arguments):
    spec_directory = pathlib.Path(".kiro/specs") / arguments.spec
    spec = json.loads((spec_directory / "spec.json").read_text(encoding="utf-8"))
    failed = False

    def report(label, passed, detail=""):
        nonlocal failed
        failed = failed or not passed
        print(f"[{'OK' if passed else 'NG'}] {label}{' — ' + detail if detail else ''}")

    # 番号つきの最上位taskだけを数える（下位の箇条書きはcheckboxを持たない）。
    task_states = re.findall(
        r"^- \[([ x])\] \d+\. ", (spec_directory / "tasks.md").read_text(encoding="utf-8"), re.M
    )
    completed_count = task_states.count("x")
    progress = spec.get("implementation_progress", {})
    report(
        "spec.jsonのimplementation_progressがtasks.mdと一致",
        progress == {"completed": completed_count, "total": len(task_states)},
        f"tasks.md {completed_count}/{len(task_states)}、spec.json {progress}",
    )
    if task_states and completed_count == len(task_states):
        report(
            "全task完了のspecのphaseが実装中でない",
            spec.get("phase") != "implementation-in-progress",
            str(spec.get("phase")),
        )
    # 現在の状態を書く場所だけを見る。review.md等の経過の記録は当時の状態を残すので対象にしない。
    state_lines = [
        ("README.md", line)
        for line in (spec_directory / "README.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("状態:")
    ]
    for document_path in PROGRESS_DOCUMENT_PATHS:
        state_lines += [
            (document_path, line)
            for line in pathlib.Path(document_path).read_text(encoding="utf-8").splitlines()
            if arguments.spec in line
        ]
    for document_name, line in state_lines:
        found_words = sorted(set(UNFINISHED_STATE_PATTERN.findall(line)))
        if found_words:
            print(f"[--] 要確認 {document_name}: {found_words} — {line[:120]}")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    names_parser = subparsers.add_parser("names")
    names_parser.add_argument("spec")
    names_parser.add_argument("files", nargs="+")
    names_parser.add_argument("--naming")
    identity_parser = subparsers.add_parser("identity")
    identity_parser.add_argument("spec")
    identity_parser.add_argument("--junit")
    identity_parser.add_argument("--rev")
    subparsers.add_parser("progress").add_argument("spec")
    arguments = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    checks = {"names": check_names, "identity": check_identity, "progress": check_progress}
    return checks[arguments.command](arguments)


if __name__ == "__main__":
    sys.exit(main())
