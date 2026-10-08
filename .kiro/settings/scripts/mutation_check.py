"""対象関数の変異を機械的に作り、対象testが検出するかを確かめる。worktreeルートで実行する。

    python .kiro/settings/scripts/mutation_check.py --source <src.py> --functions <名前,名前> \
        --tests <test.py>... --evidence <証拠ディレクトリ>

specごとに変異scriptを書く代わりに使う。対象関数の本体（最上位の文）から次の4種類を作る。
    delete: 名前を束縛しない文（検査のif/for、式文の呼出し、assert）を1つ消す。
            最上位のfor/whileの本体の中の文（要素ごとの検査など）も、1つずつ消す。
    defer:  例外を出しうる文（raiseを含む文、式文の呼出し）を、後続の最初の代入文の直後へ移す
            （「検査を更新の後へ移す」。代入の右辺が状態を更新する呼出しであることを想定する）。
    swap:   隣り合う「検査でない文」2つの順を入れ替える（後の文が前の文の束縛名を読まない場合だけ）。
    relax:  `type(x) is not T` を `not isinstance(x, T)` へ緩める。
1種ずつsourceを書き換えて対象testを実行し、毎回元byteへ戻す。変異後のsourceは実行前にcompileし、
収集失敗は検出に数えない。結果は<証拠ディレクトリ>/report.jsonと変異ごとのlog。

未検出（exit 1）は、testの穴か等価な変異（例: 読取りだけの代入の後へ検査を移した）のどちらかである。
どちらであるかは人が判断し、等価なら理由を対象specの証拠文書へ書く。機械的に作れない変異
（引数の差替え、別のownerへ渡す、条件の向きなど）は、必要なときだけ--extraで足す:
    --extra <名前>::<置換前の文字列>::<置換後の文字列>   （sourceに1回だけ現れる文字列。\\nは改行）
"""

import argparse
import ast
import hashlib
import json
import pathlib
import subprocess
import sys


def contains_raise(statement):
    return any(isinstance(node, ast.Raise) for node in ast.walk(statement))


def is_check_statement(statement):
    """例外を出すことが目的の文。raiseを含む複合文と、戻り値を捨てる呼出し。"""
    if isinstance(statement, (ast.If, ast.For, ast.While, ast.Assert, ast.Raise)):
        return contains_raise(statement) or isinstance(statement, ast.Assert)
    return False


def binds_names(statement):
    return isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return))


def bound_names(statement):
    return {
        node.id
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


def loaded_names(statement):
    return {
        node.id
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def line_span(statement):
    return statement.lineno - 1, statement.end_lineno


def build_mutations(source_text, function_names):
    """(名前, 説明, 変異後のsource) のlistを返す。"""
    tree = ast.parse(source_text)
    lines = source_text.splitlines(keepends=True)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    mutations = []
    for function_name in function_names:
        function = functions[function_name]
        statements = [
            statement
            for statement in function.body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
        ]
        for position, statement in enumerate(statements):
            begin, end = line_span(statement)
            label = f"{function_name}@L{statement.lineno}"
            first_line = lines[begin].strip()
            if not binds_names(statement):
                replacement = (
                    [] if len(statements) > 1 else [lines[begin][: statement.col_offset] + "pass\n"]
                )
                mutations.append(
                    (
                        f"delete:{label}",
                        f"消す: {first_line}",
                        "".join(lines[:begin] + replacement + lines[end:]),
                    )
                )
            if is_check_statement(statement) or isinstance(statement, ast.Expr):
                following_assignments = [
                    later
                    for later in statements[position + 1 :]
                    if isinstance(later, (ast.Assign, ast.AnnAssign))
                ]
                if following_assignments:
                    target_end = line_span(following_assignments[0])[1]
                    mutations.append(
                        (
                            f"defer:{label}",
                            f"L{following_assignments[0].lineno}の代入の後へ移す: {first_line}",
                            "".join(
                                lines[:begin]
                                + lines[end:target_end]
                                + lines[begin:end]
                                + lines[target_end:]
                            ),
                        )
                    )
            if position + 1 < len(statements):
                following = statements[position + 1]
                if (
                    not is_check_statement(statement)
                    and not is_check_statement(following)
                    and not isinstance(statement, ast.Return)
                    and not isinstance(following, ast.Return)
                    and not (bound_names(statement) & loaded_names(following))
                ):
                    following_begin, following_end = line_span(following)
                    mutations.append(
                        (
                            f"swap:{label}",
                            f"次の文と入れ替える: {first_line}",
                            "".join(
                                lines[:begin]
                                + lines[following_begin:following_end]
                                + lines[end:following_begin]
                                + lines[begin:end]
                                + lines[following_end:]
                            ),
                        )
                    )
        for loop_statement in statements:
            if not isinstance(loop_statement, (ast.For, ast.While)):
                continue
            for nested_statement in loop_statement.body:
                if binds_names(nested_statement):
                    continue
                begin, end = line_span(nested_statement)
                replacement = (
                    []
                    if len(loop_statement.body) > 1
                    else [lines[begin][: nested_statement.col_offset] + "pass\n"]
                )
                mutations.append(
                    (
                        f"delete:{function_name}@L{nested_statement.lineno}",
                        f"loopの中の文を消す: {lines[begin].strip()}",
                        "".join(lines[:begin] + replacement + lines[end:]),
                    )
                )
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Compare)
                and len(node.ops) == 1
                and isinstance(node.ops[0], ast.IsNot)
                and isinstance(node.left, ast.Call)
                and isinstance(node.left.func, ast.Name)
                and node.left.func.id == "type"
                and len(node.left.args) == 1
                and node.lineno == node.end_lineno
            ):
                checked_value = ast.get_source_segment(source_text, node.left.args[0])
                required_type = ast.get_source_segment(source_text, node.comparators[0])
                line = lines[node.lineno - 1]
                relaxed_line = (
                    line[: node.col_offset]
                    + f"not isinstance({checked_value}, {required_type})"
                    + line[node.end_col_offset :]
                )
                mutations.append(
                    (
                        f"relax:{function_name}@L{node.lineno}",
                        f"exact型検査をisinstanceへ緩める: {line.strip()}",
                        "".join(lines[: node.lineno - 1] + [relaxed_line] + lines[node.lineno :]),
                    )
                )
    return mutations


def run_tests(test_paths):
    return subprocess.run(
        [sys.executable, "-m", "pytest", *test_paths, "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def summarize(result):
    output_lines = result.stdout.strip().splitlines()
    return output_lines[-1] if output_lines else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--functions", required=True)
    parser.add_argument("--tests", nargs="+", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--extra", action="append", default=[])
    arguments = parser.parse_args()
    source_path = pathlib.Path(arguments.source)
    evidence = pathlib.Path(arguments.evidence)
    evidence.mkdir(parents=True, exist_ok=True)
    original = source_path.read_bytes()
    source_text = original.decode("utf-8")
    if "\r\n" in source_text:
        raise SystemExit("sourceはLFであること")
    mutations = build_mutations(source_text, arguments.functions.split(","))
    for extra in arguments.extra:
        name, before, after = (part.replace("\\n", "\n") for part in extra.split("::"))
        if source_text.count(before) != 1:
            raise SystemExit(
                f"--extra {name}: 置換前の文字列がsourceに{source_text.count(before)}回現れる"
            )
        mutations.append((f"extra:{name}", "手で指定した置換", source_text.replace(before, after)))
    baseline = run_tests(arguments.tests)
    if baseline.returncode != 0:
        raise SystemExit("変異の前に対象testが成功していない: " + summarize(baseline))
    results = []
    for index, (name, description, mutated_text) in enumerate(mutations, 1):
        if mutated_text == source_text:
            continue
        try:
            compile(mutated_text, str(source_path), "exec")
        except SyntaxError:
            results.append(
                {"mutation": name, "description": description, "outcome": "not_compilable"}
            )
            continue
        try:
            source_path.write_bytes(mutated_text.encode("utf-8"))
            result = run_tests(arguments.tests)
        finally:
            source_path.write_bytes(original)
        (evidence / f"{index:02d}.log").write_text(
            name + "\n" + result.stdout + result.stderr, encoding="utf-8"
        )
        summary = summarize(result)
        detected = (
            result.returncode == 1
            and " failed" in summary
            and "ERROR collecting" not in result.stdout
        )
        results.append(
            {
                "mutation": name,
                "description": description,
                "outcome": "detected" if detected else "NOT_DETECTED",
                "summary": summary,
                "log": f"{index:02d}.log",
            }
        )
        print(("detected    " if detected else "NOT DETECTED"), name, "|", description, flush=True)
    restored = run_tests(arguments.tests)
    executed = [item for item in results if item["outcome"] != "not_compilable"]
    undetected = [item["mutation"] for item in executed if item["outcome"] == "NOT_DETECTED"]
    report = {
        "source": arguments.source,
        "functions": arguments.functions.split(","),
        "tests": arguments.tests,
        "mutations": results,
        "detected": len(executed) - len(undetected),
        "total": len(executed),
        "undetected": undetected,
        "source_sha256_bytes": hashlib.sha256(original).hexdigest(),
        "restored_source_sha256_bytes": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "restored_summary": summarize(restored),
        "restored_exit_code": restored.returncode,
    }
    (evidence / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"detected {report['detected']} / {report['total']} | restored: {report['restored_summary']}"
    )
    if undetected:
        print("未検出（testの穴か等価な変異かを判断すること）:", ", ".join(undetected))
    return 1 if undetected or restored.returncode != 0 else 0


if __name__ == "__main__":
    sys.exit(main())
