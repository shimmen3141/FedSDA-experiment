"""runごとのトレース（警報の数と共有率を上げた回数）を、変種ごとに1つのJSONへまとめる。

`run.py` の worker は並列に書くので、run の途中は `<変種>/weight_rule_trace/<run>.json` に1 run 1ファイルで書く。
失敗の無い掃引の後に、`<変種>/weight_rule_trace.json`（run の識別子で並べた一覧）へまとめ、書いたものを読み直して
一致を確かめてから、まとめた run のファイルだけを消す（ほかのファイルがあればディレクトリも残す）。
まとめた後に run を足して流し直したときは、前にまとめた一覧と新しいファイルを合わせて書き直す。

既存の結果には `python -m tools.research_prediction_weight_rules.traces <出力先>` で適用する（変種のディレクトリごと）。
"""

import argparse
import json
from pathlib import Path

TRACE_DIRECTORY_NAME = "weight_rule_trace"
CONSOLIDATED_TRACE_NAME = "weight_rule_trace.json"


def read_traces(variant_directory):
    """まとめた一覧と、まだまとめていない run のファイルを合わせて、run の識別子 → 記録を返す。"""
    traces = {}
    consolidated = variant_directory / CONSOLIDATED_TRACE_NAME
    if consolidated.exists():
        for record in json.loads(consolidated.read_text(encoding="utf-8")):
            traces[record["run_identifier"]] = record
    directory = variant_directory / TRACE_DIRECTORY_NAME
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["run_identifier"] != path.stem:
                raise ValueError(f"ファイル名と run の識別子が違います: {path}")
            traces[record["run_identifier"]] = record
    return traces


def consolidate_traces(variant_directory):
    directory = variant_directory / TRACE_DIRECTORY_NAME
    paths = sorted(directory.glob("*.json")) if directory.is_dir() else []
    if not paths:
        return 0
    traces = read_traces(variant_directory)
    records = [traces[key] for key in sorted(traces)]
    consolidated = variant_directory / CONSOLIDATED_TRACE_NAME
    temporary = consolidated.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    temporary.replace(consolidated)
    written = {record["run_identifier"]: record for record in json.loads(consolidated.read_text(encoding="utf-8"))}
    if written != traces:
        raise RuntimeError(f"まとめたトレースが元と一致しません: {consolidated}")
    for path in paths:
        path.unlink()
    if not any(directory.iterdir()):
        directory.rmdir()
    return len(paths)


def main(command_arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="run.py の --output（変種のディレクトリを含む）")
    arguments = parser.parse_args(command_arguments)
    for variant_directory in sorted(path for path in arguments.output.resolve().iterdir() if path.is_dir()):
        if (variant_directory / TRACE_DIRECTORY_NAME).is_dir():
            count = consolidate_traces(variant_directory)
            print(f"{variant_directory.name}: {count} ファイルを {CONSOLIDATED_TRACE_NAME} へまとめた")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
