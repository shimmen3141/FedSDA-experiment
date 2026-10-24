"""検証済みの除去計測Bを未実行seedへ拡張する。新seedの独立参照照合は行わない。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


if __name__ == "__main__":
    from tools.research_shadow_learner_diagnostic import run_followup
    from tools.research_adopted_candidate_removal import execute_job

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--validated-result", type=Path, required=True)
    parser.add_argument("--task", choices=["adoption"], default="adoption")
    args, remaining = parser.parse_known_args()
    if "--disabled" in remaining:
        parser.error("即時採用だけを実行します")
    sys.argv = [sys.argv[0], *remaining, "--task", "adoption"]
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    meta = json.loads(
        (args.validated_result / "research-source.json").read_text(encoding="utf-8")
    )
    for name, h in meta["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != h:
            raise ValueError(f"検証済み計測ソースが変化しています: {name}")
    hashes = dict(meta["sha256"])
    hashes[Path(__file__).relative_to(ROOT).as_posix()] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    previous = json.loads(
        (args.validated_result / "results.json").read_text(encoding="utf-8")
    )
    old_seeds = {r["job"]["seed"] for r in previous}
    # 共通CLIのprint-planを読み取り専用で呼び、重複seedの追加を拒否する。
    import contextlib
    import io

    planned = "--print-plan" in sys.argv
    plan_args = list(sys.argv)
    if not planned:
        sys.argv.append("--print-plan")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_followup.main()
    sys.argv = plan_args
    plan = json.loads(buf.getvalue())
    if any(j["seed"] in old_seeds for j in plan["jobs"]):
        raise ValueError("既存seedを重複実行しません")
    if planned:
        print(buf.getvalue())
        raise SystemExit()
    existed = output.exists()
    try:
        with patch.object(run_followup, "execute_job", execute_job):
            run_followup.main()
    finally:
        path = output / "research-source.json"
        if not existed and path.exists():
            new = json.loads(path.read_text(encoding="utf-8"))
            new["sha256"].update(hashes)
            new["extension"] = dict(
                validated_result=str(args.validated_result),
                new_seed_reference_pairs_verified=0,
                validated_input_sha256={
                    n: hashlib.sha256(
                        (args.validated_result / n).read_bytes()
                    ).hexdigest()
                    for n in ("results.json", "research-source.json")
                },
            )
            path.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
