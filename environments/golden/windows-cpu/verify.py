"""再構築したvenvの環境を照合し、goldenと関連機能を検証する。"""

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from capture import ROOT

HERE = Path(__file__).resolve().parent
TESTS = (
    "test_option_schema.py", "test_parameter_schema.py", "test_metric_schema.py",
    "test_shared_backbone.py", "test_provisional_model.py", "test_clustering_decision.py",
    "test_fedsda_configuration.py", "test_regression.py", "test_proposed_regression.py",
)


def main():
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("検証にはvenvのPythonを使用してください。")
    output = Path(sys.prefix) / "golden-verification"
    output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "MPLCONFIGDIR": str(output / "matplotlib-cache")}
    (output / "matplotlib-cache").mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(HERE / "capture.py"), "--output-dir", str(output)],
                   cwd=ROOT, env=env, check=True)
    reference = json.loads((HERE / "environment.json").read_text(encoding="utf-8"))
    current = json.loads((output / "environment.json").read_text(encoding="utf-8"))
    for key in ("python", "libraries", "build", "requirements_sha256_lf", "golden_sha256_lf"):
        if current[key] != reference[key]:
            raise RuntimeError(f"基準環境との不一致: {key}。goldenは更新せず原因を確認してください。")
    for key in ("system", "machine"):
        if current["os"][key] != reference["os"][key]:
            raise RuntimeError(f"OSとの不一致: {key}")
    if current["execution"] != reference["execution"]:
        raise RuntimeError("dtypeまたはスレッド設定が基準と異なります。")
    if current["cpu"] != reference["cpu"] or current["os"] != reference["os"]:
        print("CPUまたはOSビルドが異なります。以下の回帰検証で一致を確認します。")
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    junit = output / "pytest.xml"
    command = [sys.executable, "-m", "pytest", "--ignore=.venv",
               *(f"tests/{test}" for test in TESTS), "-q", f"--junitxml={junit}"]
    result = subprocess.run(command, cwd=ROOT, env=env)
    suites = ET.parse(junit).getroot().findall("testsuite")
    counts = {key: sum(int(suite.get(key, "0")) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    passed = result.returncode == 0 and counts["tests"] > 0 and not any(
        counts[key] for key in ("failures", "errors", "skipped"))
    verification = dict(
        verified_at_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=current["source_commit"], package_index=reference["package_index"],
        package_freeze_match=True, numerical_build_match=True,
        golden_sha256_lf=current["golden_sha256_lf"],
        test_files=list(TESTS), test_counts=counts,
        elapsed_seconds=sum(float(suite.get("time", "0")) for suite in suites),
        status="PASS" if passed else "FAIL",
    )
    (output / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError(f"検証に失敗しました: {counts}")
    print(f"再構築環境の検証: PASS ({counts['tests']} tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
