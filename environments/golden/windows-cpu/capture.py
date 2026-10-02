"""指定した出力先に、実行中のPython環境とgoldenの来歴を記録する。"""

import argparse
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]


def sha256_lf(path):
    """Gitの改行変換で内容照合が変わらないよう、LFへ正規化する。"""
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def cpu_model():
    if platform.system() == "Windows":
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor()


def capture(output):
    output.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "--isolated", "freeze", "--all"], text=True,
    )
    # ローカルパスや認証付きURLを環境記録へ混入させない。
    lines = sorted(freeze.splitlines(), key=str.casefold)
    if not all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^\s]+", line) for line in lines):
        raise ValueError("通常の版固定以外の依存があります。保存前に取得元を確認してください。")
    (output / "requirements-freeze.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    numpy_build = io.StringIO()
    with redirect_stdout(numpy_build):
        np.show_config()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    # 取得元は再構築で検証する方針。過去のインストール元が証明されたとは記録しない。
    environment = dict(
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=commit,
        os=dict(system=platform.system(), release=platform.release(),
                version=platform.version(), win32_version=list(platform.win32_ver()),
                machine=platform.machine()),
        python=dict(version=platform.python_version(), implementation=platform.python_implementation(),
                    compiler=platform.python_compiler(), bitness=platform.architecture()[0]),
        cpu=dict(model=cpu_model(), logical_processors=os.cpu_count(),
                 torch_capability=torch.backends.cpu.get_cpu_capability()),
        execution=dict(device="cpu", dtype=str(torch.get_default_dtype()),
                       torch_threads=torch.get_num_threads(),
                       torch_interop_threads=torch.get_num_interop_threads(),
                       OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"),
                       MKL_NUM_THREADS=os.environ.get("MKL_NUM_THREADS")),
        libraries=dict(numpy=np.__version__, torch=str(torch.__version__),
                       torch_cuda=torch.version.cuda, torch_git_version=torch.version.git_version),
        build=dict(numpy=numpy_build.getvalue(), torch=torch.__config__.show()),
        package_index="https://pypi.org/simple",
        package_index_status="rebuild_verification_required",
        requirements_sha256_lf=sha256_lf(output / "requirements-freeze.txt"),
        golden_sha256_lf={name: sha256_lf(ROOT / "tests" / name) for name in (
            "regression_golden.json", "proposed_regression_golden.json")},
    )
    (output / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Recorded: {output.name} ({len(lines)} packages)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    capture(parser.parse_args().output_dir)
