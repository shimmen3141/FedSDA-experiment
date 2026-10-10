# Design Document: environment-keyed-regression-goldens

## Overview

**Purpose**: 最終構成のgoldenを、実行環境ごとのファイルにして、testが、実行環境に合うものを自動で選ぶ。

**Impact**: 前のspec（linux-proposed-regression-golden）で足したtestのfileとgoldenのfileを、置き換える。sourceは変えない。

## Boundary Commitments

### This Spec Owns

- `tests/refactoring/proposed_regression_goldens/`（新規のディレクトリ。環境ごとのgolden）。
- `tests/refactoring/test_environment_proposed_regression.py`（`test_linux_proposed_regression.py`を、名前を変えて作り直す）。
- `tests/refactoring/test_fedsda_run_metric_derivation.py`の、goldenの照合のtest（goldenの選び方）。
- `docs/experiments/refactoring-baseline.md`の、該当の節。

### Out of Boundary

- `tests/test_proposed_regression.py`、`tests/proposed_regression_golden.json`、`federated_drift_experiment/`、`src/`。

## Components

### test_environment_proposed_regression.py

```python
GOLDEN_DIRECTORY: Path            # tests/refactoring/proposed_regression_goldens
FIXED_GOLDEN_PATH: Path           # tests/proposed_regression_golden.json（固定。Windowsの基準環境）

def make_golden_file_name(environment: dict) -> str: ...
def list_golden_paths(golden_directory: Path = GOLDEN_DIRECTORY) -> tuple[Path, ...]: ...
def find_golden_path(environment: dict, golden_directory: Path = GOLDEN_DIRECTORY) -> Path | None: ...
def build_golden_payload(*, source_commit: str | None = None) -> dict: ...
def write_environment_golden(*, golden_directory: Path = GOLDEN_DIRECTORY, overwrite: bool = False,
                             source_commit: str | None = None) -> Path: ...
```

- ファイル名: `<os>-<機種>-python<版>-numpy<版>-torch<版>.json`（小文字。例: `linux-x86_64-python3.14.4-numpy2.4.6-torch2.12.1+cpu.json`）。
- `list_golden_paths`: 固定のgoldenと、ディレクトリの`*.json`（名前の順）。
- `find_golden_path`: `_env`が、渡した環境の記録と完全に一致するgoldenのpath。なければ`None`。2つ以上あれば、例外。
- `write_environment_golden`: 実行環境に合うgoldenが、固定のgoldenなら拒否。ディレクトリにあって、`overwrite`がなければ拒否。拒否は、計算の前。計算は、既存の`compute_all()`。
- test: (1)全goldenの形・定義・名前と`_env`の対応・`_env`の重なりのなさ（どの環境でも）。(2)選択と作成の規則（一時ディレクトリで）。(3)実行環境に合うgoldenがディレクトリにあれば、旧実装の3ケースを照合。固定のgoldenの環境なら、既存の回帰testに任せてskip。なければ、足し方を書いてskip。
- コマンド: `python tests/refactoring/test_environment_proposed_regression.py --update [--overwrite] [--source-commit <hash>]`で作成、引数なしで照合。

### test_fedsda_run_metric_derivation.py

- goldenの照合のtestが、`find_golden_path(legacy_regression.environment())`で選ぶ。`None`ならskip（理由に、環境の記録と、足し方）。

## 移行

- `tests/refactoring/proposed_regression_golden_linux.json`を、`proposed_regression_goldens/linux-x86_64-python3.14.4-numpy2.4.6-torch2.12.1+cpu.json`へ移す（中身は変えない）。
- 研究室サーバ用のgoldenは、ユーザーが、サーバで`--update`を1回実行して作る（手順を、文書に書く）。
