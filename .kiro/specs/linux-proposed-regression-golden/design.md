# Design Document: linux-proposed-regression-golden

## Overview

**Purpose**: Linuxでも、最終構成の旧実装と新実装の結果を、固定した値と照合できるようにする。

**Users**: リファクタリングを進める研究者。

**Impact**: testのfileを1つと、goldenのfileを1つ足す。既存のgoldenの照合のtestを、実行環境のgoldenを選ぶ形にする。sourceは変えない。

### Goals

- Linux（WSL Ubuntu）で、旧実装の3ケースが、Linux用のgoldenと一致する。
- Linuxで、新実装の3ケースが、Linux用のgoldenと一致する。
- Windowsの結果は、変わらない。

### Non-Goals

- Windows用のgoldenと既存の回帰testの変更。旧の代表11ケースのLinux版。研究室サーバでの実行。

## Boundary Commitments

### This Spec Owns

- `tests/refactoring/test_linux_proposed_regression.py`（新規。照合と、`--update`での作成）。
- `tests/refactoring/proposed_regression_golden_linux.json`（新規。WSL Ubuntuで作る）。
- `tests/refactoring/test_fedsda_run_metric_derivation.py`の、goldenの照合のtest（実行環境のgoldenを選ぶ）。
- `docs/experiments/refactoring-baseline.md`への追記。

### Out of Boundary

- `tests/test_proposed_regression.py`、`tests/proposed_regression_golden.json`、`federated_drift_experiment/`、`src/`。

### Revalidation Triggers

- 最終構成の条件の定義の変更（Windows用のgoldenを更新するとき、Linux用も作り直す）。Linuxの基準環境の変更。

## Components

### test_linux_proposed_regression.py

```python
LINUX_GOLDEN_PATH: Path   # tests/refactoring/proposed_regression_golden_linux.json

def load_linux_golden() -> dict: ...
def build_linux_golden_payload() -> dict: ...   # Linux以外では拒否する
def test_linux_golden_has_the_same_form_and_definition_as_windows_golden(): ...   # どの環境でも
def test_legacy_final_configuration_matches_linux_golden(): ...   # Linuxだけ。ほかはskip
```

- `build_linux_golden_payload`: `platform.system() != "Linux"`なら`RuntimeError`。既存の回帰testの`compute_all()`・`environment()`・`definition()`を呼び、Windows用のgoldenと同じキーのpayloadを作る（`source_commit`は`git rev-parse HEAD`、`legacy_golden_sha256`は旧goldenのfile、`mnist_sha256`は`default_data_dir()`の2つのfile）。
- 形の検査: キーの集合、`_env`の`system`が`Linux`、`definition`が現在の定義と一致、3ケースの指標名・離散列名がWindows用と同じ、`mnist_sha256`と`legacy_golden_sha256`がWindows用と同じ。
- Linuxの照合: 環境が違えば`warnings.warn`の後、既存の`compare(compute_all(), golden["cases"])`。
- `python tests/refactoring/test_linux_proposed_regression.py --update`で、作成して書く。引数なしなら、照合する。

### test_fedsda_run_metric_derivation.py

- goldenの照合のtestが、`platform.system()`で、Windows用（`legacy_regression.GOLDEN_PATH`）かLinux用（`LINUX_GOLDEN_PATH`）を選ぶ。ほかのOSではskip。
- goldenの条件ごとの期待（経路、候補の判定の理由）が、Linuxで違う場合は、環境ごとの期待にする（実行して確かめる）。

## 手順

1. testと道具を書く（Linux用のgoldenがない間、形の検査は失敗する）。
2. WSL Ubuntuで、`--update`で作る。別のprocessで、照合のtestを実行して、再現することを確かめる。
3. WSL Ubuntuで、新実装の照合を実行する。
4. Windowsで、全pytestを実行する。
