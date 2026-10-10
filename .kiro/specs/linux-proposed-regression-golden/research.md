# Research & Design Decisions

## Summary

- **Feature**: `linux-proposed-regression-golden`
- **Discovery Scope**: Extension（testと基準のファイルを足す。sourceは変えない）
- **Key Findings**:
  - 既存の回帰test（tests/test_proposed_regression.py）は、計算（`compute_all`）、比較（`compare`）、環境（`environment`）、条件の定義（`definition`）を、関数として持つ。importして、そのまま使える。
  - Linux（WSL Ubuntu）で、旧実装の結果は再現する。Windows用のgoldenとは、sine2とsea2で違う。
  - WSLの環境: Python 3.14.4、NumPy 2.4.6、torch 2.12.1+cpu、pytest 9.1.1（pytest-xdistはない）。MNISTのファイルは、リポジトリ直下の`data/mnist`。

## Research Log

### 既存の回帰testと、goldenの形

- **Sources Consulted**: tests/test_proposed_regression.py、tests/proposed_regression_golden.json、docs/experiments/refactoring-baseline.md、tests/refactoring/test_fedsda_run_metric_derivation.py（`test_golden_condition_metrics_and_traces_match_windows_golden`）。
- **Findings**:
  - goldenのキー: `_env`、`definition`、`cases`（datasetごとに`metrics`・`traces`・`coverage`）、`source_commit`、`legacy_golden_sha256`、`mnist_sha256`。
  - 環境が違うときは、警告して比較する（skipしない）。
  - 既存の回帰testは、Windows用のgoldenだけを読む。WSLでは、このtestは失敗する（既知。変更しない）。

### 研究室サーバ

- 主担当は、研究室サーバへ入れない。WSL Ubuntuと研究室サーバで、結果が同じかどうかは、分からない（OSは同じLinuxでも、CPU・ライブラリの版が違えば、浮動小数点の結果が違うことがある）。

## Design Decisions

### Decision: Linux用のgoldenは、WSL Ubuntuで作る

- **Alternatives Considered**: 研究室サーバで作る——実験を実行する環境そのものだが、主担当が実行できない。
- **Selected Approach**: 主担当が実行できるLinux（WSL Ubuntu）で作る。研究室サーバでは、ユーザーが、照合のtestを1回実行する（手順を、基準環境の文書に書く）。
- **Follow-up**: 研究室サーバで一致しなかったときは、goldenを自動で直さない。どちらを基準にするか（研究室サーバで作り直すか、環境ごとに持つか）を、ユーザーが決める。

### Decision: 作成の道具と照合のtestを、1つのfileに置く（既存の回帰testと同じ形）

- **Selected Approach**: tests/refactoring/test_linux_proposed_regression.py。testとして実行すれば照合、`--update`を付けて実行すれば作成（Linuxでだけ）。計算・比較・環境・定義は、既存の回帰testの関数を呼ぶ。
- **Rationale**: 既存の回帰testの使い方（`python tests/test_proposed_regression.py --update`）と、同じ使い方になる。

### Decision: 新実装の照合は、既存のgoldenの照合のtestを、実行環境のgoldenを選ぶ形にする

- **Selected Approach**: tests/refactoring/test_fedsda_run_metric_derivation.pyの照合のtestが、WindowsではWindows用、LinuxではLinux用のgoldenを読む。ほかのOSでは、skipする。
- **Rationale**: 照合の中身（33指標、31の離散列）は、同じ。

### Synthesis

- **Simplification**: 旧の代表11ケースのLinux版は、作らない（新実装の照合に使わない）。環境の版ごとのgoldenは、作らない。

## Risks & Mitigations

- 研究室サーバとWSLで、結果が違う — 照合のtestが失敗して、分かる。扱いは、ユーザーが決める（文書に書く）。
- WSLのPythonの版が変わる — 警告が出る。結果が変われば、照合が失敗する。

## References

- [fedsda-run-metric-derivation](../fedsda-run-metric-derivation/)（goldenの照合のtest）、docs/experiments/refactoring-baseline.md。
