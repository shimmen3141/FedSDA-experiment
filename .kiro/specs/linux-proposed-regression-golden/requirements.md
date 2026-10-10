# Requirements Document

## Project Description (Input)

誰の問題か: FedSDAとFedDriftの比較実験コードをリファクタリングしている研究者（主担当と、後で読む人）。実験は、Linuxの研究室サーバで実行する。

現在の状況: 最終構成のgolden（tests/proposed_regression_golden.json。3ケース: sine2・sea2・mnist2）は、Windowsの基準環境で作った。新実装は、Windowsで、この3ケースの全部と一致する。Linuxでは、旧実装の結果自体が、このgoldenと違う（WSL Ubuntuでの確認: sine2は指標10・離散列15、sea2は指標1・離散列2が違い、mnist2は一致）。そのため、Linuxでは、新実装を、固定した値と照合できていない（同じprocessの中の実旧との照合は、できる）。

ユーザーの決定（2026-10-08、順序は2026-10-10）: 旧実装をLinuxで2回実行して、結果が一致すれば、Linux用のgoldenを作る。既存のWindows用goldenと回帰testは変更せず、別ファイル・別testにする。新の全体runを、WindowsとLinuxの両方で照合する。datasetの移植の後に行う。

調査で確かめたこと:

- 旧実装のLinuxでの再現性（2026-10-10、WSL Ubuntu、Python 3.14.4、NumPy 2.4.6、torch 2.12.1+cpu、1 thread）: 別のprocessで2回実行して、完全に一致した。
- 既存の回帰test（tests/test_proposed_regression.py。固定で、変更しない）: `compute_all`（3ケースを、1 threadで実行する）、`compare`（イベント件数・離散列・指標の比較。指標は絶対誤差1e-9）、`environment`・`definition`、`--update`（Windows用のgoldenを書く）。環境が違うときは、警告して、比較を実行する。
- 新実装とgoldenの照合（tests/refactoring/test_fedsda_run_metric_derivation.py）: Windows以外では、skipしている。

## Introduction

研究者が、Linuxでも、旧実装と新実装の最終構成の結果を、固定した値（Linux用のgolden）と照合できるようにする。

## Boundary Context

- **In scope**: Linux用のgoldenのファイル。それを作る手順（道具）。Linuxでの、旧実装とLinux用goldenの照合のtest。新実装とLinux用goldenの照合。基準環境の文書への追記。
- **Out of scope**: Windows用のgoldenと、既存の回帰test（tests/test_proposed_regression.py・tests/test_regression.py）の変更。旧の代表11ケースのgolden（tests/regression_golden.json）のLinux版。研究室サーバでの実行（主担当は、研究室サーバへ入れない。照合の手順を、文書に書く）。sourceの変更。
- **Adjacent expectations**: Windowsでの全pytestの結果は、変えない（Linux用の照合は、Windowsではskipする）。固定旧実装とWindows用のgoldenは、変えない。

## Requirements

### Requirement 1: Linux用のgolden

**Objective:** As a 研究者, I want Linuxでの最終構成の結果を、固定した値として持ちたい, so that Linuxで、実装の変更が結果を変えていないことを確かめられる

#### Acceptance Criteria

1. The Linux用のgolden shall Windows用のgoldenと同じ形（環境、条件の定義、3ケースの33指標・31の離散列・イベント件数、作成時のcommit、旧goldenのSHA-256、MNISTのファイルのSHA-256）を持ち、別のファイルとして保存される。
2. The Linux用のgolden shall 条件の定義が、既存の回帰testの現在の定義（Windows用のgoldenと同じ）と一致し、MNISTのファイルのSHA-256が、Windows用のgoldenと一致する。
3. When Linux以外で、作成の手順を実行したとき, the 作成の道具 shall 何も書かずに拒否する。
4. The 作成 shall 既存の回帰testの計算（`compute_all`）を、そのまま使う（旧実装と既存のtestのファイルは、変えない）。

### Requirement 2: Linuxでの、旧実装の照合

#### Acceptance Criteria

1. When Linuxで実行したとき, the 照合のtest shall 旧実装の3ケースの結果が、Linux用のgoldenと、既存の回帰testと同じ基準（イベント件数と離散列は完全に一致、指標は絶対誤差1e-9）で一致することを確かめる。
2. While 実行環境（Pythonほかの版）が、goldenの作成時と違うとき, the 照合のtest shall 警告して、比較を実行する（既存の回帰testと同じ扱い。自動のskipや、goldenの自動の更新は、行わない）。
3. While Linux以外のとき, the 照合のtest shall skipする。
4. The goldenの形の検査（1.1・1.2） shall どの環境でも実行される。

### Requirement 3: 新実装の照合

#### Acceptance Criteria

1. When Linuxで、goldenの条件（3ケース）の新の全体runを実行したとき, the 新の結果 shall 33指標と31の離散列が、Linux用のgoldenと一致する。
2. When Windowsで実行したとき, the 照合 shall これまでどおり、Windows用のgoldenと一致する。
3. The 同じprocessの中の実旧との照合 shall LinuxでもWindowsでも成功する（goldenの条件ごとの、経路の期待を含む）。

### Requirement 4: 手順の記録

#### Acceptance Criteria

1. The 基準環境の文書 shall Linux用のgoldenの作成時の環境、検証のコマンド、更新の条件、研究室サーバで照合するときの手順と、一致しなかったときの扱いを書く。
