# Implementation Plan

依存: 1.1 → 1.2 → 1.3 → 1.4 → 2.1。どれも`test_single_run_dependency_boundaries.py`か`test_experiment_sweep.py`を編集するので、順に進める。

- [ ] 1. runの成果物のまとめ
- [ ] 1.1 まとめ・読み出し・戻しのmodule
  - 設計書のTesting Strategyの「module」のtestを先に書き、`sweep/packed_run_artifact_archive.py`を実装する。依存境界に新しいmoduleを登録する。
  - 完了条件: 新しいtest（研究ツールの`unpack`で戻ることを含む）と依存境界のtestが通り、`ruff check`が通る。
  - _Requirements: 1.2, 1.3, 1.4, 1.6, 3.1, 3.2, 4.1_
  - _Boundary: packed_run_artifact_archive_
- [ ] 1.2 zipからの状態の判定
  - 先に書くtest:
    - zipにだけrunがあるときの判定（済んだ、食い違う、基準版の違う、無い、壊れたzip）。
    - `--print-plan`の表示と、出力先に何も作らないこと。
    - 再開の候補として見つかること。
  - zipはtestの中で1.1の関数で作る。その後、`find_existing_run_state`を変え、`experiment_sweep_execution.py`の許可リストを更新する。
  - 完了条件: 新しいtest・既存の掃引と再開のtest・依存境界のtestが通る。
  - _Requirements: 2.1, 2.3, 2.4_
  - _Boundary: experiment_sweep_execution_
  - _Depends: 1.1_
- [ ] 1.3 計算の前の戻し
  - 先に書くtest:
    - zipに無いrunを足したとき、戻してから計算すること。
    - 全runがzipにあるときは戻さないこと。
    - 戻せないとき（壊れたzip、違う中身のファイル）、その構成のrunが計算されずに失敗になること。
  - その後、`execute_experiment_sweep`に戻しを入れる。
  - 完了条件: 新しいtest・既存の掃引のtestが通る。
  - _Requirements: 3.1, 3.2_
  - _Boundary: experiment_sweep_execution_
  - _Depends: 1.2_
- [ ] 1.4 終わりのまとめの接続
  - 先に書くtest（掃引のコマンドと比較script）:
    - まとまること。
    - 流し直しで出力先が変わらないこと。
    - seedの追加で全runがまとまり直すこと。
    - 失敗・読めない要約があるときはまとめないこと。
    - まとめの失敗で警告が出て、終了コードが変わらないこと。
  - その後、両方の終わりから`pack_completed_sweep_run_artifacts`を呼び、許可リストを更新し、fresh smokeに確認を足す。
  - 完了条件: 新しいtest・掃引と比較scriptのtest・依存境界のtest・fresh smokeが通る。
  - _Requirements: 1.1, 1.5, 1.7, 2.2, 3.3_
  - _Depends: 1.3_

- [ ] 2. 文書と確認
- [ ] 2.1 文書・全pytest・実装レビュー
  - `docs/experiments/experiment-sweep.md`と研究ツールの冒頭の説明に足す（設計書の「文書」）。
  - 全pytest（`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`、`tests/test_regression.py`・`tests/test_proposed_regression.py`を含む全部）、`ruff check src tests/refactoring`、pyrightを通す。
  - 独立CLIの実装レビューを1回受ける。
  - 完了条件: 文書があり、全pytestが通り（JUnitでgoldenのtestがskipされていない）、実装レビューがPASSである。
  - _Requirements: 4.2, 5.1, 5.2, 6.1, 6.2_
  - _Depends: 1.4_

## Implementation Notes
