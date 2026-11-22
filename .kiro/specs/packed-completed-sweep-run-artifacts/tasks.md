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
    - zipにだけrunがあるときの判定（済んだ、食い違う（要約のrun設定が違う、壊れたzip）、基準版の違う（掃引を始めずに止まる）、無い）。
    - `--print-plan`の表示と、出力先に何も作らないこと。
    - 再開の候補として見つかること。
  - zipはtestの中で1.1の関数で作る。その後、`find_existing_run_state`を変え、`experiment_sweep_execution.py`の許可リストを更新する。
  - 520 runのまとめた出力先（手元の`results/`のまとめた結果）で、`--print-plan`の所要時間を測り、Implementation Notesに記録する。
  - 完了条件: 新しいtest・既存の掃引と再開のtest・依存境界のtestが通る。
  - _Requirements: 2.1, 2.3, 2.4_
  - _Boundary: experiment_sweep_execution_
  - _Depends: 1.1_
- [ ] 1.3 計算の前の戻し
  - 先に書くtest:
    - zipに無いrunを足したとき、戻してから計算すること。
    - `runs/`とzipが両方あるとき、戻してから続けること。
    - 全runがzipにあるときは戻さないこと。
    - 戻せないとき（壊れたzip、違う中身のファイル）、その構成のrunが計算されずに失敗になること（例外が外へ出ない）。
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
    - 失敗・読めない要約・例外（中断を注入）があるときはまとめず、`runs/`が残ること。
    - zipが残る構成では`run_summary.csv`を書き直さないこと。
    - 比較scriptで排他を取れないとき、まとめずに警告を出すこと。
    - 比較の集計scriptが、まとめた構成で`unpack`を示して止まること。
    - まとめの失敗で警告が出て、終了コードが変わらないこと。
  - その後、両方の終わりから`pack_completed_sweep_run_artifacts`を呼び、表を書き直す条件を変え、比較の集計scriptに停止を入れ、許可リストを更新し、fresh smokeに確認を足す。
  - 完了条件: 新しいtest・掃引と比較scriptのtest・依存境界のtest・fresh smokeが通る。
  - _Requirements: 1.1, 1.5, 1.7, 2.2, 2.5, 3.3, 6.3_
  - _Boundary: experiment_sweep_command, run_behavior_fix_comparison, analyze_behavior_fix_comparison_
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

### 仕様書群のレビュー（2026-11-23）
- 1回目: Claude Haiku 5.5（effort medium、読むだけ、session `14f54d85-2db5-4f63-a2bf-6d551584d4ea`、`modelUsage`はclaude-haiku-5-5）。判定NEEDS_FIXES（Blocker 0、Major 6、Minor 8、任意1）。状態の判定・排他・ファイル操作の順序が絡む設計なので、Haikuを選んだ。
- 採用（全部）: 途中で止まった後（`runs/`とzipが両方ある）も戻す。壊れたzipで一覧を読めないときも、その構成のrunを失敗にする（例外を外へ出さない）。zipが残る構成では`run_summary.csv`を書き直さない。流し直しのtestの対象を、zip・表・`.resume/`に限る（掃引の記録は新しく増える）。比較の集計script（`analyze_behavior_fix_comparison.py`）が`runs/`を読むことを一覧に足し、まとめた構成では止める。中断のtestを足す。まとめを掃引の記録の後へ移す。manifestのキーを研究ツールと同じ`byte_count`・`format_version`にする。戻り値で「まとめた」と「警告」を分ける。圧縮方式は`ZipInfo`で指定する。研究ツールの`pack`が飛ばすことのtestを足す。tasksの不足を足す。比較scriptで排他を取れないときは警告にする。読み出しの関数の名前を`read_packed_run_summary_and_record_table`にし、非公開の処理の重複を設計に書く。zipを何度も開く所要時間を実測する。
