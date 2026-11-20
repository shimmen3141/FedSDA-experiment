# Implementation Plan

依存: 1.1 → 1.2 → 2.1 → 2.2 → 2.3 → 3.1（1と2は内容は独立だが、どれも`test_single_run_dependency_boundaries.py`を編集するので順に進める）。

- [ ] 1. 記録の表のLZMA
- [ ] 1.1 書き出し関数
  - `write_lzma_compressed_record_tables`のtest（全エントリがLZMA、`savez_compressed`と同じ配列）を先に書き、`runtime/lzma_record_table_files.py`を実装する。依存境界に新しいmoduleを登録する。
  - 完了条件: 新しいtestと依存境界のtestが通る。
  - _Requirements: 1.1, 1.2_
  - _Boundary: lzma_record_table_files_
- [ ] 1.2 4手法の書き出しへの接続
  - 4手法の保存のtest（LZMA、同じ配列、objectの配列が無い、deflateのNPZも読める、失敗で途中のファイルが残らない）を先に書き、4つの書き出しを置き換える。既存の`savez_compressed`のmonkeypatchを新しい関数へ替える。4つの許可リストを更新する。
  - 完了条件: 成果物のtest・依存境界のtest・全版のgoldenが通る。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.1_
  - _Depends: 1.1_

- [ ] 2. 開始の記録
- [ ] 2.1 開始の記録の読み方（個別のファイル→zip）
  - zipにだけ記録があるとき（zipはtestの中で手で作る）の`can_recover_interrupted_run`（直せる）と`write_run_start_record`（同じ内容は受け入れて個別のファイルを書かない、違う内容は拒否）、壊れたzipでの扱いのtestを先に書き、`_read_run_start_record`を実装して両方から使う。`experiment_sweep_resumption.py`の許可リストに、読込みで実際にimportする名前を足す。
  - 完了条件: 新しいtest・既存の再開のtest・依存境界のtestが通る。
  - _Requirements: 2.5_
  - _Boundary: experiment_sweep_resumption_
  - _Depends: 1.2_
- [ ] 2.2 まとめる関数
  - `pack_completed_sweep_run_start_records`のtest（設計書のTesting Strategyの「(B) まとめ」。重複だけならzipを書き直さないことを含む）を先に書き、実装する。`experiment_sweep_resumption.py`の許可リストに、まとめで足した名前を足す。
  - 完了条件: 新しいtest・依存境界のtestが通る。
  - _Requirements: 2.1, 2.2, 2.4, 2.6_
  - _Boundary: experiment_sweep_resumption_
  - _Depends: 2.1_
- [ ] 2.3 掃引の終わりへの接続
  - 掃引のtest（設計書の「(B) 掃引」）を先に書き、`execute_experiment_sweep`から呼ぶ。`test_experiment_sweep.py`の、完了した掃引から開始の記録を消して中断を作るtestを、設計書の「既存のtestの書き換え」のとおり直す（zipに証拠があれば直して再実行、証拠の無いzipの別のcaseでは今と同じく失敗）。`experiment_sweep_execution.py`の許可リストを更新する。fresh smokeにまとめの確認を足す。
  - 完了条件: 掃引と再開のtest・依存境界のtest・fresh smokeが通る。
  - _Requirements: 2.1, 2.3, 2.5, 2.6_
  - _Depends: 2.2_

- [ ] 3. 文書と確認
- [ ] 3.1 文書・全pytest・実装レビュー
  - `docs/experiments/experiment-sweep.md`に足す（記録の表のLZMA、`run_start_records.zip`、uraniumと分析に使うPCで`import lzma`を確かめる手順、掃引の途中で更新しない注意）。全pytest、ruff、pyright。独立CLIの実装レビュー1回。
  - 完了条件: 文書があり、全pytestが通り、実装レビューがPASSである。
  - _Requirements: 3.1, 3.2, 4.1_
  - _Depends: 1.2, 2.3_
