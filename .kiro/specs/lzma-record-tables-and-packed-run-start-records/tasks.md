# Implementation Plan

依存: 1.1 → 1.2 → 2.1 → 2.2 → 2.3 → 3.1（1と2は内容は独立だが、どれも`test_single_run_dependency_boundaries.py`を編集するので順に進める）。

- [x] 1. 記録の表のLZMA
- [x] 1.1 書き出し関数
  - `write_lzma_compressed_record_tables`と`ensure_lzma_compression_available`のtest（全エントリがLZMA、`savez_compressed`と同じ配列、lzmaが無ければ`RuntimeError`）を先に書き、`runtime/lzma_record_table_files.py`を実装する。依存境界に新しいmoduleを登録する。
  - 完了条件: 新しいtestと依存境界のtestが通り、`ruff check`が通る。
  - _Requirements: 1.1, 1.2_
  - _Boundary: lzma_record_table_files_
- [x] 1.2 4手法の書き出しへの接続
  - 4手法の保存のtest（LZMA、同じ配列、objectの配列が無い、deflateのNPZも読める、失敗で途中のファイルが残らない）を先に書き、4つの書き出しを置き換える。既存の`savez_compressed`のmonkeypatchを新しい関数へ替える。4つの許可リストを更新する。
  - 完了条件: 成果物のtest・依存境界のtest・全版のgoldenが通る。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.1_
  - _Depends: 1.1_

- [x] 2. 開始の記録
- [x] 2.1 開始の記録の読み方（個別のファイル→zip）
  - zipにだけ記録があるとき（zipはtestの中で手で作る）の`can_recover_interrupted_run`（直せる）と`write_run_start_record`（同じ内容は受け入れて個別のファイルを書かない、違う内容は拒否）、壊れたzip・途中で切れたzip・`lzma`を読み込めない状態での扱いのtestを先に書き、`_read_run_start_record`を実装して両方から使う。`experiment_sweep_resumption.py`の許可リストに、読込みで実際にimportする名前を足す。
  - 完了条件: 新しいtest・既存の再開のtest・依存境界のtestが通る。
  - _Requirements: 2.5_
  - _Boundary: experiment_sweep_resumption_
  - _Depends: 1.2_
- [x] 2.2 まとめる関数
  - `pack_completed_sweep_run_start_records`のtest（設計書のTesting Strategyの「(B) まとめ」。重複だけならzipを書き直さないことを含む）を先に書き、実装する。`experiment_sweep_resumption.py`の許可リストに、まとめで足した名前を足す。
  - 完了条件: 新しいtest・依存境界のtestが通る。
  - _Requirements: 2.1, 2.2, 2.4, 2.6_
  - _Boundary: experiment_sweep_resumption_
  - _Depends: 2.1_
- [x] 2.3 掃引の終わりへの接続
  - 掃引のtest（設計書の「(B) 掃引」と、lzmaが無ければrunの前に止まること）を先に書き、`execute_experiment_sweep`から`ensure_lzma_compression_available`とまとめを呼ぶ。`test_experiment_sweep.py`の`test_conflicting_existing_files_are_reported_and_not_overwritten`の`missing_summary`の場合を、設計書の「既存のtestの書き換え」のとおり直す（zipの中の記録を除いて今と同じく失敗を確かめ、証拠の残るcaseは直して再実行を確かめる）。`experiment_sweep_execution.py`の許可リストを更新する。fresh smokeにまとめの確認を足す。
  - 完了条件: 掃引と再開のtest・依存境界のtest・fresh smokeが通る。
  - _Requirements: 2.1, 2.3, 2.5, 2.6_
  - _Depends: 2.2_

- [x] 3. 文書と確認
- [x] 3.1 文書・全pytest・実装レビュー
  - `docs/experiments/experiment-sweep.md`に足す（記録の表のLZMA、`run_start_records.zip`、壊れたzipの影響と復旧の手順、uraniumと分析に使うPCで`import lzma`を確かめる手順、掃引の途中で更新しない注意）。全pytest（`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`、`tests/test_regression.py`・`tests/test_proposed_regression.py`を含む全部）、`ruff check src tests/refactoring`、pyright。独立CLIの実装レビュー1回。
  - 完了条件: 文書があり、全pytestが通り（JUnitでgoldenのtestがskipされていない）、実装レビューがPASSである。
  - _Requirements: 3.1, 3.2, 4.1_
  - _Depends: 1.2, 2.3_

## Implementation Notes

### 仕様書群のレビュー（2026-11-20）
- 7回。全てClaude Haiku 5.5（effort medium、読むだけ）。GPT-6 Lunaは使用量の上限で使えなかった。session・判定はspec.json。
- 主な指摘と採否（全部採用）: まとめた後の再開（直す判定・開始の記録の食い違いの拒否）が今と同じでなくなる→開始の記録を個別のファイル→zipの順に読む関数を作る。zipを書き直さない条件、重複の扱い、壊れたzip・切れたzipの読込みの例外、既存testの期待の変化（要約の無いNPZのrunが直せるようになる）、依存境界の許可リスト、`import lzma`が未使用になる問題（掃引の前に`importlib.import_module`で確かめる）、全pytestの条件。
- 7回目のPASSのMinor 3件（`numpy.ndarray`の許可、tasks 2.1の文言、研究ツールで先にまとめた出力先の扱い）は文書だけで直した。

### 実装（2026-11-20、主担当）
- commit: 1.1 `92f82a1b`、1.2 `6cba87cb`、2.1 `edb07f5d`、2.2 `bcb6b040`、2.3 `90ceb687`、文書 `ae31a446`、やり直し `3076db50`・`8ae37435`、レビューの反映 `916f4c66`。
- RED: 1.1・1.2・2.1・2.2はtestを先に書いて失敗を確かめた。2.3は接続を先に入れてからtestを書いたので、REDを確かめていない（手順からの逸脱）。
- 2.1: lzmaの無い環境では、zipの目次は読めるので、zipに無いrunの開始の記録は書ける（中身を読むときだけ失敗する）。設計の範囲内と判断し、testは「zipにあるrunの直す判定と開始が失敗する」ことを確かめる形にした。
- 設計に無い追加: `_replace_with_retries`。Windowsの全pytestの負荷の下だけで、作ったばかりの`run_start_records.zip`の置き換えがWinError 5（アクセス拒否）になった（単独・部分的な負荷では再現しない。ウイルス対策のzipの検査と推定、未確認）。`PermissionError`のときだけ、合計約14秒まで待ってやり直す。
- 全pytestの途中経過: 1回目（`ae31a446`）は2 failed（zipの置き換えのアクセス拒否1件と、並列の掃引のtest 1件。後者は単独・負荷の再現で再現せず、理由は記録できなかったので、失敗の理由を出すようtestを直した）。2回目は1 failed（やり直しのtestが待ち時間0で本物の拒否と重なった。待ち時間を既定に直した）。3回目は、主担当のコマンドの誤りで環境変数（スレッド数1）が付かずに走ったので使わない。
- 全pytest（最終、`2e24df29`、OMP/MKL=1、`-n 12 --dist loadfile`）: 14,110 passed・23 skipped・失敗0、21分56秒。goldenのskipは環境のgoldenのtest 1件（いつもどおり）。前の`dwm-neural-network-baseline`・`stagger-agrawal-random-rbf-recurring-datasets`で出たsklearnのDLLの拒否は、今回は出なかった。
- fresh smoke・pyright（変更したsource 7ファイル、0 errors）・ruffが通った。

### 実装レビュー（2026-11-20）
- Claude Haiku 5.5（effort medium）。対象test 2,519 passed、ruff、`spec_checks.py names`を独立に実行。全pytest・fresh smoke・pyrightは主担当の実測を照合。
- 採用（Minor）: 依存境界の一覧の重複を削った。zipの名前の定数を非公開にした（命名表に無い公開名だったため）。研究ツールの指摘（説明の正確さ、失敗したrunのある掃引を飛ばす、書き込み中の失敗で途中のファイルを消す）は研究ツール側で反映した（`2e24df29`）。
- 不採用（任意）: 研究ツールの並列で1つの構成が失敗したときに他の構成の処理が続く点は、元のファイルを消すのは照合の後だけなので安全側として残す。`spec_checks.py names`の表示の文言は、このspecの外。
