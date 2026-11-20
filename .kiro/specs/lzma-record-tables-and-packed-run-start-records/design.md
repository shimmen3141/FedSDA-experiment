# Design Document

## Overview
2つの独立した変更を1つのspecで行う。(A) 4手法の記録の表のNPZを、共通の書き出し関数でLZMAにする。(B) 掃引の終わりに、runごとの開始の記録を1つのzipへまとめる。どちらもrunの値と乱数の消費を変えない。

## Boundary Commitments
- 持つもの: 新しい`runtime/lzma_record_table_files.py`（記録の表をLZMAのNPZとして書く関数）、4つの`*_run_artifact_files.py`の書き出しの1行、`sweep/experiment_sweep_resumption.py`の開始の記録のまとめと読み方（個別のファイル→zip）、`sweep/experiment_sweep_execution.py`のまとめの呼び出し、依存境界の許可リスト、文書。
- 持たないもの: 読込み（`numpy.load`のまま。zipfileがLZMAを読む）、runの要約、掃引の計画・再開の判定、研究ツール。
- 許す依存: 新しいmoduleは`io`・`zipfile`・`numpy`（`numpy.lib.format.write_array`、`ndarray`）だけ。
- 再検証が要る変更: NPZのファイルのバイト列は変わる（配列は同じ）。NPZのファイルのhashで照合する処理は無い（主担当が`runtime/*_run_artifact_files.py`と`tools/experiments/`を確認）。

## (A) 記録の表のLZMA
- `write_lzma_compressed_record_tables(record_table_stream, record_tables)`: numpyの`_savez`と同じく、各配列を`<名前>.npy`として`ZipFile.open(名前, "w", force_zip64=True)`へ`numpy.lib.format.write_array(..., allow_pickle=False)`で書く。zipの圧縮方式だけを`ZIP_LZMA`にする。配列の順は辞書の順。記録の表にobjectの配列は無い（今の4手法の表は数値・bool）ので`allow_pickle=False`にする（objectの配列が来たら書き出しが失敗し、要求1.4の扱いになる）。
- 環境: LZMAはPythonの標準の`lzma`（liblzma）を要る。手元のWindowsのvenvでは使える。uraniumの`.venv`では、最初の掃引の前に`python -c "import lzma"`で確かめる（文書に書く）。
- 4つの書き出し（`save_fedsda_experiment_run`ほか）で、`savez_compressed(record_table_stream, **record_tables)`をこの関数に置き換える。途中のファイル・置く順序の処理は変えない。
- 実測（主担当、2026-11-20、FedSDAのMNIST4・間隔50の1 run）: deflate 0.06秒・0.99MB、LZMA 0.75秒・0.87MB。1 runの所要時間（数分）に比べて小さい。FedDriftの表は約1/5になる。
- 既存のtestで`savez_compressed`をmonkeypatchして書き出しの失敗を作るもの（`test_fedsda_experiment_run_artifacts.py`・`test_feddrift_experiment_run_artifacts.py`）は、新しい関数の名前に替える。

## (B) 開始の記録のまとめ
- `pack_completed_sweep_run_start_records(output_directory)`（`experiment_sweep_resumption.py`）:
  1. 残っている`.resume/run_start_records.zip.partial`があれば消す。
  2. `.resume/`直下のディレクトリのうち`recovered`以外（構成ごと）の`*.json`を集める（計画に無いrunの記録も含む）。無ければ何もせずに戻る（既にあるzipを書き直さない）。
  3. 既にzipがあれば、その中身を読む。同じ名前がzipと個別のファイルの両方にあり、バイト列が同じなら、個別のファイルは消す対象にだけ入れる（zipへ二重に入れない）。違えば、まとめずに戻る。
  4. zipの中のパス`<構成>/<run>.json`で`.partial`へ書き、fsyncしてから、非公開の`_verify_run_start_record_archive`で読み戻して全記録のバイト列を照合する。一致したら`replace`でzipを置き、その後で個別のファイルと空になったディレクトリを消す（削除の途中で止まっても、各記録はzipに残る）。
  5. 照合の失敗とOSErrorでは、`.partial`を消して元を残し、まとめなかったことを返す（例外を出さない。掃引は報告に警告を出す）。
- 開始の記録の読み方: 非公開の`_read_run_start_record(output_directory, planned_run)`が、個別のファイル、無ければzipの`<構成>/<run>.json`を読む（どちらも無ければ無い）。`can_recover_interrupted_run`と`write_run_start_record`（違う内容の記録があれば拒否する検査）の両方がこれを使う。新しく書く記録は今と同じく個別のファイル。
- `execute_experiment_sweep`: 出力先の排他の中（内側の呼び出し）で、全runの結果がそろった後（並列ならprocess poolを`shutdown(wait=True)`した後）、`try`の外で、`failed`が無いときだけ、返す直前に呼ぶ。例外で抜ける経路では呼ばない。
- 再開: 候補を探す処理（`find_sweep_resumption_candidates`）は、要約の無いNPZがあるrunについて`can_recover_interrupted_run`を呼び、そこで開始の記録を読む。上の読み方で、まとめた後もこの判定と直す処理は今と同じになる。後からrunの成果物を消して続けたときは、そのrunの開始の記録を`.resume/<構成>/`へ書く（zipに同じ内容があれば今と同じく受け入れ、違えば拒否する）。
- 研究ツール`research_results_archive`との関係: 製品はtoolsをimportしないので、NPZをLZMAで書く処理は重複して持つ。研究ツールは`.resume/<構成>/`の個別のファイルだけをまとめるので、`run_start_records.zip`は対象外で、そのまま残る。

## Testing Strategy
- (A): 4手法で小さいrunを保存し、NPZの全エントリが`ZIP_LZMA`であること、同じ表を`savez_compressed`で書いたものと、`numpy.load`で読んだ配列の名前の順・型・形・値（NaNは同じとみなす）が同じこと。deflateのNPZを置いた成果物を読込み関数が今と同じに読むこと。書き出しの失敗（新しい関数をmonkeypatch）で途中のファイルが残らないこと。
- (B): 小さい掃引（2 run）を流し、全部済めば`.resume/run_start_records.zip`ができ、元の開始の記録が消え、`sweep.json`が残ること。同じ計画で続けると候補として見つかり全runが飛ばされ、出力先のファイルが変わらないこと。1 runを失敗させた掃引ではまとめないこと。照合の失敗（`_verify_run_start_record_archive`をmonkeypatch）とOSErrorで元が残ること。既にzipがある出力先での追加（同じ内容は二重にしない、違う内容ならまとめない）。まとめた後に、要約を消したrunを直せることと、違う開始の記録を持つrunの開始を拒否すること。
- 既存のtestの書き換え: `tests/refactoring/test_experiment_sweep.py`の、完了した掃引の複製から`.resume/fedsda/<run>.json`を消して中断を作るtestは、まとめた後はzipの中にあるので、前提を書き換える（zipから記録を除く、または、まとめる前の状態を作る）。まとめた後の掃引の個別のファイルを見るtestは、zipの中身を見るように直す。
- 依存境界: 4つの`*_run_artifact_files`の許可リストの`numpy.savez_compressed`を新しい関数に替える（許可が使われていることも確かめるtestがある）。`experiment_sweep_execution.py`・`experiment_sweep_resumption.py`の許可リストに新しいimportを足す。新しいmoduleを登録する。
- 回帰: 全版のgolden、既存の成果物と掃引の再開のtest。依存境界に新しいmoduleを登録する。
