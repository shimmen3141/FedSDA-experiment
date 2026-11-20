# Design Document

## Overview
2つの独立した変更を1つのspecで行う。(A) 4手法の記録の表のNPZを、共通の書き出し関数でLZMAにする。(B) 掃引の終わりに、runごとの開始の記録を1つのzipへまとめる。どちらもrunの値と乱数の消費を変えない。

## Boundary Commitments
- 持つもの: 新しい`runtime/lzma_record_table_files.py`（記録の表をLZMAのNPZとして書く関数）、4つの`*_run_artifact_files.py`の書き出しの1行、`sweep/experiment_sweep_resumption.py`の開始の記録のまとめ、`sweep/experiment_sweep_execution.py`のまとめの呼び出し、文書。
- 持たないもの: 読込み（`numpy.load`のまま。zipfileがLZMAを読む）、runの要約、掃引の計画・再開の判定、研究ツール。
- 許す依存: 新しいmoduleは`io`・`zipfile`・`numpy`（`numpy.lib.format.write_array`、`ndarray`）だけ。
- 再検証が要る変更: NPZのファイルのバイト列は変わる（配列は同じ）。NPZのファイルのhashで照合する処理は無い（主担当が`runtime/*_run_artifact_files.py`と`tools/experiments/`を確認）。

## (A) 記録の表のLZMA
- `write_lzma_compressed_record_tables(record_table_stream, record_tables)`: `numpy.savez_compressed`と同じく、各配列を`<名前>.npy`として、`numpy.lib.format.write_array(..., allow_pickle=False)`で書き、zipの`ZIP_LZMA`で入れる。配列の順は辞書の順（`savez_compressed`と同じ）。`savez_compressed`との違いはzipの圧縮方式だけ。
- 4つの書き出し（`save_fedsda_experiment_run`ほか）で、`savez_compressed(record_table_stream, **record_tables)`をこの関数に置き換える。途中のファイル・置く順序の処理は変えない。
- 実測（主担当、2026-11-20、FedSDAのMNIST4・間隔50の1 run）: deflate 0.06秒・0.99MB、LZMA 0.75秒・0.87MB。1 runの所要時間（数分）に比べて小さい。FedDriftの表は約1/5になる。
- 既存のtestで`savez_compressed`をmonkeypatchして書き出しの失敗を作るもの（`test_fedsda_experiment_run_artifacts.py`・`test_feddrift_experiment_run_artifacts.py`）は、新しい関数の名前に替える。

## (B) 開始の記録のまとめ
- `pack_completed_sweep_run_start_records(output_directory)`（`experiment_sweep_resumption.py`）: `.resume/`の下の構成のフォルダ（`sweep.json`・`recovered/`・zip以外のディレクトリ）の`*.json`を集め、zipの中のパス`<構成>/<run>.json`で`.resume/run_start_records.zip.partial`へ書く。既にzipがあれば、その中身も入れる（同じ名前でバイト列が違えば、まとめずに戻る）。読み戻して全ファイルのバイト列を照合し、一致したら`replace`でzipを置き、元のファイルと空になった構成のフォルダを消す。照合に失敗したら`.partial`を消して元を残す（例外は出さず、まとめなかったことを返す）。
- `execute_experiment_sweep`: 出力先の排他の中で、全runの結果がそろい、`failed`が無いときだけ、返す直前に呼ぶ。例外で抜ける経路では呼ばない。
- 再開: 候補を探す処理は`sweep.json`と`runs/`の成果物で判定し、開始の記録を読まない。直す処理（`can_recover_interrupted_run`）は、要約の無いNPZがあるrunだけで開始の記録を読む。まとめるのは全runが済んだ後だけなので、まとめた出力先で直す必要があるrunは無い。後からrunの成果物を消して続けたときは、そのrunの開始の記録が新しく`.resume/<構成>/`へ書かれる（今と同じ）。

## Testing Strategy
- (A): 4手法で小さいrunを保存し、NPZの全エントリが`ZIP_LZMA`であること、同じ表を`savez_compressed`で書いたものと、`numpy.load`で読んだ配列の名前の順・型・形・値（NaNは同じとみなす）が同じこと。deflateのNPZを置いた成果物を読込み関数が今と同じに読むこと。書き出しの失敗（新しい関数をmonkeypatch）で途中のファイルが残らないこと。
- (B): 小さい掃引（2 run）を流し、全部済めば`.resume/run_start_records.zip`ができ、元の開始の記録が消え、`sweep.json`が残ること。同じ計画で続けると候補として見つかり全runが飛ばされること。1 runを失敗させた掃引ではまとめないこと。照合の失敗（zipの読み戻しをmonkeypatch）で元が残ること。既にzipがある出力先での追加。
- 回帰: 全版のgolden、既存の成果物と掃引の再開のtest。依存境界に新しいmoduleを登録する。
