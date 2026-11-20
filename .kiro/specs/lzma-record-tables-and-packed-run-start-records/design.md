# Design Document

## Overview
2つの独立した変更を1つのspecで行う。(A) 4手法の記録の表のNPZを、共通の書き出し関数でLZMAにする。(B) 掃引の終わりに、runごとの開始の記録を1つのzipへまとめ、開始の記録を個別のファイル→zipの順に読む。どちらもrunの値と乱数の消費を変えない。

## Boundary Commitments
- 持つもの: 新しい`runtime/lzma_record_table_files.py`（記録の表をLZMAのNPZとして書く関数）、4つの`*_run_artifact_files.py`の書き出しの1行、`sweep/experiment_sweep_resumption.py`の開始の記録の読み方とまとめ、`sweep/experiment_sweep_execution.py`のまとめの呼び出し、依存境界の許可リスト、共用のfresh smoke、文書。
- 持たないもの: 読込み（`numpy.load`のまま。zipfileがLZMAを読む）、runの要約、掃引の計画、研究ツール。
- 許す依存: 新しいmoduleは`io`・`lzma`（moduleの読込み時に存在を確かめるためだけ）・`zipfile`・`numpy`（`numpy.lib.format.write_array`、`ndarray`）だけ。
- 再検証が要る変更: NPZのファイルのバイト列は変わる（配列は同じ）。NPZのファイルのhashで照合する処理は無い（主担当とレビュー担当が`runtime/*_run_artifact_files.py`・`tools/experiments/`・goldenを確認。goldenは配列単位のhash）。
- 運用の注意: 開始の記録は`src/`の全`.py`のhash（`execution_source_sha256`）を含むので、`src/`へファイルを足すこの変更を、途中の掃引の途中で取り込むと、その掃引の中断したrunは直せない（既存の性質）。掃引の途中で更新しない（文書に書く）。

## (A) 記録の表のLZMA
- `write_lzma_compressed_record_tables(record_table_stream, record_tables)`: 各配列を`<名前>.npy`として、`zipfile.ZipFile(stream, "w", compression=ZIP_LZMA, allowZip64=True)`の`open(名前, "w", force_zip64=True)`へ`numpy.lib.format.write_array(..., allow_pickle=False)`で書く。配列の順は辞書の順。numpyの`savez_compressed`との違い: 圧縮方式（deflate→LZMA）と、`force_zip64`を常に立てること（numpyは配列が1GiB以上のときだけ立てる。ファイルのバイト列だけが違い、読んだ配列は同じ）。
- `allow_pickle=False`: 今の4手法の記録の表はobjectの配列を持たない（testで全表の`dtype.kind != "O"`を確かめる）。objectの配列が来たら書き出しが失敗し、要求1.4の扱いになる。
- 環境: LZMAはPythonの標準の`lzma`（liblzma）を要る。新しいmoduleの先頭で`import lzma`するので、使えない環境では、掃引の入口がこのmoduleを読み込む時点（runの計算の前）で失敗する。手元のWindowsのvenvでは使える。uraniumでは`python -c "import lzma"`で確かめる手順を文書に書く。
- 4つの書き出し（`save_fedsda_experiment_run`ほか）で、`savez_compressed(record_table_stream, **record_tables)`をこの関数に置き換える。途中のファイル・置く順序の処理は変えない。
- 実測（主担当、2026-11-20、FedSDAのMNIST4・間隔50の1 run）: deflate 0.06秒・0.99MB、LZMA 0.75秒・0.87MB。1 runの所要時間（数分）に比べて小さい。FedDriftの表は約1/5になる。

## (B) 開始の記録の読み方とまとめ
- 開始の記録の読み方: 非公開の`_read_run_start_record(output_directory, planned_run)`が、個別のファイル`.resume/<構成>/<run>.json`、無ければ`.resume/run_start_records.zip`の`<構成>/<run>.json`を読む（どちらも無ければ無い）。`can_recover_interrupted_run`と`write_run_start_record`の両方がこれを使う。
  - `write_run_start_record`: 記録がzipにあり同じ内容なら、個別のファイルを書かずに受け入れる（今の「同じ内容の既存の記録は受け入れる」と同じ）。違えば今と同じく拒否する。どちらにも無ければ今と同じく個別のファイルへ書く。
  - zipはrunごとに開く。今の件数（1つの掃引で数百）では問題にならない。件数が大きくなったら、1回の読込みにまとめる（今回は行わない）。
- `pack_completed_sweep_run_start_records(output_directory)`（公開）:
  1. `.resume/`のパスは既存の`_safe_path`で作り、`.resume/`と、走査するディレクトリが`_is_link`ならまとめない。残っている`.resume/run_start_records.zip.partial`があれば消す。
  2. `.resume/`直下のディレクトリのうち`recovered`以外（構成ごと）の`*.json`を集める（計画に無いrunの記録も含む。構成の名前が`recovered`なら対象外になるが、今の構成名にその名前は無い）。`_write_atomic`の途中のファイル（`*.json.<uuid>.partial`）は`*.json`に一致しないので対象外のまま残す。集めた物が無ければ何もせずに戻る（既にあるzipを書き直さない）。
  3. 既にzipがあれば、その中身を読む。同じ名前がzipと個別のファイルの両方にあり、バイト列が同じなら、個別のファイルは消す対象にだけ入れる（zipへ二重に入れない）。1つでも違えば、zipを作らずに戻り、警告を返す。
  4. zipの中のパス`<構成>/<run>.json`で`.partial`へ書き、fsyncしてから、非公開の`_verify_run_start_record_archive`で読み戻して全記録のバイト列を照合する。一致したら`os.replace`でzipを置き、その後で個別のファイルと空になったディレクトリを消す（削除の途中で止まっても、各記録はzipに残り、上の読み方で読める）。
  5. 手順1〜4で起きる例外のうち、読み書き・照合の失敗（`OSError`、`zipfile.BadZipFile`、`KeyError`、`ValueError`、`lzma.LZMAError`、照合の不一致）は、`.partial`を消して元を残し、まとめなかったこととその理由を返す（例外を出さない）。
- `execute_experiment_sweep`: 出力先の排他の中（内側の呼び出し）で、全runの結果がそろった後（並列ならprocess poolを`shutdown(wait=True)`した後）、`try`の外で、`failed`が無いときだけ、返す直前に呼ぶ。まとめなかった理由があれば、進み具合の報告（`report_progress`）へ警告として出す。例外で抜ける経路では呼ばない。
- 再開: `find_sweep_resumption_candidates`は、完了でも不在でもない全run（食い違うrun、基準版の違うrunを含む）について`can_recover_interrupted_run`を呼び、そこで開始の記録を読む。上の読み方で、まとめた後もこの判定と直す処理は今と同じになる。
- 研究ツール`research_results_archive`との関係: 製品はtoolsをimportしないので、NPZをLZMAで書く処理は重複して持つ。研究ツールのCLIの`pack`（関数は`pack_configuration`）は`.resume/<構成>/`の個別のファイルだけをまとめるので、`run_start_records.zip`は対象外で、そのまま残る。

## Testing Strategy
- (A): `write_lzma_compressed_record_tables`の全エントリが`ZIP_LZMA`で、同じ表を`savez_compressed`で書いたものと、`numpy.load`で読んだ配列の名前の順・型・形・値（NaNは同じとみなす）が同じこと。4手法で小さいrunを保存し、同じことと、全表に`dtype.kind == "O"`が無いことを確かめる。deflateのNPZを置いた成果物を読込み関数が今と同じに読むこと。書き出しの失敗（新しい関数をmonkeypatch）で途中のファイルが残らないこと。
- (B) 読み方: まとめた後に、要約を消したrunを直せること、違う開始の記録を持つrunの開始を拒否すること、同じ内容なら個別のファイルを書かずに受け入れること。
- (B) まとめ: まとめ・照合・元の削除、`sweep.json`と`recovered/`を残す、個別のファイルが無ければzipを書き直さない、既存のzipへの追加、同じ名前で同じ中身なら二重にしない・違う中身ならまとめない、照合の失敗（`_verify_run_start_record_archive`をmonkeypatch）・書き込みのOSError・壊れた既存のzipで元が残ること、残った`.partial`を消すこと、`.resume`がリンクならまとめないこと。
- (B) 掃引: 小さい掃引（2 run）で、全部済めばまとまり、同じ計画で続けると候補として見つかり全runが飛ばされ出力先のファイルが変わらないこと。1 runを失敗させた掃引ではまとめないこと。まとめの失敗で警告を出し、掃引は失敗にしないこと。
- 既存のtestの書き換え: `tests/refactoring/test_experiment_sweep.py`の、完了した掃引の複製から`.resume/fedsda/<run>.json`を消して中断を作るtestは、まとめた後はzipの中にあるので、前提を書き換える。まとめた後の掃引の個別のファイルを見るtestは、zipの中身を見るように直す。既存の`savez_compressed`のmonkeypatch（`test_fedsda_experiment_run_artifacts.py`・`test_feddrift_experiment_run_artifacts.py`）は、新しい関数の名前に替える。
- 依存境界（許可リストは完全一致）: 4つの`*_run_artifact_files`は`numpy.savez_compressed`を外し、`federated_learning_experiments.runtime.lzma_record_table_files.write_lzma_compressed_record_tables`を足す。新しいmoduleは`io.BytesIO`は使わず、`lzma`・`zipfile.ZipFile`・`zipfile.ZIP_LZMA`・`numpy.lib.format.write_array`など実際にimportする名前の組で登録する。`experiment_sweep_resumption.py`は`zipfile`・`os`・`lzma`の実際にimportする名前を足す。`experiment_sweep_execution.py`は`pack_completed_sweep_run_start_records`を足す。正確な名前の組は実装でimportした物に合わせる（未使用の許可は既存のtestが拒否する）。
- 共用のfresh smoke（`tests/refactoring/fresh_process_smoke.py`）: 既存の掃引の2回実行（再開）の流れが、まとめた後も通ること。まとめた`run_start_records.zip`ができていることを足す。
- 回帰: 全版のgolden、既存の成果物と掃引の再開のtest。
