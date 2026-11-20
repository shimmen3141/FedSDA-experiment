# Naming

| 名前 | 意味・似た名前との違い |
|---|---|
| `lzma_record_table_files` | runの記録の表を、各配列をLZMAで圧縮したNPZとして書くmodule（`runtime/`）。読込みは持たない |
| `ensure_lzma_compression_available` | Pythonの`lzma`が使えることを確かめ、使えなければ`RuntimeError`を出す。掃引がrunの計算の前に呼ぶ |
| `write_lzma_compressed_record_tables` | `numpy.savez_compressed`と同じ並び・形式で、zipの圧縮方式だけをLZMAにして書く |
| `pack_completed_sweep_run_start_records` | 全runが済んだ掃引の、runごとの開始の記録を`.resume/run_start_records.zip`へまとめ、照合してから元を消す。`sweep.json`・`recovered/`は残す。研究ツール`research_results_archive`のCLIの`pack`（関数は`pack_configuration`。`runs/`と`.resume/<構成>/`をまとめる）とは対象と置き場所が違う |
| `write_run_start_record`（既存） | 役割は同じ（runの開始の記録を書く。同じ内容の既存の記録は受け入れ、違えば拒否）。記録をzipで受け入れたときはzipのパスを返す点だけが変わる |
| `run_start_records.zip` | まとめた開始の記録のファイル名（`.resume/`の下） |
