# Naming

| 名前 | 意味・似た名前との違い |
|---|---|
| `lzma_record_table_files` | runの記録の表を、各配列をLZMAで圧縮したNPZとして書くmodule（`runtime/`）。読込みは持たない |
| `write_lzma_compressed_record_tables` | `numpy.savez_compressed`と同じ並び・形式で、zipの圧縮方式だけをLZMAにして書く |
| `pack_completed_sweep_run_start_records` | 全runが済んだ掃引の、runごとの開始の記録を`.resume/run_start_records.zip`へまとめ、照合してから元を消す。`sweep.json`・`recovered/`は残す |
| `run_start_records.zip` | まとめた開始の記録のファイル名（`.resume/`の下） |
