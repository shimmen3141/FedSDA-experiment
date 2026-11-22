# Naming

| 名前 | 意味・似た名前との違い |
|---|---|
| `packed_run_artifact_archive` | 構成ごとのrunの成果物（`runs/`の要約と記録の表）を`runs.zip`へまとめ、zipから読み、`runs/`へ戻すmodule（`sweep/`）。開始の記録のまとめ（`experiment_sweep_resumption`の`pack_completed_sweep_run_start_records`）とは対象と置き場所が違う |
| `RUN_ARTIFACT_ARCHIVE_NAME` | まとめたzipのファイル名`runs.zip`（構成のディレクトリの下）。研究ツールの`ARCHIVE_NAME`と同じ値 |
| `RunArtifactPackingResult` | 掃引の終わりのまとめの結果（まとめた構成の名前と、警告）。判定はしない記録の型 |
| `pack_completed_sweep_run_artifacts` | 全runが済んだ掃引の、計画の各構成の`runs/`をまとめる。呼出し側が、失敗と読めない要約が無いことを確かめ、出力先の排他を持つ。既存の`pack_completed_sweep_run_start_records`（開始の記録を`.resume/run_start_records.zip`へまとめる）とは対象が違う（こちらはrunの成果物） |
| `pack_configuration_run_artifacts` | 1つの構成の`runs/`を`runs.zip`へまとめ、照合してから元を消す。例外を出さず、(zipを置いたか, 警告)を返す。研究ツールの`pack_configuration`（NPZの中を書き直し、`.resume/<構成>/`も入れる）とは形と対象が違う |
| `read_packed_run_summary_and_record_table` | `runs.zip`から1つのrunの要約と記録の表のバイト列を読み、manifestと照合して返す。zipに無ければ`None`。runの成果物を保存・読込みする`*_run_artifact_files`のmoduleとは別で、ファイルを書かない |
| `list_packed_run_identifiers` | `runs.zip`にある（要約と記録の表の両方がある）runの識別子の集合 |
| `restore_packed_run_artifacts` | `runs.zip`の全ファイルを`runs/`へ戻し、照合してからzipを消す。研究ツールの`unpack_configuration`と同じ結果になるが、既にある同じ中身のファイルを受け入れる点が違う。既存の`recover_interrupted_run`（中断したrunのNPZを退避する）とは目的が違う |
| `find_existing_run_state`（既存） | 役割は同じ（出力先にそのrunの済んだ結果があるかを調べる、何も変えない）。`runs/`に無ければ`runs.zip`からも調べる点だけが変わる |
| `execute_experiment_sweep`（既存） | 役割は同じ。計画にzipに無いrunがある構成と、`runs/`とzipが両方ある構成を、計算の前に戻す点が変わる |
