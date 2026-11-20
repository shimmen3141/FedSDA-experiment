# Requirements Document

## Project Description (Input)
- 誰が困っているか: 研究者（ユーザー）。研究室サーバ（uranium）で流した結果を手元へ転送するとき、ファイルの数が多く時間がかかる。結果の置き場所（`results/`）は49GB・12万5千ファイルに増えた。
- 今の状況: 全手法（FedSDA・FedDrift・FedAvg・DWM）のrunの記録の表（NPZ）は`numpy.savez_compressed`（deflate）で書く。掃引は、runごとの開始の記録（`.resume/<構成>/<run>.json`。中断したrunを安全に直すためだけに使う）を、全runが済んだ後も残す。
- 何を変えるか（2026-11-20ユーザー指示「1・2・3はやってください」の2。2026-11-17の指示「`.resume`を1つの圧縮ファイルにまとめる」を含む）:
  (1) 記録の表のNPZを、配列を変えずに、各配列をLZMAで圧縮して書く（実測でFedSDAの1 runで約12%小さく、FedDriftでは約1/5）。
  (2) 掃引の全runが済んだら、runごとの開始の記録を1つのzipへまとめる。
- 守ること: runの値・配列・goldenを変えない。既に書いたdeflateのNPZも読める。

## Boundary Context
- In scope: 4手法の記録の表の書き方、掃引の終わりの開始の記録のまとめ、文書。
- Out of scope: runの要約（JSON）の圧縮、`runs/`のまとめ（研究ツール`research_results_archive`が扱う）、単一ストリームのパッケージのNPZ、既存の結果の書き直し。

## Requirements

### Requirement 1: 記録の表のLZMA圧縮
1.1 When FedSDA・FedDrift・FedAvg・DWMのrunの成果物を保存するとき, the run artifact writer shall 記録の表のNPZの各配列を、zipのLZMAで圧縮して書く。ファイル名・拡張子・置き場所・書く順序（表を先に置き、要約を最後に置く）・途中のファイルの扱いは今と同じにする。
1.2 The written NPZ shall `numpy.load`でそのまま読め、配列の名前の順・型・形・値が、今の`savez_compressed`で書いた場合と同じになる。
1.3 The run artifact loaders shall LZMAのNPZと、既に書いたdeflateのNPZの両方を、今と同じ結果で読む。
1.4 If 記録の表の書き出しに失敗したら, the writer shall 今と同じく、途中のファイルを残さず、要約も置かない。

### Requirement 2: 掃引の終わりの開始の記録のまとめ
2.1 When 掃引の全runの結果が「済んだ」か「済んだ結果があり飛ばした」で、失敗が無いとき, the sweep shall 出力先の排他を持ったまま、`.resume/<構成>/<run>.json`の全ファイルを、`.resume/run_start_records.zip`（LZMA）へまとめる。zipを読み戻して全ファイルのバイト列が元と一致したら元のファイルを消し、一致しなければzipを消して元を残す。
2.2 The sweep shall `.resume/sweep.json`（再開の候補を探すのに使う）と`.resume/recovered/`（直したrunの退避）は、まとめずに残す。
2.3 If 失敗したrunがある、または、掃引が例外（中断を含む）で終わったら, the sweep shall 開始の記録をまとめない（今と同じに残す）。
2.4 When 既に`.resume/run_start_records.zip`がある出力先で掃引を続け、全runが済んだとき, the sweep shall 既にあるzipの記録と新しく書いた記録を合わせた新しいzipを作り、照合してから置き換える。同じ名前の記録がzipとファイルの両方にあり、バイト列が違うときは、まとめずに両方を残す（掃引は失敗にしない）。
2.5 The resumption shall 開始の記録をまとめた出力先を、今と同じく再開の候補として見つけ、済んだrunを飛ばす。

### Requirement 3: 既存の値の維持と検証
3.1 The goldens shall 全版のgoldenの値を変えない。`BEHAVIOR_BASELINE_VERSION`は上げない。
3.2 The tests shall 次を確かめる: (i) 4手法の書いたNPZの全配列がLZMAで、`savez_compressed`で書いた同じ表と配列が同じ。(ii) 4手法の読込みが、deflateのNPZとLZMAのNPZを同じに読む。(iii) 書き出しの失敗で途中のファイルが残らない。(iv) 全runが済んだ掃引で開始の記録がまとまり、`sweep.json`が残り、まとめた後も再開の候補として見つかり全runが飛ばされる。(v) 失敗したrunがある掃引ではまとめない。(vi) zipの照合が失敗したら元が残る。

### Requirement 4: 文書
4.1 The docs shall `docs/experiments/experiment-sweep.md`の出力の説明に、記録の表のLZMAと、`.resume/run_start_records.zip`を足す。
