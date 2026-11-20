# 実験の結果のファイルの数と容量を減らす（2026-11-20）

ユーザーの依頼（2026-11-20）: 情報を保ったまま、ファイルの数を少なく・容量を小さく・扱いやすくする。既存の結果にも適用する。
1（済んだ結果をまとめる研究ツール）・2（製品の書き出しの変更）・3（Windowsの書き出し直しの削除）を行う判断（ユーザー）。

## 今の状態の測定（手元の`results/`、2026-11-20）
- 全体49GB・125,313ファイル。runごとのNPZ 41.5GB（45,568個）、JSON 3.4GB（73,118個）、`.resume`は2,893ファイル・0.04GB。
- NPZはFedSDA系が27.4GB、FedDrift系2.1GB、FedAvg系0.4GB。runの外のNPZ 10.4GB（ほぼ1クライアントの版の系列）。
- runのNPZは既にdeflateで圧縮されている。可逆の試算（各60ファイル、読み戻して配列が一致することを確認）: LZMAにすると、FedSDA系で0.87倍、FedDrift系で約0.22倍、1クライアントの版のrunで0.90倍。整数を最小の型にしても、FedSDA系では0.98倍しか減らない（容量の大半は標本ごとの浮動小数の診断の配列）。
- runのJSON（200個の試料）: gzipで0.14倍、LZMAで0.13倍。
- 1クライアントの版の系列（表現の特徴、float64）: 値は全てfloat32で正確に表せる。float32＋LZMAで0.80倍（22.3MB→17.9MB、1ファイル）。

## 3. 削除（ユーザーの判断）
- `results/fedhasp-single-client-pilot-v16/streams-pretrained-shared-features-seeds0-29-windows-reexport`（4.5GB、npz 300・json 301）を削除した。手元のWindowsで表現の系列を書き出し直した診断の退避物で、比較の正本はuraniumで書き出した`streams-pretrained-shared-features-seeds0-29`（[記録](fedhasp-single-client-pilot-v16.md)の照合ができていない理由の節）。2組は最大1e-4違う（事前学習の浮動小数点の環境差）。

## 1. 研究ツール`tools/research_results_archive`
- `pack`: 構成のフォルダごとに、`runs/`と`.resume/<構成>/`を1つの`runs.zip`へまとめる。JSONなどはLZMA、NPZは配列を変えずに中をLZMAへ書き直して無圧縮で入れる。zipの中に元の各ファイルのパス・バイト数・sha256（NPZは配列の名前・型・形）を記録する。読み戻して全ファイルを照合してから元を消す（NPZ以外はバイト列、NPZは配列の名前・型・形・値）。`run_summary.csv`が無い・runの数が合わない構成は済んでいないとみなして飛ばす。
- `unpack`: 元の場所へ戻し、manifestと照合してからzipを消す。NPZはファイルのバイト列が元と変わるが、配列は同じ。
- `--workers N`で構成のフォルダを並列に処理する。
- 検査`checks_results_archive.py`（8件）: 往復で全ファイルが同じ、済んでいない構成を飛ばす、dry-runは何も変えない、照合の失敗で元を残す、戻す先にファイルがあれば止める、まとめ済みを飛ばす、並列の往復。
- 実結果のコピーでの試し（手元、1 process）: `method-comparison-recurring-synthetic-v16`と`fedhasp-simplified-v16`の2,331 run分。4,792ファイル・1,369MBが62ファイル・1,150MB（0.84倍）、32分。戻してから元の結果（`results/`）と全ファイルを照合し、548ファイル・NPZ 1,400個とも一致した（NPZ以外はバイト列、NPZは配列）。LZMAの速さは1 processで約0.7MB/秒なので、全体（約45GB）を1 processで処理すると約18時間かかる。並列で縮める。
- 実結果への適用は、ユーザーが対象を決めてから行う（元のファイルを消すため）。

## 2. 製品の書き出しの変更
- spec `lzma-record-tables-and-packed-run-start-records`（[tasks](../../.kiro/specs/lzma-record-tables-and-packed-run-start-records/tasks.md)）。runの記録の表をLZMAのNPZで書く。全runが済んだ掃引の開始の記録を`.resume/run_start_records.zip`へまとめる（2026-11-17のユーザー指示を含む）。手順と注意は[掃引の文書](../experiments/experiment-sweep.md)。
- 実測（FedSDAのMNIST4の1 run）: deflate 0.06秒・0.99MB、LZMA 0.75秒・0.87MB。

## 削除と、既存の結果のまとめ（2026-11-20〜21、ユーザーの判断）
- 削除（ユーザーの了承）: 作業の一時物（`_test_tmp`（pytestの一時フォルダ、20,677ファイル）、`_sol_tmp_b1`・`_sol_tmp_c`・`_s3_preview`・`_tool-checks`、空のフォルダ2つ）、参照の無い10月27日のオラクルの実行ログ4つ、`feddrift-tuned-delta-recurring-synthetic-v16`の直下にあったagrawal4の掃引の複製（転送でできたもの。中のフォルダとバイト列まで同じことを確かめた）。完了したspecが検証の証拠として名指しする`_feddrift_*`のログ7つは残した。
- 古い結果（基準版7〜15の比較とablation、10月の診断）は残す（ユーザーの判断。後で消すこともできるため）。目的・結果・対応は、`server-capacity`（uraniumの所要時間とメモリの測定、記録への参照なし）を除き、台帳・研究記録・spec・論文のどれかから追える（研究の問いの名前での照合。掃引ごとの照合はしていない）。
- まとめ（1回目、2026-11-20 22:57〜11-21 00:43、手元、12 workers）: 済んだ掃引の構成のフォルダ816個（79,158ファイル）を`runs.zip`へ。31.6GB→26.2GB、エラー0。`results/`のファイル数は104,734→26,393、容量は39GB（この日の初めは125,313ファイル・49GB）。ログは`results/_archive_pack_20261120.log`。
- まとめ（2回目）: 1回目で飛ばした29か所（`run_summary.csv`を作らない形式の結果、打ち切ったδの調整の掃引3つ、研究ツールの検査の出力）を、済んでいるかの判定を省く設定`--include-incomplete`でまとめる。ログは`results/_archive_pack_20261121_incomplete.log`。29か所・702MB→192MB、エラー0。`results/`のファイル数は12,463、容量は39GB（残りのファイルは主に、1クライアントの版の系列、`_sol_prompts`・`_sol_tmp`、README・`run_summary.csv`・掃引の記録）。
- 注意: まとめた構成のフォルダには`runs/`が無い。runのファイルを読む分析や、同じ出力先での掃引の流し直しの前に、`python -m tools.research_results_archive.archive unpack <フォルダ>`で戻す。
