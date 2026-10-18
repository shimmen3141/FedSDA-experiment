# uraniumの元CLI・診断scriptの測定結果（2026-10-19）

**最新の進行:** 現状の証拠で試作の価値は判断できるため、ユーザーの依頼に従い[実行側に閉じた回収試作](../experiments/run-memory-reclamation-prototype.md)を実装・手元検証した。以下の「次の対照」は実装前の追加調査ではなく、この試作のサーバ検証として行う。製品の既定採用は結果返却後に判断する。

## 結論

uraniumでは**元CLIでも数GiBの使用量を再観測した**。一方、製品が持つ観測/モデル/記録が手元より数GiB増えたという証拠は得られなかった。**格納形式や学習フローを変える前に、実行中の解放済み領域の回収を研究用の対照で調べる価値が高い。**

元CLIの最大RSSは5.49 GiBで、従来の6.76 GiBと数値自体は一致しない。診断はさらに8.89 GiBに増え、計測自体の影響も切り分ける必要がある。今回だけでglibc、Python、スレッドのどれか一つを原因と断定しない。本体改修・allocator設定の既定採用はまだ行わない。

## 入力と再照合

ユーザーが実行して戻したディレクトリ:

`results/fedsda-memory-diagnosis/uranium-comparison-20261019/uranium_20261019_060505_190543/`

Python 3.10.16、NumPy 2.2.6、torch 2.12.1+cpu、glibc 2.35。commit `8ace43cefd5cd0acf77d1babccf3df9903ab9c37`、clean。src hashは手元と同じ`3861f16d77fad2449baf39e37c433bb5343b56f4291efa315ce9c3aea8cf71e2`。

FedSDA、MNIST2、seed0、10client、5000/client、間隔50、workers1。OpenMP/MKL/OpenBLASは各1。記録したallocator環境変数は全て未設定。測定前後のmetadataは一致。

主担当はユーザー側の`comparison.json`を読むだけでなく、元JSON/NPZから次を再照合した。

- CLIも診断もGNU timeのexit0。元CLIはcompleted1、skipped0、failed0。
- 完全設定・全指標が一致。
- 全133配列のキー・shape・dtype・bytesが一致（NaNや符号付き0もbytesで比較）。NPZの各配列を読み出して照合。
- 診断の`finished`を確認し、run返却までと全工程のhigh-waterを再計算。
- 入力ファイルをSHA-256で記録し、元記録は変更しない。

解析scriptと出力は`results/fedsda-memory-diagnosis/uranium-analysis-20261019/{analyze.py,analysis.json,input-sha256.json}`（git管理外）。CLIが保存しない最終パラメータ、および全乱数源の状態は、CLI対診断の照合対象外。

## 最大RSSと時間

| 項目 | 元CLI | 診断script |
| --- | ---: | ---: |
| GNU time最大RSS KiB | 5754512 | 9319656 |
| GNU time最大RSS GiB | 5.488 | 8.888 |
| wall秒 | 892.71 | 939.36 |
| user CPU秒 | 876.84 | 888.56 |
| system CPU秒 | 15.54 | 258.92 |
| CPU使用率 | 99% | 122% |
| swap回数 | 0 | 0 |

診断のrun返却までのhigh-waterは9054.14 MiB（8.842 GiB）、全工程は9101.23 MiB（8.888 GiB）。保存後の測定期間を含めるかどうかだけで、元CLIとの差は説明できない。1回ずつの逐次観測であり、一般的な性能悪化率の推定には使わない。

診断は0.2秒ごとに別スレッドで`/proc/self/status`と`smaps_rollup`を読む。元CLIとの差は計測に伴う割当・スレッド・PSS読取りの影響を含み得る。system CPUの増加も無視できず、この監視方式を軽微な負荷と一般化しない。どの部分がRSS増加を起こしたかは未同定。

## どの段階で増えたか

以下は**診断script側**のRSSであり、元CLIの段階別RSSではない。

| 時点 | RSS MiB |
| --- | ---: |
| import後 | 221.02 |
| 観測列生成前 | 556.37 |
| 観測列生成後 | 847.89 |
| round0同期後 | 1134.65 |
| round20同期後 | 4147.96 |
| round40同期後 | 5629.45 |
| round60同期後 | 6416.63 |
| round80同期後 | 8007.34 |
| run返却時 | 9054.14 |
| 所有量走査後 | 9096.30 |
| 保存後 | 9096.34 |
| 参照解放・GC後 | 8932.75 |
| その後の`malloc_trim(0)` | 409.32 |

増加は学習/同期の進行中に起き、最後の保存ではほとんど増えていない。run返却時のPSSも9049.77 MiBで、RSSの大きさは共有ライブラリの重複計上だけでは説明できない。

GCだけではほぼ減らず、その後のtrimで8523.43 MiB（約8.32 GiB）減った。**その時点では**OSへ返せる解放済み領域が大きかったことを示す。一方、参照を解放した後の測定なので、「生存中のモデルを保ったまま8.32 GiB削減できる」とはまだ言えない。循環参照リークや全ラウンドのモデル保持を、RSSの増加だけから断定しない。

## 手元との保存状態の比較

| 計数 | uranium | 手元Ubuntu |
| --- | ---: | ---: |
| 結果から辿れるPython浅いサイズ MiB | 354.60 | 351.40 |
| Tensor storage MiB | 275.99 | 275.99 |
| Tensor storage数 | 100552 | 100552 |

所有量走査はgrad/autograd内部やruntime/global cache全体を網羅せず、RSSの完全な内訳ではない。ただし、数GiB増える主因をこの観測表現・Tensor payloadに求める根拠は弱まった。保存用全133配列も4.38 MiBだった。

手元の5000/client対照との完全設定・全指標も一致し、**診断が保存した最終サーバパラメータのhashも全て一致した**。133配列中129列はbytesで一致。残る4列の差は以下で、環境を跨いだ完全一致とは扱わない。

| 列 | 差のある要素数 | 最大絶対差 |
| --- | ---: | ---: |
| sample_prediction.maximum_prediction_weight | 23188 | 4.44e-16 |
| sample_prediction.effective_model_count | 24232 | 2.66e-15 |
| sample_prediction.fixed_share_effective_model_count | 24232 | 2.66e-15 |
| sample_loss_monitoring.monitored_log_e_value | 8733 | 1.78e-15 |

大きく異なる採用/統合経路やモデルpayloadの増大によってメモリ差が生じた、という説明は今回の保存証拠には合わない。ただし記録していない内部状態の同一性まで保証しない。goldenや許容誤差は更新しない。

## 次に確認する対照

次は研究用script内で、**追加の監視スレッドを作らない対照**と、**その同じ実行で同期末に定期的にtrimする対照**を、同じuranium環境で比較するのがよい。

1. 0.2秒ごとの監視を外し、区切りでだけRSS/high-waterを読む。診断自体の上乗せを減らす。
2. モデル・学習標本・診断記録の参照を全て保持した同期末にtrimを行い、その直前/直後を測る。run終了後の回収と区別する。GC追加やallocator環境変数の変更は混ぜない。
3. 対照間で完全設定・全指標・全133配列bytes・最終パラメータhashを再照合し、GNU timeと実行時間を比較する。

これで生存中の状態を維持した回収が大きく効き、時間コストも許容できると確認できた場合に、実験ランナー境界の小さな対策としてspec化する。標本形式の横断改修や、学習処理への省略・分岐追加を先に行わない。今回、新たなサーバ実験や本体改修は開始していない。

## 検証と独立レビュー

主担当は元21ファイルをhash記録し、元JSON/NPZから133配列・設定/指標、metadata前後、high-water、GNU time、手元との4列差と最終パラメータhashを再照合した。新規学習・全pytest・golden更新は行っていない。

独立CLI GPT-6 Luna high、session `01a15106-bb50-7ee1-ac9c-6c1f89a37987`、**PASS**、Major/Minor/任意の指摘なし。実モデルとeffortをログで確認。独立に元記録hash、CLI/診断の全133配列bytes・設定/指標、手元との129列一致/4列差と最終パラメータ、RSS換算と解釈を照合した。学習とpytestは独立未実行。ログは作業rootの`venv/refactoring-tests/uranium-memory-results-review.log`。
