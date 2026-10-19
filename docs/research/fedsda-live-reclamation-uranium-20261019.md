# uraniumの同期末メモリ回収：本規模の対照結果

## 判断

**省メモリ化は可能で、この環境では採用する価値が十分ある。** MNIST2・seed0・10 clients・5000標本/client・集約間隔50・1 workerの対照で、最大RSSを81.54%削減し、所要時間の増加は1.79%だった。完全設定・全指標・全133保存配列のdtype/shape/bytes・最終サーバパラメータhashは一致した。

この対照の完了時点では、Linux/glibcの実行側に限定した同期末回収を通常の掃引へ接続することを推奨した。アルゴリズム・標本表現・保存形式の変更や、allocator環境変数の一律変更は必要ない。

**後続作業（2026-10-19）:** [fedsda-runtime-memory-reclamation](../../.kiro/specs/fedsda-runtime-memory-reclamation/tasks.md)で通常FedSDA全体run・掃引への接続と検証を完了した。現在の運用・非対応環境でのno-op・適用対象は[運用手順](../experiments/fedsda-runtime-memory-reclamation.md)を参照する。以下は独立研究ランナーで測った対照結果で、導入後の多worker性能を測ったものではない。

## 条件と比較

入力：`results/fedsda-memory-diagnosis/live-reclamation-20261019_073006/`。
実行手順と回収位置は[試作の仕様・検証](../../.kiro/specs/run-memory-reclamation-prototype/tasks.md)、[実行手順](../experiments/run-memory-reclamation-prototype.md)を参照。

- Python 3.10.16、NumPy 2.2.6、PyTorch 2.12.1+cpu、glibc 2.35。各数値ライブラリ1 thread、記録対象のallocator環境変数は未設定。
- control→reclaimを独立processで逐次実行。両側とも同じ記録処理を持ち、追加の監視threadはない。
- reclaimだけ、正常なモデル同期の終了後に`malloc_trim(0)`を実行。モデル・標本・戻り値を保持したまま、明示的なGCやキャッシュ破棄を行わない。
- source hash：`3861f16d77fad2449baf39e37c433bb5343b56f4291efa315ce9c3aea8cf71e2`。
- runner hash：`2d27763ab52b2239920d691cccae6049773db64d359a837d8a1a6138550f1732`。配布した最終ZIPのrunnerと一致。

| GNU timeの実測 | 回収なし | 同期末回収あり | 差 |
| --- | ---: | ---: | ---: |
| 最大RSS | 9081.27 MiB（8.868 GiB） | 1676.76 MiB（1.637 GiB） | −7404.51 MiB、−81.54% |
| 経過時間 | 915.31秒 | 931.67秒 | +16.36秒、+1.79% |
| user CPU | 884.32秒 | 892.01秒 | +7.69秒 |
| system CPU | 30.60秒 | 39.30秒 | +8.70秒 |

両側ともexit0、100回の同期と正常終了を記録。run返却時までの最大RSSも表と同じで、保存後だけに発生したピークではない。回収ありの100回すべてが`trim_result=1`、呼出し時間合計は4.284秒。回収なしではtrimを呼んでいない。

この時間差は1対の観測値であり、実行順・ホスト負荷を反転した反復から求めた性能悪化率ではない。

## 何が効いたか

回収なしでは同期末RSSが徐々に増えた。一方、回収ありでは毎回数百MiBをOSへ返し、蓄積を抑えた。

| 同期回数 | 回収なしの同期末RSS（MiB） | 回収あり・回収前（MiB） | 回収あり・回収後（MiB） |
| --- | ---: | ---: | ---: |
| 1 | 1192.43 | 1222.68 | 904.43 |
| 10 | 3187.18 | 1413.94 | 853.84 |
| 25 | 4900.68 | 1339.03 | 874.59 |
| 50 | 7032.66 | 1481.35 | 966.18 |
| 75 | 8324.01 | 1568.66 | 1057.37 |
| 100 | 9081.01 | 1509.31 | 1147.70 |

回収後RSSは全100回で782.16〜1147.70 MiB。1回の最大RSS低下は696.54 MiBであり、7.23 GiBを一度に回収したわけではない。繰り返し回収して最大RSSの増加を防いだ結果が、表の81.54%削減である。各回の低下量を足してピーク削減量とすることもしない。

この観測は、必要なモデル・標本を保持していても、allocatorが保持する解放済み領域の返却で大幅に削減できることを示す。必要な生存データだけで約9 GiBを使うという説明は支持しない。ただし、全RSSの内訳や領域を作った個別演算、特定ライブラリの不具合までは同定していない。

[前回の入口比較](fedsda-memory-uranium-results-20261019.md)では、監視付き診断の大きなRSSに監視thread等が影響した可能性を残した。今回は追加監視threadがないcontrolでも8.87 GiBになったため、監視threadだけを大容量化の原因とする説明はできない。以前の元CLIの5.49 GiBとの差の原因までは、この対照で分離していない。

## 同一性と限界

返却元の`comparison.json`だけを信用せず、主担当が現行runnerの純粋読取り関数`_compare`で再照合し、返却集計と完全一致することを確認した。両側の実行前後の環境・コードidentityも一致し、計画と配布runner hashを照合した。元31ファイルは照合前後のSHA256が一致した。

解析script：`results/fedsda-memory-diagnosis/analyze_live_reclamation.py`。
再集計・入力hash：`results/fedsda-memory-diagnosis/live-reclamation-analysis-20261019/{analysis.json,input-sha256.json}`。元データは変更していない。

一致を確認したのは今回の同一環境・同一条件の2run。全データセット・seed・手法、異なるライブラリ環境、32/64 workerの総スループットについて同じ削減率や速度を保証する結果ではない。単独runの削減率をそのままworker数へ掛けて並列数を決めない。

**採用の価値を判断するための追加実験は不要。** 通常掃引へ接続するときには、Linux/glibc専用の実行資源管理として責務を分離し、未対応環境の扱いを明示する。数学的な手法設定・runの意味・既存成果物形式を増やさず、通常経路での結果同一性を実装の受け入れ条件にする。複数workerでの実効速度とホスト総メモリは、その接続後の運用確認として測る。

今回、学習・全pytestの再実行、製品source・既定・goldenの変更、mainへのマージは行っていない。

独立CLIのGPT-6 Luna highでPASS（Blocker0/Major0/Minor1）。session `01a15146-2ca4-7cb3-801a-eb2b58eac1cf`、ログ`../../venv/refactoring-tests/live-reclamation-results-review.log`。基点`2ca681b`、本報告とresumeの未commit追記が対象。GNU time、各100同期、133配列dtype/shape/bytes、4モデル・32テンソルの最終hash、元31ファイルhash、comparison/analysisの数値を独立照合した。Minorは単一条件・単一対の一般化制限の注意で、本報告の限界欄で対応済み。追加修正・再レビューは不要。学習・pytest・主担当解析script自体の再実行は独立未実施。
