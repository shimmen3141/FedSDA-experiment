# Bの本規模確認（新規240run）

[研究計画と検証記録](../../docs/research/fedsda-s3-fullscale-20261025.md)。
既定DとBだけを比較する。W/S3は今回の主評価に不要なので実行しない。
Bの実行は検証済み`research_s3_validation.run.execute`を再利用し、製品は変更しない。

6データセット×4間隔×10seed×各client5000標本・10clients。
既定240件を再利用すれば、新規実行はBの240run。比較する値は計480run相当。

## uraniumで実行

`refactor/architecture`の作業ディレクトリでtmux内から実行する。
下記の既存既定結果ディレクトリがuranium側にも必要。

```bash
git pull --ff-only origin refactor/architecture
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u -m tools.research_s3_fullscale.run \
  --output results/fedsda-s3-fullscale/20261025_uranium-b-seeds0-9-n5000 \
  --baseline results/method-comparison-v12/20261023_220120_fedsda-v12-default-seeds0-9/fedsda/runs \
  --datasets sea2 sea4 sine2 circle2 mnist2 mnist4 \
  --intervals 50 100 200 500 --seeds 0 1 2 3 4 5 6 7 8 9 \
  --sample-count 5000 --workers 64

python -m tools.research_s3_fullscale.analyze \
  results/fedsda-s3-fullscale/20261025_uranium-b-seeds0-9-n5000
```

表示例:

```text
START new_runs=0/240 reused_default=240 remaining_new=240
NEW 37/240 (compared=277/480 reused_default=240) mnist4 i500 s2 full-score-head-train acc=...
```

`NEW`は完了した新規run数。再開時も過去の完了分を含む。実行中は`runs/*.json`へ
完了runを原子的に保存する。途中で止まった場合は**同じコマンドに`--resume`を追加**する。
完了済みは再実行せず、失敗・未完了runだけを実行する。workersだけは変更可能で試行履歴へ記録する。
数値ソース・比較条件・Python/NumPy/Torch等の環境が変わった場合は再開を拒否する。
同じ出力先を複数の親processから同時実行しない。実行中のpull・数値ソース更新もしない。

事前の内部確認は`--print-plan`。出力ディレクトリを作らず、480比較値・240再利用・240新規を確認できる。
手元からのdry planでは環境照合を実行時へ延期することを表示し、実行開始時には必ず照合する。

## 既定結果の再利用条件

元記録はclean commit `8e34d4fdb4169bfe9ac4f6bdd61d6b4c62f449b4`、基準版12。
そのGit treeの全製品Pythonファイル（集合とblob内容）を現在と照合し、
全run設定・50000予測の件数と精度・Python/NumPy/Torch/OS/archの一致も必須化する。
既存記録の環境はLinux/x86_64、Python3.10.16、NumPy2.2.6、Torch2.12.1+cpu。
コピーした元記録とSHAを結果内に保存し、再開・解析時に再検査する。
比較元のGit commitもローカルGit履歴から読める必要がある。

不一致時には流用せず停止する。既定も同時に再計測する場合は、未使用出力先で
`--baseline`の指定を外す。その場合は既定240＋B240の480runとなる。
W/S3を含めた960runへ自動で増やすことはない。

旧既定には入力画像hash・秒数・準備MAC・概念別精度が無く、厳密な画像列の直接照合はできない。
コード・全設定・実行環境に基づく再利用であり、画像hash照合済みとは表現しない。
欠ける指標はN/A（JSONではnull）。旧時間の補完やBの値の代入はしない。
`--baseline`なしで両方を再計測する場合は入力署名も構成間で照合する。

## 指標と出力

各データ×間隔のseed等重みで、精度%、既定差ppとSD/正負seed数/最悪値、
上り/下りbytes、順伝播＋推定逆伝播MAC、秒を出す。Bの準備MAC・概念別精度と
両構成の安定期/回復期精度もJSONへ残す。異なるデータや間隔を先に平均しない。
時間は同時負荷・試行条件が異なるため、方式固有の速度比較には使わない。

- `runs/*.json`: 新規runと再利用した既定のcheckpoint。
- `baseline/*.json`: 再利用元のネイティブ既定記録。
- `attempts.jsonl`: 開始時のworkers・再開数、失敗履歴。
- `complete.json`/`results.json`: 全条件を照合した後だけ確定。
- `summary.json`/`standard_metrics_by_seed.csv`/`analysis_standard_metrics.md`: 解析結果。

失敗・欠落・重複・source変化・原記録との不一致は解析を拒否する。
これはBの一般性と費用を見る研究実験であり、製品の既定化やcc-sddの本実装は行わない。
