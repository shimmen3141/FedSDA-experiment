# 出力層限定学習者の独立対照

S3の研究ツール。製品の実装・設定には登録しない。詳しい条件・結果は
[研究記録](../../docs/research/fedsda-output-layer-learner-s3-20261025.md)を参照する。

| `--variants` | warm符号長 | warm選択時の実候補 |
|---|---|---|
| `product-default` | 現在モデルのadapter＋出力層を更新 | 既定の初期値と学習 |
| `warm-matched` | 既定と同じ | 現在モデルからadapter＋出力層を更新 |
| `output-layer-matched` | 現在モデルの出力層だけを更新 | 現在モデルから出力層だけを更新 |

S3の出力層は再初期化しない。初期化し直す学習者は既定のまま、2学習者の`ln(2)`を維持する。
候補の共有部は既定どおり固定。採用後の学習は通常へ戻す。
製品型の検査を保つため内部の学習者名は`warm_started_learner`を再利用する。
結果は`research_variant`で識別し、共通runnerの`variant=product-default`だけで区別しない。

```bash
python -m tools.research_output_layer_learner.run \
  --output results/fedsda-output-layer-learner/<unique-label> \
  --datasets sea4 sine2 --intervals 100 --seeds 0 1 2 3 \
  --sample-count 2500 --workers 12
```

MNIST4は別実行で`--datasets mnist4 --intervals 500`、SEA4の無変化対照は
`--datasets sea4 --fixed-concept 2`。手元の並列実行のworkers合計は14以下。
uraniumでは通常`--workers 64`。各workerの数値ライブラリは1threadとする。
実行前の内部確認には`--print-plan`を足す。この確認は出力先を作らない。
出力先が既にある場合や同じコマンド内に重複条件がある場合は拒否する。

精度・概念別精度・期間別精度・上り/下りbytes・モデルMAC・準備MAC・秒数などを
共通runnerから保存する。研究イベントではwarm評価回数、選択仮説、実候補の
初期値変更、S3の共有部/adapter不変性を記録する。S3候補が作られなかった場合、
精度差を出力層だけで学習した候補の効果と解釈しない。

```bash
python -m tools.research_output_layer_learner.analyze \
  --inputs results/fedsda-output-layer-learner/<unique-label> \
  --output results/fedsda-output-layer-learner/<analysis-label>
```

既定結果を再利用するときは実験側に`--variants warm-matched output-layer-matched`を指定できる。
解析側の`--baselines`へB診断形式のディレクトリを渡す。各`baseline`だけを使い、
製品全ソース・共通runner・固定入力のhashと条件・seedの対応を照合する。
対応する既定runが欠ける、重複する、各群が4seed未満なら解析を完了しない。
入力ごとのhash検証件数と原結果のhash、解析器自身のhashを`summary.json`に保存する。
解析器は学習で実行しないため、数値計算ソースとは別に来歴を持つ。

`complete.json`のない実験は解析しない。今回のCLIは中断後の再開に対応していない。
再実行前に保存済みの条件と完了数を確認し、既存結果を上書きしない。
