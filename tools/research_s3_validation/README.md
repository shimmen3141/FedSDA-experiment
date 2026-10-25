# Bの未使用seed確認とseed3追跡

[研究記録](../../docs/research/fedsda-s3-validation-20261025.md)。製品と既存の数値実行部は変更しない。共用追跡解析器の片側走査の不備だけは修正した（研究記録参照）。

## uraniumでの固定比較

MNIST4・間隔500・2500標本/client・seeds 4–7。
既定D、初期値整合W、B（評価full・候補学習head）、S3の4構成・16runを全て同じ環境で計測する。
過去結果の転送は不要。各seedで入力画像/ラベル/概念の署名を照合する。
主指標はBの既定Dからの精度差pp。W/S3差、全seedの精度差、SD、通信上り/下りbytes、
モデル順伝播＋推定逆伝播MAC、秒、概念別/安定期/回復期精度を保存する。
今回の結果を見てBの規則やseedを選び直さない。探索済み0–3をCLIで拒否する。

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u -m tools.research_s3_validation.run \
  --output results/fedsda-s3-validation/20261025_uranium-holdout-seeds4-7-n2500 \
  --seeds 4 5 6 7 --sample-count 2500 --workers 64
python -m tools.research_s3_validation.analyze \
  results/fedsda-s3-validation/20261025_uranium-holdout-seeds4-7-n2500
```

tmux内で実行する。64は上限で、今回は16jobsなので最大16process。
実行前の内部確認は`--print-plan`。既存出力先は拒否し、再開は未対応。
途中結果は原子的に保存し、失敗があればfailures.jsonを残して解析を拒否する。
完了時に全矩形・同一入力・数値ソースhashを確認し、解析時も再検査する。
実行中に数値ソースを更新しない。解析器は別hashを持つ。

## 手元のseed3非介入追跡

未使用seedの性能実験と独立に実行可能。
既観測seed3のBを1回だけ再実行し、保存済みBと秒以外の全指標・イベントの一致を要求する。
4seed実験の代用ではなく、悪化例1件の診断。既存の寄与trackerを再利用し、追加のNN推論はしない。
保存確率から正誤を復元し、全client/sample被覆・製品digestを検査する。
既存D/W/S3の同seedのtraceとの入力対応・最初の判断/予測分岐・概念/期間/client別正誤差を求める。
追跡結果から未使用seed向けのゲートや閾値を調整しない。

```bash
python -m tools.research_s3_validation.trace \
  --reference results/fedsda-s3-factorial/20261025_mnist4-seeds0-3-n2500 \
  --output results/fedsda-s3-validation/20261025_local-seed3-trace-n2500
python -m tools.research_s3_validation.trace --analyze \
  --output results/fedsda-s3-validation/20261025_local-seed3-trace-n2500
```

追跡には過去のfactorial結果と、そこから参照する元結果・入力proof・seed3の生traceが必要。
未使用seedのCLIにはこれらは不要。新しい本実装・spec・採用ゲートは作らない。
