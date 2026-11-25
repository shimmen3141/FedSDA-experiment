# モデルの混合の重みの規則の変種（研究ツール）

FedHASP（基準版16の既定 `fedsda`）の、モデルの混合の重み（Fixed Share）の更新の規則だけを変えた変種を流す。予測以外（学習・割り当て・通信・計算量）は既定と同じで、`analyze.py` が対の run で確かめる。変種の中身は `run.py` の冒頭。

- `product-default`: 製品のまま（包むだけで、何も差し替えない）。
- `log-loss-weights`: 重みの更新を log loss・学習率1（ベイズの更新）にする。
- `alarm-share-boost`: 警報の後の1回の更新だけ、共有率を 1/T から 0.25 に上げる。
- `log-loss-weights-alarm-share-boost`: 両方。
- `bounded-loss-unit-learning-rate`（対照、2026-11-25追加）: 損失は製品のまま（1 − p_y）、学習率だけを1で一定にする。

runごとのトレース（警報の数・共有率を上げた回数）は、run の途中は `<変種>/weight_rule_trace/<run>.json` に書き、失敗の無い掃引の後に `<変種>/weight_rule_trace.json` の1つへまとめる（`traces.py`。まとめる前の結果には `python -m tools.research_prediction_weight_rules.traces <OUT>` で適用できる）。

## uranium での実行（ブランチ `research/single-client` の作業ディレクトリで、tmux の中から）

最初の4変種（2026-11-25に実行済み、`results/fedhasp-prediction-weight-rules/20261125_094022_all-datasets-intervals50-500-seeds0-9`）:

```bash
git pull --ff-only origin research/single-client
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT=results/fedhasp-prediction-weight-rules/$(date +%Y%m%d_%H%M%S)_all-datasets-intervals50-500-seeds0-9
mkdir -p "$OUT"
python -u -m tools.research_prediction_weight_rules.run --output "$OUT" \
  --variants product-default log-loss-weights alarm-share-boost log-loss-weights-alarm-share-boost \
  --datasets sea2 sea4 sine2 circle2 mnist2 mnist4 fmnist2 fmnist4 rotmnist2 rotmnist4 \
  --intervals 50 500 --seeds 0 1 2 3 4 5 6 7 8 9 --workers 120 2>&1 | tee "$OUT/run.log"
```

対照の変種を同じ出力先へ足す（`product-default` と対にするので、出力先は上と同じにする）:

```bash
git pull --ff-only origin research/single-client
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT=results/fedhasp-prediction-weight-rules/20261125_094022_all-datasets-intervals50-500-seeds0-9
python -u -m tools.research_prediction_weight_rules.run --output "$OUT" \
  --variants bounded-loss-unit-learning-rate \
  --datasets sea2 sea4 sine2 circle2 mnist2 mnist4 fmnist2 fmnist4 rotmnist2 rotmnist4 \
  --intervals 50 500 --seeds 0 1 2 3 4 5 6 7 8 9 --workers 120 2>&1 | tee "$OUT/run-bounded-loss-unit-learning-rate.log"
```

データの環境変数（MNIST・Fashion-MNIST）は、これまでの uranium の実行と同じものを使う。中断したら、同じコマンドを流し直せば残りを流す（済んだrunは製品の掃引のコマンドが飛ばす）。

## 分析（手元、再実験なし）

```bash
python -m tools.research_prediction_weight_rules.analyze <OUT>
python -m tools.research_prediction_weight_rules.selection_versus_mixture <OUT>/<変種>/fedsda/runs.zip --output <OUT>/analysis_selection_versus_mixture_<変種>.md
```

`<OUT>/analysis_prediction_weight_rules.md` に、学習が同じことの確認、精度の差（全体・変化の後100標本・それ以外・概念別・概念の変化からの経過の期間別）、run の対ごとの正負の数、通信量・MAC（既定と同じことの確認つき）、変種ごとの掃引の開始〜終了の時刻、共有率を上げた回数と警報の数を書く。run ごとの経過時間は記録に残らない（掃引の表示の秒は掃引の開始からの累積）。
