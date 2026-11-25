# モデルの混合の重みの規則の変種（研究ツール）

FedHASP（基準版16の既定 `fedsda`）の、モデルの混合の重み（Fixed Share）の更新の規則だけを変えた変種を流す。予測以外（学習・割り当て・通信・計算量）は既定と同じで、`analyze.py` が対の run で確かめる。変種の中身は `run.py` の冒頭。

- `product-default`: 製品のまま（包むだけで、何も差し替えない）。
- `log-loss-weights`: 重みの更新を log loss・学習率1（ベイズの更新）にする。
- `alarm-share-boost`: 警報の後の1回の更新だけ、共有率を 1/T から 0.25 に上げる。
- `log-loss-weights-alarm-share-boost`: 両方。
- `bounded-loss-unit-learning-rate`（対照、2026-11-25追加）: 損失は製品のまま（1 − p_y）、学習率だけを1で一定にする。
- `tempered-log-loss-weights-0.5`・`tempered-log-loss-weights-0.25`（2026-11-25追加）: log loss の学習率を 0.5・0.25 にする。
- `learning-rate-mixture-log-loss-weights`（2026-11-25追加）: 学習率 1・1/2・1/4 の log loss の重みを、上の層の log loss の Fixed Share で混ぜる。
- `log-loss-weights-log-loss-version-mixture`（2026-11-25追加）: `log-loss-weights` に加え、版の混合（同期の版と手元の版）の重みも log loss・学習率1で更新する。

`--record-prediction-probabilities-for <変種>` を付けた変種の run では、重みの制御への呼出しの列と、予測ごとのモデル別の確率を記録し、失敗の無い掃引の後に `<変種>/prediction_probability_trace.zip` へまとめる（`probability_trace.py`）。`replay.py` が、この記録から別の規則の混合FS の精度を計算し直す。

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

簡素版の最小形（`--base minimal-soft-prediction`。[積み上げの記録](../../docs/research/core-component-ladder-v16-20261119.md)の節3）を基点に、既定の重みと log loss の重みを比べる（2026-11-25追加。担当教授の指摘を受けて単純な構成で実験し直す方針に合わせる）。`product-default` の200 runの run の識別子は、既存の `results/fedhasp-simplified-v16/20261119_minimal-seeds0-9/minimal-soft-prediction` と全件同じ（同じ run 設定）:

```bash
git pull --ff-only origin research/single-client
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT=results/fedhasp-prediction-weight-rules/$(date +%Y%m%d_%H%M%S)_minimal-soft-prediction-intervals50-500-seeds0-9
mkdir -p "$OUT"
python -u -m tools.research_prediction_weight_rules.run --output "$OUT" --base minimal-soft-prediction \
  --variants product-default log-loss-weights \
  --datasets sea2 sea4 sine2 circle2 mnist2 mnist4 fmnist2 fmnist4 rotmnist2 rotmnist4 \
  --intervals 50 500 --seeds 0 1 2 3 4 5 6 7 8 9 --workers 120 2>&1 | tee "$OUT/run.log"
```

最小形の上の学習率の変種と版の混合の log loss（2026-11-25追加）。上の最小形の出力先へ足す（`product-default` と対にする）。`tempered-log-loss-weights-0.5` の run では、確率を記録する（`replay.py` が使う。記録は全体で約750MBの見込み、1つの zip）:

```bash
git pull --ff-only origin research/single-client
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT=results/fedhasp-prediction-weight-rules/20261125_172606_minimal-soft-prediction-intervals50-500-seeds0-9
python -u -m tools.research_prediction_weight_rules.run --output "$OUT" --base minimal-soft-prediction \
  --variants tempered-log-loss-weights-0.5 tempered-log-loss-weights-0.25 learning-rate-mixture-log-loss-weights log-loss-weights-log-loss-version-mixture \
  --record-prediction-probabilities-for tempered-log-loss-weights-0.5 \
  --datasets sea2 sea4 sine2 circle2 mnist2 mnist4 fmnist2 fmnist4 rotmnist2 rotmnist4 \
  --intervals 50 500 --seeds 0 1 2 3 4 5 6 7 8 9 --workers 120 2>&1 | tee "$OUT/run-learning-rate-variants.log"
```

データの環境変数（MNIST・Fashion-MNIST）は、これまでの uranium の実行と同じものを使う。中断したら、同じコマンドを流し直せば残りを流す（`$(date …)` で出力先を作るコマンドは、`OUT=` の行を前のパスに書き換え、`mkdir` の行は飛ばす）（済んだrunは製品の掃引のコマンドが飛ばす）。

## 分析（手元、再実験なし）

```bash
python -m tools.research_prediction_weight_rules.analyze <OUT>
python -m tools.research_prediction_weight_rules.selection_versus_mixture <OUT>/<変種>/fedsda/runs.zip --output <OUT>/analysis_selection_versus_mixture_<変種>.md
```

確率を記録した変種があれば、別の重みの規則を再実験なしに計算し直す（`<OUT>/analysis_replay_<記録した変種>.md`。混合FS の精度と log loss。版の混合は計算し直さない。冒頭に、記録した重みとの差と、同じ出力先の実際の run との正誤の不一致の数を書く）:

```bash
python -m tools.research_prediction_weight_rules.replay <OUT> --recorded-variant tempered-log-loss-weights-0.5 --workers 8
```

`<OUT>/analysis_prediction_weight_rules.md` に、学習が同じことの確認、精度の差（全体・変化の後100標本・それ以外・概念別・概念の変化からの経過の期間別）、run の対ごとの正負の数、通信量・MAC（既定と同じことの確認つき）、変種ごとの掃引の開始〜終了の時刻、共有率を上げた回数と警報の数を書く。run ごとの経過時間は記録に残らない（掃引の表示の秒は掃引の開始からの累積）。
