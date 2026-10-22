# 基準版11の既定での構成要素比較（ablation）

2026-10-23。7構成1680runの実測結果。**採否は未決で、既定・学習コード・goldenは変更していない。**
許容精度低下幅は未設定で、非劣性や要素の不要性を自動判定しない。前回（基準版10の既定、seed 0〜4）の比較は[こちら](fedsda-component-ablation-v10-20261021.md)。

## 結果と比較の成立条件

- 変更側: `results/fedsda-component-ablation-v11/20261022_235734_core-components-v11-seeds0-9/`（研究室サーバ、`run_experiment_sweep.py`、7構成×6データ×間隔50・100・200・500×seed 0〜9、10 client、各5000標本、1680 run。2026-10-22 23:57〜2026-10-23 04:47、64 workers）。全runのcommitは`9222b0e`（`src/`は基準版11の`fd6a89a`と同じ）、基準版11。
- 基準側: `results/method-comparison-v11/20261022_221105_fedsda-v11-default-seeds0-9/fedsda/`（240 run、commit `fd6a89a`、研究室サーバ）。
- 同じデータ・間隔・seedの対の差。集計は`run_summary.csv`から（Git管理外の一時script）。
- 各値は、変更−基準の全体精度（pp）の10 seedの平均と、基準以上のseedの数。下りの比・clientの順伝播の計算の比は、変更÷基準の10 seedの平均。
- 前回（版10）はseed 0〜4。今回はseed 0〜9（損失の距離のクラスタリングで、5 seedと10 seedの結論が違ったため）。

## 要約

| 構成 | 精度の差の範囲（pp） | 基準以上の条件（/24） | （参考）版10・5 seed |
|---|---|---|---|
| `fedsda-without-candidate-validation`（候補の学習で共有部も更新する） | −0.17〜+0.23 | 16 | −1.83〜+0.63、11 |
| `fedsda-without-prediction-weight-recalibration` | −0.48〜+0.05 | 5 | −0.42〜+0.04、5 |
| `fedsda-overall-loss-esr-detection` | −1.12〜+0.19 | 7 | −1.95〜+0.21、7 |
| `fedsda-without-concept-specific-adapter` | −1.45〜+0.08 | 3 | −1.39〜+0.10、4 |
| `fedsda-loss-distance-clustering` | −0.34〜+0.11 | 13 | −0.62〜+0.46、13 |
| `fedsda-local-parameter-prediction` | −1.91〜+0.53 | 1 | −2.07〜+0.24、1 |
| `fedsda-single-model-prediction` | −3.89〜−0.19 | 0 | −4.90〜−0.46、0 |

## 構成ごとの結果（精度の差pp（基準以上のseed数）、間隔50／100／200／500）

### `fedsda-without-candidate-validation`（候補の学習で共有部も更新する）

| データ | 精度の差 | 下りの比 |
|---|---|---|
| SEA2 | +0.02(4)／+0.00(6)／−0.02(6)／+0.01(7) | 0.97〜1.06 |
| SEA4 | −0.17(3)／−0.12(4)／+0.10(5)／+0.23(5) | 0.99〜1.02 |
| SINE2 | −0.07(1)／+0.08(4)／−0.00(5)／+0.03(4) | 1.00〜1.06 |
| CIRCLE2 | +0.18(8)／+0.12(6)／+0.21(7)／+0.18(6) | 0.95〜1.06 |
| MNIST2 | +0.01(6)／+0.01(6)／−0.03(5)／+0.02(5) | 0.99〜1.00 |
| MNIST4 | +0.04(4)／+0.00(7)／−0.09(2)／−0.01(4) | 1.00〜1.01 |

- 版10（5 seed）では−1.83ppまで下がる条件があったが、版11（10 seed）では−0.17〜+0.23ppで、24条件のうち16で基準以上。CIRCLE2は4間隔とも+0.12〜+0.21pp。版11で、候補の共有部を固定する既定の効果がほとんど見えなくなった（版10との差が、版11の変更によるのか、seedの数によるのかは未確認）。

### `fedsda-without-prediction-weight-recalibration`

SEA2・SEA4は±0.05pp。SINE2（−0.09〜−0.22、0/10）、CIRCLE2（−0.14〜−0.21）、MNIST2（−0.23〜−0.38）、MNIST4（−0.21〜−0.48）は、全間隔でほぼ全seedが下がる。通信・計算は同じ。再較正は版10と同じく必要。

### `fedsda-overall-loss-esr-detection`

CIRCLE2（−0.41〜−1.12）、MNIST2（−0.21〜−0.58）、MNIST4（−0.41〜−0.78）で下がる。SEA・SINE2は±0.2pp以内。clientの順伝播の計算は0.74〜0.90倍（CIRCLE2・MNIST）。正解クラス別の監視は版10と同じく必要。

### `fedsda-without-concept-specific-adapter`

SINE2（−0.63〜−1.45、0/10）、CIRCLE2（−0.76〜−1.31）、MNIST2（−0.48〜−0.67）、MNIST4（−0.70〜−1.16）で下がる。SEAは−0.24〜+0.08。下りは0.28〜0.94倍、計算は0.53〜0.98倍。adapterは版10と同じく必要。

### `fedsda-loss-distance-clustering`

| データ | 精度の差 | 下りの比 |
|---|---|---|
| SEA2 | −0.02(8)／+0.00(9)／−0.01(9)／+0.00(10) | 0.96〜1.02 |
| SEA4 | +0.00(7)／+0.04(7)／+0.11(9)／−0.00(9) | 1.00〜1.03 |
| SINE2 | −0.00(5)／+0.03(9)／−0.01(9)／+0.02(8) | 0.97〜1.03 |
| CIRCLE2 | +0.09(8)／−0.06(4)／+0.04(5)／−0.07(6) | **0.76〜0.89** |
| MNIST2 | +0.01(6)／+0.03(8)／+0.02(9)／−0.10(6) | 0.99〜1.00 |
| MNIST4 | +0.02(8)／−0.03(8)／−0.12(6)／**−0.34(5)** | 1.00〜1.01 |

- 版10の10 seed（[IMPROVE-025](improvement-candidates/improve-025-simplify-clustering-decision-to-loss-distance.md)）より差が小さい（−0.34〜+0.11、13/24）。MNIST4・500は−0.34pp。CIRCLE2の下りは0.76〜0.89倍。評価のclientを絞る変更（LEGACY-022）でも、CIRCLE2の下りは0.67〜0.81倍に減る（下の参照）ので、通信の利点は重なる。

### `fedsda-local-parameter-prediction`

CIRCLE2・500（+0.53、10/10）を除き、全条件で下がる（MNIST2・MNIST4で−1.0〜−1.9pp）。

### `fedsda-single-model-prediction`

全条件で下がる（−0.19〜−3.89pp。SINE2で最大−3.89pp）。予測の結合は版10と同じく最も効く要素。

## まとめ（暫定）

- 予測の結合、局所パラメータでない同期の予測、正解クラス別の監視、adapter、予測の重みの再較正は、版11でも、版10と同じく明確に必要。
- 候補の共有部の固定は、版11（10 seed）では効果がほとんど見えない（外しても−0.17〜+0.23pp）。採否は未決。
- 損失の距離のクラスタリングは、精度がほぼ同じでCIRCLE2の通信が減るが、MNIST4・500で下がる。評価のclientを絞る変更（LEGACY-022）の後に、改めて比べる必要がある。

参考: 評価のclientを絞る変更（`product-default-sample-holding-cross-evaluation`）の本規模の結果は[IMPROVE-028](improvement-candidates/improve-028-evidence-accumulating-model-consolidation.md)の9。
