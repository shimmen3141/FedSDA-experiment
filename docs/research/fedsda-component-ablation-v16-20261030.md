# 基準版16の構成要素の比較（6データと新しい4データ、2026-10-30〜31）

- 結果: `results/fedsda-component-ablation-v16/20261029_231413_core-components-v16-seeds0-9`（9構成×240 run、commit `de65e59c`）と`20261030_052220_without-warm-started-learner-seeds0-9`（warm startの学習者を外す構成の240 run、commit `095dc66a`）。研究室サーバ、64 workers、全2,400 run完了・失敗0。
- 条件: SINE2、CIRCLE2、SEA2、SEA4、MNIST2、MNIST4×集約間隔50・100・200・500×seed 0〜9、10 client×5000標本。比べる相手は同じ実行の`fedsda`（基準版16）。
- 集計の一時script: Claudeのscratchpadの`ablation_v16.py`（Git管理外）。

## 一致の確認

- 版16の`fedsda`は、基準版15のときに流した構成`fedsda-every-sample-change-interval-search`の結果（`results/fedsda-every-sample-change-interval-search/20261029_151540_tuned-datasets-seeds0-9`）と、240 runすべてで精度・上りと下りの通信が一致した。
- 構成`fedsda-minimum-interval-multiples-change-interval-search`（5件刻みの前向きの探し方）は、基準版15の`fedsda`（同じ実行の240 run）と240 runすべてで一致した。
- したがって、新しい4データでは、この2構成を流し直さず、基準版15のときの結果（`fedsda-every-sample-change-interval-search/20261029_172159_new-datasets-seeds0-9`と`new-datasets-v15/20261029_024740_fedsda-seeds0-9`）を使える。

## 結果

外した構成と`fedsda`の精度の差（pp、10 seedの平均）。左の数値はデータ×間隔の24セルの範囲と平均、括弧は10 seedすべてで下がったセルの数。通信とMACは`fedsda`との比の範囲。

| 外したもの（構成） | 精度の差 | 全seedで低下 | 通信の比 | MACの比 |
|---|---|---|---|---|
| 予測を現在のモデルだけで行う（`fedsda-single-model-prediction`） | −2.53〜+0.60、平均−1.15 | 22 | 1.00 | 0.98〜0.99 |
| 局所のパラメータだけで予測（`fedsda-local-parameter-prediction`） | −1.96〜+0.62、平均−0.89 | 23 | 1.00 | 0.99 |
| 概念固有のadapter（`fedsda-without-concept-specific-adapter`） | −1.53〜+0.11、平均−0.60 | 15 | 0.41〜0.96 | 0.61〜0.98 |
| 集約後の再較正（`fedsda-without-prediction-weight-recalibration`） | −0.47〜+0.10、平均−0.18 | 16 | 1.00 | 1.00 |
| クラス別の監視（全体の損失だけ、`fedsda-overall-loss-esr-detection`） | −0.32〜+0.27、平均−0.02 | 0 | 0.97〜1.05 | 0.89〜1.06 |
| 1件刻み・逆向きの探し方（5件刻みに戻す） | −0.23〜+0.22、平均−0.01 | 0 | 0.96〜1.05 | 0.95〜1.04 |
| クラス別の機能的なクラスタリング（損失の距離に替える） | −0.40〜+0.18、平均−0.01 | 0 | 0.88〜1.01 | 0.90〜1.01 |
| 候補の共有部の固定（`fedsda-without-candidate-validation`） | −0.24〜+0.33、平均−0.03 | 0 | 0.96〜1.03 | 0.93〜1.04 |
| warm startの学習者（`fedsda-without-warm-started-learner`） | −0.22〜+0.05、平均−0.03 | 0 | 0.98〜1.02 | 0.93〜1.04 |

データ別の傾向:

- 予測の側の4つ（現在のモデルだけ、局所のパラメータだけ、adapter、再較正）は、ほとんどのデータと間隔で10 seedすべてが下がる。MNISTでの低下が大きい（現在のモデルだけ: −1.46〜−2.53pp）。例外はCIRCLE2・500で、予測を現在のモデルだけにすると+0.60pp（10/10で上）、局所のパラメータだけにすると+0.62pp（10/10）。adapterを外すとSEA2はほぼ変わらない（−0.00〜+0.11）。
- 固い側の細部（監視、探し方、クラスタリング、候補の共有部の固定、warm start）は、どのセルでも10 seedすべてが下がることはなく、差は±0.4pp以内。目立つのは、損失の距離のクラスタリングでSEA4が4間隔とも下がる（−0.14〜−0.40pp）、全体の損失だけの監視でMNIST4が4間隔とも下がる（−0.10〜−0.32pp）、warm startを外すとMNIST4・500が−0.22pp（10 seed中9で下がる）。
- warm startの学習者を外したとき、MNISTで概念が取り込まれたrun（概念0以外の精度が88%未満。NEW-006の指標）は、`fedsda`の0に対して1（MNIST4・間隔500・seed 1、概念1の精度84.7%）。

## 読み

- 版13の構成要素の比較（[記録](fedsda-component-ablation-v13-20261027.md)）と同じく、精度は予測の側（全モデルの重み付け、同期と局所の混合、adapter、再較正）から来る。論文の主軸（固い割り当てと柔らかい予測の分離）の「精度は柔らかい予測から来る」という主張を支える。
- 固い側の細部は、外しても6データの平均の精度はほとんど変わらない。これらは精度ではなく、閾値の要らなさ、変化点の正確さ（[IMPROVE-032](improvement-candidates/improve-032-reverse-order-prequential-code-for-all-change-intervals.md)）、特定の状況での崩れの防止（MNIST4の取り込み、SEA4の統合）として位置づける。
- warm startの学習者は、版11で入れたときほどの効果は、版16の6データでは見えない。版11以降の変更（初期の統計の交差検証、監視の基準の引下げ、評価clientの上限の撤廃）で、取り込みが起きにくくなったとみる（未確認）。一部のクラスだけが変わる概念が多いfmnist（warm startが新しいモデルの約6割で選ばれる）を含む新しい4データの比較で、改めて確かめる。

## 新しい4データ（2026-10-31）

- 結果: `results/fedsda-component-ablation-v16/20261030_181023_other-components-seeds0-9`（6構成×160 run、commit `71595383`、研究室サーバ、64 workers）。全960 run完了。
- 条件: fmnist2、fmnist4、rotmnist2、rotmnist4（調整に使っていないデータ）×集約間隔50・100・200・500×seed 0〜9、10 client×5000標本。比べる相手は、値が同じ基準版15のときの構成`fedsda-every-sample-change-interval-search`の結果（`results/fedsda-every-sample-change-interval-search/20261029_172159_new-datasets-seeds0-9`、160 run）。
- 予測を現在のモデルだけで行う構成と、warm startの学習者を外す構成（320 run）は、まだ結果が無い。
- 集計の一時script: Claudeのscratchpadの`new4_analysis.py`（Git管理外）。

| 外したもの（構成） | 精度の差（16セルの範囲、平均） | 全seedで低下 | 通信の比 | MACの比 | 6データの平均 |
|---|---|---|---|---|---|
| 局所のパラメータだけで予測（`fedsda-local-parameter-prediction`） | −2.28〜−1.05、平均−1.87 | 16 | 1.00 | 0.99 | −0.89 |
| 概念固有のadapter（`fedsda-without-concept-specific-adapter`） | −0.99〜−0.18、平均−0.53 | 8 | 0.93〜0.95 | 0.91〜0.97 | −0.60 |
| 集約後の再較正（`fedsda-without-prediction-weight-recalibration`） | −0.49〜−0.17、平均−0.33 | 16 | 1.00 | 1.00 | −0.18 |
| 候補の共有部の固定（`fedsda-without-candidate-validation`） | −0.31〜+0.14、平均−0.11 | 0 | 0.98〜1.01 | 0.94〜1.08 | −0.03 |
| クラス別の監視（全体の損失だけ、`fedsda-overall-loss-esr-detection`） | −0.32〜+0.15、平均−0.05 | 0 | 0.99〜1.00 | 0.83〜1.05 | −0.02 |
| クラス別の機能的なクラスタリング（損失の距離に替える） | −0.41〜+0.18、平均−0.04 | 0 | 0.98〜1.00 | 0.87〜1.00 | −0.01 |

- 読み: 6データと同じ形になった。予測の側の3つは、ほとんどのセルで10 seedすべてが下がる（局所のパラメータと再較正は16セルすべて）。局所のパラメータだけにしたときの低下は、6データの約2倍。固い側の細部は±0.4pp以内で、10 seedすべてが下がるセルは無い。損失の距離のクラスタリングは、fmnistでやや下がり（−0.03〜−0.41）、rotmnistでやや上がる（−0.00〜+0.18）。調整に使っていないデータでも、「精度は柔らかい予測から来る」という読みは変わらない。

## 新しい4データの手法比較（2026-10-31）

- 結果: `results/new-datasets-baselines`（FedDrift〈公式の既定のδ。新しい4データでδを調整していない〉、`fedavg`、`fedavg-local-updates-per-aggregation`の3構成×160 run、commit `71595383`、研究室サーバ）。全480 run完了。FedSDAは上と同じ160 run。

| データ | FedSDA（間隔50/100/200/500） | FedDrift | FedAvg（標本ごと） | FedAvg（集約ごと） |
|---|---|---|---|---|
| fmnist2 | 85.93/85.60/85.05/83.79 | 82.94/83.57/81.85/78.15 | 77.24/77.23/77.04/76.60 | 78.94/78.93/78.71/78.04 |
| fmnist4 | 85.26/84.76/84.28/82.89 | 81.62/82.04/79.73/74.94 | 73.57/73.57/73.42/73.00 | 75.18/75.18/75.04/74.44 |
| rotmnist2 | 94.17/93.53/92.59/90.47 | 91.12/88.20/82.54/78.07 | 89.37/89.17/88.63/87.65 | 91.48/90.80/89.52/86.92 |
| rotmnist4 | 92.32/91.74/90.56/87.79 | 89.68/86.98/81.32/69.06 | 83.25/83.19/82.69/81.62 | 83.30/81.58/78.71/73.49 |

（全体の累積精度、%、10 seedの平均）

- FedSDAは、16セルすべてで3構成より高く、どのセルでも10 seedすべてで上回る。差はFedDriftに対して+2.03〜+18.73pp（平均+6.18）、FedAvg（標本ごと）に対して+2.82〜+11.69pp、FedAvg（集約ごと）に対して+2.69〜+14.30pp。FedDriftとの差は集約間隔が長いほど大きい（rotmnist4で50→500: +2.64→+18.73pp）。
- 資源（FedSDAとの比）: FedDriftの通信は1.59〜3.72倍、MACは0.63〜2.08倍。FedAvgの通信は0.90〜0.94倍、MACは0.34〜0.49倍。

## 新しい4データの、予測を現在のモデルだけで行う構成とwarm startを外す構成（2026-11-01）

- 結果: `results/fedsda-component-ablation-new-datasets-v16/20261031_062435_warm-start-and-single-model-seeds0-9`（2構成×160 run、commit `71595383`、研究室サーバ）。全320 run完了。比べる相手は上と同じ版15の160 run。上の6構成の結果は、`results/fedsda-component-ablation-new-datasets-v16/20261030_181023_other-components-seeds0-9`にも置かれ、`results/fedsda-component-ablation-v16/`の同名の結果とCSVが同じ。
- 集計の一時script: Claudeのscratchpadの`new4_warm_single.py`・`new4_warm_concepts.py`（Git管理外）。

| 外したもの（構成） | 精度の差（16セルの範囲、平均） | 全seedで低下 | 通信の比 | MACの比 | 6データの平均 |
|---|---|---|---|---|---|
| 予測を現在のモデルだけで行う（`fedsda-single-model-prediction`） | −3.21〜−1.27、平均−2.44 | 16 | 1.00 | 0.99 | −1.15 |
| warm startの学習者（`fedsda-without-warm-started-learner`） | −0.78〜+0.12、平均−0.11 | 0 | 0.98〜1.00 | 0.84〜1.01 | −0.03 |

- 予測を現在のモデルだけにすると、16セルすべてで10 seedすべてが下がり、低下は6データの約2倍。新しい4データでも、要素の中で最も大きい。
- warm startを外すと、fmnist4で4間隔とも下がる（−0.20〜−0.78pp。seedごとでは最大−3.37pp）。fmnist2は−0.16〜+0.07pp、rotmnistは−0.04〜+0.12pp。概念0以外で最も低い概念の精度を同じseedの既定と比べると、3pp以上下がったrunはfmnist4で40 run中7（上がったのは1）、fmnist2とrotmnist2では0、rotmnist4では上がったのが1。warm startを外すと、切替の回数（`training_model_switch_detection_metrics.detection_count`）がfmnistで1 runあたり約5〜11回減る（新しいモデルを作る回数が減る）。一部のクラスだけが変わる概念が多いfmnistで、warm startが概念の取り込みを防ぐというNEW-006の見立てと合う。全体が変わるrotmnistでは効かない。
- 6データの読み（固い側の細部は精度の平均ではほとんど効かない）は変わらないが、warm startは、一部のクラスだけが変わる概念が多いデータで、取り込みの防止として効く（fmnist4）。

