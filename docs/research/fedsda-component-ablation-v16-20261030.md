# 基準版16の構成要素の比較（6データ、2026-10-30）

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
