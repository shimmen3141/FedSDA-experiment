# 再来の合成3データでの手法比較（連合学習、基準版16、2026-11-20）

1クライアントの単一ストリームの比較で、FedHASPの木の版が木のアンサンブルに負けたデータ（STAGGER・Agrawal・Random RBF）で、連合学習でもFedHASPが有望かを確かめる。
データの定義は[データの説明](../experiments/dataset-characteristics.md)（spec `stagger-agrawal-random-rbf-recurring-datasets`）。

## 条件と実行
- `fedsda`（FedHASP、既定）・`feddrift`（既定のδ0.10）・`fedavg-local-updates-per-aggregation`（表1のFedAvg）×`stagger3`・`agrawal4`・`randomrbf4`×間隔50・500×seed 0〜9、10 client×5000標本、180 run。
- `results/method-comparison-recurring-synthetic-v16/20261120_111227_fedhasp-feddrift-fedavg-seeds0-9`（uranium、commit `620c735b`、未commitの変更なし、基準版16、120 workersで19分、180 run完了・失敗0）。
- 集計: `tools/research_feddrift_delta_comparison.py compare`（出力`analysis_method_comparison.csv`）と、変化の直後（各clientの真の概念の変化から間隔I件）とそれ以外の分割（`research_post_change_accuracy_breakdown`の`post_change_mask`。各runで正誤の平均が要約の精度と一致した）。
- FedDriftは既定のδ。選んだδでの本評価（[δの調整](feddrift-delta-tuning-recurring-synthetic-v16-20261120.md)）は別に流す。`agrawal4`・間隔50の選んだδは既定と同じ0.10なので、ここの値がそのまま調整後の値。

## 結果（seed 0〜9の平均、%）

| データ | 間隔 | 手法 | 全体 | 直後 | それ以外 | 通信MB/run | client MAC/run |
|---|---|---|---|---|---|---|---|
| stagger3 | 50 | FedHASP | 97.78 | 82.28 | 98.56 | 24.0 | 1.90e10 |
| | | FedDrift | 96.92 | 45.87 | 99.51 | 26.5 | 1.63e10 |
| | | FedAvg | 66.29 | 56.31 | 66.79 | 9.7 | 5.49e9 |
| stagger3 | 500 | FedHASP | 95.42 | 91.92 | 98.06 | 2.9 | 2.15e10 |
| | | FedDrift | 74.80 | 47.32 | 96.03 | 3.4 | 1.28e10 |
| | | FedAvg | 65.86 | 56.39 | 72.96 | 1.0 | 5.49e9 |
| agrawal4 | 50 | FedHASP | 80.78 | 67.68 | 81.42 | 25.2 | 1.95e10 |
| | | FedDrift | 87.98 | 55.37 | 89.58 | 39.0 | 2.08e10 |
| | | FedAvg | 66.27 | 59.43 | 66.61 | 11.3 | 6.12e9 |
| agrawal4 | 500 | FedHASP | 82.22 | 76.64 | 86.32 | 3.3 | 2.22e10 |
| | | FedDrift | 74.34 | 55.34 | 88.28 | 4.1 | 1.49e10 |
| | | FedAvg | 65.96 | 59.49 | 70.64 | 1.1 | 6.12e9 |
| randomrbf4 | 50 | FedHASP | 88.26 | 75.01 | 88.91 | 27.7 | 2.24e10 |
| | | FedDrift | 89.45 | 54.89 | 91.15 | 39.1 | 2.06e10 |
| | | FedAvg | 79.23 | 73.57 | 79.51 | 11.5 | 6.23e9 |
| randomrbf4 | 500 | FedHASP | 83.81 | 79.69 | 86.78 | 3.3 | 2.49e10 |
| | | FedDrift | 76.91 | 61.10 | 88.51 | 4.3 | 1.56e10 |
| | | FedAvg | 76.14 | 71.84 | 79.27 | 1.2 | 6.23e9 |

- 直後の割合は間隔50で約4.7%、間隔500で約43%。経過時間はrunごとに保存されていない（掃引全体で19分）。
- FedHASP−FedDrift（全体、seedで対、上回ったseed）: 間隔50でstagger3 +0.86（10/10）、agrawal4 −7.20（0/10）、randomrbf4 −1.19（0/10）。間隔500でstagger3 +20.62、agrawal4 +7.88、randomrbf4 +6.90（いずれも10/10）。
- 3データ平均のFedHASP−FedDrift: 間隔50で全体−2.51（0/10）、それ以外−3.78（0/10）。間隔500で全体+11.80（10/10）、それ以外−0.55（3/10）。FedAvgには全セル・全seedで上（3データ平均+17.8〜+18.3）。
- 変化の直後を除く精度（それ以外）は、6セルのうち5セルでFedDriftがFedHASPより高い（差は−0.95〜−8.16pp。stagger3の間隔500だけFedHASPが+2.03）。元の10データ（[直後の内訳](post-change-accuracy-breakdown-v16-20261119.md)）では、それ以外でもFedHASPが平均+1.70pp上だった。
- agrawal4・間隔50のFedHASPはseed間の差が大きい（SD 5.7、69.36〜87.57%）。精度の低いseedほど警報が多い（真の変化は1 runで42〜55回に対し、警報はseed 7で71回・87.57%、seed 9で249回・69.36%）。誤った警報が多いseedで精度が落ちている。
- 通信はFedHASPがFedDriftの0.64〜0.91倍。clientのMACは間隔50でFedDriftの0.94〜1.17倍、間隔500で1.49〜1.68倍。

## まとめ（数値をそのまま述べる）
- 1クライアントで負けた3データでも、連合学習では、変化の直後を含む全体の精度で、間隔500では3データともFedDriftを大きく上回り、間隔50ではstagger3だけ上回った。
- 一方、変化の直後を除く安定した期間の精度は、6セル中5セルでFedDriftより低く、間隔50のagrawal4では誤った警報の多いseedで大きく落ちる。元の10データで見えた「安定した期間でも上」は、この3データでは成り立たない。FedHASPの優位は、主に変化の直後の速い切り替えから来ている。
- 安定した期間の差の原因（誤った警報による学習の分断、モデルあたりの学習量、Agrawalの摂動の雑音への検出器の反応）は切り分けていない。
