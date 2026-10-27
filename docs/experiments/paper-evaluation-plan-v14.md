# 論文の評価の実験の計画（基準版14、2026-10-28）

論文の主な比較（FedSDA・FedDrift・FedAvg）を、基準版14の同じ条件で取るための計画と、2026-10-28の時点の状態。今の状態の正本として、実験が進んだら更新する。

## 共通の条件
- 10 client、各client 5000標本、集約間隔50・100・200・500、評価のseed 0〜9。
- 主指標は5000標本の全体の累積精度。補助に、期間別の精度（ウォームアップ・回復・定常）と累積の曲線、上り・下りの通信byte、clientのMAC（[研究の課題一覧の§9](../research/research-backlog.md)）。
- 集計: 期間別は`python -m tools.experiments.accuracy_periods`、手法を並べる表とFedDriftのδの選択は`tools/research_feddrift_delta_comparison.py`（`select`・`compare`）。runの対応はデータ・間隔・seedで取る（runの識別子のhashは使わない）。
- データ: 調整に使った6つ（SINE2、CIRCLE2、SEA2、SEA4、MNIST2、MNIST4）と、調整に使っていない4つ（`fmnist2`、`fmnist4`、`rotmnist2`、`rotmnist4`。[データの説明](dataset-characteristics.md)）。
  新しい4データでは、FedSDAを調整しない。

## 手法と構成
| 手法 | 構成 | 備考 |
|---|---|---|
| FedSDA | `fedsda`（基準版14の既定） | 版14の変更はFedDriftだけなので、版13の既定と同じ設定・同じ値 |
| FedDrift（δを調整） | `feddrift`、データ×間隔ごとに選んだδ | 調整用のseed 100〜102で、δ 0.02〜0.20（論文の付録と同じ10値）から全体の累積精度が最大のδ（同じなら小さいδ）。[FedDriftとの差](differences-from-feddrift.md) |
| FedDrift（公式の既定のδ） | `feddrift`（δはデータ別の既定） | 参考。新しい4データは0.10（MNISTの公式の既定） |
| FedAvg | `fedavg-local-updates-per-aggregation`（主表）、`fedavg`（参考） | 主表は配布されたモデルで予測し塊の終わりに学習する形（FedDriftの公式のFedAvgと同じ時間の流れ）。`fedavg`は1標本ごとに局所で適応する参考 |

## 状態（2026-10-28）
| 実験 | 6データ | 新しい4データ |
|---|---|---|
| FedSDA | 済み: `results/fedsda-component-ablation-v13/20261027_044431_core-components-v13-seeds0-9/fedsda`（240 run、版13、版14と同じ設定） | 未実行（下の「新しいデータ」に含む） |
| FedDrift（δの調整、seed 100〜102） | 実行中（uranium、720 run） | 当面は行わない（公式の既定の0.10。時間があるときに他の値も試す、2026-10-28のユーザー判断） |
| FedDrift（調整したδ、seed 0〜9） | 未実行（調整の後、`select`が書き出すコマンド） | — |
| FedDrift（公式の既定のδ、seed 0〜9） | 未確認（`--study feddrift-official-default-v14`のコマンド、240 run） | 未実行（下の「新しいデータ」に含む） |
| FedAvg・FedAvg（塊の終わりに学習） | 未実行（下の「FedAvg」、480 run）。基準版5〜6の頃の結果（`results/method-baselines/`）は間隔・seedが一部だけで、版14との一致も確かめていないので使わない | 未実行（下の「新しいデータ」に含む） |
| 構成要素の比較（ablation） | 済み: [基準版13の構成要素比較](../research/fedsda-component-ablation-v13-20261027.md)（版14でも同じ） | 行わない（必要なら後で） |

## 残りのコマンド（uranium、workersの合計は上限に収まるよう分ける）
新しいデータ（640 run。先に`data/fashion-mnist/`へFashion-MNISTの2ファイルを置き、MD5を照合する）:
```
python run_experiment_sweep.py --configurations fedsda feddrift fedavg fedavg-local-updates-per-aggregation --datasets fmnist2 fmnist4 rotmnist2 rotmnist4 --seeds 0-9 --aggregation-intervals 50 100 200 500 --client-count 10 --per-client-sample-count 5000 --workers 64 --study new-datasets-v14 --label fmnist-rotmnist-seeds0-9 --purpose "調整に使っていない新しいデータ（Fashion-MNISTのラベルの入れ替え、MNISTの回転）で、基準版14の各手法を比べる"
```
FedAvg（6データ、480 run）:
```
python run_experiment_sweep.py --configurations fedavg fedavg-local-updates-per-aggregation --datasets sea2 sea4 sine2 circle2 mnist2 mnist4 --seeds 0-9 --aggregation-intervals 50 100 200 500 --client-count 10 --per-client-sample-count 5000 --workers 64 --study method-comparison-v14 --label fedavg-seeds0-9 --purpose "基準版14のFedAvgの2構成を、6データで評価する"
```
FedDrift（調整したδ）: δの調整が終わったら、`python tools/research_feddrift_delta_comparison.py select --tuning-root <調整の出力先> --output <出力先>/selected_delta.csv`の書き出すコマンド。

## 候補検証を消す作業（IMPROVE-023）との関係
[候補検証を消すspec](../../.kiro/specs/remove-post-alarm-candidate-validation/requirements.md)の後も、候補検証を使わないFedSDAの計算は変えない（goldenと乱数の状態で確かめる）。
保存の形とFedSDAのrunの識別子は変わるが、上の集計はデータ・間隔・seedで対応させるので、版14の結果と、消した後の結果を同じ表で比べられる。過去の掃引の置き場所への再開はできない。
