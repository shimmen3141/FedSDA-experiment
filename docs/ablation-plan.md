# 主要構成のablation計画

## 比較の基準

主要構成は、ランダムスケジュール、6データセット、seed 0--4、5000サンプル、
集約間隔 50/100/200/500 で評価する。手法側は次を基準とする。

- ClassESRによる全体・正解クラス別の損失監視
- 共有バックボーンとrank 8の概念別Residual Adapter
- joint学習と平均勾配
- モデル追従Fixed-ShareによるSwitching SoftRouting
- 集約後のFIFO replay
- `forward_persistent`による新規モデル作成
- `class_functional_confidence`、average linkage、通常merge

基準結果は
`results/results_20260831_232910_routing-selection/switching-routing`
にある。

## 再利用する既存結果

次の比較は、基準と同じ規模・主要条件を持つ既存結果を再利用する。再実験しない。

| 比較軸 | 既存結果 |
|---|---|
| 提案構成のRouting基準 | `results_20260831_232910_routing-selection/switching-routing` |
| SoftRoutingなし | `results_20260831_232910_routing-selection/hard-routing` |
| Global routingへの置換 | `results_20260831_232910_routing-selection/global-routing` |
| Meta routingへの置換 | `results_20260831_232910_routing-selection/meta-routing` |
| 階層Meta-switchingへの置換 | `results_20260830_212508_residual-class-functional-confidence-average-full` |

FedDriftは`results/baselines/feddrift`を固定比較対象として再利用する。
adapter rank、共有表現の学習方法、平均勾配とPCGrad、Meta-switchingのleaderとmixtureは、
既存の感度実験で比較済みである。averageとconnected linkage、距離閾値、クラスタリング無効化、
ClassADWINにも既存結果はあるが、これらは旧Meta-switching基準である。Switchingを基準にした
主要ablationとは区別し、必要な比較は下記スイートで揃える。

## スイートが定義する比較

`tools/run_main_ablation_suite.sh`は、Switching基準と次の比較を同じ条件で実行できる。

| variant | 基準から変える要素 | 確認する寄与 |
|---|---|---|
| `independent` | 共有表現を独立モデルへ変更 | 共有表現全体 |
| `shared-backbone` | Residual Adapterを外す | 概念別補正部分 |
| `hard-routing` | Switching SoftRoutingを外す | 予測時混合 |
| `no-recalibration` | FIFO replayを外す | 集約後再較正 |
| `immediate-creation` | forward検証を外す | 検証付き新規モデル作成 |
| `distance-average` | 統合判定を距離へ変更 | class-functional判定 |
| `overall-esr` | クラス別e-SRを外す | ESRのクラス条件付け |
| `class-adwin` | ClassESRをClassADWINへ置換 | 検出器ファミリ |
| `overall-adwin` | ClassADWINからクラス別系列を外す | ADWINのクラス条件付け |

`reference`は要素を除かない提案構成そのものであり、ablationではない。`global-routing`、
`meta-routing`、`meta-switching-routing`も単一要素の除去ではなく、Switchingを別のrouting方式へ
置き換える感度比較である。これら4方式と`hard-routing`は上表の既存結果で比較済みなので、
現在不足している実験だけを行う場合は再実行しない。

`class-adwin`はClassESRの構成から検出器だけをClassADWINへ置換する。`overall-adwin`は
そのClassADWIN構成からクラス別系列を除く。提案構成と`overall-adwin`を直接比較すると検出器と
クラス条件付けの二要素が同時に変わるため、両variantを一組として解釈する。

未実行の主要ablationに近い過去実験もあるが、旧クラスタリング、別routing、部分データセット、
一部の集約間隔だけなど、複数条件が同時に異なる。そのため、主要構成に対する単一要素の
寄与としては使用しない。一方、上表の再利用対象は、必要な精度・通信・計算・検出・routing・
モデル数指標とrawを備えているため再実験しない。

## 実行方法

個別variant名または`all`を指定する。同じ`FDE_RUN_DIR`を使えば、中断後も完了済みvariantを
飛ばして再開できる。通常の重複検査も`error`のまま有効にする。

```bash
FDE_WORKERS=14 bash tools/run_main_ablation_suite.sh independent shared-backbone no-recalibration immediate-creation distance-average overall-esr class-adwin overall-adwin
```

個別実行例は次のとおりである。

```bash
FDE_WORKERS=14 bash tools/run_main_ablation_suite.sh independent
```
