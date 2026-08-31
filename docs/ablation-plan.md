# 主要構成のablation計画

## 比較の基準

主要構成は、ランダムスケジュール、6データセット、seed 0--4、5000サンプル、
集約間隔 50/100/200/500 で評価する。手法側は次を基準とする。

- ClassESRによる全体・正解クラス別の損失監視
- 共有バックボーンとrank 8の概念別Residual Adapter
- joint学習と平均勾配
- Meta-switchingによるRestarting SoftRouting
- 集約後のFIFO replay
- `forward_persistent`による新規モデル作成
- `class_functional_confidence`、average linkage、通常merge

基準結果は
`results/results_20260830_212508_residual-class-functional-confidence-average-full`
にある。

## 再利用する既存結果

次の比較は、基準と同じ規模・主要条件を持つ既存結果を再利用する。再実験しない。

| 比較軸 | 既存結果 |
|---|---|
| 最終候補 | `results_20260830_212508_residual-class-functional-confidence-average-full` |
| ClassESRからClassADWINへの置換 | `results_20260831_054846_residual-class-adwin-functional-confidence-average-full` |
| averageからconnected linkageへの置換 | `results_20260830_212532_residual-class-functional-confidence-connected-full` |
| 距離閾値感度 `0.05` | `results_20260831_075440_class-esr-average-gamma005` |
| 距離閾値感度 `0.20` | `results_20260831_075450_class-esr-average-gamma020` |
| クラスタリング無効化 | `results_20260816_043238_clustering-disabled-full` |

FedDriftは`results/baselines/feddrift`を固定比較対象として再利用する。
adapter rank、共有表現の学習方法、平均勾配とPCGrad、Meta-switchingのleaderとmixtureは、
既存の感度実験で比較済みである。これらは主要構成からの除去ablationとは分けて扱う。

## 不足している比較

`tools/run_main_ablation_suite.sh`は、次の不足分だけを実行する。

| variant | 基準から変える要素 | 確認する寄与 |
|---|---|---|
| `independent` | 共有表現を独立モデルへ変更 | 共有表現全体 |
| `shared-backbone` | Residual Adapterを外す | 概念別補正部分 |
| `hard-routing` | SoftRoutingを外す | 予測時混合 |
| `global-routing` | クラス文脈と上位切替を外す | 文脈依存routing全体 |
| `meta-routing` | 上位switchingだけを外す | 上位switching |
| `no-recalibration` | FIFO replayを外す | 集約後再較正 |
| `immediate-creation` | forward検証を外す | 検証付き新規モデル作成 |
| `distance-average` | 統合判定を距離へ変更 | class-functional判定 |
| `overall-esr` | クラス別e-SRを外す | ESRのクラス条件付け |
| `overall-adwin` | クラス別ADWINを外す | ADWINのクラス条件付け |

過去にはこれらに近い実験もあるが、旧クラスタリング、別routing、部分データセット、
一部の集約間隔だけなど、複数条件が同時に異なる。そのため、主要構成に対する単一要素の
寄与としては使用しない。一方、上表の再利用対象は、必要な精度・通信・計算・検出・routing・
モデル数指標とrawを備えているため再実験しない。

## 実行方法

個別variant名または`all`を指定する。同じ`FDE_RUN_DIR`を使えば、中断後も完了済みvariantを
飛ばして再開できる。通常の重複検査も`error`のまま有効にする。

```bash
FDE_WORKERS=14 bash tools/run_main_ablation_suite.sh all
```

個別実行例は次のとおりである。

```bash
FDE_WORKERS=14 bash tools/run_main_ablation_suite.sh independent
```
