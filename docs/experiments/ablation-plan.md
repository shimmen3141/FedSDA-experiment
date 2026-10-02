# 主要構成のablation計画

> 文書の役割: 実験比較の索引
>
> 対象と採用状況: 最終Switching構成の主要ablation・routing比較・既存成果との対応を管理する。
>
> 最終構成の正本: [FedSDAの最終提案構成](../overview/proposed-method.md)

## 実行状況（2026-10-02確認）

最終提案はSwitching SoftRouting構成である。基準、主要8 ablation、4つのrouting比較は、
それぞれ6データセット×5 seed×4集約間隔の120条件が揃っている。CSVとrawの主要指標も一致し、
主要ablationの追加実験は不要である。[成果監査と論文用索引](experiment-results-audit.md)を参照する。

主要8 ablationの正本は`results/results_20260901_113537_main-ablation-suite/<variant>/`、
基準とrouting比較の正本は以下に記載する既存結果である。

## 比較の基準

主要構成は、ランダムスケジュール、6データセット、seed 0--4、各クライアント5000サンプル、
集約間隔 50/100/200/500 で評価する。採用要素と固定設定は
[最終提案構成](../overview/proposed-method.md#2-最終構成の固定設定)を正本とする。

基準結果は
`results/results_20260831_232910_routing-selection/switching-routing`
にある。

## 再利用する既存結果

手法全体のFedDrift比較、採用要素の主要ablation、routing等の設計選択・感度分析を分けて集計する。
旧構成で探索した結果は、最終構成の主要ablationを代替するものとして扱わない。

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
主要ablationとは区別する。必要な主要比較は下記スイートの既存結果で揃っている。

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
現在の成果整理では再実行しない。hard-routingではSoftRoutingに付随するFIFO再較正も無効になる
ため、混合だけを変更した比較として解釈しない。

`class-adwin`はClassESRの構成から検出器だけをClassADWINへ置換する。`overall-adwin`は
そのClassADWIN構成からクラス別系列を除く。提案構成と`overall-adwin`を直接比較すると検出器と
クラス条件付けの二要素が同時に変わるため、両variantを一組として解釈する。

主要ablationに近い旧実験には、旧クラスタリング、別routing、部分データセット、
一部の集約間隔など複数条件が同時に異なるものがある。主要構成の寄与の確認には、条件の揃った
上記スイートとrouting-selectionの結果を使用する。global/meta/meta-switchingは実装ハッシュが
基準と異なるため、監査報告のコード差分に関する留保も付記する。

## 今後の利用

`tools/run_main_ablation_suite.sh`は個別variant名または`all`を指定できる再現用の定義として維持する。
現在のまとめ作業では、実行済みのスイートを再投入する必要はない。意図的な再実験を計画する場合は、
既存manifestとの重複を内部で確認し、通常の重複検査は`error`を維持する。

FedDriftのbyte指標は既定MLP・float32を仮定して復元できた。正式なbyte単位の比較に採用する前に、
実行時のモデル構造・dtypeを確認する。最終Switchingの固定スケジュール、
追加seed、長系列は、論文で必要とする主張に応じて追加を判断する。
