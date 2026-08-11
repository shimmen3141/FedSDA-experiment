# 固定ベースラインの構成

固定比較結果は `results/baselines/` に保存する。FedSDA は実験オプションを優先した構成に統一し、
各方式の下でデータセットを分ける。

```text
results/baselines/fedsda/
├── manifest.json
├── reference/<dataset>/
├── detectors/<detector>/<dataset>/
├── creation_policies/<policy>/<dataset>/
├── protocols/<protocol>/<dataset>/
├── parameter_profiles/<profile>/<dataset>/
├── architectures/<architecture>/<dataset>/
├── routing/<routing>/<dataset>/
├── studies/<study_id>/
│   ├── manifest.json
│   └── variants/<variant_id>/<dataset>/
└── schedules/<schedule>/<mode>/<dataset>/
```

各データセットディレクトリには集計値の `metrics.csv` とシード別の `.npz` を置く。
各方式の `manifest.json` は、対象データセット、固定パラメータ、掃引パラメータ、欠測データ、
および元実験のパスを記録する。したがって、方式名だけで変更対象を判別し、正確な固定値・
掃引値は同じ階層の manifest で確認する。

`reference/` は現在の比較基準であり、ClassESR、random schedule、forward persistent、
`N_FIFO=30`、`N_forward=10`、距離クラスタリングを基本構成とする。完全な値と掃引範囲は
ルートの `manifest.json` を正本とする。

`architectures/`は検出・モデル作成・クラスタリング等を固定し、モデル構造だけを変更した比較を置く。
`residual_adapter_rank8/`には、joint学習、FIFO再較正、Restarting SoftRoutingを用いた低ランク残差
adapterのrandom schedule・全6データセット・5シード・集約間隔`50/100/200/500`の結果を保存する。

## 単一軸baselineと多因子ablation

`detectors/`や`architectures/`などは、一つの主な比較軸を持つ方式の索引として使う。複数オプションの
組合せを同時に比較する場合、いずれか一つのカテゴリへ無理に分類せず、`studies/<study_id>/`へ置く。
study直下の`manifest.json`には、研究上の比較目的、全variantに共通する設定、比較軸、基準variantを
記録する。各`variants/<variant_id>/manifest.json`には、そのvariant固有のmode・上書き設定・元実験・
欠測データセット・出力ハッシュを記録する。

`studies/residual_adapter_routing_ablation/`は、Residual Adapter rank 8について、hard routing、
Restarting SoftRouting、集約後FIFO再較正の寄与を分離した比較である。全variantはrandom schedule、
5シード、集約間隔`50/200`を共通条件とする。再較正なしSoftRoutingだけSEA2・SEA4が未実験であり、
この欠測はstudy manifestとvariant manifestの両方に明記する。

ディレクトリ名は検索用の短い安定IDに留め、完全な条件を名前へ埋め込まない。条件の正本はmanifest、
数値結果の正本は各データセットの`metrics.csv`、シード別時系列の正本は`.npz`とする。これにより、
条件追加でディレクトリ名が長大化することと、名前に現れない固定値が生じることを避ける。

FedDrift は方式が一つであるため、`results/baselines/feddrift/<dataset>/` の
データセット優先構成を維持する。
