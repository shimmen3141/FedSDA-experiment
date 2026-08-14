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
同じ構成の`feddrift_fixed` schedule結果は
`schedules/feddrift_fixed/residual_adapter_rank8/<dataset>/`へ保存し、FedDrift固定baselineと直接比較できるようにする。

`routing/restarting_soft_routing_class_esr/`には、独立モデル構造へRestarting SoftRoutingを適用した
random schedule・全6データセット・5シード・集約間隔`50/100/200/500`の比較結果を保存する。

`routing/meta_soft_routing_zero_one/`には、Residual Adapter rank 8、joint学習、FIFO再較正を固定し、
予測クラス文脈のMeta routingを0/1損失で更新した結果を保存する。random schedule・全6データセット・
5シード・集約間隔`50/100/200/500`を収録し、`architectures/residual_adapter_rank8/`のGlobal routingと
同一条件で比較できるようにする。

## 単一軸baselineと多因子ablation

`detectors/`や`architectures/`などは、一つの主な比較軸を持つ方式の索引として使う。複数オプションの
組合せを同時に比較する場合、いずれか一つのカテゴリへ無理に分類せず、`studies/<study_id>/`へ置く。
study直下の`manifest.json`には、研究上の比較目的、全variantに共通する設定、比較軸、基準variantを
記録する。各`variants/<variant_id>/manifest.json`には、そのvariant固有のmode・上書き設定・元実験・
欠測データセット・出力ハッシュを記録する。

実行manifestは実験開始時に機械生成する。一方、studyの`title`と`question`はログから推定せず、
実験群をbaselineへ整理する時点でUTF-8のstudy定義へ人間が記述する。定義と各variant manifestから
study manifestを再生成するコマンドは次のとおりである。

```bash
python -m tools.baselines.build_fedsda_study \
  --definition tools/baselines/studies/residual_adapter_routing_ablation.json \
  --study-root results/baselines/fedsda/studies/residual_adapter_routing_ablation
```

整理後の検証では末尾へ`--check`を付け、定義・variant・生成済みmanifestの不一致を検出する。

`studies/residual_adapter_routing_ablation/`は、Residual Adapter rank 8について、hard routing、
Restarting SoftRouting、集約後FIFO再較正の寄与を分離した比較である。全variantはrandom schedule、
5シード、集約間隔`50/200`を共通条件とする。再較正なしSoftRoutingだけSEA2・SEA4が未実験であり、
この欠測はstudy manifestとvariant manifestの両方に明記する。

ディレクトリ名は検索用の短い安定IDに留め、完全な条件を名前へ埋め込まない。条件の正本はmanifest、
数値結果の正本は各データセットの`metrics.csv`、シード別時系列の正本は`.npz`とする。これにより、
条件追加でディレクトリ名が長大化することと、名前に現れない固定値が生じることを避ける。

FedDrift は方式が一つであるため、`results/baselines/feddrift/<dataset>/` の
データセット優先構成を維持する。

## 論文でのablationの使い分け

本文の主要ablationでは、最終候補を基準に有力な要素を一つずつ無効化し、各要素の寄与を示す。
たとえば、Residual Adapter、Restarting SoftRouting、FIFO再較正の有無を同一条件で比較する。
一方、`joint / sequential / frozen`や`distance / confidence / confidence_margin`のような選択肢間比較は、
採用理由を示す設計選択・感度分析として補足または付録に置く。代替選択肢自体が研究上の主張でない限り、
これらを主要ablationの代わりにはしない。
