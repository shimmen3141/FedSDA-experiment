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
├── routing/<routing>/<dataset>/
└── schedules/<schedule>/<mode>/<dataset>/
```

各データセットディレクトリには集計値の `metrics.csv` とシード別の `.npz` を置く。
各方式の `manifest.json` は、対象データセット、固定パラメータ、掃引パラメータ、欠測データ、
および元実験のパスを記録する。したがって、方式名だけで変更対象を判別し、正確な固定値・
掃引値は同じ階層の manifest で確認する。

`reference/` は現在の比較基準であり、ClassESR、random schedule、forward persistent、
`N_FIFO=30`、`N_forward=10`、距離クラスタリングを基本構成とする。完全な値と掃引範囲は
ルートの `manifest.json` を正本とする。

FedDrift は方式が一つであるため、`results/baselines/feddrift/<dataset>/` の
データセット優先構成を維持する。
