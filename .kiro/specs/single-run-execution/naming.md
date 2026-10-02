# 単一run specの命名

- 状態: 新しい実装名は未確定・未承認。要件承認後の設計で必要な型・関数・引数・状態と役割を一覧化する。
- 新しい名前の承認はgpt-6-lunaレビューと主担当の有用な指摘反映による。ユーザーへ命名だけの承認を再要求しない。
- 現段階では新src・実装を先取りするテストを作らない。

## 引き継ぐ正式名と利用範囲

| 名前 | この段階での扱い |
|---|---|
| `federated_learning_experiments` | 承認済みの新パッケージ |
| `sine2` | 既存ベンチマーク識別子。変更しない |
| `ExperimentRunConditions` | dataset・seed・規模・同期区間の固定条件 |
| `ValidatedExperimentRunSettingsSubset` | 機能の一部を束ねた検証済み部分型。完全な実験実行設定へ昇格させない |

既存の名称・値域の正本は`../configuration-foundation/naming.md`。
調査表の旧名は移植元の位置を示すための表記であり、新APIのaliasではない。
