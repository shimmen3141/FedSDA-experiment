# 公開名

| 名前 | 役割・区別 |
| --- | --- |
| `federated_learning_experiments.sweep.seed_argument_parsing` | CLIのseed表記を整数列へ正規化する。実験設定や乱数生成を扱わない。 |
| `SeedListArgumentAction` | argparseでseed列挙/範囲を受け取るAction。一般の整数掃引軸へ同じ範囲規則を強制しない。 |
