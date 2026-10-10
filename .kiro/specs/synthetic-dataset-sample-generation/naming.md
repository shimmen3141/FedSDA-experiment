# 命名表: synthetic-dataset-sample-generation

sourceの公開する名前（module、class、公開の関数・メソッド）だけを載せる。確認は、全taskの後の独立レビューで受ける。

## module・class

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `dataset_definitions` / `DatasetDefinition` | `data/` | datasetごとの、入力の特徴数・概念数・クラス数 |
| `observed_sample_generation` | `data/` | 生成器の共通の型、生成器を作る関数、観測列を作る関数 |
| `ObservedSampleGenerator` | 同上 | 概念IDから観測標本を1件作る、生成器の共通の型（Protocol） |
| `sea_sample_generation` / `SeaSampleGenerator` | `data/sea/` | SEA（sea2・sea4）の観測標本の生成 |
| `circle_sample_generation` / `CircleSampleGenerator` | `data/circle/` | CIRCLE-2の観測標本の生成 |

## 関数・メソッド・定数

| 名前 | 役割 |
|---|---|
| `get_dataset_definition` | dataset名から、定義を返す |
| `create_observed_sample_generator` | dataset名と、借りた乱数から、生成器を作る |
| `build_client_observed_streams` | 概念列と生成器から、clientごとの観測列を作る（`build_sine_client_observed_streams`を、名前を変えて移す） |
| `OBSERVED_SAMPLE_GENERATOR_TYPES` | 生成器を作る関数が返す、型の一覧（受け取る側のexact型の検査に使う） |
| `SeaSampleGenerator.generate_sample`・`CircleSampleGenerator.generate_sample` | 既存の`SineSampleGenerator.generate_sample`と同じ名前・同じ役割 |

## 判断が必要な点

- 「dataset definition」は、datasetを決める、手法に依らない値（特徴数・概念数・クラス数）。旧の`DatasetSpec`は、隠れ層の幅と学習率も持つが、新は、手法の設定の束が持つ。
- 「observed sample generator」は、既存の`ObservedSample`・`SineSampleGenerator`の語に合わせる。
