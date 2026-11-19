# Naming

| 名前 | 意味・似た名前との違い |
|---|---|
| `stagger3`・`agrawal4`・`randomrbf4` | 製品の3データ。単一ストリームの版の同じ名前の系列と、概念の中身（分類の関数・中心）が同じ。標本の乱数の列は違う |
| `stagger_sample_generation` | STAGGERの生成器と分類の関数を定義するmodule（`data/stagger/`） |
| `StaggerSampleGenerator` | runが貸すNumPyの乱数でSTAGGERの標本を作る。特徴は値/2 |
| `StaggerSampleGenerator.generate_sample` | 概念を検査してから、size・color・shapeを選び、ラベルを決めて標本を返す |
| `compute_stagger_class_label` | 尺度を写す前のsize・color・shapeと概念から、Riverの`STAGGER`の分類の関数と同じラベルを返す |
| `agrawal_sample_generation` | Agrawalの生成器と分類の関数を定義するmodule（`data/agrawal/`） |
| `AgrawalSampleGenerator` | runが貸すNumPyの乱数で、Riverの`Agrawal`（摂動0.05）と同じ順序・分布の標本を作る。特徴は固定の範囲で0〜1へ写す |
| `AgrawalSampleGenerator.generate_sample` | 概念を検査してから、9特徴を作り、摂動の前の値でラベルを決め、摂動を加えて標本を返す |
| `compute_agrawal_class_label` | 尺度を写す前の9特徴と概念（0〜3）から、Riverの`Agrawal`の分類の関数と同じラベルを返す |
| `random_rbf_sample_generation` | Random RBFの生成器と中心の生成を定義するmodule（`data/random_rbf/`） |
| `RandomRbfSampleGenerator` | 概念ごとの固定の中心から、runが貸すNumPyの乱数で標本を作る。特徴の尺度は変えない |
| `RandomRbfSampleGenerator.generate_sample` | 概念を検査してから、中心を選び、特徴を作り、中心のラベルの標本を返す |
| `generate_random_rbf_centroids` | 概念の番号をseedとするPythonの`random.Random`で、Riverの`RandomRBF._generate_centroids`と同じ中心（位置・ラベル・標準偏差・重み）を返す。runのseedを使わない |
| `RandomRbfCentroid` | 1つの中心の位置・ラベル・標準偏差・重みを持つ不変の記録 |
