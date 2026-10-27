# Naming

| 名前 | 意味 |
|---|---|
| `fmnist2`・`fmnist4` | Fashion-MNISTの、ラベルの入れ替えのドリフト（概念2つ・4つ） |
| `rotmnist2`・`rotmnist4` | MNISTの、画像の回転のドリフト（概念2つ: 0°・180°、4つ: 0°・90°・180°・270°） |
| `FDE_FASHION_MNIST_DATA_DIR` | Fashion-MNISTのIDXファイルの置き場所の環境変数（既定はリポジトリ直下の`data/fashion-mnist`） |
| `fashion_mnist_sample_generation` | Fashion-MNISTのラベル交換の生成器を定義するmodule。MNIST形式の読込みと生成を扱う`data/mnist/`に置く |
| `FashionMnistSampleGenerator` | `MnistSampleGenerator`のラベル交換・画像選択を継承し、元の画像がFashion-MNISTであることを生成器の型で区別する |
| `rotated_mnist_sample_generation` | MNISTの回転の生成器と、概念ごとの画素の並べ替えを定義するmodule |
| `RotatedMnistSampleGenerator` | MNISTの概念0の標本を同じ乱数で選び、ラベルを保ったまま画素を回す。ラベル交換の生成器と型で区別する |
| `RotatedMnistSampleGenerator.generate_sample` | 概念を検査してから標本を選び、2概念はk=0/2、4概念はk=0/1/2/3の回転を適用する |
| `resolve_fashion_mnist_data_directory` | Fashion-MNISTの環境変数または既定の置き場所を解決する。MNISTの置き場所の解決とは独立 |
| `load_fashion_mnist_training_data` | Fashion-MNISTの学習用IDXを読み、MNISTと共通の形式検査とディレクトリ別cacheを使う |
| `MnistTrainingData` | 既存のMNIST形式の画素とラベルを持つ型を同じ役割で再利用する。Fashion-MNISTでも`digit_labels`は元のクラスラベルを表す |

文書での呼び方: 「Fashion-MNISTのラベルの入れ替え」「MNISTの回転」。実装担当がモジュール・型の名前を足すときは、既存の命名（`MnistSampleGenerator`、`MnistTrainingData`）に合わせ、この表へ足す。
