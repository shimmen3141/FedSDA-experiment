# 命名表: mnist-sample-generation

sourceの公開する名前（module、class、公開の関数・メソッド）だけを載せる。確認は、全taskの後の独立レビューで受ける。

## module・class

| 名前 | 置き場所 | 役割 |
|---|---|---|
| `mnist_training_data` / `MnistTrainingData` | `data/mnist/` | MNISTの学習用データ（画素と、数字のラベル）と、その読込み |
| `mnist_sample_generation` / `MnistSampleGenerator` | `data/mnist/` | MNIST（mnist2・mnist4）の観測標本の生成 |

## 関数・メソッド・field

| 名前 | 役割 |
|---|---|
| `resolve_mnist_data_directory` | MNISTのファイルの置き場所（環境変数、なければ既定）を返す |
| `load_mnist_training_data` | ディレクトリから、学習用データを読む（processの中で使い回す） |
| `MnistTrainingData.pixel_values` | uint8の画素（件数×784） |
| `MnistTrainingData.digit_labels` | 画像の数字（概念で交換する前のラベル） |
| `MnistSampleGenerator.generate_sample` | 既存の生成器の`generate_sample`と同じ名前・同じ役割 |

## 判断が必要な点

- 「training data」は、MNISTの学習用の6万件（検証用の1万件は、読まない）を指す。モデルの学習に使う標本（`training_samples`）とは別なので、`mnist_`を前に付ける。
- 「digit label」は、画像の数字。観測標本の`class_label`（概念で交換した後）と区別する。
