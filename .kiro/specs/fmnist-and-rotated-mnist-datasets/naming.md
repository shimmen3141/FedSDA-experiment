# Naming

| 名前 | 意味 |
|---|---|
| `fmnist2`・`fmnist4` | Fashion-MNISTの、ラベルの入れ替えのドリフト（概念2つ・4つ） |
| `rotmnist2`・`rotmnist4` | MNISTの、画像の回転のドリフト（概念2つ: 0°・180°、4つ: 0°・90°・180°・270°） |
| `FDE_FASHION_MNIST_DATA_DIR` | Fashion-MNISTのIDXファイルの置き場所の環境変数（既定はリポジトリ直下の`data/fashion-mnist`） |

文書での呼び方: 「Fashion-MNISTのラベルの入れ替え」「MNISTの回転」。実装担当がモジュール・型の名前を足すときは、既存の命名（`MnistSampleGenerator`、`MnistTrainingData`）に合わせ、この表へ足す。
